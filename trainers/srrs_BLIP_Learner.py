"""
@author: Zhihao Li
@date: 2024-11-11
@homepage: https://zhihaoli.top/
"""
from dassl.engine import TRAINER_REGISTRY, TrainerX
import time
import datetime
import numpy as np
import os
from collections import deque
from sklearn.mixture import GaussianMixture

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights
from dassl.utils import MetricMeter, AverageMeter, load_pretrained_weights
from dassl.modeling.ops.utils import sharpen_prob, create_onehot
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from blip.blip_itm import blip_itm

from datasets.data_manager import SRRSDataManager

_tokenizer = _Tokenizer()
from trainers.loss import GeneralizedCrossEntropy


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)
    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    
    model = clip.build_model(state_dict or model.state_dict())
    
    return model

def load_blip_to_cpu(cfg):
    pretrained = cfg.TRAINER.SRRS.BLIP_PATH
    img_size = cfg.INPUT.SIZE[0]
    blip = blip_itm(pretrained=pretrained, image_size=img_size, vit='base')
    blip = blip.to(device="cpu")
    
    return blip


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x
    

class CLIPPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.SRRS.N_CTX
        ctx_init = cfg.TRAINER.SRRS.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init

        else:
            # random initialization
            if cfg.TRAINER.SRRS.CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = cfg.TRAINER.SRRS.CLASS_TOKEN_POSITION

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,     # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,      # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,     # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,   # (1, name_len, dim)
                        ctx_i,     # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = CLIPPromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image):
        image_features = self.image_encoder(image.type(self.dtype))

        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logits

class BLIPPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, blip_model, features):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.SRRS.N_CTX
        ctx_init = "a photo of a" # .join(["X"] * n_ctx) #cfg.TRAINER.SRRS.CTX_INIT
        n_ctx = len(ctx_init.split(" "))
        tokenizer = blip_model.tokenizer
        embeddings = blip_model.text_encoder.embeddings
        
        classnames = [name.replace("_", " ") for name in classnames]
        feature_lens = [len(_tokenizer.encode(features[name][0])) for name in classnames]
        ctx_prompts = [ctx_init + " " + name + ", " + features[name][0] + '.' for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]

        self.token_sos = []
        self.token_cls = []
        #ctx_vectors_prefix_list = []
        attention_mask = []
        for i, prompt in enumerate(ctx_prompts):
            prompt = prompt.replace("_", " ")	
            ctx_prompt = tokenizer(prompt, padding='max_length', truncation=True, max_length=35, 
                                return_tensors="pt")
            attention_mask.append(ctx_prompt.attention_mask)
            with torch.no_grad():
                ctx_embedding = embeddings(input_ids=ctx_prompt.input_ids)
        
            self.token_sos.append(ctx_embedding[0, : 1, :])
            #ctx_vectors_prefix_list.append(ctx_embedding[0, 1 : 1 + n_ctx, :])
            ctx_vectors_prefix = ctx_embedding[0, 1 : 1 + n_ctx, :]
            self.token_cls.append(ctx_embedding[0, 1 + n_ctx :, :])
        
        self.attention_mask = torch.stack(attention_mask, dim=0)
        #ctx_vectors_prefix = torch.cat(ctx_vectors_prefix_list, dim=0)

        print(f'Initial context: "{ctx_init}"')
        print(f"Max Number of context words (tokens): {max(feature_lens)}")

        self.ctx_prefix = nn.Parameter(ctx_vectors_prefix)  # to be optimized
        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.name_lens = name_lens
        self.feature_lens = feature_lens
        self.class_token_position = cfg.TRAINER.SRRS.CLASS_TOKEN_POSITION
    
    def forward(self):
        ctx_prefix = self.ctx_prefix
        
        prompts = []
        for i in range(self.n_cls):
            token_sos_i = self.token_sos[i]
            token_cls_i = self.token_cls[i]

            prompts.append(torch.cat(
                [
                    token_sos_i.to(ctx_prefix.device),   # (1, dim)
                    ctx_prefix,                          # (n_ctx, dim)
                    token_cls_i.to(ctx_prefix.device),   # (n_cls, dim)
                ],
                dim=0,
            ))

        prompts = torch.stack(prompts, dim=0) # (n_cls, n_ctx, dim)
        return prompts
    

class CustomBLIP(nn.Module):
    def __init__(self, cfg, classnames, blip_model, features):
        super().__init__()
        self.prompt_learner = BLIPPromptLearner(cfg, classnames, blip_model, features)
        self.attention_mask = self.prompt_learner.attention_mask
        self.image_encoder = blip_model.visual_encoder
        self.text_encoder = blip_model.text_encoder
        self.itm_head = blip_model.itm_head
        self.vision_proj = blip_model.vision_proj
        self.text_proj = blip_model.text_proj

        self.blip = blip_model
        self.classnames = classnames
        self.cfg = cfg

    def forward(self, image, refurbished_label=None, match_head='itm'):
        image_embeds = self.image_encoder(image)
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)

        prompts = self.prompt_learner()  # (n_cls, n_ctx, dim)
        if match_head == 'itm':
            refurbished_prompts = prompts[refurbished_label, :, :]
            output = self.text_encoder(encoder_embeds = refurbished_prompts,
                                attention_mask = self.attention_mask[refurbished_label, :].to(refurbished_prompts.device),
                                encoder_hidden_states = image_embeds,
                                encoder_attention_mask = image_atts,        
                                return_dict = True,
                                )
            itm_output = self.itm_head(output.last_hidden_state[:,0,:])           
            return itm_output    # samples_num * 2
       
        elif match_head == 'itc':
            text_output = self.text_encoder(encoder_embeds = prompts,
                                attention_mask = self.attention_mask.to(prompts.device),                    
                                return_dict = True, 
                                mode = 'text')                     
            image_feat = F.normalize(self.vision_proj(image_embeds[:,0,:]),dim=-1)   
            text_feat = F.normalize(self.text_proj(text_output.last_hidden_state[:,0,:]),dim=-1)    
            sim = image_feat @ text_feat.t()        
            return sim

class NegEntropy(object):
    def __call__(self, outputs, single=False):
        probs = torch.softmax(outputs, dim=1)
        probs = probs.clamp(min=1e-10)   # eps
        if single:
            return torch.sum(probs.log() * probs, dim=1)
        else:
            return torch.mean(torch.sum(probs.log() * probs, dim=1))
    
@TRAINER_REGISTRY.register()
class SRRS(TrainerX):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.GCE = GeneralizedCrossEntropy(q=0.5)
        self.temp = cfg.TRAINER.SRRS.TEMP
        self.theta = 1e-6
        self.beta = cfg.TRAINER.SRRS.BETA
        self.beta1 = cfg.TRAINER.SRRS.BETA1
        self.beta2 = cfg.TRAINER.SRRS.BETA2
        self.alpha = cfg.TRAINER.SRRS.ALPHA
        self.loss = deque(maxlen=5)
        self.match_probs = deque(maxlen=5)
        self.negloss = NegEntropy()

    def check_cfg(self, cfg):
        assert cfg.TRAINER.SRRS.PREC in ["fp16", "fp32", "amp"]
    
    def build_data_loader(self):
        """Create essential data-related attributes.

        A re-implementation of this method must create the
        same attributes (self.dm is optional).
        """
        _, preprocess = clip.load(self.cfg.MODEL.BACKBONE.NAME)
        dm = SRRSDataManager(self.cfg, custom_tfm_test=preprocess)

        self.train_loader = dm.train_loader
        self.val_loader = dm.val_loader  # optional, can be None
        self.test_loader = dm.test_loader
        self.num_classes = dm.num_classes
        self.num_source_domains = dm.num_source_domains
        self.lab2cname = dm.lab2cname

        self.dm = dm

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames
        features = self.dm.dataset.features

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        blip_model = load_blip_to_cpu(cfg)

        if cfg.TRAINER.SRRS.PREC == "fp32" or cfg.TRAINER.SRRS.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()
            blip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)
        print("Building custom BLIP")
        self.discriminator = CustomBLIP(cfg, classnames, blip_model, features)      # discriminator
        
        print("Turning off gradients in both the image and the text encoder")
        print("The params need to be learned in Generator:")
        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)
            if param.requires_grad:
                print(name)
        print("The params need to be learned in Discriminator:")
        for name, param in self.discriminator.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)
            if param.requires_grad:
                print(name)

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)
            load_pretrained_weights(self.discriminator.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        self.discriminator.to(self.device)
        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner_generator", self.model.prompt_learner, self.optim, self.sched)

        self.optimD = build_optimizer(self.discriminator.prompt_learner, cfg.OPTIM)
        self.schedD = build_lr_scheduler(self.optimD, cfg.OPTIM)
        self.register_model("prompt_learner_discriminator", self.discriminator.prompt_learner, self.optimD, self.schedD)

        self.scaler = GradScaler() if cfg.TRAINER.SRRS.PREC == "amp" else None
        self.scalerD = GradScaler() if cfg.TRAINER.SRRS.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)
            self.discriminator = nn.DataParallel(self.discriminator)

    def run_epoch(self):
        self.set_model_mode("train")
        losses = MetricMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()

        # co-divide
        match_ID, match_Labels = self.eval_train()
        self.dm.build_dataloader_HQ(match_ID)
        self.train_loader_HQ = self.dm.train_loader_HQ
        print(f"learning samples: {len(self.train_loader_HQ.dataset)} total samples: {self.num_classes*16} rate: {len(self.train_loader_HQ.dataset)/(self.num_classes*16):.2f}%")

        self.num_batches = len(self.train_loader_HQ)

        end = time.time()
        for self.batch_idx, batch in enumerate(self.train_loader_HQ):
            data_time.update(time.time() - end)
            loss_summary = self.forward_backward(batch, match_Labels)
            batch_time.update(time.time() - end)
            losses.update(loss_summary)

            meet_freq = (self.batch_idx + 1) % self.cfg.TRAIN.PRINT_FREQ == 0
            only_few_batches = self.num_batches < self.cfg.TRAIN.PRINT_FREQ
            if meet_freq or only_few_batches:
                nb_remain = 0
                nb_remain += self.num_batches - self.batch_idx - 1
                nb_remain += (
                                     self.max_epoch - self.epoch - 1
                             ) * self.num_batches
                eta_seconds = batch_time.avg * nb_remain
                eta = str(datetime.timedelta(seconds=int(eta_seconds)))

                info = []
                info += [f"epoch [{self.epoch + 1}/{self.max_epoch}]"]
                info += [f"batch [{self.batch_idx + 1}/{self.num_batches}]"]
                info += [f"time {batch_time.val:.3f} ({batch_time.avg:.3f})"]
                info += [f"data {data_time.val:.3f} ({data_time.avg:.3f})"]
                info += [f"{losses}"]
                info += [f"lr {self.get_current_lr():.4e}"]
                info += [f"eta {eta}"]
                print(" ".join(info))

            n_iter = self.epoch * self.num_batches + self.batch_idx
            for name, meter in losses.meters.items():
                self.write_scalar("train/" + name, meter.avg, n_iter)
            self.write_scalar("train/lr", self.get_current_lr(), n_iter)

            end = time.time()

    def eval_train(self):
        self.set_model_mode("eval")
        
        data_len = len(self.train_loader.dataset)

        all_label = torch.zeros(data_len, dtype=torch.long)       # just for analysis
        predict_label = torch.zeros(data_len, dtype=torch.long)   # just for analysis
        gt_label_ID = torch.zeros(data_len, dtype=torch.bool)     # just for analysis

        #--- Step 1: do eval for splitting the dataset
        losses = torch.zeros(data_len)     # for GMM modeling
        with torch.no_grad():
            for self.batch_id, batch in enumerate(self.train_loader):
                input, label, index, impath, _ = self.parse_batch(batch)
                output = self.model(input)

                predict = F.softmax(output, dim=1)
                all_label[index] = label.cpu()
                predict_label[index] = predict.argmax(dim=1).cpu()

                loss = F.cross_entropy(output, label, reduction='none')
                for b in range(input.size(0)):
                    losses[index[b]] = loss[b]
                    if impath[b].rsplit('/', 2)[-2] == self.lab2cname[label[b].item()]:
                        gt_label_ID[index[b]] = True
                    else:
                        gt_label_ID[index[b]] = False

        losses = (losses - losses.min()) / (losses.max() - losses.min())
        self.loss.append(losses)

        if self.cfg.TRAINER.SRRS.AVERAGE_LOSS:  # average loss over last 5 epochs to improve convergence stability
            history = torch.stack(list(self.loss), dim=0)
            input_loss = history.mean(0)
            input_loss = input_loss.reshape(-1, 1)
        else:
            input_loss = losses.reshape(-1, 1)

        # fit a two-component GMM to the loss
        gmm = GaussianMixture(n_components=2, max_iter=100, tol=1e-3, reg_covar=5e-4)
        gmm.fit(input_loss)
        prob = gmm.predict_proba(input_loss)
        mean = gmm.means_.reshape(-1)
        std = gmm.covariances_.reshape(-1)
        idx_clean = mean.argmin()
        idx_noise = mean.argmax()

        mean_clean = torch.tensor(mean[idx_clean]).cuda()
        mean_noise = torch.tensor(mean[idx_noise]).cuda()
        std_clean = torch.tensor(std[idx_clean]).cuda()
        std_noise = torch.tensor(std[idx_noise]).cuda()

        # calculate the thredhold
        alpha_1 = mean_clean + torch.sqrt(-2 * (std_clean ** 2) * torch.log(self.theta * 
            std_clean * torch.sqrt(torch.tensor(2 * torch.pi)) + 1e-8))
        alpha_2 = mean_noise - torch.sqrt(-2 * (std_noise ** 2) * torch.log(self.theta *
            std_noise * torch.sqrt(torch.tensor(2 * torch.pi)) + 1e-8))
	
        #print(f"alpha_1: {alpha_1} alpha_2: {alpha_2}")
        if alpha_1 > alpha_2:
            clean_ID = input_loss < alpha_2.item()
            noisy_ID = input_loss > alpha_1.item()
        else:
            #thres = ( alpha_1 + alpha_2 ) / 2
            #print(f"alpha_1: {alpha_1} alpha_2: {alpha_2} thres: {thres}")
            clean_ID = input_loss < alpha_1.item()
            #clean_ID = input_loss < thres.item()
            noisy_ID = input_loss > alpha_2.item()
            #noisy_ID = input_loss >= thres.item()
        confused_ID = ~(clean_ID | noisy_ID)     # confusing samples

        # clean probalities for the label
        clean_prob = prob[:, idx_clean]

        # # samples ID
        # clean_ID = torch.tensor(clean_prob >= self.beta2, dtype=bool) & (all_label == predict_label)  # high confidence clean samples
        # noisy_ID = torch.tensor(clean_prob <= self.beta1, dtype=bool) & (all_label != predict_label)  # high confidence noisy samples
        # confused_ID = ~(clean_ID | noisy_ID)     # confusing samples
        # visualization
        # if self.epoch % 5 == 0:
        #     self.visualize_samples("Pre Losses", input_loss[clean_ID], input_loss[noisy_ID], input_loss[confused_ID])
        #     self.visualize_samples("GT Losses", input_loss[gt_label_ID], input_loss[~gt_label_ID])
        #     self.visualize_samples("GMM Losses", input_loss[clean_prob > 0.5], input_loss[clean_prob <= 0.5])

        clean_ID = torch.nonzero(clean_ID, as_tuple=True)[0]
        noisy_ID = torch.nonzero(noisy_ID, as_tuple=True)[0]
        confused_ID = torch.nonzero(confused_ID, as_tuple=True)[0]
        
        #--- Step 2: do label refinement for the three subsets
        all_labels = torch.zeros(data_len, dtype=torch.long)
        all_inputs = torch.zeros(data_len, 3, self.cfg.INPUT.SIZE[0], self.cfg.INPUT.SIZE[1], device=self.device)
        with torch.no_grad():
            for self.batch_id, batch in enumerate(self.train_loader):
                input, label, index, _, label_onehot = self.parse_batch(batch)

                # simple prompt learning
                output_G = self.model(input)
                # prompt learning customized by class-specific features
                output_D = self.discriminator(input, match_head='itc')
                
                predict_G = F.softmax(output_G, dim=1)
                predict_D = F.softmax(output_D, dim=1)
                pred_label_G = sharpen_prob(predict_G, self.temp)
                pred_label_D = sharpen_prob(predict_D, self.temp)

                probs = torch.tensor(clean_prob[index], device=self.device).reshape(-1, 1)
                # label refinement
                refined_predict = probs * label_onehot + (1 - probs) * predict_G
                refined_label = sharpen_prob(refined_predict, self.temp)
                # label mixrefinement
                mixrefined_predict = probs * label_onehot + (1 - probs) * (predict_G + predict_D) / 2
                mixrefined_label = sharpen_prob(mixrefined_predict, self.temp)

                for i, id in enumerate(index):
                    if id in clean_ID:
                        # Label absorb of labeled samples
                        all_labels[id] = label[i]
                    elif id in noisy_ID:
                        # Label refinement for unlabeled data
                        all_labels[id] = refined_label[i].argmax()
                    else:
                        # mixrefine confused samples
                        all_labels[id] = mixrefined_label[i].argmax()

                    all_inputs[id] = input[i]

        #--- Step 3: do label evaluation for all pseduo labels
        # evaluate the quality
        with torch.no_grad():
            outputs_eval = self.discriminator(all_inputs, all_labels, 'itm')        # Samples_Num * 2
            match_prob = outputs_eval[:, 1]

        match_prob = (match_prob - match_prob.min()) / (match_prob.max() - match_prob.min())
        self.match_probs.append(match_prob)

        if self.cfg.TRAINER.SRRS.AVERAGE_MATCH:  # average loss over last 5 epochs to improve convergence stability
            history = torch.stack(list(self.match_probs), dim=0)
            input_match_prob = history.mean(0)
            input_match_prob = input_match_prob.reshape(-1, 1)
        else:
            input_match_prob = match_prob.reshape(-1, 1)

        # fit a two-component GMM to the match probality
        input_match_prob = input_match_prob.cpu()
        gmm = GaussianMixture(n_components=2, max_iter=100, tol=1e-3, reg_covar=5e-4)
        gmm.fit(input_match_prob)
        prob = gmm.predict_proba(input_match_prob)
        w = prob[:, gmm.means_.argmax()]
        match_ID = torch.tensor(w > -1)
        # if self.epoch % 5 == 0:
        #     self.visualize_samples("GT eval", w[gt_label_ID], w[~gt_label_ID])
        #     self.visualize_samples("GMM eval", w[match_ID], w[~match_ID])

        match_ID = torch.nonzero(match_ID, as_tuple=True)[0]
        return match_ID, all_labels[match_ID]
        
    def forward_backward(self, batch, match_Labels):
        input, label, index, _, _ = self.parse_batch(batch)
        
        for i, id in enumerate(index):
            label[i] = match_Labels[id.item()]

        prec = self.cfg.TRAINER.SRRS.PREC
        if prec == "amp":
            with autocast():
                output_G = self.model(input)
                output_D = self.discriminator(input, match_head='itc')

                if self.cfg.TRAINER.SRRS.GCE:
                    Lx_G = self.GCE(F.softmax(output_G, dim=1), label)
                    Lx_D = self.GCE(F.softmax(output_D, dim=1), label)
                else:
                    Lx_G = F.cross_entropy(F.softmax(output_G, dim=1), label)
                    Lx_D = F.cross_entropy(F.softmax(output_D, dim=1), label)
                # regularization
                prior = torch.ones(self.num_classes) / self.num_classes
                prior = prior.to(self.device)
                pred_mean_G = torch.softmax(output_G, dim=1).mean(0)
                pred_mean_D = torch.softmax(output_D, dim=1).mean(0)
                penalty_G = torch.sum(prior * torch.log(prior / pred_mean_G))
                penalty_D = torch.sum(prior * torch.log(prior / pred_mean_D))

                loss_G = Lx_G + self.alpha * penalty_G
                loss_D = Lx_D + self.alpha * penalty_D
            self.optim.zero_grad()
            self.scaler.scale(loss_G).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
            
            self.optimD.zero_grad()
            self.scalerD.scale(loss_D).backward()
            self.scalerD.step(self.optimD)
            self.scalerD.update()
        else:
            output_G = self.model(input)
            output_D = self.discriminator(input, match_head='itc')

            if self.cfg.TRAINER.SRRS.GCE:
                Lx_G = self.GCE(F.softmax(output_G, dim=1), label)
                Lx_D = self.GCE(F.softmax(output_D, dim=1), label)
            else:
                Lx_G = F.cross_entropy(F.softmax(output_G, dim=1), label)
                Lx_D = F.cross_entropy(F.softmax(output_D, dim=1), label)
            # regularization
            prior = torch.ones(self.num_classes) / self.num_classes
            prior = prior.to(self.device)
            pred_mean_G = torch.softmax(output_G, dim=1).mean(0)
            pred_mean_D = torch.softmax(output_D, dim=1).mean(0)
            penalty_G = torch.sum(prior * torch.log(prior / pred_mean_G))
            penalty_D = torch.sum(prior * torch.log(prior / pred_mean_D))

            loss_G = Lx_G + self.alpha * penalty_G
            loss_D = Lx_D + self.alpha * penalty_D
            self.model_backward_and_update(loss_G, "prompt_learner_generator")
            self.model_backward_and_update(loss_D, "prompt_learner_discriminator")
        
        loss_summary = {
            "loss_G": loss_G.item(),
            "loss_D": loss_D.item(),
            "acc_G": compute_accuracy(output_G, label)[0].item(),
            "acc_D": compute_accuracy(output_D, label)[0].item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch(self, batch):
        input = batch["img"]
        label = batch["label"]
        index = batch["index"]
        impath = batch["impath"]
        label_onehot = create_onehot(label, self.num_classes).to(self.device)
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label, index, impath, label_onehot

    def visualize_samples(self, data_name, data_clean, data_noisy=None, data_confused=None):
        import matplotlib.pyplot as plt
        import seaborn as sns
        from matplotlib.font_manager import FontProperties
        # settings
        # Fonts style
        font_prop = FontProperties(fname="./fonts/ARLRDBD.TTF")
        font_size = 10
        sns.set(style="whitegrid")
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), dpi=500)

        # histplot for sub-figure1
        sns.histplot(data_clean, color="blue", label="Clean", kde=False, stat="probability", bins=20, alpha=0.6, ax=axes[0])
        if data_noisy is not None:
            sns.histplot(data_noisy, color="red", label="Noisy", kde=False, stat="probability", bins=20, alpha=0.6, ax=axes[0])
        if data_confused is not None:
            sns.histplot(data_confused, color="orange", label="Confused", kde=False, stat="probability", bins=20, alpha=0.6, ax=axes[0])
        axes[0].set_title(f'{data_name} Density (Histogram)', fontproperties=font_prop, fontsize=font_size + 3)
        axes[0].set_xlabel(r'$L_{Divide}$', fontproperties=font_prop, fontsize=font_size + 1)
        axes[0].set_ylabel('Probability Density', fontproperties=font_prop, fontsize=font_size + 1)
        axes[0].legend(prop=font_prop)
        
        # kdeplot for sub-figure2
        sns.kdeplot(data_clean, color="blue", label="Clean", fill=True, ax=axes[1], alpha=0.3)
        if data_noisy is not None:
            sns.kdeplot(data_noisy, color="red", label="Noisy", fill=True, ax=axes[1], alpha=0.3)
        if data_confused is not None:
            sns.kdeplot(data_confused, color="orange", label="Confused", fill=True, ax=axes[1], alpha=0.3)
        axes[1].set_title(f'{data_name} Density (KDE)', fontproperties=font_prop, fontsize=font_size + 3)
        axes[1].set_xlabel(r'$L_{Divide}$', fontproperties=font_prop, fontsize=font_size + 1)
        axes[1].set_ylabel('Probability Density', fontproperties=font_prop, fontsize=font_size + 1)
        axes[1].legend(prop=font_prop)
        
        plt.tight_layout()

        folder_path = os.path.join(self.output_dir, 'images')

        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        plt.savefig(os.path.join(folder_path, f"{data_name}_Probability_Density_Epoch{self.epoch+1}.png"), format="png")
        plt.show()

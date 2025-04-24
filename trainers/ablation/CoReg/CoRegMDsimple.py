"""
@author: Zhihao Li
@date: 2024-11-11
@homepage: https://zhihaoli.top/
"""
from dassl.engine import TRAINER_REGISTRY, TrainerX
import time
from collections import deque
import datetime
import numpy as np
from sklearn.mixture import GaussianMixture
import os.path as osp
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import ( MetricMeter, AverageMeter, mkdir_if_missing, load_pretrained_weights )
from dassl.optim import build_optimizer, build_lr_scheduler
from dassl.modeling.ops.utils import sharpen_prob, create_onehot

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
    

class PromptLearner(nn.Module):
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


class FeaturedPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model, features):
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
        prompts = [prompt_prefix + " " + name + ", " + features[name] + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        #self.suffix_ctx = nn.Parameter(torch.cat([embedding[:, 1 + n_ctx + nl + 1 : 1 + n_ctx + nl + 1 + 13, :] for nl in name_lens]))  # to be optimized
        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.name_lens = name_lens
        self.tokenized_prompts = tokenized_prompts
        self.class_token_position = cfg.TRAINER.SRRS.CLASS_TOKEN_POSITION

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        
        prefix = self.token_prefix
        suffix = self.token_suffix
        #suffix_ctx = self.suffix_ctx

        prompts = []
        for i in range(self.n_cls):
            name_len = self.name_lens[i]
            prefix_i = prefix[i : i + 1, :, :]
            ctx_i = ctx[i : i + 1, :, :]
            class_i = suffix[i : i + 1, : name_len + 1, :]
            #suffix_ctx_i = suffix_ctx[i : i + 1, :, :]
            suffix_i = suffix[i : i + 1, name_len + 1 :, :]
            prompt = torch.cat(
                [
                    prefix_i,     # (1, 1, dim)
                    ctx_i,  # (1, n_ctx//2, dim)
                    class_i,      # (1, name_len, dim)
                    #suffix_ctx_i,  # (1, n_ctx//2, dim)
                    suffix_i,     # (1, *, dim)
                ],
                dim=1,
            )
            prompts.append(prompt)
        prompts = torch.cat(prompts, dim=0)

        return prompts


class FeaturedCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model, features):
        super().__init__()
        self.prompt_learner = FeaturedPromptLearner(cfg, classnames, clip_model, features)
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
    
    

class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
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


def kl_loss_compute(pred, soft_targets, reduce=True):

    kl = F.kl_div(F.log_softmax(pred, dim=1),F.softmax(soft_targets, dim=1), reduction='none')

    if reduce:
        return torch.mean(torch.sum(kl, dim=1))
    else:
        return torch.sum(kl, 1)

class NegEntropy(object):
    def __call__(self, outputs):
        probs = torch.softmax(outputs, dim=1)
        return torch.mean(torch.sum(probs.log() * probs, dim=1))

@TRAINER_REGISTRY.register()
class SRRS(TrainerX):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.GCE = GeneralizedCrossEntropy(q=0.5)
        self.warmup_epoch = cfg.TRAINER.SRRS.WARMUP_EPOCH
        self.temp = cfg.TRAINER.SRRS.TEMP
        self.beta = cfg.TRAINER.SRRS.BETA
        self.alpha1 = cfg.TRAINER.SRRS.ALPHA1
        self.alpha2 = cfg.TRAINER.SRRS.ALPHA2
        self.theta = 0.01
        self.co_lambda = cfg.TRAINER.SRRS.CO_LAMBDA
        self.loss = deque(maxlen=5)
        self.refined_noisy_rates = []

    def check_cfg(self, cfg):
        assert cfg.TRAINER.SRRS.PREC in ["fp16", "fp32", "amp"]

    def build_data_loader(self):
        """Create essential data-related attributes.

        A re-implementation of this method must create the
        same attributes (except self.dm).
        """
        _, preprocess = clip.load(self.cfg.MODEL.BACKBONE.NAME)
        dm = SRRSDataManager(self.cfg, custom_tfm_test=preprocess)
        
        self.train_loader_x = dm.train_loader_x
        self.train_loader_u = dm.train_loader_u  # optional, can be None
        self.val_loader = dm.val_loader  # optional, can be None
        self.test_loader = dm.test_loader
        self.num_classes = dm.num_classes
        self.num_source_domains = dm.num_source_domains
        self.lab2cname = dm.lab2cname  # dict {label: classname}

        self.dm = dm

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames
        features = self.dm.dataset.features

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.SRRS.PREC == "fp32" or cfg.TRAINER.SRRS.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)
        self.fmodel = FeaturedCLIP(cfg, classnames, clip_model, features)

        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)
        for name, param in self.fmodel.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        self.fmodel.to(self.device)
        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)

        self.foptim = build_optimizer(self.fmodel.prompt_learner, cfg.OPTIM)
        self.fsched = build_lr_scheduler(self.foptim, cfg.OPTIM)
        self.register_model("featured_prompt_learner", self.fmodel.prompt_learner, self.foptim, self.fsched)

        self.scaler = GradScaler() if cfg.TRAINER.SRRS.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)
            self.fmodel = nn.DataParallel(self.fmodel)

    def train(self):
        """Generic training loops."""

        print("Start WarmUp")
        for self.epoch in range(0, self.warmup_epoch):
            self.warmup()

        self.before_train()

        for self.epoch in range(self.start_epoch + self.warmup_epoch, self.max_epoch):
            self.before_epoch()
            self.run_epoch()
            self.after_epoch()
        
        self.after_train()

    def after_train(self):
        print("Finish training")
        print(f"* noise rate: {self.refined_noisy_rates}")

        do_test = not self.cfg.TEST.NO_TEST

        if do_test:
            if self.cfg.TEST.FINAL_MODEL == "best_val":
                print("Deploy the model with the best val performance")
                self.load_model(self.output_dir)
            else:
                print("Deploy the last-epoch model")
            self.test()
            
        # Show elapsed time
        elapsed = round(time.time() - self.time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print(f"Elapsed: {elapsed}")

        # Close writer
        self.close_writer()

    def run_epoch(self):
        self.set_model_mode("train")
        losses = MetricMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()

        # co-divide
        if (self.epoch - self.warmup_epoch) % 2 == 0:
            self.refined_labels, self.refined_labels_expand = self.eval_train()

        self.num_batches = len(self.train_loader_x)
        end = time.time()
        for self.batch_idx, batch in enumerate(self.train_loader_x):
            data_time.update(time.time() - end)
            loss_summary = self.forward_backward_mixmatch(batch)
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

    def warmup(self):
        self.set_model_mode("train")
        losses = MetricMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()

        self.num_batches = len(self.train_loader_x)

        end = time.time()
        for self.batch_idx, batch in enumerate(self.train_loader_x):
            data_time.update(time.time() - end)
            loss_summary = self.forward_backward_warmup(batch)
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
        
        data_len = len(self.train_loader_x.dataset)
        #--- Step 1: do eval for splitting the dataset
        losses = torch.zeros(data_len)     # for GMM modeling
        with torch.no_grad():
            for self.batch_id, batch in enumerate(self.train_loader_x):
                input, label, index, _, _, _ = self.parse_batch(batch)
                output_simple, output_featured = 0, 0
                for input_i in input:
                    output_simple_i = self.model(input_i)
                    output_featured_i = self.fmodel(input_i)
                    output_simple += output_simple_i
                    output_featured += output_featured_i      
                output_simple /= len(input)
                output_featured /= len(input)
                probs_simple = torch.softmax(output_simple, dim=1)
                probs_featured = torch.softmax(output_featured, dim=1)

                co_reg = kl_loss_compute(probs_simple, probs_featured, reduce=False) + kl_loss_compute(probs_featured, probs_simple, reduce=False)
                loss_simple = F.cross_entropy(output_simple, label, reduction='none')
                loss_featured = F.cross_entropy(output_featured, label, reduction='none')

                regular_simple = -torch.sum(probs_simple.log() * probs_simple, dim=1)
                regular_featured = -torch.sum(probs_featured.log() * probs_featured, dim=1)

                loss = loss_simple + loss_featured + self.co_lambda * co_reg + regular_simple + regular_featured
                for b in range(label.size(0)):
                    losses[index[b]] = loss[b]

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
        std = np.sqrt(gmm.covariances_).reshape(-1)
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
	
        if alpha_1 > alpha_2:
            clean_ID = (input_loss < alpha_2.item())
            noisy_ID = (input_loss > alpha_1.item())
        else:
            clean_ID = (input_loss < alpha_1.item())
            noisy_ID = (input_loss > alpha_2.item())
        confused_ID = ~(clean_ID | noisy_ID)     # confusing samples

        # clean probalities for the label
        clean_prob = torch.tensor(prob[:, idx_clean], device=self.device).reshape(-1, 1)

        clean_ID = torch.nonzero(clean_ID, as_tuple=True)[0]
        noisy_ID = torch.nonzero(noisy_ID, as_tuple=True)[0]
        confused_ID = torch.nonzero(confused_ID, as_tuple=True)[0]
        
        #--- Step 2: do label refinement for the three subsets
        noisy_labels = torch.zeros(data_len, dtype=torch.long)
        gt_labels = torch.zeros(data_len, dtype=torch.long)
        refined_labels = torch.zeros(data_len, dtype=torch.long)
        refined_labels_expand = torch.zeros((data_len, self.num_classes))
        with torch.no_grad():
            for self.batch_id, batch in enumerate(self.train_loader_x):
                input, label, index, _, label_onehot, gt_label = self.parse_batch(batch)

                output_simple, output_featured = 0, 0
                for input_i in input:
                    output_simple_i = self.model(input_i)
                    output_featured_i = self.fmodel(input_i)
                    output_simple += output_simple_i
                    output_featured += output_featured_i            
                output_simple /= len(input)
                output_featured /= len(input)
                probs_simple = torch.softmax(output_simple, dim=1)
                probs_featured = torch.softmax(output_featured, dim=1)

                probs = clean_prob[index]

                # label refinement
                refined_predict = (probs_simple + probs_featured) / 2
                refined_label = sharpen_prob(refined_predict, self.temp)
                mixrefined_predict = probs * label_onehot + (1 - probs) * (probs_simple + probs_featured) / 2 
                mixrefined_label = sharpen_prob(mixrefined_predict, self.temp)

                refined_batch_labels = label.detach().clone()
                for i, id in enumerate(index):
                    if id in clean_ID:
                        # Label absorb of labeled samples
                        refined_labels[id] = label[i]
                        refined_labels_expand[id] = label_onehot[i]
                    elif id in noisy_ID:
                        # Label refinement for unlabeled data
                        refined_labels[id] = refined_label[i].argmax()
                        refined_labels_expand[id] = refined_label[i]
                    else:
                        # mixrefine confused samples
                        refined_labels[id] = mixrefined_label[i].argmax()
                        refined_labels_expand[id] = mixrefined_label[i]

                    noisy_labels[id] = label[i]
                    gt_labels[id] = gt_label[i]

        # effectiveness for nosiy label refinement
        noisy_rate = sum(noisy_labels != gt_labels) / data_len
        refined_noisy_rate = sum(refined_labels != gt_labels) / data_len
        print(f">>> noisy rate: {noisy_rate:.2f} --> refined noisy rate: {refined_noisy_rate:.2f} <<<")
        self.refined_noisy_rates.append(round(refined_noisy_rate.item(), 2))

        return refined_labels, refined_labels_expand
    

    def forward_backward(self, batch):
        input, label, index, _, label_onehot, _ = self.parse_batch(batch)

        for i, id in enumerate(index):
            label_onehot[i] = self.refined_labels_expand[id.item()]
            label[i] = self.refined_labels[id.item()]

        index = [index] * len(input)
        index = torch.cat(index, 0)
        input = torch.cat(input, 0)

        inputs, labels = [], []
        for i, id in enumerate(index):
            inputs.append(input[i])
            labels.append(self.refined_labels_expand[id.item()])

        all_inputs = torch.stack(inputs, dim=0).to(self.device)
        all_labels = torch.stack(labels, dim=0).to(self.device)
        
        prec = self.cfg.TRAINER.SRRS.PREC
        if prec == "amp":
            with autocast():
                output_simple = self.model(input)
                output_featured = self.fmodel(input)

                predict_simple = F.softmax(output_simple, dim=1)
                predict_featured = F.softmax(output_featured, dim=1)

                co_reg = kl_loss_compute(predict_simple, predict_featured) + kl_loss_compute(predict_featured, predict_simple)
                if self.cfg.TRAINER.SRRS.GCE:
                    loss_simple = self.GCE(output_simple, label)
                    loss_featured = self.GCE(output_featured, label)
                else:
                    loss_simple = F.cross_entropy(output_simple, label)
                    loss_featured = F.cross_entropy(output_featured, label)
                loss = loss_simple + loss_featured + self.co_lambda * co_reg
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            output_simple = self.model(all_inputs)
            probs_simple = torch.softmax(output_simple, dim=1)

            regular_simple = -torch.mean(torch.sum(probs_simple.log() * probs_simple, dim=1))

            # regularization
            prior = torch.ones(self.num_classes) / self.num_classes
            prior = prior.to(self.device)
            pred_mean_simple = torch.softmax(output_simple, dim=1).mean(0)
            penalty = torch.sum(prior * torch.log(prior / pred_mean_simple))

            if self.cfg.TRAINER.SRRS.GCE:
                loss_simple = self.GCE(output_simple, all_labels.argmax(dim=1))
            else:
                loss_simple = F.cross_entropy(output_simple, all_labels)
            loss = loss_simple +  self.alpha2 * penalty
            self.model_backward_and_update(loss)

        loss_summary = {
            "loss": loss.item(),
            "acc": compute_accuracy(output_simple, all_labels.argmax(dim=1))[0].item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary
    
    def forward_backward_mixmatch(self, batch):
        input, label, index, _, label_onehot, _ = self.parse_batch(batch)

        for i, id in enumerate(index):
            label_onehot[i] = self.refined_labels_expand[id.item()]
            label[i] = self.refined_labels[id.item()]

        index = [index] * len(input)
        index = torch.cat(index, 0)
        input = torch.cat(input, 0)

        inputs, labels = [], []
        for i, id in enumerate(index):
            inputs.append(input[i])
            labels.append(self.refined_labels_expand[id.item()])

        all_inputs = torch.stack(inputs, dim=0).to(self.device)
        all_labels = torch.stack(labels, dim=0).to(self.device)

        # # mixmatch
        # l = np.random.beta(self.beta, self.beta)
        # l = max(l, 1 - l)
        # idx = torch.randperm(all_inputs.size(0))

        # input_a, input_b = all_inputs, all_inputs[idx]
        # label_a, label_b = all_labels, all_labels[idx]
        # mixed_input = l * input_a + (1 - l) * input_b
        # mixed_label = l * label_a + (1 - l) * label_b

        # all_inputs = mixed_input.to(self.device)
        # all_labels = mixed_label.to(self.device)
        
        prec = self.cfg.TRAINER.SRRS.PREC
        if prec == "amp":
            with autocast():
                output_simple = self.model(input)
                output_featured = self.fmodel(input)

                predict_simple = F.softmax(output_simple, dim=1)
                predict_featured = F.softmax(output_featured, dim=1)

                co_reg = kl_loss_compute(predict_simple, predict_featured) + kl_loss_compute(predict_featured, predict_simple)
                if self.cfg.TRAINER.SRRS.GCE:
                    loss_simple = self.GCE(output_simple, label)
                    loss_featured = self.GCE(output_featured, label)
                else:
                    loss_simple = F.cross_entropy(output_simple, label)
                    loss_featured = F.cross_entropy(output_featured, label)
                loss = loss_simple + loss_featured + self.co_lambda * co_reg
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            output_simple = self.model(all_inputs)
            output_featured = self.fmodel(all_inputs)

            probs_simple = torch.softmax(output_simple, dim=1)
            probs_featured = torch.softmax(output_featured, dim=1)

            co_reg = kl_loss_compute(probs_simple, probs_featured) + kl_loss_compute(probs_featured, probs_simple)

            regular_simple = -torch.mean(torch.sum(probs_simple.log() * probs_simple, dim=1))
            regular_featured = -torch.mean(torch.sum(probs_featured.log() * probs_featured, dim=1))

            # regularization
            prior = torch.ones(self.num_classes) / self.num_classes
            prior = prior.to(self.device)
            pred_mean_simple = torch.softmax(output_simple, dim=1).mean(0)
            pred_mean_featured = torch.softmax(output_featured, dim=1).mean(0)
            penalty = torch.sum(prior * torch.log(prior / pred_mean_simple)) + torch.sum(prior * torch.log(prior / pred_mean_featured))

            if self.cfg.TRAINER.SRRS.GCE:
                loss_simple = self.GCE(output_simple, all_labels.argmax(dim=1))
                loss_featured = self.GCE(output_featured, all_labels.argmax(dim=1))
            else:
                loss_simple = F.cross_entropy(output_simple, all_labels)
                loss_featured = F.cross_entropy(output_featured, all_labels)
            loss = loss_simple + loss_featured + self.co_lambda * co_reg + self.alpha2 * penalty
            self.model_backward_and_update(loss)

        loss_summary = {
            "loss": loss.item(),
            "acc": (compute_accuracy(output_simple, all_labels.argmax(dim=1))[0].item() + compute_accuracy(output_featured, all_labels.argmax(dim=1))[0].item()) / 2,
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    
    def forward_backward_warmup(self, batch):
        input, label, index, _, _, _ = self.parse_batch(batch)
        negloss = NegEntropy()
        label = [label] * len(input)
        label = torch.cat(label, 0)
        input = torch.cat(input, 0)

        prec = self.cfg.TRAINER.SRRS.PREC
        if prec == "amp":
            with autocast():
                output = self.model(input)
                loss = F.cross_entropy(output, label)
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            output_simple = self.model(input)
            loss_simple = F.cross_entropy(output_simple, label)

            penalty_simple = negloss(output_simple)
            loss = loss_simple + self.alpha1 * penalty_simple
            self.model_backward_and_update(loss)

        loss_summary = {
            "loss": loss.item(),
            "acc": compute_accuracy(output_simple, label)[0].item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch(self, batch):
        input = []
        for k in range(self.cfg.DATALOADER.K):
            keyname = "img"
            if (k + 1) > 1:
                keyname += str(k + 1)
            input.append(batch[keyname].to(self.device))
        label = batch["label"]
        gt_label = batch["gt_label"]
        index = batch["index"]
        impath = batch["impath"]
        label_onehot = create_onehot(label, self.num_classes).to(self.device)
        label = label.to(self.device)
        gt_label = gt_label.to(self.device)
        return input, label, index, impath, label_onehot, gt_label

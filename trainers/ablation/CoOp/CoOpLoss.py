import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
import time
import datetime
from collections import deque
from sklearn.mixture import GaussianMixture

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.optim import build_optimizer, build_lr_scheduler
from dassl.modeling.ops.utils import sharpen_prob, create_onehot
from dassl.utils import ( MetricMeter, AverageMeter, mkdir_if_missing, load_pretrained_weights )
from trainers.utils import update_or_create_csv
from dassl.modeling.ops.utils import create_onehot

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

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


@TRAINER_REGISTRY.register()
class SRRS(TrainerX):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.GCE = GeneralizedCrossEntropy(q=0.5)
        self.csv_path = cfg.TRAINER.SRRS.CSV
        self.fixed_thres = 0.5
        self.loss_epoch = {}

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

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.SRRS.PREC == "fp32" or cfg.TRAINER.SRRS.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.SRRS.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def run_epoch(self):
        self.set_model_mode("train")
        losses = MetricMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()

        if (self.epoch + 1) == 1 or (self.epoch + 1) % 5 == 0:
            self.refined_labels_expand = self.eval_train()

        self.num_batches = len(self.train_loader_x)
        end = time.time()
        for self.batch_idx, batch in enumerate(self.train_loader_x):
            data_time.update(time.time() - end)
            loss_summary = self.forward_backward(batch)
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
        losses = torch.zeros(data_len)
        noisy_labels = torch.zeros(data_len, dtype=torch.long)
        gt_labels = torch.zeros(data_len, dtype=torch.long)
        pred_labels = torch.zeros(data_len, dtype=torch.long)
        with torch.no_grad():
            for batch_id, batch in enumerate(self.train_loader_x):
                input, label, index, _, _, gt_label = self.parse_batch(batch)
                
                output = self.model(input)     
                probs = torch.softmax(output, dim=1)
                loss = F.cross_entropy(probs, label, reduction='none')
                pred_label = probs.argmax(dim=1)

                for b in range(label.size(0)):
                    losses[index[b]] = loss[b]
                    noisy_labels[index[b]] = label[b]
                    gt_labels[index[b]] = gt_label[b]
                    pred_labels[index[b]] = pred_label[b]

        norm_losses = (losses - losses.min()) / (losses.max() - losses.min())
        input_loss = norm_losses.reshape(-1, 1)

        # fit a two-component GMM to the loss
        gmm = GaussianMixture(n_components=2, max_iter=100, tol=1e-3, reg_covar=5e-4)
        gmm.fit(input_loss)
        prob = gmm.predict_proba(input_loss)
        mean = gmm.means_.reshape(-1)
        idx_clean = mean.argmin()
        idx_noise = mean.argmax()

        # clean probalities for the label
        clean_prob = torch.tensor(prob[:, idx_clean])
        noisy_id = clean_prob <= self.fixed_thres

        if "noisy id" not in self.loss_epoch:
            self.loss_epoch["noisy_id"] = noisy_labels != gt_labels
        
        self.loss_epoch[f"ep_{self.epoch+1}_norm_loss"] = norm_losses.numpy()
        self.loss_epoch[f"ep_{self.epoch+1}_origin_loss"] = losses.numpy()
        self.loss_epoch[f"ep_{self.epoch+1}_clean_probs"] = clean_prob.numpy()
        self.loss_epoch[f"ep_{self.epoch+1}_pred_noisy_id"] = pred_labels != gt_labels
        self.loss_epoch[f"ep_{self.epoch+1}_gmm_noisy_id"] = noisy_id
        update_or_create_csv(self.loss_epoch, self.csv_path)

        noisy_id = torch.nonzero(noisy_id, as_tuple=True)[0]

        #--- Step 2: do label refinement for the three subsets
        refined_labels = torch.zeros(data_len, dtype=torch.long)
        refined_labels_expand = torch.zeros((data_len, self.num_classes))
        with torch.no_grad():
            for self.batch_id, batch in enumerate(self.train_loader_x):
                input, label, index, _, label_onehot, gt_label = self.parse_batch(batch)

                output = self.model(input)     
                probs = torch.softmax(output, dim=1)

                clean_p = clean_prob[index].reshape(-1, 1).to(self.device)

                # label refinement
                refined_label = probs

                # label mixrefinement
                mixrefined_label = clean_p * label_onehot + (1 - clean_p) * probs
                for i, id in enumerate(index):
                    if id in noisy_id:
                        # Label refinement for unlabeled data
                        refined_labels[id] = refined_label[i].argmax()
                        refined_labels_expand[id] = refined_label[i]
                    else:
                        # Label absorb of labeled samples
                        refined_labels[id] = label[i]
                        refined_labels_expand[id] = label_onehot[i]

        return refined_labels_expand

    def forward_backward(self, batch):
        image, label, index, _, _, _ = self.parse_batch(batch)
        
        label_expand = []
        for i, id in enumerate(index):
            label_expand.append(self.refined_labels_expand[id.item()])
            #label[i] = self.refined_labels[id.item()]
        label_expand = torch.stack(label_expand, dim=0).to(self.device)
        prec = self.cfg.TRAINER.SRRS.PREC
        if prec == "amp":
            with autocast():
                output = self.model(image)
                if self.cfg.TRAINER.SRRS.GCE:
                    loss = self.GCE(output, label_expand)
                else:
                    loss = F.cross_entropy(output, label_expand)
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            output = self.model(image)
            if self.cfg.TRAINER.SRRS.GCE:
                loss = self.GCE(output, label_expand)
            else:
                loss = F.cross_entropy(output, label_expand)
            self.model_backward_and_update(loss)

        loss_summary = {
            "loss": loss.item(),
            "acc": compute_accuracy(output, label_expand.argmax(dim=1))[0].item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary
    
    def parse_batch(self, batch):
        input = batch["img"].to(self.device)
        label = batch["label"]
        gt_label = batch["gt_label"]
        index = batch["index"]
        impath = batch["impath"]
        label_onehot = create_onehot(label, self.num_classes).to(self.device)
        label = label.to(self.device)
        gt_label = gt_label.to(self.device)
        return input, label, index, impath, label_onehot, gt_label


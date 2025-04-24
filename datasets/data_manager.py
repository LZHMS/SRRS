from dassl.data import DataManager
from dassl.data.data_manager import DatasetWrapper
from dassl.data.samplers import build_sampler
from dassl.utils import Registry, check_availability
from dassl.data.transforms import INTERPOLATION_MODES, build_transform
from dassl.data.datasets import DATASET_REGISTRY,build_dataset
from dassl.utils import read_image
from dassl.data.data_manager import build_data_loader

from tabulate import tabulate
from collections import defaultdict
import torch

def build_data_loader(
    cfg,
    sampler_type="SequentialSampler",
    sampler=None,
    gt_labels=None,
    data_source=None,
    batch_size=64,
    n_domain=0,
    n_ins=2,
    tfm=None,
    is_train=True,
    dataset_wrapper=None,
    tag=None
):
    # Build sampler
    if sampler_type is not None:
        sampler = build_sampler(
            sampler_type,
            cfg=cfg,
            data_source=data_source,
            batch_size=batch_size,
            n_domain=n_domain,
            n_ins=n_ins,
        )
    else:
        sampler = sampler

    if dataset_wrapper is None:
        dataset_wrapper = DatasetWrapper(cfg, data_source, transform=tfm, is_train=is_train)
    else:
        dataset_wrapper = dataset_wrapper(cfg, gt_labels, data_source, transform=tfm, is_train=is_train)

    # Build data loader
    if tag is None:
        data_loader = torch.utils.data.DataLoader(
            dataset_wrapper,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=cfg.DATALOADER.NUM_WORKERS,
            drop_last=is_train and len(data_source) >= batch_size,
            pin_memory=(torch.cuda.is_available() and cfg.USE_CUDA),
        )
    else:
        data_loader = torch.utils.data.DataLoader(
            dataset_wrapper,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=cfg.DATALOADER.NUM_WORKERS,
            drop_last=False,
            pin_memory=(torch.cuda.is_available() and cfg.USE_CUDA),
        )

    return data_loader

class SRRSDataManager(DataManager):
    def __init__(self,
                cfg,
                custom_tfm_train=None,
                custom_tfm_test=None,
                dataset_wrapper=None):
        super().__init__(cfg, custom_tfm_train, custom_tfm_test, dataset_wrapper)

        # Build transform
        if custom_tfm_train is None:
            tfm_train = build_transform(cfg, is_train=True)
        else:
            print("* Using custom transform for training")
            tfm_train = custom_tfm_train

        # Build train_loader_x/clean data
        train_loader_x = build_data_loader(
            cfg,
            sampler_type=cfg.DATALOADER.TRAIN_X.SAMPLER,
            gt_labels=self.dataset.gt_labels,
            data_source=self.dataset.train_x,
            batch_size=cfg.DATALOADER.TRAIN_X.BATCH_SIZE,
            tfm=tfm_train,
            is_train=True,
            dataset_wrapper=DatasetWrapper_XU,
            tag="keep_last"
        )
        self.train_loader_x = train_loader_x
        # for batch_idx, batch in enumerate(self.train_loader_x):
        #     print(f"Batch {batch_idx} - img: {batch['img'].shape}, label: {batch['label']}")



class DatasetWrapper_XU(DatasetWrapper):
    def __init__(self, cfg, gt_labels, data_source, transform=None, is_train=False):
        super().__init__(cfg, data_source, transform, is_train)
        self.K = cfg.DATALOADER.K
        self.gt_labels = gt_labels

    def __getitem__(self, idx):
        item = self.data_source[idx]
        gt_label = self.gt_labels[item.impath]

        output = {
            "label": item.label,
            "gt_label": gt_label,
            "domain": item.domain,
            "impath": item.impath,
            "index": idx
        }

        img0 = read_image(item.impath)

        if self.transform is not None:

            if isinstance(self.transform, (list, tuple)):
                for i, tfm in enumerate(self.transform):
                    img = self._transform_image(tfm, img0)
                    keyname = "img"
                    if (i + 1) > 1:
                        keyname += str(i + 1)
                    output[keyname] = img
            else:
                for k in range(self.K):            # image augmentation for K times
                    img = self._transform_image(self.transform, img0)
                    keyname = "img"
                    if (k + 1) > 1:
                        keyname += str(k + 1)
                    output[keyname] = img
        else:
            output["img"] = img0

        if self.return_img0:
            output["img0"] = self.to_tensor(img0)  # without any augmentation
        return output

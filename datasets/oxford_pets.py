import os
import pickle
import random
import math
from collections import defaultdict

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import read_json, write_json, mkdir_if_missing

from .add_noise import generate_fewshot_dataset_with_symflip_noise, generate_fewshot_dataset_with_pairflip_noise


@DATASET_REGISTRY.register()
class OxfordPets(DatasetBase):

    dataset_dir = "oxford_pets"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "images")
        self.anno_dir = os.path.join(self.dataset_dir, "annotations")
        self.split_path = os.path.join(self.dataset_dir, "split_zhou_OxfordPets.json")

        if os.path.exists(self.split_path):
            train, val, test = self.read_split(self.split_path, self.image_dir)
        else:
            trainval = self.read_data(split_file="trainval.txt")
            test = self.read_data(split_file="test.txt")
            train, val = self.split_trainval(trainval)
            self.save_split(train, val, test, self.split_path, self.image_dir)

        # generate the ground truth labels
        self.gt_labels = {}
        for sample in train:
            self.gt_labels[sample.impath] = sample.label

        # generate the noisy labels
        num_shots = cfg.DATASET.NUM_SHOTS
        num_fp = cfg.DATASET.NUM_FP
        fp_type = cfg.DATASET.FP_TYPE
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot", f"shots_{num_shots}_{fp_type}")
        mkdir_if_missing(self.split_fewshot_dir)
        if num_shots >= 1:
            seed = cfg.SEED
            preprocessed = os.path.join(self.split_fewshot_dir, f"fp_{num_fp}-seed_{seed}.pkl")

            if os.path.exists(preprocessed):
                print(f"Loading preprocessed few-shot data from {preprocessed}")
                with open(preprocessed, "rb") as file:
                    data = pickle.load(file)
                    train, val = data["train"], data["val"]
            else:
                if fp_type == "symflip":
                    train = generate_fewshot_dataset_with_symflip_noise(train, num_shots=num_shots,
                                                                        num_fp=num_fp, seed=seed)
                elif fp_type == "pairflip":
                    train = generate_fewshot_dataset_with_pairflip_noise(train, num_shots=num_shots,
                                                                         num_fp=num_fp, seed=seed)
                else:
                    raise ValueError(f"There is no such type of noise!")
                val = self.generate_fewshot_dataset(val, num_shots=min(num_shots, 4))
                data = {"train": train, "val": val}
                print(f"Saving preprocessed few-shot data to {preprocessed}")
                with open(preprocessed, "wb") as file:
                    pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

        super().__init__(train_x=train, val=val, test=test)

        self.features = {
            "abyssinian": "which has a sleek short-haired cat with large pointed ears features",
            "american bulldog": "which has a muscular sturdy dog with a broad head and powerful body features",
            "american pit bull terrier": "which has a strong athletic dog with a broad chest and short coat features",
            "basset hound": "which has a long-bodied dog with short legs large ears and sad eyes features",
            "beagle": "which has a small hound with a distinctive bark and tricolor fur features",
            "bengal": "which has a wild-looking domestic cat with striking spotted or marbled coat features",
            "birman": "which has a long-haired cat with blue eyes and white gloves on paws features",
            "bombay": "which has a sleek black-coated cat resembling a mini panther with copper eyes features",
            "boxer": "which has a medium-sized muscular dog with a square muzzle and playful energy features",
            "british shorthair": "which has a stocky cat with a round face and dense plush coat features",
            "chihuahua": "which has a tiny dog with large eyes and ears often seen in handbags features",
            "egyptian mau": "which has a spotted short-haired cat known for its speed and agility features",
            "english cocker spaniel": "which has a medium-sized dog with long ears wavy coat and friendly demeanor features",
            "english setter": "which has a graceful dog with feathered coat and distinctive speckled markings features",
            "german shorthaired": "which has an energetic hunting dog with a sleek coat and athletic build features",
            "great pyrenees": "which has a large fluffy dog with thick white fur and protective instincts features",
            "havanese": "which has a small dog with long silky fur and a cheerful affectionate personality features",
            "japanese chin": "which has a small dog with a flat face flowing coat and playful temperament features",
            "keeshond": "which has a medium-sized dog with thick double coat and fox-like face features",
            "leonberger": "which has a giant dog with a lion-like mane and gentle affectionate nature features",
            "maine coon": "which has a large cat with tufted ears thick fur and friendly personality features",
            "miniature pinscher": "which has a small sleek dog with pointed ears and an energetic personality features",
            "newfoundland": "which has a giant gentle dog with a thick coat and strong swimming ability features",
            "persian": "which has a long-haired cat with a flat face and large expressive eyes features",
            "pomeranian": "which has a tiny fluffy dog with a fox-like face and lively personality features",
            "pug": "which has a small wrinkly-faced dog with a curly tail and playful nature features",
            "ragdoll": "which has a large cat with blue eyes soft fur and a relaxed temperament features",
            "russian blue": "which has a sleek short-haired cat with a silvery blue coat and green eyes features",
            "saint bernard": "which has a giant dog with a thick coat known for rescue work in mountains features",
            "samoyed": "which has a fluffy white dog with a permanent smile and friendly nature features",
            "scottish terrier": "which has a small sturdy dog with a wiry coat and distinct beard features",
            "shiba inu": "which has a small fox-like dog with pointed ears curled tail and bold personality features",
            "siamese": "which has a sleek cat with blue eyes and color-pointed short fur features",
            "sphynx": "which has a hairless cat with wrinkled skin large ears and a playful demeanor features",
            "staffordshire bull terrier": "which has a muscular compact dog with a broad head and loyal temperament features",
            "wheaten terrier": "which has a medium-sized dog with a soft wavy coat and friendly nature features",
            "yorkshire terrier": "which has a tiny dog with silky fur and a bold personality features"
        }

    def read_data(self, split_file):
        filepath = os.path.join(self.anno_dir, split_file)
        items = []

        with open(filepath, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                imname, label, species, _ = line.split(" ")
                breed = imname.split("_")[:-1]
                breed = "_".join(breed)
                breed = breed.lower()
                imname += ".jpg"
                impath = os.path.join(self.image_dir, imname)
                label = int(label) - 1  # convert to 0-based index
                item = Datum(impath=impath, label=label, classname=breed)
                items.append(item)

        return items

    @staticmethod
    def split_trainval(trainval, p_val=0.2):
        p_trn = 1 - p_val
        print(f"Splitting trainval into {p_trn:.0%} train and {p_val:.0%} val")
        tracker = defaultdict(list)
        for idx, item in enumerate(trainval):
            label = item.label
            tracker[label].append(idx)

        train, val = [], []
        for label, idxs in tracker.items():
            n_val = round(len(idxs) * p_val)
            assert n_val > 0
            random.shuffle(idxs)
            for n, idx in enumerate(idxs):
                item = trainval[idx]
                if n < n_val:
                    val.append(item)
                else:
                    train.append(item)

        return train, val

    @staticmethod
    def save_split(train, val, test, filepath, path_prefix):
        def _extract(items):
            out = []
            for item in items:
                impath = item.impath
                label = item.label
                classname = item.classname
                impath = impath.replace(path_prefix, "")
                if impath.startswith("/"):
                    impath = impath[1:]
                out.append((impath, label, classname))
            return out

        train = _extract(train)
        val = _extract(val)
        test = _extract(test)

        split = {"train": train, "val": val, "test": test}

        write_json(split, filepath)
        print(f"Saved split to {filepath}")

    @staticmethod
    def read_split(filepath, path_prefix):
        def _convert(items):
            out = []
            for impath, label, classname in items:
                impath = os.path.join(path_prefix, impath)
                item = Datum(impath=impath, label=int(label), classname=classname)
                out.append(item)
            return out

        print(f"Reading split from {filepath}")
        split = read_json(filepath)
        train = _convert(split["train"])
        val = _convert(split["val"])
        test = _convert(split["test"])

        return train, val, test
    
    @staticmethod
    def subsample_classes(*args, subsample="all"):
        """Divide classes into two groups. The first group
        represents base classes while the second group represents
        new classes.

        Args:
            args: a list of datasets, e.g. train, val and test.
            subsample (str): what classes to subsample.
        """
        assert subsample in ["all", "base", "new"]

        if subsample == "all":
            return args

        dataset = args[0]
        labels = set()
        for item in dataset:
            labels.add(item.label)
        labels = list(labels)
        labels.sort()
        n = len(labels)
        # Divide classes into two halves
        m = math.ceil(n / 2)

        print(f"SUBSAMPLE {subsample.upper()} CLASSES!")
        if subsample == "base":
            selected = labels[:m]  # take the first half
        else:
            selected = labels[m:]  # take the second half
        relabeler = {y: y_new for y_new, y in enumerate(selected)}

        output = []
        for dataset in args:
            dataset_new = []
            for item in dataset:
                if item.label not in selected:
                    continue
                item_new = Datum(
                    impath=item.impath,
                    label=relabeler[item.label],
                    classname=item.classname
                )
                dataset_new.append(item_new)
            output.append(dataset_new)

        return output

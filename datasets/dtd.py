import os
import pickle
import random

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import listdir_nohidden, mkdir_if_missing
from .add_noise import generate_fewshot_dataset_with_symflip_noise, generate_fewshot_dataset_with_pairflip_noise

from .oxford_pets import OxfordPets


@DATASET_REGISTRY.register()
class DescribableTextures(DatasetBase):

    dataset_dir = "dtd"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "images")
        self.split_path = os.path.join(self.dataset_dir, "split_zhou_DescribableTextures.json")

        if os.path.exists(self.split_path):
            train, val, test = OxfordPets.read_split(self.split_path, self.image_dir)
        else:
            train, val, test = self.read_and_split_data(self.image_dir)
            OxfordPets.save_split(train, val, test, self.split_path, self.image_dir)

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
                    # for item in train:
                    #     print(item.label, item.impath, item.classname)
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
        
        subsample = cfg.DATASET.SUBSAMPLE_CLASSES
        train, val, test = OxfordPets.subsample_classes(train, val, test, subsample=subsample)

        super().__init__(train_x=train, val=val, test=test)
    
    @property
    def features(self):
        return {
            'banded': 'which has alternating parallel bands with varying widths and colors features',
            'blotchy': 'which has irregular patches of uneven tones and unclear shapes features',
            'braided': 'which has interwoven strands creating a complex and tight pattern features',
            'bubbly': 'which has round raised bubbles scattered evenly across the surface features',
            'bumpy': 'which has uneven surface with small bumps creating noticeable tactile variation features',
            'chequered': 'which has alternating square tiles with contrasting colors forming a grid features',
            'cobwebbed': 'which has thin web-like lines crossing each other forming a net structure features',
            'cracked': 'which has visible cracks with rough uneven edges spread across the surface features',
            'crosshatched': 'which has overlapping lines forming a dense grid-like crisscross pattern features',
            'crystalline': 'which has sharp angular facets forming a geometric crystalline structure features',
            'dotted': 'which has small circular dots scattered evenly or randomly throughout the surface features',
            'fibrous': 'which has thread-like fibers running parallel or intertwined across the surface features',
            'flecked': 'which has small specks scattered randomly across an uneven surface with variation features',
            'freckled': 'which has tiny irregular spots scattered unevenly across the surface in random features',
            'frilly': 'which has delicate ruffled edges creating decorative flowing shapes across the surface features',
            'gauzy': 'which has thin translucent fabric-like material with a soft flowing surface texture features',
            'grid': 'which has intersecting straight lines creating regular evenly spaced square patterns features',
            'grooved': 'which has deep linear grooves running parallel across the surface in a repeating pattern features',
            'honeycombed': 'which has hexagonal cells forming a regular honeycomb pattern across the surface features',
            'interlaced': 'which has interwoven elements forming a complex layered pattern across the surface features',
            'knitted': 'which has looped threads tightly connected to form a textured fabric-like surface features',
            'lacelike': 'which has delicate openwork patterns resembling lace spread evenly across the surface features',
            'lined': 'which has straight parallel lines running consistently across the surface in regular intervals features',
            'marbled': 'which has swirling veins of different colors creating a marble-like pattern surface features',
            'matted': 'which has a dense flattened texture without much shine or smoothness across the surface features',
            'meshed': 'which has interconnected open spaces forming a mesh-like structure across the entire surface features',
            'paisley': 'which has curved teardrop-shaped motifs with intricate inner details across the surface features',
            'perforated': 'which has small evenly spaced holes punctured through the surface forming a pattern features',
            'pitted': 'which has small depressions spread randomly across the surface creating an uneven texture features',
            'pleated': 'which has evenly folded pleats running parallel to each other across the surface features',
            'polka-dotted': 'which has round dots evenly spaced across the surface in regular repeating patterns features',
            'porous': 'which has small pores spread evenly across the surface creating a breathable texture features',
            'potholed': 'which has irregular holes of varying sizes spread randomly across the surface features',
            'scaly': 'which has overlapping scales forming a textured pattern across the surface like reptiles features',
            'smeared': 'which has irregular smudges spread unevenly across the surface creating blurry patterns features',
            'spiralled': 'which has continuous spiral lines radiating outward from a central point on the surface features',
            'sprinkled': 'which has small particles or dots scattered randomly across the surface in varying patterns features',
            'stained': 'which has uneven patches of color spread irregularly across the surface creating distinct marks features',
            'stratified': 'which has distinct layers of material stacked on top of each other across the surface features',
            'striped': 'which has parallel stripes of varying widths running across the surface in alternating patterns features',
            'studded': 'which has small raised studs evenly spaced across the surface creating a textured pattern features',
            'swirly': 'which has swirling lines or shapes that curve around each other across the surface features',
            'veined': 'which has thin lines resembling veins running throughout the surface in intricate patterns features',
            'waffled': 'which has a repeating grid-like pattern forming small square indentations across the surface features',
            'woven': 'which has interlaced threads or fibers creating a tight textured pattern across the surface features',
            'wrinkled': 'which has uneven folds and creases spread across the surface creating a textured appearance features',
            'zigzagged': 'which has sharp angular lines zigzagging across the surface in regular repeating patterns features'
        }

    @staticmethod
    def read_and_split_data(image_dir, p_trn=0.5, p_val=0.2, p_tst=0.3, ignored=[], new_cnames=None):
        # The data are supposed to be organized into the following structure
        # =============
        # images/
        #     dog/
        #     cat/
        #     horse/
        # =============
        categories = listdir_nohidden(image_dir)
        categories = [c for c in categories if c not in ignored]
        categories.sort()

        print(f"Splitting into {p_trn:.0%} train, {p_val:.0%} val, and {p_tst:.0%} test")

        def _collate(ims, y, c):
            items = []
            for im in ims:
                item = Datum(impath=im, label=y, classname=c)  # is already 0-based
                items.append(item)
            return items

        train, val, test = [], [], []
        for label, category in enumerate(categories):
            category_dir = os.path.join(image_dir, category)
            images = listdir_nohidden(category_dir)
            images = [os.path.join(category_dir, im) for im in images]
            random.shuffle(images)
            n_total = len(images)
            n_train = round(n_total * p_trn)
            n_val = round(n_total * p_val)
            n_test = round(n_total * p_tst)
            assert n_train > 0 and n_val > 0 and n_test > 0

            if new_cnames is not None and category in new_cnames:
                category = new_cnames[category]

            train.extend(_collate(images[:n_train], label, category))
            val.extend(_collate(images[n_train: n_train + n_val], label, category))
            test.extend(_collate(images[n_train + n_val: n_train + n_val + n_test], label, category))

        return train, val, test

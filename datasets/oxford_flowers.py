import os
import random
import pickle
from scipy.io import loadmat
from collections import defaultdict

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import read_json, mkdir_if_missing
from .add_noise import generate_fewshot_dataset_with_symflip_noise, generate_fewshot_dataset_with_pairflip_noise

from .oxford_pets import OxfordPets


@DATASET_REGISTRY.register()
class OxfordFlowers(DatasetBase):

    dataset_dir = "oxford_flowers"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "jpg")
        self.label_file = os.path.join(self.dataset_dir, "imagelabels.mat")
        self.lab2cname_file = os.path.join(self.dataset_dir, "cat_to_name.json")
        self.split_path = os.path.join(self.dataset_dir, "split_zhou_OxfordFlowers.json")

        if os.path.exists(self.split_path):
            train, val, test = OxfordPets.read_split(self.split_path, self.image_dir)
        else:
            train, val, test = self.read_data()
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

        super().__init__(train_x=train, val=val, test=test)

        self.features = {
                'pink primrose': 'which has delicate pink petals with a yellow center features',
                'hard-leaved pocket orchid': 'which has tough green leaves and small bright blooms features',
                'canterbury bells': 'which has bell-shaped flowers in shades of purple and blue features',
                'sweet pea': 'which has soft petals and a pleasant sweet fragrance features',
                'english marigold': 'which has bright orange petals arranged in a circular pattern features',
                'tiger lily': 'which has orange petals with black spots and curved shapes features',
                'moon orchid': 'which has white petals with a central yellow spot features',
                'bird of paradise': 'which has orange blue petals resembling a bird in flight features',
                'monkshood': 'which has deep purple hooded flowers with long stems features',
                'globe thistle': 'which has spiky round blue flowers and silvery leaves features',
                'snapdragon': 'which has tall stems with bright snap-like flowers blooming features',
                'colt\'s foot': 'which has yellow flowers resembling dandelions in early spring features',
                'king protea': 'which has large pink petals and a spiky central structure features',
                'spear thistle': 'which has spiny leaves and purple thistle-shaped flowers features',
                'yellow iris': 'which has yellow petals with long sword-shaped green leaves features',
                'globe-flower': 'which has round bright yellow flowers with layered petals features',
                'purple coneflower': 'which has purple petals surrounding a spiky orange center features',
                'peruvian lily': 'which has spotted colorful petals with intricate patterns features',
                'balloon flower': 'which has star-shaped blue flowers and inflated buds features',
                'giant white arum lily': 'which has large white flowers with a yellow central spike features',
                'fire lily': 'which has bright red petals with slightly curled edges features',
                'pincushion flower': 'which has small round blooms with spiky petals arranged symmetrically features',
                'fritillary': 'which has checkered patterned petals in purple and white tones features',
                'red ginger': 'which has tall spikes with vibrant red tropical flowers features',
                'grape hyacinth': 'which has clusters of small purple bell-shaped flowers features',
                'corn poppy': 'which has delicate red petals with a black center features',
                'prince of wales feathers': 'which has tall feathery green leaves and tiny purple blooms features',
                'stemless gentian': 'which has deep blue star-shaped flowers with a short stem features',
                'artichoke': 'which has thick green petals surrounding a spiky purple center features',
                'sweet william': 'which has clusters of small flowers in white pink red features',
                'carnation': 'which has ruffled petals in a variety of bright colors features',
                'garden phlox': 'which has dense clusters of small five-petaled flowers features',
                'love in the mist': 'which has delicate blue flowers surrounded by feathery leaves features',
                'mexican aster': 'which has bright daisy-like flowers with yellow centers features',
                'alpine sea holly': 'which has spiky blue flowers with silver-tipped petals features',
                'ruby-lipped cattleya': 'which has large purple petals with a deep red center features',
                'cape flower': 'which has vibrant pink petals with dark centers and yellow stamens features',
                'great masterwort': 'which has tiny white flowers arranged in a dome-shaped cluster features',
                'siam tulip': 'which has pink bracts surrounding small white and yellow flowers features',
                'lenten rose': 'which has nodding flowers in shades of pink or white features',
                'barbeton daisy': 'which has large daisy-like flowers with vibrant red petals features',
                'daffodil': 'which has yellow trumpet-shaped flowers with long green leaves features',
                'sword lily': 'which has tall spikes with rows of colorful trumpet-shaped flowers features',
                'poinsettia': 'which has bright red leaves and small yellow flowers features',
                'bolero deep blue': 'which has deep blue petals with a delicate round shape features',
                'wallflower': 'which has clusters of bright orange and yellow fragrant flowers features',
                'marigold': 'which has bright orange or yellow flowers with many layered petals features',
                'buttercup': 'which has shiny yellow petals and grows close to the ground features',
                'oxeye daisy': 'which has white petals surrounding a bright yellow center features',
                'common dandelion': 'which has yellow flower heads that turn into white seed puffs features',
                'petunia': 'which has trumpet-shaped flowers in a variety of bright colors features',
                'wild pansy': 'which has heart-shaped petals with vivid purple and yellow features',
                'primula': 'which has clusters of bright flowers with green basal leaves features',
                'sunflower': 'which has large yellow petals surrounding a dark central disk features',
                'pelargonium': 'which has brightly colored flowers and aromatic leaves features',
                'bishop of llandaff': 'which has dark red flowers with yellow-tipped petals features',
                'gaura': 'which has delicate white or pink flowers on long slender stems features',
                'geranium': 'which has rounded leaves and clusters of brightly colored flowers features',
                'orange dahlia': 'which has large orange blooms with many layered petals features',
                'pink-yellow dahlia': 'which has pink petals with yellow centers and curved shapes features',
                'cautleya spicata': 'which has tall red spikes with yellow tubular flowers features',
                'japanese anemone': 'which has white or pink flowers with golden yellow centers features',
                'black-eyed susan': 'which has bright yellow petals with a dark brown center features',
                'silverbush': 'which has small white flowers with silver-tinted leaves features',
                'californian poppy': 'which has orange petals that form a cup-like shape features',
                'osteospermum': 'which has daisy-like flowers with purple centers and white petals features',
                'spring crocus': 'which has cup-shaped purple or yellow flowers blooming in spring features',
                'bearded iris': 'which has ruffled petals in shades of purple and yellow features',
                'windflower': 'which has delicate white or pink petals and green foliage features',
                'tree poppy': 'which has large white petals with a yellow center features',
                'gazania': 'which has daisy-like flowers with bright orange or yellow petals features',
                'azalea': 'which has clusters of brightly colored trumpet-shaped flowers features',
                'water lily': 'which has large flat leaves and flowers floating on water features',
                'rose': 'which has layered petals in various colors and a strong fragrance features',
                'thorn apple': 'which has large white or purple trumpet-shaped flowers features',
                'morning glory': 'which has heart-shaped leaves and funnel-shaped flowers features',
                'passion flower': 'which has intricate petals with a star-like appearance features',
                'lotus': 'which has large flat leaves and pink or white flowers features',
                'toad lily': 'which has speckled petals in purple and white colors features',
                'anthurium': 'which has bright red spathes and a yellow central spike features',
                'frangipani': 'which has white or pink petals with a yellow center features',
                'clematis': 'which has star-shaped flowers in purple pink or white features',
                'hibiscus': 'which has large trumpet-shaped flowers in bright colors features',
                'columbine': 'which has spurred petals in shades of purple pink or blue features',
                'desert-rose': 'which has thick succulent stems and bright pink flowers features',
                'tree mallow': 'which has large purple flowers and tall bushy stems features',
                'magnolia': 'which has large white or pink petals and a strong fragrance features',
                'cyclamen': 'which has heart-shaped leaves and bright upturned flowers features',
                'watercress': 'which has small white flowers and grows near freshwater streams features',
                'canna lily': 'which has large red orange or yellow blooms with broad leaves features',
                'hippeastrum': 'which has large trumpet-shaped flowers in red or pink features',
                'bee balm': 'which has clusters of tubular red or purple flowers features',
                'ball moss': 'which has thin green leaves growing in spherical clusters features',
                'foxglove': 'which has tall spikes of tubular purple or pink flowers features',
                'bougainvillea': 'which has bright pink or purple papery bracts and small flowers features',
                'camellia': 'which has large glossy leaves and showy pink or white flowers features',
                'mallow': 'which has soft pink or purple flowers with five distinct petals features',
                'mexican petunia': 'which has purple trumpet-shaped flowers and long green leaves features',
                'bromelia': 'which has spiky green leaves and bright red central flowers features',
                'blanket flower': 'which has daisy-like red and yellow flowers with flat petals features',
                'trumpet creeper': 'which has long tubular orange flowers and climbing vines features',
                'blackberry lily': 'which has orange petals with dark spots and slender leaves features'
            }

    def read_data(self):
        tracker = defaultdict(list)
        label_file = loadmat(self.label_file)["labels"][0]
        for i, label in enumerate(label_file):
            imname = f"image_{str(i + 1).zfill(5)}.jpg"
            impath = os.path.join(self.image_dir, imname)
            label = int(label)
            tracker[label].append(impath)

        print("Splitting data into 50% train, 20% val, and 30% test")

        def _collate(ims, y, c):
            items = []
            for im in ims:
                item = Datum(impath=im, label=y - 1, classname=c)  # convert to 0-based label
                items.append(item)
            return items

        lab2cname = read_json(self.lab2cname_file)
        train, val, test = [], [], []
        for label, impaths in tracker.items():
            random.shuffle(impaths)
            n_total = len(impaths)
            n_train = round(n_total * 0.5)
            n_val = round(n_total * 0.2)
            n_test = n_total - n_train - n_val
            assert n_train > 0 and n_val > 0 and n_test > 0
            cname = lab2cname[str(label)]
            train.extend(_collate(impaths[:n_train], label, cname))
            val.extend(_collate(impaths[n_train : n_train + n_val], label, cname))
            test.extend(_collate(impaths[n_train + n_val :], label, cname))

        return train, val, test
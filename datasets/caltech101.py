import os
import pickle

from dassl.data.datasets import DATASET_REGISTRY, DatasetBase
from dassl.utils import mkdir_if_missing

from .oxford_pets import OxfordPets
from .dtd import DescribableTextures as DTD
from .add_noise import generate_fewshot_dataset_with_symflip_noise,generate_fewshot_dataset_with_pairflip_noise

IGNORED = ["BACKGROUND_Google", "Faces_easy"]
NEW_CNAMES = {
    "airplanes": "airplane",
    "Faces": "face",
    "Leopards": "leopard",
    "Motorbikes": "motorbike",
}


@DATASET_REGISTRY.register()
class Caltech101(DatasetBase):

    dataset_dir = "caltech-101"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "101_ObjectCategories")
        self.split_path = os.path.join(self.dataset_dir, "split_zhou_Caltech101.json")

        if os.path.exists(self.split_path):
            train, val, test = OxfordPets.read_split(self.split_path, self.image_dir)
        else:
            train, val, test = DTD.read_and_split_data(self.image_dir, ignored=IGNORED, new_cnames=NEW_CNAMES)
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

        super().__init__(train_x=train, train_u=None, val=val, test=test)

        self.features = {
                "face": "which has human facial features including eyes nose mouth and overall expression features",
                "leopard": "which has spotted big cat with sleek fur and powerful muscular build features",
                "motorbike": "which has two-wheeled vehicle with engine handlebars and distinctive design elements features",
                "accordion": "which has musical instrument with bellows and keys producing rich melodic sounds features",
                "airplane": "which has aerial vehicle with wings engines and streamlined body for flight features",
                "anchor": "which has heavy metal device used to secure boats to the seabed features",
                "ant": "which has small industrious insect with segmented body and strong social structure features",
                "barrel": "which has cylindrical container typically made of wood used for storage features",
                "bass": "which has deep-sounding musical instrument or large aquatic fish species features",
                "beaver": "which has large rodent with flat tail and strong teeth for building features",
                "binocular": "which has optical device with two lenses for enhanced distant viewing features",
                "bonsai": "which has miniature cultivated tree artfully shaped and maintained in pots features",
                "brain": "which has complex organ responsible for thought memory and neural processing features",
                "brontosaurus": "which has extinct long-necked dinosaur known for massive size and herbivorous diet features",
                "buddha": "which has symbolic representation of enlightened figure in various cultural contexts features",
                "butterfly": "which has insect with colorful wings undergoing metamorphosis from caterpillar stage features",
                "camera": "which has device for capturing photographs or recording videos through a lens features",
                "cannon": "which has large artillery weapon designed to launch projectiles over distances features",
                "car side": "which has view showing the lateral profile and details of an automobile features",
                "ceiling fan": "which has rotating mechanical device mounted on ceilings to circulate air features",
                "cellphone": "which has portable electronic device used for communication and accessing applications features",
                "chair": "which has seating furniture typically with a back and four legs features",
                "chandelier": "which has decorative hanging light fixture adorned with crystals or glass elements features",
                "cougar body": "which has large feline with muscular build agile movements and tawny fur features",
                "cougar face": "which has facial features of cougar including eyes ears and muzzle shape features",
                "crab": "which has crustacean with a hard exoskeleton and pincers for defense features",
                "crayfish": "which has freshwater crustacean resembling small lobsters with slender bodies features",
                "crocodile": "which has large reptile with powerful jaws armored skin and aquatic habits features",
                "crocodile head": "which has distinctive head shape of crocodile featuring strong jaws and eyes features",
                "cup": "which has small bowl-shaped container typically used for drinking beverages features",
                "dalmatian": "which has spotted dog breed known for its distinctive coat and active nature features",
                "dollar bill": "which has paper currency featuring prominent national symbols and portraits features",
                "dolphin": "which has intelligent marine mammal known for playful behavior and echolocation features",
                "dragonfly": "which has insect with elongated body large eyes and two pairs of wings features",
                "electric guitar": "which has stringed musical instrument powered by electronic amplification for diverse sounds features",
                "elephant": "which has massive mammal with trunk tusks and thick wrinkled skin features",
                "emu": "which has large flightless bird native to Australia with long neck and legs features",
                "euphonium": "which has brass musical instrument resembling a small tuba with mellow tone features",
                "ewer": "which has decorative pitcher with a wide mouth and handle used for pouring features",
                "ferry": "which has boat or ship designed to carry passengers and vehicles across water features",
                "flamingo": "which has tall wading bird with pink feathers long legs and curved neck features",
                "flamingo head": "which has distinctive head shape of flamingo featuring curved beak and slender features features",
                "garfield": "which has famous comic strip cat known for laziness and love of lasagna features",
                "gerenuk": "which has long-necked antelope with slender legs and ability to stand on hind legs features",
                "gramophone": "which has early device for playing recorded music using a rotating horn features",
                "grand piano": "which has large keyboard instrument with extensive range and rich tonal quality features",
                "hawksbill": "which has sea turtle species with sharp beak and beautiful shell patterns features",
                "headphone": "which has audio device worn on ears for personal sound listening features",
                "hedgehog": "which has small nocturnal mammal covered in spines and known for curling up features",
                "helicopter": "which has aircraft with rotating blades allowing vertical takeoff and landing features",
                "ibis": "which has wading bird with long curved bill and distinctive plumage features",
                "inline skate": "which has footwear with wheels arranged in a single line for skating features",
                "joshua tree": "which has unique tree species native to desert regions with twisted branches features",
                "kangaroo": "which has marsupial with powerful hind legs large feet and pouch for young features",
                "ketch": "which has two-masted sailing vessel with specific rigging configuration features",
                "lamp": "which has lighting device typically consisting of a base and a shade features",
                "laptop": "which has portable computer with integrated screen keyboard and battery power features",
                "llama": "which has domesticated South American camelid known for wool and gentle demeanor features",
                "lobster": "which has large marine crustacean with hard shell and prominent claws features",
                "lotus": "which has aquatic flowering plant with large fragrant blooms and sacred significance features",
                "mandolin": "which has stringed musical instrument played by plucking with a plectrum features",
                "mayfly": "which has insect with very short adult lifespan emerging near water sources features",
                "menorah": "which has candelabrum with multiple branches used in religious ceremonies features",
                "metronome": "which has device used by musicians to keep a consistent tempo timing features",
                "minaret": "which has tower associated with mosques from which the call to prayer is made features",
                "nautilus": "which has marine mollusk with a spiraled chambered shell and tentacles features",
                "octopus": "which has eight-armed cephalopod known for intelligence and ability to camouflage features",
                "okapi": "which has forest-dwelling mammal related to giraffes with striped legs features",
                "pagoda": "which has tiered tower with multiple eaves commonly found in Asian architecture features",
                "panda": "which has black and white bear known for eating bamboo and endangered status features",
                "pigeon": "which has common bird species often found in urban areas with diverse colors features",
                "pizza": "which has popular Italian dish consisting of dough topped with sauce and cheese features",
                "platypus": "which has unique mammal with duck bill webbed feet and egg-laying abilities features",
                "pyramid": "which has ancient monumental structure with triangular sides built as tombs features",
                "revolver": "which has handheld firearm with a rotating cylinder holding multiple bullets features",
                "rhino": "which has thick-skinned large herbivore with one or two horns on snout features",
                "rooster": "which has male chicken known for crowing at dawn and vibrant plumage features",
                "saxophone": "which has brass musical instrument with keys and a distinctive curved shape features",
                "schooner": "which has sailing vessel with two or more masts and fore-and-aft sails features",
                "scissors": "which has handheld tool with two blades for cutting various materials features",
                "scorpion": "which has arachnid with pincers and a venomous stinger on tail end features",
                "sea horse": "which has marine fish with horse-like head and curled prehensile tail features",
                "snoopy": "which has iconic cartoon beagle known for adventures and imaginative persona features",
                "soccer ball": "which has round ball used in the sport of soccer kicked by players features",
                "stapler": "which has office device used to bind papers together with metal staples features",
                "starfish": "which has marine echinoderm with radial symmetry and ability to regenerate limbs features",
                "stegosaurus": "which has dinosaur known for large plates along back and spiked tail features",
                "stop sign": "which has octagonal traffic sign indicating drivers must come to a complete stop features",
                "strawberry": "which has sweet red fruit with seeds on exterior and juicy flesh features",
                "sunflower": "which has tall plant with large yellow petals and seed-filled center features",
                "tick": "which has small blood-sucking arachnid often parasitic on animals and humans features",
                "trilobite": "which has extinct marine arthropod with segmented body and hard exoskeleton features",
                "umbrella": "which has portable device used to protect from rain or sunlight features",
                "watch": "which has wrist-worn timepiece with various functionalities and designs features",
                "water lilly": "which has aquatic plant with floating leaves and fragrant colorful flowers features",
                "wheelchair": "which has mobility device with wheels used by individuals with movement impairments features",
                "wild cat": "which has undomesticated feline with agile body and hunting instincts features",
                "windsor chair": "which has stylish wooden chair with upholstered seat and carved back features",
                "wrench": "which has hand tool used for gripping and turning nuts or bolts features",
                "yin yang": "which has symbol representing duality and balance in Chinese philosophy features"
            }




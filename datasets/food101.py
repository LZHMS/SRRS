import os
import pickle

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import mkdir_if_missing
from .add_noise import generate_fewshot_dataset_with_symflip_noise, generate_fewshot_dataset_with_pairflip_noise

from .oxford_pets import OxfordPets
from .dtd import DescribableTextures as DTD
import random
import numpy as np

@DATASET_REGISTRY.register()
class Food101(DatasetBase):
    dataset_dir = "food-101"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "images")
        self.meta_dir = os.path.join(self.dataset_dir, "meta")
        self.split_path = os.path.join(self.dataset_dir, "split_zhou_Food101.json")
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
        mkdir_if_missing(self.split_fewshot_dir)

        if os.path.exists(self.split_path):
            train, val, test = OxfordPets.read_split(self.split_path, self.image_dir)
        else:
            classnames = []
            with open(os.path.join(self.meta_dir, "classes.txt"), "r") as f:
                lines = f.readlines()
                for line in lines:
                    classnames.append(line.strip())
            cname2lab = {c: i for i, c in enumerate(classnames)}

            trainval = self.read_data(cname2lab, "train.txt")
            train, val = self.split_data(trainval, 0.8)
            test = self.read_data(cname2lab, "test.txt")
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
            'apple pie': "which has flaky crust warm spiced apples and golden brown top",
            'baby back ribs': "which has tender meat savory sauce and caramelized grill marks",
            'baklava': "which has flaky pastry honey layers and crunchy nut filling",
            'beef carpaccio': "which has thinly sliced raw beef drizzled oil and greens",
            'beef tartare': "which has minced raw beef capers egg yolk and spices",
            'beet salad': "which has fresh beets leafy greens crumbled cheese and nuts",
            'beignets': "which has light dough soft center and powdered sugar topping",
            'bibimbap': "which has colorful veggies sliced meat and runny egg on rice",
            'bread pudding': "which has soft bread custard flavor and caramelized crust",
            'breakfast burrito': "which has tortilla wrap eggs cheese and savory fillings",
            'bruschetta': "which has toasted bread fresh tomatoes garlic basil and drizzle",
            'caesar salad': "which has crisp romaine cheese croutons and creamy Caesar dressing",
            'cannoli': "which has crispy shell creamy ricotta filling and chocolate chips",
            'caprese salad': "which has fresh tomatoes mozzarella basil and balsamic drizzle",
            'carrot cake': "which has moist cake cream cheese frosting and spiced flavor",
            'ceviche': "which has citrus-marinated fish fresh herbs and diced vegetables",
            'cheese plate': "which has various cheeses fruits nuts and crackers for pairing",
            'cheesecake': "which has creamy filling graham crust and smooth top layer",
            'chicken curry': "which has tender chicken rich spices and creamy sauce",
            'chicken quesadilla': "which has crispy tortilla melted cheese and seasoned chicken",
            'chicken wings': "which has crispy skin juicy meat and tangy sauce",
            'chocolate cake': "which has rich cocoa flavor moist texture and frosting layer",
            'chocolate mousse': "which has smooth creamy texture intense chocolate and whipped topping",
            'churros': "which has crispy exterior soft center and cinnamon sugar coating",
            'clam chowder': "which has creamy base tender clams and potato chunks",
            'club sandwich': "which has stacked layers of meat cheese lettuce and bread",
            'crab cakes': "which has crispy crust moist crab meat and flavorful seasoning",
            'creme brulee': "which has smooth custard caramelized sugar and creamy vanilla taste",
            'croque madame': "which has toasted bread ham melted cheese and runny egg",
            'cup cakes': "which has moist cake fluffy frosting and colorful sprinkles",
            'deviled eggs': "which has creamy yolk filling paprika and garnish on eggs",
            'donuts': "which has soft dough round shape and glazed topping",
            'dumplings': "which has dough wrapper flavorful filling and steamed or fried exterior",
            'edamame': "which has steamed soybeans firm texture and sprinkle of salt",
            'eggs benedict': "which has poached eggs hollandaise sauce and English muffin base",
            'escargots': "which has tender snails garlic butter sauce and parsley garnish",
            'falafel': "which has crispy chickpea exterior soft center and herbs",
            'filet mignon': "which has tender beef juicy center and grill sear",
            'fish and chips': "which has crispy battered fish and golden fries",
            'foie gras': "which has creamy texture rich flavor and golden sear",
            'french fries': "which has crispy golden exterior fluffy center and salted finish",
            'french onion soup': "which has caramelized onions melted cheese and rich broth",
            'french toast': "which has golden brown bread egg custard and syrup",
            'fried calamari': "which has crispy batter tender squid rings and lemon wedge",
            'fried rice': "which has fluffy rice mixed veggies and savory flavor",
            'frozen yogurt': "which has smooth texture tangy flavor and creamy consistency",
            'garlic bread': "which has crispy edges soft center and garlic butter",
            'gnocchi': "which has soft doughy texture potato flavor and light sauce",
            'greek salad': "which has fresh veggies feta cheese olives and olive oil",
            'grilled cheese sandwich': "which has melted cheese toasted bread and buttery crust",
            'grilled salmon': "which has flaky fish golden crust and smoky flavor",
            'guacamole': "which has creamy avocado fresh lime and chunky texture",
            'gyoza': "which has crispy bottom juicy filling and thin wrapper",
            'hamburger': "which has juicy patty lettuce tomato and toasted bun",
            'hot and sour soup': "which has tangy flavor tofu and mushroom slices",
            'hot dog': "which has grilled sausage soft bun and toppings",
            'huevos rancheros': "which has fried eggs tortillas beans and salsa",
            'hummus': "which has creamy chickpeas olive oil and smooth texture",
            'ice cream': "which has creamy base smooth texture and sweet flavor",
            'lasagna': "which has layered pasta melted cheese and meat sauce",
            'lobster bisque': "which has smooth creamy soup and rich lobster flavor",
            'lobster roll sandwich': "which has buttery roll fresh lobster and creamy sauce",
            'macaroni and cheese': "which has creamy cheese sauce soft pasta and crispy top",
            'macarons': "which has crispy shell chewy center and smooth filling",
            'miso soup': "which has savory broth tofu and seaweed",
            'mussels': "which has steamed shellfish rich broth and herbs",
            'nachos': "which has crispy chips melted cheese and jalapenos",
            'omelette': "which has fluffy eggs cheese and fresh vegetables",
            'onion rings': "which has crispy batter soft onion and golden exterior",
            'oysters': "which has briny flavor fresh shucked and lemon wedge",
            'pad thai': "which has rice noodles tangy sauce and peanuts",
            'paella': "which has saffron rice seafood and rich flavor",
            'pancakes': "which has fluffy texture golden edges and syrup",
            'panna cotta': "which has creamy custard smooth texture and fresh berries",
            'peking duck': "which has crispy skin tender meat and savory sauce",
            'pho': "which has beef broth rice noodles and fresh herbs",
            'pizza': "which has melted cheese crispy crust and toppings",
            'pork chop': "which has juicy meat seared crust and seasoned",
            'poutine': "which has crispy fries melted cheese and gravy",
            'prime rib': "which has tender beef marbling and seasoned crust",
            'pulled pork sandwich': "which has tender pork tangy sauce and soft bun",
            'ramen': "which has broth noodles and sliced pork",
            'ravioli': "which has pasta shell stuffed filling and sauce",
            'red velvet cake': "which has soft texture cream cheese and red color",
            'risotto': "which has creamy texture soft rice and parmesan",
            'samosa': "which has crispy pastry spicy filling and fried",
            'sashimi': "which has raw fish fresh slices and garnishes",
            'scallops': "which has seared crust tender interior and buttery",
            'seaweed salad': "which has tangy flavor green seaweed and sesame",
            'shrimp and grits': "which has creamy grits spicy shrimp and sauce",
            'spaghetti bolognese': "which has rich sauce ground beef and pasta",
            'spaghetti carbonara': "which has creamy sauce bacon bits and pasta",
            'spring rolls': "which has crispy wrapper fresh filling and dipping sauce",
            'steak': "which has juicy interior grilled sear and seasoning",
            'strawberry shortcake': "which has sweet strawberries whipped cream and soft cake",
            'sushi': "which has rice fish seaweed and wasabi",
            'tacos': "which has tortilla filling meat and toppings",
            'takoyaki': "which has batter octopus pieces and savory",
            'tiramisu': "which has coffee flavor mascarpone and cocoa",
            'tuna tartare': "which has raw tuna seasoning and herbs",
            'waffles': "which has crispy edges fluffy center and syrup"
        }
    
    def read_data(self, cname2lab, split_file):
        filepath = os.path.join(self.meta_dir, split_file)
        items = []

        with open(filepath, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                imname = line + ".jpg"
                line = line.split("/")
                classname = line[0]
                impath = os.path.join(self.image_dir, imname)
                label = cname2lab[classname]
                item = Datum(impath=impath, label=label, classname=classname)
                items.append(item)

        return items

    def split_data(self, trainval, p_train):
        print(f"Splitting trainval into {p_train:.0%} train and {1-p_train:.0%} val")
        random.seed(1)
        np.random.seed(1)
        random.shuffle(trainval)
        n_total = len(trainval)
        n_train = round(n_total * p_train)
        return trainval[:n_train], trainval[n_train:]
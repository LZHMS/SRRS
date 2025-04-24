import os
import pickle
from scipy.io import loadmat

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase

from .oxford_pets import OxfordPets
from dassl.utils import mkdir_if_missing
from .add_noise import generate_fewshot_dataset_with_symflip_noise, generate_fewshot_dataset_with_pairflip_noise


@DATASET_REGISTRY.register()
class StanfordCars(DatasetBase):

    dataset_dir = "stanford_cars"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.split_path = os.path.join(self.dataset_dir, "split_zhou_StanfordCars.json")

        if os.path.exists(self.split_path):
            train, val, test = OxfordPets.read_split(self.split_path, self.dataset_dir)
        else:
            trainval_file = os.path.join(self.dataset_dir, "devkit", "cars_train_annos.mat")
            test_file = os.path.join(self.dataset_dir, "cars_test_annos_withlabels.mat")
            meta_file = os.path.join(self.dataset_dir, "devkit", "cars_meta.mat")
            trainval = self.read_data("cars_train", trainval_file, meta_file)
            test = self.read_data("cars_test", test_file, meta_file)
            train, val = OxfordPets.split_trainval(trainval)
            OxfordPets.save_split(train, val, test, self.split_path, self.dataset_dir)

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
                    for item in train:
                        print(item.label, item.impath, item.classname)
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
            '2000 AM General Hummer SUV': 'which has a large rugged body and off-road capability features',
            '2012 Acura RL Sedan': 'which has a sleek design with luxurious leather interior features',
            '2012 Acura TL Sedan': 'which has a modern design with advanced safety technology features',
            '2008 Acura TL Type-S': 'which has sporty styling and a powerful V6 engine features',
            '2012 Acura TSX Sedan': 'which has an elegant appearance with spacious interior features',
            '2001 Acura Integra Type R': 'which has a lightweight body with a high-performance engine features',
            '2012 Acura ZDX Hatchback': 'which has a unique sloping roofline with premium materials features',
            '2012 Aston Martin V8 Vantage Convertible': 'which has a sleek design and powerful V8 engine features',
            '2012 Aston Martin V8 Vantage Coupe': 'which has a luxurious interior and high-performance engine features',
            '2012 Aston Martin Virage Convertible': 'which has an elegant appearance with advanced suspension system features',
            '2012 Aston Martin Virage Coupe': 'which has sleek aerodynamic lines with premium leather interior features',
            '2008 Audi RS 4 Convertible': 'which has a sporty design with high-speed handling capabilities features',
            '2012 Audi A5 Coupe': 'which has a sleek design with a powerful turbocharged engine features',
            '2012 Audi TTS Coupe': 'which has a sporty exterior with a turbocharged four-cylinder engine features',
            '2012 Audi R8 Coupe': 'which has a supercar design with a mid-engine layout features',
            '1994 Audi V8 Sedan': 'which has a classic design with a powerful V8 engine features',
            '1994 Audi 100 Sedan': 'which has a simple design with a reliable engine and features',
            '1994 Audi 100 Wagon': 'which has a spacious interior with a practical body design features',
            '2011 Audi TT Hatchback': 'which has a compact design with agile handling and turbocharged engine features',
            '2011 Audi S6 Sedan': 'which has a luxurious interior with advanced driver-assistance systems features',
            '2012 Audi S5 Convertible': 'which has a stylish design with powerful engine and all-wheel drive features',
            '2012 Audi S5 Coupe': 'which has an aggressive design with all-wheel drive performance features',
            '2012 Audi S4 Sedan': 'which has a sporty design with turbocharged engine and quick acceleration features',
            '2007 Audi S4 Sedan': 'which has a classic design with powerful V8 engine and sporty features',
            '2012 Audi TT RS Coupe': 'which has a sleek body with turbocharged performance and agility features',
            '2012 BMW ActiveHybrid 5 Sedan': 'which has a hybrid powertrain with luxurious interior and premium features',
            '2012 BMW 1 Series Convertible': 'which has a compact design with rear-wheel drive and sporty features',
            '2012 BMW 1 Series Coupe': 'which has a compact design with powerful engine and precise handling features',
            '2012 BMW 3 Series Sedan': 'which has a refined design with advanced technology and sporty handling features',
            '2012 BMW 3 Series Wagon': 'which has a spacious interior with practical design and sporty features',
            '2007 BMW 6 Series Convertible': 'which has a sleek body with luxurious interior and advanced technology features',
            '2007 BMW X5 SUV': 'which has a large body with luxurious interior and all-wheel drive features',
            '2012 BMW X6 SUV': 'which has a coupe-like roofline with sporty handling and features',
            '2012 BMW M3 Coupe': 'which has a high-performance engine with precise handling and sporty features',
            '2010 BMW M5 Sedan': 'which has a powerful V10 engine with luxurious interior and sporty features',
            '2010 BMW M6 Convertible': 'which has a sleek convertible body with a high-performance engine features',
            '2012 BMW X3 SUV': 'which has a practical design with all-wheel drive and spacious interior features',
            '2012 BMW Z4 Convertible': 'which has a compact design with agile handling and a powerful engine features',
            '2012 Bentley Continental Supersports Conv. Convertible': 'which has a luxurious convertible design with a powerful engine features',
            '2009 Bentley Arnage Sedan': 'which has a classic design with luxurious materials and powerful engine features',
            '2011 Bentley Mulsanne Sedan': 'which has a stately design with premium interior and advanced features features',
            '2012 Bentley Continental GT Coupe': 'which has a sleek design with a powerful W12 engine and features',
            '2007 Bentley Continental GT Coupe': 'which has a luxurious body with a powerful engine and premium materials features',
            '2007 Bentley Continental Flying Spur Sedan': 'which has a spacious interior with premium materials and advanced features',
            '2009 Bugatti Veyron 16.4 Convertible': 'which has an incredibly fast engine with a luxurious open-top design features',
            '2009 Bugatti Veyron 16.4 Coupe': 'which has a record-breaking top speed with luxurious interior features',
            '2012 Buick Regal GS': 'which has a sporty exterior with turbocharged engine and advanced technology features',
            '2007 Buick Rainier SUV': 'which has a large body with a practical interior and smooth ride features',
            '2012 Buick Verano Sedan': 'which has a comfortable interior with quiet ride and modern technology features',
            '2012 Buick Enclave SUV': 'which has a spacious interior with premium materials and all-wheel drive features',
            '2012 Cadillac CTS-V Sedan': 'which has a high-performance engine with sporty handling and luxurious features',
            '2012 Cadillac SRX SUV': 'which has a stylish design with premium interior and advanced technology features',
            '2007 Cadillac Escalade EXT Crew Cab': 'which has a rugged design with luxurious interior and powerful engine features',
            '2012 Chevrolet Silverado 1500 Hybrid Crew Cab': 'which has a hybrid powertrain with a large towing capacity and practical features',
            '2012 Chevrolet Corvette Convertible': 'which has a sleek design with high-speed performance and open-top driving features',
            '2012 Chevrolet Corvette ZR1': 'which has a supercharged engine with track-focused performance and features',
            '2007 Chevrolet Corvette Ron Fellows Edition Z06': 'which has a limited edition design with a powerful engine features',
            '2012 Chevrolet Traverse SUV': 'which has a spacious interior with practical design and all-wheel drive features',
            '2012 Chevrolet Camaro Convertible': 'which has a sporty design with open-top driving and powerful engine features',
            '2010 Chevrolet HHR SS': 'which has a retro-inspired design with a turbocharged engine features',
            '2007 Chevrolet Impala Sedan': 'which has a classic design with a spacious interior and smooth ride features',
            '2012 Chevrolet Tahoe Hybrid SUV': 'which has a hybrid powertrain with a large interior and towing capacity features',
            '2012 Chevrolet Sonic Sedan': 'which has a compact design with modern technology and fuel efficiency features',
            '2007 Chevrolet Express Cargo Van': 'which has a large cargo capacity with practical design and powerful engine features',
            '2012 Chevrolet Avalanche Crew Cab': 'which has a versatile body design with powerful engine and spacious interior features',
            '2010 Chevrolet Cobalt SS': 'which has a sporty design with turbocharged performance and agile handling features',
            '2010 Chevrolet Malibu Hybrid Sedan': 'which has a hybrid powertrain with fuel efficiency and modern technology features',
            '2009 Chevrolet TrailBlazer SS': 'which has a sporty exterior with powerful engine and all-wheel drive features',
            '2012 Chevrolet Silverado 2500HD Regular Cab': 'which has a rugged design with powerful engine and towing capacity features',
            '2007 Chevrolet Silverado 1500 Classic Extended Cab': 'which has a spacious interior with powerful engine and practical features',
            '2007 Chevrolet Express Van': 'which has a practical design with large cargo capacity and powerful engine features',
            '2007 Chevrolet Monte Carlo Coupe': 'which has a sporty exterior with powerful engine and modern technology features',
            '2007 Chevrolet Malibu Sedan': 'which has a comfortable interior with modern technology and fuel efficiency features',
            '2012 Chevrolet Silverado 1500 Extended Cab': 'which has a rugged design with powerful engine and towing capacity features',
            '2012 Chevrolet Silverado 1500 Regular Cab': 'which has a practical body design with powerful engine and towing capacity features',
            '2009 Chrysler Aspen SUV': 'which has a large body with a comfortable interior and towing capacity features',
            '2010 Chrysler Sebring Convertible': 'which has a stylish design with open-top driving and modern technology features',
            '2012 Chrysler Town and Country Minivan': 'which has a spacious interior with family-friendly features and practicality',
            '2010 Chrysler 300 SRT-8': 'which has a muscular design with a powerful V8 engine and luxurious features',
            '2008 Chrysler Crossfire Convertible': 'which has a sleek design with open-top driving and sporty performance features',
            '2008 Chrysler PT Cruiser Convertible': 'which has a retro-inspired design with open-top driving and practical features',
            '2002 Daewoo Nubira Wagon': 'which has a practical design with spacious interior and fuel efficiency features',
            '2012 Dodge Caliber Wagon': 'which has a compact design with versatile interior and modern technology features',
            '2007 Dodge Caliber Wagon': 'which has a unique design with spacious interior and practical features',
            '1997 Dodge Caravan Minivan': 'which has a family-friendly design with spacious interior and comfortable features',
            '2010 Dodge Ram Pickup 3500 Crew Cab': 'which has a rugged design with powerful engine and large towing capacity features',
            '2009 Dodge Ram Pickup 3500 Quad Cab': 'which has a practical design with a powerful engine and rugged towing capacity features',
            '2009 Dodge Sprinter Cargo Van': 'which has a large cargo capacity with practical interior and fuel efficiency features',
            '2012 Dodge Journey SUV': 'which has a versatile design with spacious interior and modern technology features',
            '2010 Dodge Dakota Crew Cab': 'which has a practical design with powerful engine and spacious interior features',
            '2007 Dodge Dakota Club Cab': 'which has a rugged design with a practical body and towing capacity features',
            '2008 Dodge Magnum Wagon': 'which has a muscular design with a powerful engine and practical features',
            '2011 Dodge Challenger SRT8': 'which has a retro-inspired design with a powerful V8 engine and sporty features',
            '2012 Dodge Durango SUV': 'which has a large body with a powerful engine and family-friendly features',
            '2007 Dodge Durango SUV': 'which has a spacious interior with a rugged design and powerful engine features',
            '2012 Dodge Charger Sedan': 'which has a muscular design with a powerful engine and modern technology features',
            '2009 Dodge Charger SRT-8': 'which has a sporty design with a powerful V8 engine and performance features',
            '1998 Eagle Talon Hatchback': 'which has a compact design with a sporty engine and agile handling features',
            '2012 FIAT 500 Abarth': 'which has a compact design with a turbocharged engine and sporty features',
            '2012 FIAT 500 Convertible': 'which has a retro-inspired design with open-top driving and compact features',
            '2012 Ferrari FF Coupe': 'which has a sleek design with a powerful V12 engine and luxurious features',
            '2012 Ferrari California Convertible': 'which has a stylish design with open-top driving and high-speed performance features',
            '2012 Ferrari 458 Italia Convertible': 'which has a supercar design with a powerful engine and open-top features',
            '2012 Ferrari 458 Italia Coupe': 'which has a sleek design with a powerful engine and track-focused performance features',
            '2012 Fisker Karma Sedan': 'which has a futuristic design with an eco-friendly hybrid powertrain and luxurious features',
            '2012 Ford F-450 Super Duty Crew Cab': 'which has a rugged design with a powerful engine and large towing capacity features',
            '2007 Ford Mustang Convertible': 'which has a classic design with open-top driving and powerful engine features',
            '2007 Ford Freestar Minivan': 'which has a practical design with family-friendly features and spacious interior',
            '2009 Ford Expedition EL SUV': 'which has a large body with a powerful engine and spacious interior features',
            '2012 Ford Edge SUV': 'which has a modern design with a spacious interior and advanced technology features',
            '2011 Ford Ranger SuperCab': 'which has a rugged design with a practical body and powerful engine features',
            '2006 Ford GT Coupe': 'which has a supercar design with a powerful engine and track-focused performance features',
            '2012 Ford F-150 Regular Cab': 'which has a rugged design with a powerful engine and large towing capacity features',
            '2007 Ford F-150 Regular Cab': 'which has a practical body with a powerful engine and rugged features',
            '2007 Ford Focus Sedan': 'which has a compact design with fuel efficiency and modern technology features',
            '2012 Ford E-Series Wagon Van': 'which has a large body with spacious interior and practical design features',
            '2012 Ford Fiesta Sedan': 'which has a compact design with fuel efficiency and modern technology features',
            '2012 GMC Terrain SUV': 'which has a modern design with a spacious interior and advanced technology features',
            '2012 GMC Savana Van': 'which has a large body with a practical interior and powerful engine features',
            '2012 GMC Yukon Hybrid SUV': 'which has a hybrid powertrain with a large body and towing capacity features',
            '2012 GMC Acadia SUV': 'which has a spacious interior with modern technology and all-wheel drive features',
            '2012 GMC Canyon Extended Cab': 'which has a practical body with a powerful engine and towing capacity features',
            '1993 Geo Metro Convertible': 'which has a compact design with fuel efficiency and open-top driving features',
            '2010 HUMMER H3T Crew Cab': 'which has a rugged design with a powerful engine and off-road capabilities features',
            '2009 HUMMER H2 SUT Crew Cab': 'which has a large body with a powerful engine and off-road capabilities features',
            '2012 Honda Odyssey Minivan': 'which has a family-friendly design with spacious interior and practical features',
            '2007 Honda Odyssey Minivan': 'which has a spacious interior with modern technology and family-friendly features',
            '2012 Honda Accord Coupe': 'which has a sleek design with a powerful engine and sporty handling features',
            '2012 Honda Accord Sedan': 'which has a practical design with modern technology and fuel efficiency features',
            '2012 Hyundai Veloster Hatchback': 'which has a unique design with a sporty engine and modern technology features',
            '2012 Hyundai Santa Fe SUV': 'which has a spacious interior with advanced technology and practical features',
            '2012 Hyundai Tucson SUV': 'which has a compact design with a powerful engine and all-wheel drive features',
            '2012 Hyundai Veracruz SUV': 'which has a luxurious interior with advanced technology and spacious features',
            '2012 Hyundai Sonata Hybrid Sedan': 'which has a hybrid powertrain with fuel efficiency and modern technology features',
            '2007 Hyundai Elantra Sedan': 'which has a compact design with fuel efficiency and modern technology features',
            '2012 Hyundai Accent Sedan': 'which has a compact design with fuel efficiency and practical features',
            '2012 Hyundai Genesis Sedan': 'which has a luxurious interior with advanced technology and powerful engine features',
            '2012 Hyundai Sonata Sedan': 'which has a sleek design with modern technology and fuel efficiency features',
            '2012 Hyundai Elantra Touring Hatchback': 'which has a practical design with fuel efficiency and spacious interior features',
            '2012 Hyundai Azera Sedan': 'which has a luxurious interior with advanced technology and powerful engine features',
            '2012 Infiniti G Coupe IPL': 'which has a sporty design with powerful engine and luxurious interior features',
            '2011 Infiniti QX56 SUV': 'which has a large body with luxurious interior and advanced technology features',
            '2008 Isuzu Ascender SUV': 'which has a rugged design with a practical interior and off-road capabilities features',
            '2012 Jaguar XK XKR': 'which has a sleek design with a powerful engine and luxurious interior features',
            '2012 Jeep Patriot SUV': 'which has a compact design with off-road capabilities and modern technology features',
            '2012 Jeep Wrangler SUV': 'which has a rugged design with off-road capabilities and open-air driving features',
            '2012 Jeep Liberty SUV': 'which has a rugged design with off-road capabilities and practical features',
            '2012 Jeep Grand Cherokee SUV': 'which has a luxurious interior with off-road capabilities and advanced technology features',
            '2012 Jeep Compass SUV': 'which has a compact design with off-road capabilities and modern technology features',
            '2008 Lamborghini Reventon Coupe': 'which has a futuristic design with a powerful engine and track-focused performance features',
            '2012 Lamborghini Aventador Coupe': 'which has a supercar design with a powerful V12 engine and futuristic features',
            '2012 Lamborghini Gallardo LP 570-4 Superleggera': 'which has a lightweight design with a powerful engine and track-focused performance features',
            '2001 Lamborghini Diablo Coupe': 'which has a classic supercar design with a powerful engine and track-focused performance features',
            '2012 Land Rover Range Rover SUV': 'which has a luxurious interior with off-road capabilities and advanced technology features',
            '2012 Land Rover LR2 SUV': 'which has a compact design with off-road capabilities and luxurious interior features',
            '2011 Lincoln Town Car Sedan': 'which has a spacious interior with luxurious materials and advanced technology features',
            '2012 MINI Cooper Roadster Convertible': 'which has a compact design with open-top driving and sporty handling features',
            '2012 Maybach Landaulet Convertible': 'which has a luxurious design with open-top driving and advanced technology features',
            '2011 Mazda Tribute SUV': 'which has a compact design with off-road capabilities and practical interior features',
            '2012 McLaren MP4-12C Coupe': 'which has a sleek design with a powerful engine and track-focused features',
            '1993 Mercedes-Benz 300-Class Convertible': 'which has a classic design with open-top driving and luxurious features',
            '2012 Mercedes-Benz C-Class Sedan': 'which has a luxurious interior with modern technology and powerful engine features',
            '2009 Mercedes-Benz SL-Class Coupe': 'which has a sleek design with a powerful engine and open-top driving features',
            '2012 Mercedes-Benz E-Class Sedan': 'which has a luxurious interior with advanced technology and powerful engine features',
            '2012 Mercedes-Benz S-Class Sedan': 'which has a spacious interior with luxurious materials and advanced technology features',
            '2012 Mercedes-Benz Sprinter Van': 'which has a large body with a spacious interior and practical cargo features',
            '2012 Mitsubishi Lancer Sedan': 'which has a sporty design with modern technology and fuel efficiency features',
            '2012 Nissan Leaf Hatchback': 'which has an eco-friendly design with electric powertrain and modern technology features',
            '2012 Nissan NV Passenger Van': 'which has a spacious interior with practical design and modern technology features',
            '2012 Nissan Juke Hatchback': 'which has a compact design with a turbocharged engine and sporty handling features',
            '1998 Nissan 240SX Coupe': 'which has a sporty design with agile handling and a lightweight body features',
            '1999 Plymouth Neon Coupe': 'which has a compact design with fuel efficiency and practical features',
            '2012 Porsche Panamera Sedan': 'which has a sleek design with luxurious interior and powerful engine features',
            '2012 Ram C/V Cargo Van Minivan': 'which has a practical design with spacious cargo area and fuel efficiency features',
            '2012 Rolls-Royce Phantom Drophead Coupe Convertible': 'which has a luxurious design with open-top driving and advanced technology features',
            '2012 Rolls-Royce Ghost Sedan': 'which has a luxurious interior with advanced technology and powerful engine features',
            '2012 Rolls-Royce Phantom Sedan': 'which has a spacious interior with luxurious materials and modern technology features',
            '2012 Scion xD Hatchback': 'which has a compact design with fuel efficiency and modern technology features',
            '2009 Spyker C8 Convertible': 'which has a unique design with open-top driving and track-focused performance features',
            '2009 Spyker C8 Coupe': 'which has a sporty design with a powerful engine and lightweight body features',
            '2007 Suzuki Aerio Sedan': 'which has a compact design with fuel efficiency and practical features',
            '2012 Suzuki Kizashi Sedan': 'which has a sporty design with modern technology and fuel efficiency features',
            '2012 Suzuki SX4 Hatchback': 'which has a compact design with a practical interior and modern technology features',
            '2012 Suzuki SX4 Sedan': 'which has a practical design with fuel efficiency and modern technology features',
            '2012 Tesla Model S Sedan': 'which has a futuristic design with an electric powertrain and advanced technology features',
            '2012 Toyota Sequoia SUV': 'which has a large body with a spacious interior and off-road capabilities features',
            '2012 Toyota Camry Sedan': 'which has a practical design with fuel efficiency and modern technology features',
            '2012 Toyota Corolla Sedan': 'which has a compact design with fuel efficiency and modern technology features',
            '2012 Toyota 4Runner SUV': 'which has a rugged design with off-road capabilities and spacious interior features',
            '2012 Volkswagen Golf Hatchback': 'which has a compact design with fuel efficiency and modern technology features',
            '1991 Volkswagen Golf Hatchback': 'which has a classic design with fuel efficiency and practical features',
            '2012 Volkswagen Beetle Hatchback': 'which has a retro-inspired design with modern technology and fuel efficiency features',
            '2012 Volvo C30 Hatchback': 'which has a sporty design with a luxurious interior and modern technology features',
            '1993 Volvo 240 Sedan': 'which has a classic design with a spacious interior and practical features',
            '2007 Volvo XC90 SUV': 'which has a spacious interior with advanced safety features and modern technology',
            '2012 smart fortwo Convertible': 'which has a compact design with open-top driving and fuel efficiency features'
        }

    def read_data(self, image_dir, anno_file, meta_file):
        anno_file = loadmat(anno_file)["annotations"][0]
        meta_file = loadmat(meta_file)["class_names"][0]
        items = []

        for i in range(len(anno_file)):
            imname = anno_file[i]["fname"][0]
            impath = os.path.join(self.dataset_dir, image_dir, imname)
            label = anno_file[i]["class"][0, 0]
            label = int(label) - 1  # convert to 0-based index
            classname = meta_file[label][0]
            names = classname.split(" ")
            year = names.pop(-1)
            names.insert(0, year)
            classname = " ".join(names)
            item = Datum(impath=impath, label=label, classname=classname)
            items.append(item)

        return items
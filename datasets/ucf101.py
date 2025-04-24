import os
import re
import pickle

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import mkdir_if_missing
from .add_noise import generate_fewshot_dataset_with_symflip_noise, generate_fewshot_dataset_with_pairflip_noise

from .oxford_pets import OxfordPets



@DATASET_REGISTRY.register()
class UCF101(DatasetBase):

    dataset_dir = "ucf101"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "UCF-101-midframes")
        self.split_path = os.path.join(self.dataset_dir, "split_zhou_UCF101.json")

        if os.path.exists(self.split_path):
            train, val, test = OxfordPets.read_split(self.split_path, self.image_dir)
        else:
            cname2lab = {}
            filepath = os.path.join(self.dataset_dir, "ucfTrainTestlist/classInd.txt")
            with open(filepath, "r") as f:
                lines = f.readlines()
                for line in lines:
                    label, classname = line.strip().split(" ")
                    label = int(label) - 1  # conver to 0-based index
                    cname2lab[classname] = label

            trainval = self.read_data(cname2lab, "ucfTrainTestlist/trainlist01.txt")
            test = self.read_data(cname2lab, "ucfTrainTestlist/testlist01.txt")
            train, val = OxfordPets.split_trainval(trainval)
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
                    "Apply Eye Makeup": "which has a person applying makeup around their eyes features",
                    "Apply Lipstick": "which has a person putting lipstick on their lips features",
                    "Archery": "which has a person shooting arrows using a bow features",
                    "Baby Crawling": "which has a baby moving on hands and knees indoors features",
                    "Balance Beam": "which has a gymnast balancing and performing on a narrow beam features",
                    "Band Marching": "which has a group of musicians walking while playing instruments features",
                    "Baseball Pitch": "which has a player throwing a baseball towards a batter features",
                    "Basketball": "which has players dribbling and shooting a basketball during gameplay features",
                    "Basketball Dunk": "which has a player jumping and scoring by dunking the basketball features",
                    "Bench Press": "which has a person lifting weights while lying on a bench features",
                    "Biking": "which has a person riding a bicycle outdoors on a path features",
                    "Billiards": "which has players striking balls with cues on a pool table features",
                    "Blow Dry Hair": "which has a person using a hairdryer to dry their hair features",
                    "Blowing Candles": "which has a person blowing out candles typically on a cake features",
                    "Body Weight Squats": "which has a person performing squats using only their body weight features",
                    "Bowling": "which has a person rolling a ball down a lane towards pins features",
                    "Boxing Punching Bag": "which has a person practicing punches on a hanging bag features",
                    "Boxing Speed Bag": "which has a person hitting a small bag rapidly with punches features",
                    "Breast Stroke": "which has a swimmer using the breaststroke technique in a pool features",
                    "Brushing Teeth": "which has a person cleaning their teeth using a toothbrush features",
                    "Clean And Jerk": "which has a person lifting a barbell in an Olympic weightlifting movement features",
                    "Cliff Diving": "which has a person jumping off a high cliff into water features",
                    "Cricket Bowling": "which has a player delivering a cricket ball towards the batsman features",
                    "Cricket Shot": "which has a cricket player striking the ball with a bat features",
                    "Cutting In Kitchen": "which has a person using a knife to chop ingredients features",
                    "Diving": "which has a person jumping off a platform or board into water features",
                    "Drumming": "which has a person playing drums with sticks in a rhythmic fashion features",
                    "Fencing": "which has two individuals engaging in a sword fight using protective gear features",
                    "Field Hockey Penalty": "which has a player taking a penalty shot in field hockey features",
                    "Floor Gymnastics": "which has a gymnast performing routines on a padded floor features",
                    "Frisbee Catch": "which has a person jumping or running to catch a thrown frisbee features",
                    "Front Crawl": "which has a swimmer using the front crawl stroke in a pool features",
                    "Golf Swing": "which has a golfer swinging a club to hit the ball features",
                    "Haircut": "which has a person cutting hair with scissors or clippers features",
                    "Hammering": "which has a person striking nails into a surface using a hammer features",
                    "Hammer Throw": "which has an athlete spinning and throwing a heavy weight features",
                    "Handstand Pushups": "which has a person doing pushups while inverted in a handstand position features",
                    "Handstand Walking": "which has a person walking on their hands in an inverted position features",
                    "Head Massage": "which has a person receiving a soothing massage to their scalp features",
                    "High Jump": "which has an athlete leaping over a horizontal bar into a mat features",
                    "Horse Race": "which has jockeys riding horses in a competitive race event features",
                    "Horse Riding": "which has a person riding a horse typically in an outdoor setting features",
                    "Hula Hoop": "which has a person spinning a circular hoop around their waist features",
                    "Ice Dancing": "which has a pair of skaters performing artistic moves on ice features",
                    "Javelin Throw": "which has an athlete throwing a long spear-like object for distance features",
                    "Juggling Balls": "which has a person tossing and catching multiple balls in the air features",
                    "Jumping Jack": "which has a person jumping while spreading arms and legs repeatedly features",
                    "Jump Rope": "which has a person skipping over a swinging rope for exercise features",
                    "Kayaking": "which has a person paddling a kayak through water typically in a river features",
                    "Knitting": "which has a person using needles to create fabric with yarn features",
                    "Long Jump": "which has an athlete running and leaping into a sandpit for distance features",
                    "Lunges": "which has a person stepping forward and bending knees in an exercise features",
                    "Military Parade": "which has soldiers marching in formation during a formal parade features",
                    "Mixing": "which has a person combining ingredients together in a bowl or container features",
                    "Mopping Floor": "which has a person cleaning the floor using a mop and bucket features",
                    "Nunchucks": "which has a person practicing martial arts with connected sticks features",
                    "Parallel Bars": "which has a gymnast performing routines on two horizontal bars features",
                    "Pizza Tossing": "which has a person spinning pizza dough in the air by hand features",
                    "Playing Cello": "which has a musician sitting and playing the cello with a bow features",
                    "Playing Daf": "which has a person playing a large circular frame drum with hands features",
                    "Playing Dhol": "which has a person playing a double-sided drum with sticks features",
                    "Playing Flute": "which has a musician blowing into a flute to produce melodic sounds features",
                    "Playing Guitar": "which has a musician strumming or plucking strings of a guitar features",
                    "Playing Piano": "which has a person pressing keys on a piano to create music features",
                    "Playing Sitar": "which has a musician playing a traditional Indian stringed instrument features",
                    "Playing Tabla": "which has a person playing a set of hand drums from India features",
                    "Playing Violin": "which has a musician playing the violin using a bow on strings features",
                    "Pole Vault": "which has an athlete using a pole to jump over a high bar features",
                    "Pommel Horse": "which has a gymnast performing routines on a padded apparatus features",
                    "Pull Ups": "which has a person pulling their body upwards on a horizontal bar features",
                    "Punch": "which has a person throwing a forceful punch with a clenched fist features",
                    "Push Ups": "which has a person doing pushups by lowering and raising their body features",
                    "Rafting": "which has a group of people navigating a river on an inflatable raft features",
                    "Rock Climbing Indoor": "which has a person climbing a wall with handholds in a gym features",
                    "Rope Climbing": "which has a person climbing up a vertical rope using hands features",
                    "Rowing": "which has a person or team using oars to row a boat features",
                    "Salsa Spin": "which has a dancer spinning gracefully during a salsa performance features",
                    "Shaving Beard": "which has a person using a razor to shave facial hair features",
                    "Shotput": "which has an athlete throwing a heavy spherical object for distance features",
                    "Skate Boarding": "which has a person riding a skateboard and performing tricks features",
                    "Skiing": "which has a person sliding downhill on skis through snow features",
                    "Skijet": "which has a person riding a motorized watercraft on the ocean features",
                    "Sky Diving": "which has a person jumping from a plane and free falling with parachute features",
                    "Soccer Juggling": "which has a player keeping the soccer ball in the air with feet features",
                    "Soccer Penalty": "which has a soccer player taking a penalty kick towards the goal features",
                    "Still Rings": "which has a gymnast performing routines on suspended rings features",
                    "Sumo Wrestling": "which has two wrestlers competing in a sumo match in a circular ring features",
                    "Surfing": "which has a person riding waves on a surfboard at the beach features",
                    "Swing": "which has a person sitting on a suspended seat swinging back and forth features",
                    "Table Tennis Shot": "which has a player hitting a small ball over the net in table tennis features",
                    "Tai Chi": "which has a person performing slow flowing martial arts movements features",
                    "Tennis Swing": "which has a tennis player swinging the racket to hit a ball features",
                    "Throw Discus": "which has an athlete spinning and throwing a discus for distance features",
                    "Trampoline Jumping": "which has a person bouncing up and down on a trampoline with control features",
                    "Typing": "which has a person pressing keys on a keyboard to input text features",
                    "Uneven Bars": "which has a gymnast performing routines on two uneven horizontal bars features",
                    "Volleyball Spiking": "which has a player jumping to hit a volleyball downwards over net features",
                    "Walking With Dog": "which has a person walking with their dog on a leash outdoors features",
                    "Wall Pushups": "which has a person doing pushups while leaning against a vertical wall features",
                    "Writing On Board": "which has a person using chalk or marker to write on a board features",
                    "Yo Yo": "which has a person performing tricks by spinning a yo-yo on string features"
                }

    def read_data(self, cname2lab, text_file):
        text_file = os.path.join(self.dataset_dir, text_file)
        items = []

        with open(text_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split(" ")[0]  # trainlist: filename, label
                action, filename = line.split("/")
                label = cname2lab[action]

                elements = re.findall("[A-Z][^A-Z]*", action)
                renamed_action = "_".join(elements)

                filename = filename.replace(".avi", ".jpg")
                impath = os.path.join(self.image_dir, renamed_action, filename)

                item = Datum(impath=impath, label=label, classname=renamed_action)
                items.append(item)

        return items
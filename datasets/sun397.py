import os
import pickle

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase

from .oxford_pets import OxfordPets
from dassl.utils import mkdir_if_missing
from .add_noise import generate_fewshot_dataset_with_symflip_noise, generate_fewshot_dataset_with_pairflip_noise


@DATASET_REGISTRY.register()
class SUN397(DatasetBase):

    dataset_dir = "sun397"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "SUN397")
        self.split_path = os.path.join(self.dataset_dir, "split_zhou_SUN397.json")

        if os.path.exists(self.split_path):
            train, val, test = OxfordPets.read_split(self.split_path, self.image_dir)
        else:
            classnames = []
            with open(os.path.join(self.dataset_dir, "ClassName.txt"), "r") as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip()[1:]  # remove /
                    classnames.append(line)
            cname2lab = {c: i for i, c in enumerate(classnames)}
            trainval = self.read_data(cname2lab, "Training_01.txt")
            test = self.read_data(cname2lab, "Testing_01.txt")
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

        subsample = cfg.DATASET.SUBSAMPLE_CLASSES
        train, val, test = OxfordPets.subsample_classes(train, val, test, subsample=subsample)
        super().__init__(train_x=train, val=val, test=test)

        self.features = {
            'abbey': 'which has tall stone walls and stained glass windows features',
            'airplane cabin': 'which has narrow aisles with aligned seats and overhead compartments features',
            'airport terminal': 'which has spacious waiting areas with large glass windows and gates features',
            'alley': 'which has narrow pathways with tall buildings and scattered urban elements features',
            'amphitheater': 'which has tiered seating with open stage and expansive outdoor views features',
            'amusement arcade': 'which has flashing lights with gaming machines and colorful decorations features',
            'amusement park': 'which has towering rides with vibrant colors and scattered food stands features',
            'anechoic chamber': 'which has padded walls with silent surroundings and acoustic testing equipment features',
            'outdoor apartment building': 'which has multiple floors with balconies and surrounding urban landscape features',
            'indoor apse': 'which has curved walls with intricate designs and tall vaulted ceilings features',
            'aquarium': 'which has large tanks with aquatic creatures and colorful coral environments features',
            'aqueduct': 'which has towering stone arches with water channels and ancient architectural elements features',
            'arch': 'which has tall curved structures with symmetrical designs and surrounding stone walls features',
            'archive': 'which has long shelves with stacked documents and an organized storage system features',
            'outdoor arrival gate': 'which has open entryways with adjacent roads and visible transportation facilities features',
            'art gallery': 'which has white walls with hanging artworks and spacious open rooms features',
            'art school': 'which has open studios with scattered art supplies and creative workspaces features',
            'art studio': 'which has canvas setups with scattered brushes and vibrant artistic environments features',
            'assembly line': 'which has conveyor belts with neatly aligned products and mechanical systems features',
            'outdoor athletic field': 'which has green grass with visible goal posts and surrounding bleachers features',
            'public atrium': 'which has high ceilings with glass walls and large open spaces features',
            'attic': 'which has slanted ceilings with wooden beams and small dusty windows features',
            'auditorium': 'which has tiered seating with a large stage and expansive acoustics features',
            'auto factory': 'which has assembly lines with robotic arms and moving car parts features',
            'badlands': 'which has rugged landscapes with eroded hills and sparse vegetation features',
            'indoor badminton court': 'which has white lines with nets and high ceilings under bright lights features',
            'baggage claim': 'which has rotating carousels with scattered luggage and waiting passengers features',
            'shop bakery': 'which has display cases with freshly baked goods and sweet aromas features',
            'exterior balcony': 'which has metal railings with open views and surrounding tall buildings features',
            'interior balcony': 'which has railings with visible lower floors and spacious open rooms features',
            'ball pit': 'which has colorful plastic balls with low walls and playful environments features',
            'ballroom': 'which has grand chandeliers with polished floors and elegant decorations features',
            'bamboo forest': 'which has tall green stalks with dense foliage and narrow winding paths features',
            'banquet hall': 'which has long tables with formal settings and elegant chandeliers features',
            'bar': 'which has long counters with high stools and scattered bottles under dim lighting features',
            'barn': 'which has wooden beams with open lofts and scattered hay bales features',
            'barndoor': 'which has large wooden panels with sliding mechanisms and open access features',
            'baseball field': 'which has green grass with white lines and a visible pitcher’s mound features',
            'basement': 'which has low ceilings with exposed pipes and dim lighting under concrete floors features',
            'basilica': 'which has tall arched ceilings with stained glass windows and detailed stonework features',
            'outdoor basketball court': 'which has visible hoops with painted lines and surrounding urban scenery features',
            'bathroom': 'which has tiled walls with mirrors and visible sinks under bright lighting features',
            'batters box': 'which has dirt surfaces with marked boundaries and visible home plate features',
            'bayou': 'which has still waters with overhanging trees and scattered wetlands vegetation features',
            'indoor bazaar': 'which has narrow pathways with colorful stalls and densely packed crowds features',
            'outdoor bazaar': 'which has open streets with scattered stalls and vibrant outdoor markets features',
            'beach': 'which has sandy shores with gentle waves and scattered sunbathers under clear skies features',
            'beauty salon': 'which has styling chairs with mirrors and scattered hair products under soft lighting features',
            'bedroom': 'which has soft beds with plush pillows and surrounding personal belongings features',
            'berth': 'which has narrow sleeping spaces with overhead compartments and visible curtains features',
            'biology laboratory': 'which has microscopes with glass slides and scattered lab equipment under bright lighting features',
            'indoor bistro': 'which has small tables with cozy settings and visible menus under soft lighting features',
            'boardwalk': 'which has wooden planks with visible beach views and scattered tourists features',
            'boat deck': 'which has visible railings with open water views and scattered seating areas features',
            'boathouse': 'which has wooden walls with visible docks and small boats stored inside features',
            'bookstore': 'which has tall shelves with neatly stacked books and visible reading areas features',
            'indoor booth': 'which has small enclosed spaces with visible seating and surrounding partition walls features',
            'botanical garden': 'which has manicured plants with lush greenery and visible walking paths features',
            'indoor bow window': 'which has curved glass panes with surrounding seating and visible outside views features',
            'outdoor bow window': 'which has arched glass windows with visible exterior walls and surrounding views features',
            'bowling alley': 'which has polished lanes with scattered pins and visible scoreboards features',
            'boxing ring': 'which has raised platforms with surrounding ropes and visible corner stools features',
            'indoor brewery': 'which has large brewing tanks with visible pipes and scattered kegs features',
            'bridge': 'which has tall arches with visible cables and surrounding river views features',
            'building facade': 'which has ornate details with visible windows and surrounding cityscape features',
            'bullring': 'which has circular arenas with tiered seating and visible red barriers features',
            'burial chamber': 'which has stone walls with carved engravings and visible sarcophagi features',
            'bus interior': 'which has long rows with visible seats and surrounding handrails features',
            'butchers shop': 'which has display cases with visible meat cuts and hanging sausages features',
            'butte': 'which has steep slopes with flat tops and surrounding desert landscapes features',
            'outdoor cabin': 'which has wooden walls with visible porches and surrounding forest scenery features',
            'cafeteria': 'which has long tables with scattered trays and visible food counters features',
            'campsite': 'which has visible tents with surrounding trees and scattered campfires features',
            'campus': 'which has open quads with visible buildings and scattered students features',
            'natural canal': 'which has narrow water channels with overhanging trees and lush surroundings features',
            'urban canal': 'which has man-made water paths with visible bridges and surrounding buildings features',
            'candy store': 'which has colorful displays with scattered sweets and visible customers features',
            'canyon': 'which has steep rock walls with narrow paths and rugged landscapes features',
            'backseat car interior': 'which has soft upholstery with visible seatbelts and limited legroom features',
            'frontseat car interior': 'which has visible dashboard with steering wheel and scattered controls features',
            'carrousel': 'which has brightly colored horses with circular motion and visible riders features',
            'indoor casino': 'which has brightly lit slot machines with visible poker tables and scattered players features',
            'castle': 'which has tall towers with thick stone walls and visible battlements features',
            'catacomb': 'which has narrow tunnels with ancient engravings and visible skeletal remains features',
            'indoor cathedral': 'which has tall vaulted ceilings with stained glass windows and visible pews features',
            'outdoor cathedral': 'which has intricate stone carvings with visible spires and surrounding open spaces features',
            'indoor cavern': 'which has damp walls with narrow passages and scattered stalagmites features',
            'cemetery': 'which has scattered tombstones with visible mausoleums and surrounding green lawns features',
            'chalet': 'which has wooden balconies with sloped roofs and surrounding mountain scenery features',
            'cheese factory': 'which has large vats with visible conveyor belts and scattered wheels of cheese features',
            'chemistry lab': 'which has glass beakers with scattered chemicals and visible laboratory equipment features',
            'indoor chicken coop': 'which has wire cages with scattered feed and visible perching chickens features',
            'outdoor chicken coop': 'which has wooden structures with scattered nests and surrounding free-range chickens features',
            'childs room': 'which has scattered toys with small beds and colorful decorations features',
            'indoor church': 'which has wooden pews with tall windows and visible altar features',
            'outdoor church': 'which has tall spires with surrounding open grounds and visible entrance features',
            'classroom': 'which has scattered desks with visible blackboards and surrounding educational posters features',
            'clean room': 'which has sterile walls with visible lab equipment and people in protective gear features',
            'cliff': 'which has steep rocky edges with visible drops and surrounding coastal views features',
            'indoor cloister': 'which has arched ceilings with stone columns and visible courtyards features',
            'closet': 'which has hanging clothes with visible shelves and scattered shoes features',
            'clothing store': 'which has racks of clothing with scattered mannequins and visible fitting rooms features',
            'coast': 'which has sandy shores with gentle waves and visible distant horizon features',
            'cockpit': 'which has visible controls with small windows and surrounding instrument panels features',
            'coffee shop': 'which has small tables with scattered patrons and visible baristas features',
            'computer room': 'which has scattered computers with visible keyboards and surrounding desks features',
            'conference center': 'which has large meeting rooms with scattered chairs and visible projection screens features',
            'conference room': 'which has long tables with visible notepads and scattered office chairs features',
            'construction site': 'which has cranes with visible steel beams and scattered construction workers features',
            'control room': 'which has scattered monitors with visible control panels and surrounding operator seats features',
            'outdoor control tower': 'which has tall structures with visible windows and surrounding airfields features',
            'corn field': 'which has tall green stalks with scattered cobs and visible rows features',
            'corral': 'which has wooden fences with scattered horses and visible feeding troughs features',
            'corridor': 'which has long narrow walkways with visible doors and surrounding hallway lights features',
            'cottage garden': 'which has blooming flowers with visible stone paths and surrounding greenery features',
            'courthouse': 'which has tall columns with visible entrance and surrounding justice symbols features',
            'courtroom': 'which has wooden benches with visible witness stands and surrounding judges’ desks features',
            'courtyard': 'which has open spaces with visible arches and surrounding trees features',
            'exterior covered bridge': 'which has wooden planks with visible roof beams and surrounding water features',
            'creek': 'which has shallow water with visible pebbles and surrounding lush foliage features',
            'crevasse': 'which has deep cracks with visible ice walls and surrounding snowy terrain features',
            'crosswalk': 'which has white lines with visible traffic signals and surrounding urban streets features',
            'office cubicle': 'which has partition walls with scattered desks and surrounding computers features',
            'dam': 'which has tall concrete walls with visible water flow and surrounding power equipment features',
            'delicatessen': 'which has glass cases with scattered meats and visible cheese displays features',
            'dentists office': 'which has reclining chairs with visible dental tools and scattered medical charts features',
            'sand desert': 'which has rolling dunes with scattered cacti and visible expansive blue sky features',
            'vegetation desert': 'which has dry scrubland with scattered shrubs and visible rocky terrain features',
            'indoor diner': 'which has red booths with visible counters and scattered customers features',
            'outdoor diner': 'which has visible picnic tables with umbrellas and scattered patrons features',
            'home dinette': 'which has small tables with visible chairs and surrounding family photos features',
            'vehicle dinette': 'which has foldable tables with visible seating and surrounding compact storage features',
            'dining car': 'which has narrow aisles with visible tablecloths and scattered passengers features',
            'dining room': 'which has long tables with visible chairs and surrounding cabinets features',
            'discotheque': 'which has flashing lights with scattered dancers and visible DJ booth features',
            'dock': 'which has wooden platforms with scattered boats and visible water views features',
            'outdoor doorway': 'which has arched frames with visible open spaces and surrounding stone walls features',
            'dorm room': 'which has small beds with visible study desks and scattered personal belongings features',
            'driveway': 'which has paved paths with scattered vehicles and visible garage doors features',
            'outdoor driving range': 'which has open fields with scattered golf balls and visible targets features',
            'drugstore': 'which has tall shelves with scattered medications and visible checkout counters features',
            'electrical substation': 'which has tall metal towers with scattered wires and surrounding transformers features',
            'door elevator': 'which has sliding doors with visible control buttons and surrounding narrow space features',
            'interior elevator': 'which has mirrored walls with visible floor indicators and surrounding metallic surfaces features',
            'elevator shaft': 'which has narrow spaces with visible cables and surrounding concrete walls features',
            'engine room': 'which has large machinery with visible pipes and surrounding control panels features',
            'indoor escalator': 'which has moving steps with visible handrails and scattered passengers features',
            'excavation': 'which has deep pits with scattered rocks and visible construction equipment features',
            'indoor factory': 'which has assembly lines with scattered workers and visible machinery features',
            'fairway': 'which has wide green areas with scattered golf balls and visible holes features',
            'fastfood restaurant': 'which has plastic seats with visible counter and scattered customers features',
            'cultivated field': 'which has rows of crops with visible soil and surrounding farm tools features',
            'wild field': 'which has tall grass with scattered wildflowers and visible hills features',
            'fire escape': 'which has metal stairs with visible platforms and surrounding tall buildings features',
            'fire station': 'which has large red doors with visible fire trucks and surrounding equipment features',
            'indoor firing range': 'which has shooting lanes with visible targets and scattered shooters features',
            'fishpond': 'which has still water with visible koi fish and surrounding green plants features',
            'indoor florist shop': 'which has scattered bouquets with visible flower arrangements and surrounding shelves features',
            'food court': 'which has scattered tables with visible food stalls and surrounding diners features',
            'broadleaf forest': 'which has tall trees with visible green leaves and surrounding underbrush features',
            'needleleaf forest': 'which has scattered pine trees with visible needles and surrounding forest floor features',
            'forest path': 'which has narrow trails with scattered fallen leaves and visible forest trees features',
            'forest road': 'which has gravel surfaces with scattered branches and surrounding tall trees features',
            'formal garden': 'which has neatly trimmed hedges with visible pathways and scattered flower beds features',
            'fountain': 'which has flowing water with visible stone sculptures and surrounding park benches features',
            'galley': 'which has narrow aisles with visible stoves and surrounding cooking utensils features',
            'game room': 'which has scattered arcade machines with visible pool tables and surrounding game players features',
            'indoor garage': 'which has parked cars with visible toolboxes and surrounding shelves features',
            'garbage dump': 'which has piles of trash with visible discarded items and surrounding industrial machinery features',
            'gas station': 'which has fuel pumps with visible convenience store and surrounding vehicles features',
            'exterior gazebo': 'which has wooden benches with visible rooftops and surrounding green spaces features',
            'indoor general store': 'which has tall shelves with scattered goods and visible cash registers features',
            'outdoor general store': 'which has open stalls with scattered products and visible buyers features',
            'gift shop': 'which has scattered souvenirs with visible shelves and surrounding small trinkets features',
            'golf course': 'which has neatly trimmed fairways with scattered golf balls and visible holes features',
            'indoor greenhouse': 'which has glass walls with scattered plants and visible watering systems features',
            'outdoor greenhouse': 'which has transparent roofs with scattered plants and visible irrigation features',
            'indoor gymnasium': 'which has scattered exercise equipment with visible mirrors and surrounding workout areas features',
            'indoor hangar': 'which has parked aircraft with visible large doors and surrounding maintenance equipment features',
            'outdoor hangar': 'which has open spaces with scattered planes and surrounding runways features',
            'harbor': 'which has anchored boats with visible docks and surrounding water features',
            'hayfield': 'which has tall grass with scattered hay bales and visible farming equipment features',
            'heliport': 'which has helicopter pads with visible lights and surrounding landing markers features',
            'herb garden': 'which has small plants with visible walking paths and surrounding greenery features',
            'highway': 'which has long lanes with scattered cars and visible road signs features',
            'hill': 'which has sloping grassy areas with visible rocky surfaces and surrounding landscapes features',
            'home office': 'which has scattered papers with visible desks and surrounding electronic devices features',
            'hospital': 'which has sterile hallways with visible medical equipment and surrounding hospital beds features',
            'hospital room': 'which has patient beds with visible IV stands and surrounding medical devices features',
            'hot spring': 'which has steamy water with visible rocky surroundings and scattered people features',
            'outdoor hot tub': 'which has warm water with visible bubbles and surrounding wooden decks features',
            'outdoor hotel': 'which has tall buildings with scattered balconies and surrounding landscaped gardens features',
            'hotel room': 'which has neatly made beds with visible nightstands and surrounding amenities features',
            'house': 'which has visible windows with scattered front gardens and surrounding driveways features',
            'outdoor hunting lodge': 'which has wooden walls with visible game trophies and surrounding dense forests features',
            'ice cream parlor': 'which has colorful counters with scattered ice cream cones and surrounding patrons features',
            'ice floe': 'which has floating ice sheets with scattered seals and surrounding icy waters features',
            'ice shelf': 'which has thick ice with visible cracks and surrounding frigid waters features',
            'indoor ice skating rink': 'which has smooth ice with scattered skaters and visible walls features',
            'outdoor ice skating rink': 'which has frozen surfaces with scattered skaters and visible benches features',
            'iceberg': 'which has towering ice with visible cracks and surrounding cold waters features',
            'igloo': 'which has dome-shaped structures with visible ice blocks and surrounding snowy plains features',
            'industrial area': 'which has large factories with visible chimneys and surrounding machinery features',
            'outdoor inn': 'which has wooden walls with scattered rocking chairs and surrounding open fields features',
            'islet': 'which has small land masses with scattered vegetation and surrounding clear waters features',
            'indoor jacuzzi': 'which has bubbling water with scattered jets and surrounding tiled floors features',
            'indoor jail': 'which has metal bars with visible guard stations and surrounding cells features',
            'jail cell': 'which has small beds with visible bars and surrounding concrete walls features',
            'jewelry shop': 'which has glass cases with scattered necklaces and visible rings features',
            'kasbah': 'which has thick walls with visible arches and surrounding desert scenery features',
            'indoor kennel': 'which has small cages with scattered dogs and visible feeding bowls features',
            'outdoor kennel': 'which has fenced areas with scattered dogs and visible doghouses features',
            'kindergarden classroom': 'which has small desks with visible colorful drawings and surrounding toys features',
            'kitchen': 'which has visible countertops with scattered utensils and surrounding cooking appliances features',
            'kitchenette': 'which has small counters with scattered plates and surrounding compact appliances features',
            'outdoor labyrinth': 'which has tall hedges with scattered pathways and visible open courtyards features',
            'natural lake': 'which has clear waters with visible shores and surrounding green forests features',
            'landfill': 'which has piles of waste with scattered machinery and visible debris features',
            'landing deck': 'which has flat surfaces with scattered safety markings and visible ships features',
            'laundromat': 'which has washing machines with scattered clothes and visible detergent bottles features',
            'lecture room': 'which has rows of desks with visible whiteboards and surrounding projectors features',
            'indoor library': 'which has tall bookshelves with scattered tables and visible reading lamps features',
            'outdoor library': 'which has open bookshelves with scattered benches and surrounding gardens features',
            'outdoor lido deck': 'which has pool chairs with scattered towels and visible sun umbrellas features',
            'lift bridge': 'which has large mechanical arms with visible crossing lanes and surrounding water features',
            'lighthouse': 'which has tall towers with visible lights and surrounding rocky coastlines features',
            'limousine interior': 'which has leather seats with visible tinted windows and surrounding entertainment systems features',
            'living room': 'which has comfortable couches with scattered cushions and visible coffee tables features',
            'lobby': 'which has spacious seating areas with visible reception desks and surrounding plants features',
            'lock chamber': 'which has large gates with visible boats and surrounding water levels features',
            'locker room': 'which has metal lockers with scattered benches and visible sports gear features',
            'mansion': 'which has grand staircases with scattered chandeliers and visible luxurious decorations features',
            'manufactured home': 'which has prefabricated walls with visible windows and surrounding grassy yards features',
            'indoor market': 'which has bustling stalls with scattered goods and visible overhead banners features',
            'outdoor market': 'which has open-air stands with scattered vendors and visible bustling crowds features',
            'marsh': 'which has wetland areas with scattered reeds and visible shallow water features',
            'martial arts gym': 'which has padded mats with visible punching bags and scattered practice gear features',
            'mausoleum': 'which has stone walls with visible tombs and surrounding solemn monuments features',
            'medina': 'which has narrow streets with visible vendors and scattered traditional shops features',
            'water moat': 'which has calm waters with visible castle walls and surrounding grassy areas features',
            'outdoor monastery': 'which has stone buildings with visible cloisters and surrounding peaceful gardens features',
            'indoor mosque': 'which has ornate prayer halls with visible arches and surrounding domes features',
            'outdoor mosque': 'which has tall minarets with visible courtyards and surrounding gathering areas features',
            'motel': 'which has small rooms with visible parking lots and surrounding signage features',
            'mountain': 'which has steep slopes with scattered rocks and visible snow-capped peaks features',
            'mountain snowy': 'which has frosty slopes with visible icy peaks and surrounding snowdrifts features',
            'indoor movie theater': 'which has plush seats with scattered popcorn and visible projection screens features',
            'indoor museum': 'which has exhibits with visible artworks and surrounding informational plaques features',
            'music store': 'which has instrument displays with visible guitars and surrounding amplifiers features',
            'music studio': 'which has soundproof walls with visible recording equipment and surrounding microphones features',
            'outdoor nuclear power plant': 'which has cooling towers with visible steam and surrounding power lines features',
            'nursery': 'which has small cribs with scattered toys and visible baby monitors features',
            'oast house': 'which has conical roofs with visible drying hops and surrounding farmlands features',
            'outdoor observatory': 'which has large telescopes with visible domes and surrounding starry skies features',
            'ocean': 'which has endless waves with visible horizon and surrounding blue waters features',
            'office': 'which has cubicles with visible computers and surrounding office supplies features',
            'office building': 'which has glass windows with scattered workspaces and surrounding elevators features',
            'outdoor oil refinery': 'which has tall chimneys with visible pipelines and surrounding industrial equipment features',
            'oilrig': 'which has large platforms with visible drilling equipment and surrounding ocean features',
            'operating room': 'which has surgical tables with visible medical instruments and surrounding sterile equipment features',
            'orchard': 'which has rows of fruit trees with scattered baskets and surrounding green fields features',
            'outdoor outhouse': 'which has wooden structures with visible doors and surrounding grassy areas features',
            'pagoda': 'which has tiered roofs with visible wooden pillars and surrounding gardens features',
            'palace': 'which has grand halls with visible ornate decorations and surrounding royal furnishings features',
            'pantry': 'which has stacked shelves with visible canned goods and surrounding kitchen supplies features',
            'park': 'which has green lawns with scattered trees and visible benches features',
            'indoor parking garage': 'which has parked cars with scattered signs and visible concrete pillars features',
            'outdoor parking garage': 'which has multi-level structures with visible ramps and scattered vehicles features',
            'parking lot': 'which has parked cars with visible white lines and surrounding asphalt features',
            'parlor': 'which has cozy chairs with visible fireplaces and surrounding small tables features',
            'pasture': 'which has green fields with scattered grazing cows and visible fences features',
            'patio': 'which has outdoor furniture with visible stone floors and surrounding potted plants features',
            'pavilion': 'which has open-air roofs with visible columns and surrounding gathering spaces features',
            'pharmacy': 'which has medicine shelves with visible cash registers and surrounding customers features',
            'phone booth': 'which has glass doors with visible telephones and surrounding street features',
            'physics laboratory': 'which has scientific equipment with visible workbenches and surrounding computers features',
            'picnic area': 'which has scattered tables with visible barbecues and surrounding grassy fields features',
            'indoor pilothouse': 'which has navigation controls with visible steering wheels and surrounding radar screens features',
            'outdoor planetarium': 'which has domed roofs with visible telescopes and surrounding open skies features',
            'playground': 'which has swings and slides with scattered children and surrounding sand features',
            'playroom': 'which has scattered toys with visible colorful walls and surrounding small chairs features',
            'plaza': 'which has open squares with scattered fountains and visible cobblestone streets features',
            'indoor podium': 'which has elevated stages with visible microphones and surrounding seated audiences features',
            'outdoor podium': 'which has open platforms with visible speakers and surrounding gathering crowds features',
            'pond': 'which has calm waters with visible lily pads and surrounding reeds features',
            'establishment poolroom': 'which has billiard tables with visible cues and surrounding hanging lights features',
            'home poolroom': 'which has wooden tables with scattered balls and visible scoreboards features',
            'outdoor power plant': 'which has large turbines with visible electrical equipment and surrounding power lines features',
            'promenade deck': 'which has long walkways with visible benches and surrounding ocean views features',
            'indoor pub': 'which has wooden bars with scattered stools and visible beer taps features',
            'pulpit': 'which has raised platforms with visible religious symbols and surrounding seated congregations features',
            'putting green': 'which has short grass with scattered golf balls and visible flagpoles features',
            'racecourse': 'which has oval tracks with visible grandstands and surrounding cheering crowds features',
            'raceway': 'which has long tracks with visible racing cars and surrounding safety barriers features',
            'raft': 'which has wooden platforms with visible ropes and surrounding calm water features',
            'railroad track': 'which has metal rails with visible wooden ties and surrounding open landscapes features',
            'rainforest': 'which has dense vegetation with visible tall trees and surrounding humidity features',
            'reception': 'which has front desks with visible computers and surrounding waiting areas features',
            'recreation room': 'which has game tables with scattered video game consoles and surrounding seating areas features',
            'residential neighborhood': 'which has rows of houses with scattered trees and visible driveways features',
            'restaurant': 'which has dining tables with scattered menus and visible waitstaff features',
            'restaurant kitchen': 'which has stainless steel counters with visible cooking utensils and surrounding appliances features',
            'restaurant patio': 'which has outdoor tables with visible umbrellas and surrounding diners features',
            'rice paddy': 'which has flooded fields with scattered rice plants and surrounding rural scenery features',
            'riding arena': 'which has sandy floors with visible horse jumps and surrounding grandstands features',
            'river': 'which has flowing water with scattered boats and visible banks features',
            'rock arch': 'which has natural stone formations with visible curved structures and surrounding open skies features',
            'rope bridge': 'which has a long flexible structure to cross features',
            'ruin': 'which has remnants of a historical building or site features',
            'runway': 'which has a long strip for aircraft takeoff landing features',
            'sandbar': 'which has a narrow landform made of sand features',
            'sandbox': 'which has a container filled with sand for play features',
            'sauna': 'which has a small room designed for heat sessions features',
            'schoolhouse': 'which has a building for educational purposes features',
            'sea cliff': 'which has steep rocky edges overlooking the ocean features',
            'server room': 'which has a space housing computer servers equipment features',
            'shed': 'which has a small structure for storing tools equipment features',
            'shoe shop': 'which has a store selling various types of footwear features',
            'shopfront': 'which has a display window for retail goods features',
            'indoor shopping mall': 'which has a large enclosed retail complex features',
            'shower': 'which has a space designed for bathing with water features',
            'skatepark': 'which has a recreational area for skateboarding activities features',
            'ski lodge': 'which has a building providing accommodation for skiers features',
            'ski resort': 'which has a destination for skiing snow activities features',
            'ski slope': 'which has a hill or ramp for skiing purposes features',
            'sky': 'which has a vast expanse visible above the earth features',
            'skyscraper': 'which has a very tall multi-story building structure features',
            'slum': 'which has a densely populated urban area with poor conditions features',
            'snowfield': 'which has a large area covered with snow features',
            'squash court': 'which has a walled area for playing squash game features',
            'stable': 'which has a building for housing horses or livestock features',
            'baseball stadium': 'which has a venue for playing baseball games features',
            'football stadium': 'which has a venue for playing football matches features',
            'indoor stage': 'which has a raised platform for performances indoors features',
            'staircase': 'which has a series of steps for going up down features',
            'street': 'which has a public road in a city or town features',
            'subway interior': 'which has the inside part of an underground train features',
            'platform subway station': 'which has a waiting area for subway trains features',
            'supermarket': 'which has a large self-service store for groceries features',
            'sushi bar': 'which has a restaurant serving sushi dishes features',
            'swamp': 'which has a wetland area with waterlogged soil features',
            'indoor swimming pool': 'which has a pool located inside a building features',
            'outdoor swimming pool': 'which has a pool located outside in open air features',
            'indoor synagogue': 'which has a building for Jewish worship indoors features',
            'outdoor synagogue': 'which has a space for Jewish worship outside features',
            'television studio': 'which has a facility for producing TV programs features',
            'east asia temple': 'which has a place of worship in East Asia features',
            'south asia temple': 'which has a sacred site located in South Asia features',
            'indoor tennis court': 'which has a court for tennis located indoors features',
            'outdoor tennis court': 'which has a court for tennis located outdoors features',
            'outdoor tent': 'which has a temporary shelter set up outside features',
            'indoor procenium theater': 'which has a theater with a proscenium arch features',
            'indoor seats theater': 'which has a performance venue with seating features',
            'thriftshop': 'which has a store selling secondhand goods at low prices features',
            'throne room': 'which has a ceremonial room for a monarch’s throne features',
            'ticket booth': 'which has a small structure for selling tickets features',
            'toll plaza': 'which has a section of road where tolls are collected features',
            'topiary garden': 'which has a garden with trimmed shrubs shaped decoratively features',
            'tower': 'which has a tall structure used for various purposes features',
            'toyshop': 'which has a store selling toys for children features',
            'outdoor track': 'which has a running surface located outside features',
            'train railway': 'which has a set of tracks for trains to travel features',
            'platform train station': 'which has an area where trains stop to board features',
            'tree farm': 'which has a plot of land for growing trees features',
            'tree house': 'which has a structure built in the branches of a tree features',
            'trench': 'which has a long narrow excavation in the ground features',
            'coral reef underwater': 'which has a vibrant marine ecosystem below water features',
            'utility room': 'which has a space for laundry and storage equipment features',
            'valley': 'which has a low area between hills or mountains features',
            'van interior': 'which has the inside space of a van for passengers features',
            'vegetable garden': 'which has a cultivated area for growing vegetables features',
            'veranda': 'which has a roofed platform along the outside of a building features',
            'veterinarians office': 'which has a place for treating animal patients features',
            'viaduct': 'which has a bridge that carries a road or railway over a valley features',
            'videostore': 'which has a shop that rents movies and video games features',
            'village': 'which has a small community in a rural area features',
            'vineyard': 'which has a plantation where grapes are grown features',
            'volcano': 'which has a mountain with an opening for lava emissions features',
            'indoor volleyball court': 'which has a court for playing volleyball inside features',
            'outdoor volleyball court': 'which has a court for playing volleyball outside features',
            'waiting room': 'which has a space for people to sit and wait features',
            'indoor warehouse': 'which has a large building for storing goods indoors features',
            'water tower': 'which has a structure for storing and distributing water features',
            'block waterfall': 'which has a waterfall cascading over a rocky ledge features',
            'fan waterfall': 'which has a waterfall shaped like a fan structure features',
            'plunge waterfall': 'which has a waterfall that drops straight down features',
            'watering hole': 'which has a natural or artificial water source for animals features',
            'wave': 'which has a moving ridge of water on the surface features',
            'wet bar': 'which has a small bar area with water supply facilities features',
            'wheat field': 'which has a large area planted with wheat crops features',
            'wind farm': 'which has a collection of wind turbines generating energy features',
            'windmill': 'which has a structure converting wind energy into power features',
            'barrel storage wine cellar': 'which has a room for aging wine in barrels features',
            'bottle storage wine cellar': 'which has a cellar for storing wine bottles features',
            'indoor wrestling ring': 'which has a space for wrestling matches indoors features',
            'yard': 'which has an area surrounding a house for outdoor activities features',
            'youth hostel': 'which has an affordable lodging for young travelers features'
        }


    def read_data(self, cname2lab, text_file):
        text_file = os.path.join(self.dataset_dir, text_file)
        items = []

        with open(text_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                imname = line.strip()[1:]  # remove /
                classname = os.path.dirname(imname)
                label = cname2lab[classname]
                impath = os.path.join(self.image_dir, imname)

                names = classname.split("/")[1:]  # remove 1st letter
                names = names[::-1]  # put words like indoor/outdoor at first
                classname = " ".join(names)

                item = Datum(impath=impath, label=label, classname=classname)
                items.append(item)

        return items
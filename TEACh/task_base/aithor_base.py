
import numpy as np
import os
# import hyperparams as hyp
from arguments import args
from argparse import Namespace
# from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter 
from utils.ai2thor_docker.ai2thor_docker.x_server import startx
import ipdb
st = ipdb.set_trace
import utils.dist

class Base():

    def __init__(self):   

        self.W = args.W #480
        self.H = args.H #480

        # # alfred split
        # self.mapnames_train = ['FloorPlan223', 'FloorPlan13', 'FloorPlan429', 'FloorPlan222', 'FloorPlan321', 'FloorPlan22', 'FloorPlan428', 'FloorPlan324',
        #                         'FloorPlan401', 'FloorPlan207', 'FloorPlan309', 'FloorPlan210', 'FloorPlan18', 'FloorPlan8', 'FloorPlan408', 'FloorPlan227',
        #                          'FloorPlan418', 'FloorPlan230', 'FloorPlan422', 'FloorPlan318', 'FloorPlan214', 'FloorPlan3', 'FloorPlan216', 'FloorPlan212',
        #                           'FloorPlan307', 'FloorPlan24', 'FloorPlan416', 'FloorPlan426', 'FloorPlan421', 'FloorPlan423', 'FloorPlan415', 'FloorPlan228', 
        #                           'FloorPlan205', 'FloorPlan409', 'FloorPlan410', 'FloorPlan16', 'FloorPlan427', 'FloorPlan229', 'FloorPlan15', 'FloorPlan213',
        #                            'FloorPlan411', 'FloorPlan301', 'FloorPlan11', 'FloorPlan414', 'FloorPlan221', 'FloorPlan319', 'FloorPlan326', 'FloorPlan7',
        #                             'FloorPlan405', 'FloorPlan27', 'FloorPlan218', 'FloorPlan430', 'FloorPlan328', 'FloorPlan304', 'FloorPlan211', 'FloorPlan419',
        #                              'FloorPlan14', 'FloorPlan20', 'FloorPlan25', 'FloorPlan312', 'FloorPlan217', 'FloorPlan204', 'FloorPlan30', 'FloorPlan314',
        #                               'FloorPlan203', 'FloorPlan313', 'FloorPlan6', 'FloorPlan5', 'FloorPlan21', 'FloorPlan420', 'FloorPlan327', 'FloorPlan306',
        #                                'FloorPlan208', 'FloorPlan224', 'FloorPlan316', 'FloorPlan28', 'FloorPlan225', 'FloorPlan209', 'FloorPlan26', 'FloorPlan303',
        #                                 'FloorPlan311', 'FloorPlan310', 'FloorPlan322', 'FloorPlan317', 'FloorPlan413', 'FloorPlan12', 'FloorPlan323', 'FloorPlan17',
        #                                  'FloorPlan330', 'FloorPlan412', 'FloorPlan302', 'FloorPlan19', 'FloorPlan305', 'FloorPlan407', 'FloorPlan4', 'FloorPlan403',
        #                                   'FloorPlan23', 'FloorPlan2', 'FloorPlan320', 'FloorPlan417', 'FloorPlan206', 'FloorPlan402', 'FloorPlan201', 'FloorPlan220',
        #                                    'FloorPlan329', 'FloorPlan406', 'FloorPlan1', 'FloorPlan202'
        #                                    ]

        # teach split
        self.mapnames_train = [
            "FloorPlan306", "FloorPlan407", "FloorPlan207", "FloorPlan202", "FloorPlan417", "FloorPlan210", "FloorPlan428",
             "FloorPlan21", "FloorPlan304", "FloorPlan427", "FloorPlan204", "FloorPlan415", "FloorPlan327", "FloorPlan312",
              "FloorPlan318", "FloorPlan30", "FloorPlan24", "FloorPlan214", "FloorPlan313", "FloorPlan418", "FloorPlan206",
               "FloorPlan420", "FloorPlan426", "FloorPlan317", "FloorPlan307", "FloorPlan3", "FloorPlan14", "FloorPlan324",
                "FloorPlan227", "FloorPlan20", "FloorPlan320", "FloorPlan6", "FloorPlan17", "FloorPlan224", "FloorPlan25",
                 "FloorPlan328", "FloorPlan323", "FloorPlan15", "FloorPlan314", "FloorPlan13", "FloorPlan310", "FloorPlan229",
                  "FloorPlan329", "FloorPlan211", "FloorPlan212", "FloorPlan423", "FloorPlan411", "FloorPlan203", "FloorPlan406",
                   "FloorPlan2", "FloorPlan208", "FloorPlan330", "FloorPlan18", "FloorPlan213", "FloorPlan405", "FloorPlan416",
                    "FloorPlan303", "FloorPlan410", "FloorPlan228", "FloorPlan421", "FloorPlan216", "FloorPlan223", "FloorPlan311",
                     "FloorPlan225", "FloorPlan305", "FloorPlan205", "FloorPlan412", "FloorPlan16", "FloorPlan1", "FloorPlan429", 
                     "FloorPlan408", "FloorPlan419", "FloorPlan302", "FloorPlan23", "FloorPlan321", "FloorPlan309", "FloorPlan322", 
                     "FloorPlan12", "FloorPlan218", "FloorPlan5", "FloorPlan301", "FloorPlan316", "FloorPlan7", "FloorPlan409", 
                     "FloorPlan430", "FloorPlan209", "FloorPlan27", "FloorPlan401", "FloorPlan221", "FloorPlan402", "FloorPlan414", 
                     "FloorPlan220", "FloorPlan217", "FloorPlan22", "FloorPlan422", "FloorPlan403", "FloorPlan26", "FloorPlan319", "FloorPlan230", 
                     "FloorPlan11", "FloorPlan8", "FloorPlan201", "FloorPlan4", "FloorPlan28", "FloorPlan413", "FloorPlan222", "FloorPlan326", "FloorPlan19"
                     ]

        self.mapnames_val = ['FloorPlan424', 'FloorPlan219', 'FloorPlan308', 'FloorPlan10']

        # self.mapnames_val_seen = ["FloorPlan303", "FloorPlan406", "FloorPlan28", "FloorPlan224", "FloorPlan225", "FloorPlan412", "FloorPlan4", "FloorPlan314", "FloorPlan426", "FloorPlan18", "FloorPlan7", "FloorPlan22", "FloorPlan223", "FloorPlan5", "FloorPlan302", "FloorPlan14", "FloorPlan408", "FloorPlan310", "FloorPlan304", "FloorPlan26", "FloorPlan311", "FloorPlan419", "FloorPlan328", "FloorPlan405", "FloorPlan409", "FloorPlan204", "FloorPlan202", "FloorPlan415", "FloorPlan229", "FloorPlan19", "FloorPlan414", "FloorPlan16", "FloorPlan207", "FloorPlan301", "FloorPlan6", "FloorPlan218", "FloorPlan305", "FloorPlan330", "FloorPlan230", "FloorPlan11", "FloorPlan222", "FloorPlan423", "FloorPlan25", "FloorPlan417", "FloorPlan2", "FloorPlan402", "FloorPlan410", "FloorPlan24", "FloorPlan213", "FloorPlan316", "FloorPlan13", "FloorPlan17", "FloorPlan201", "FloorPlan23", "FloorPlan407", "FloorPlan326", "FloorPlan214", "FloorPlan216", "FloorPlan27", "FloorPlan401", "FloorPlan1", "FloorPlan320", "FloorPlan3", "FloorPlan203", "FloorPlan20", "FloorPlan313", "FloorPlan309", "FloorPlan227", "FloorPlan427", "FloorPlan15", "FloorPlan323", "FloorPlan212", "FloorPlan329", "FloorPlan30", "FloorPlan12", "FloorPlan324", "FloorPlan206", "FloorPlan413", "FloorPlan21", "FloorPlan327", "FloorPlan428", "FloorPlan403", "FloorPlan418", "FloorPlan429", "FloorPlan8", "FloorPlan422", "FloorPlan205", "FloorPlan318"]

        self.mapnames_test = ['FloorPlan315', 'FloorPlan29', 'FloorPlan325', 'FloorPlan425', 'FloorPlan215', 'FloorPlan9', 'FloorPlan404', 'FloorPlan226']

        self.class_agnostic = False
        self.random_select = True

        self.include_classes = [
            'ShowerDoor', 'Cabinet', 'CounterTop', 'Sink', 'Towel', 'HandTowel', 'TowelHolder', 'SoapBar', 
            'ToiletPaper', 'ToiletPaperHanger', 'HandTowelHolder', 'SoapBottle', 'GarbageCan', 'Candle', 'ScrubBrush', 
            'Plunger', 'SinkBasin', 'Cloth', 'SprayBottle', 'Toilet', 'Faucet', 'ShowerHead', 'Box', 'Bed', 'Book', 
            'DeskLamp', 'BasketBall', 'Pen', 'Pillow', 'Pencil', 'CellPhone', 'KeyChain', 'Painting', 'CreditCard', 
            'AlarmClock', 'CD', 'Laptop', 'Drawer', 'SideTable', 'Chair', 'Blinds', 'Desk', 'Curtains', 'Dresser', 
            'Watch', 'Television', 'WateringCan', 'Newspaper', 'FloorLamp', 'RemoteControl', 'HousePlant', 'Statue', 
            'Ottoman', 'ArmChair', 'Sofa', 'DogBed', 'BaseballBat', 'TennisRacket', 'VacuumCleaner', 'Mug', 'ShelvingUnit', 
            'Shelf', 'StoveBurner', 'Apple', 'Lettuce', 'Bottle', 'Egg', 'Microwave', 'CoffeeMachine', 'Fork', 'Fridge', 
            'WineBottle', 'Spatula', 'Bread', 'Tomato', 'Pan', 'Cup', 'Pot', 'SaltShaker', 'Potato', 'PepperShaker', 
            'ButterKnife', 'StoveKnob', 'Toaster', 'DishSponge', 'Spoon', 'Plate', 'Knife', 'DiningTable', 'Bowl', 
            'LaundryHamper', 'Vase', 'Stool', 'CoffeeTable', 'Poster', 'Bathtub', 'TissueBox', 'Footstool', 'BathtubBasin', 
            'ShowerCurtain', 'TVStand', 'Boots', 'RoomDecor', 'PaperTowelRoll', 'Ladle', 'Kettle', 'Safe', 'GarbageBag', 'TeddyBear', 
            'TableTopDecor', 'Dumbbell', 'Desktop', 'AluminumFoil', 'Window', 'LightSwitch', 'Lamp',
            'AppleSliced', 'BreadSliced', 'LettuceSliced', 'PotatoSliced', 'TomatoSliced']

        # self.include_classes = [
        #         'AlarmClock',
        #         'Apple',
        #         'ArmChair',
        #         'BaseballBat',
        #         'BasketBall',
        #         'Bathtub',
        #         'BathtubBasin',
        #         'Bed',
        #         'Blinds',
        #         'Book',
        #         'Boots',
        #         'Bowl',
        #         'Box',
        #         'Bread',
        #         'ButterKnife',
        #         'Cabinet',
        #         'Candle',
        #         'Cart',
        #         'CD',
        #         'CellPhone',
        #         'Chair',
        #         'Cloth',
        #         'CoffeeMachine',
        #         'CounterTop',
        #         'CreditCard',
        #         'Cup',
        #         'Curtains',
        #         'Desk',
        #         'DeskLamp',
        #         'DishSponge',
        #         'Drawer',
        #         'Dresser',
        #         'Egg',
        #         'FloorLamp',
        #         'Footstool',
        #         'Fork',
        #         'Fridge',
        #         'GarbageCan',
        #         'Glassbottle',
        #         'HandTowel',
        #         'HandTowelHolder',
        #         'HousePlant',
        #         'Kettle',
        #         'KeyChain',
        #         'Knife',
        #         'Ladle',
        #         'Laptop',
        #         'LaundryHamper',
        #         'LaundryHamperLid',
        #         'Lettuce',
        #         'LightSwitch',
        #         'Microwave',
        #         'Mirror',
        #         'Mug',
        #         'Newspaper',
        #         'Ottoman',
        #         'Painting',
        #         'Pan',
        #         'PaperTowel',
        #         'PaperTowelRoll',
        #         'Pen',
        #         'Pencil',
        #         'PepperShaker',
        #         'Pillow',
        #         'Plate',
        #         'Plunger',
        #         'Poster',
        #         'Pot',
        #         'Potato',
        #         'RemoteControl',
        #         'Safe',
        #         'SaltShaker',
        #         'ScrubBrush',
        #         'Shelf',
        #         'ShowerDoor',
        #         'ShowerGlass',
        #         'Sink',
        #         'SinkBasin',
        #         'SoapBar',
        #         'SoapBottle',
        #         'Sofa',
        #         'Spatula',
        #         'Spoon',
        #         'SprayBottle',
        #         'Statue',
        #         'StoveBurner',
        #         'StoveKnob',
        #         'DiningTable',
        #         'CoffeeTable',
        #         'SideTable',
        #         'TeddyBear',
        #         'Television',
        #         'TennisRacket',
        #         'TissueBox',
        #         'Toaster',
        #         'Toilet',
        #         'ToiletPaper',
        #         'ToiletPaperHanger',
        #         'ToiletPaperRoll',
        #         'Tomato',
        #         'Towel',
        #         'TowelHolder',
        #         'TVStand',
        #         'Vase',
        #         'Watch',
        #         'WateringCan',
        #         'Window',
        #         'WineBottle',
        #     ]

        self.object_detector_objs = ['AlarmClock', 'Apple', 'AppleSliced', 'BaseballBat', 'BasketBall', 'Book', 'Bowl', 'Box', 'Bread', 'BreadSliced', 'ButterKnife', 'CD', 'Candle', 'CellPhone', 'Cloth', 'CreditCard', 'Cup', 'DeskLamp', 'DishSponge', 'Egg', 'Faucet', 'FloorLamp', 'Fork', 'Glassbottle', 'HandTowel', 'HousePlant', 'Kettle', 'KeyChain', 'Knife', 'Ladle', 'Laptop', 'LaundryHamperLid', 'Lettuce', 'LettuceSliced', 'LightSwitch', 'Mug', 'Newspaper', 'Pan', 'PaperTowel', 'PaperTowelRoll', 'Pen', 'Pencil', 'PepperShaker', 'Pillow', 'Plate', 'Plunger', 'Pot', 'Potato', 'PotatoSliced', 'RemoteControl', 'SaltShaker', 'ScrubBrush', 'ShowerDoor', 'SoapBar', 'SoapBottle', 'Spatula', 'Spoon', 'SprayBottle', 'Statue', 'StoveKnob', 'TeddyBear', 'Television', 'TennisRacket', 'TissueBox', 'ToiletPaper', 'ToiletPaperRoll', 'Tomato', 'TomatoSliced', 'Towel', 'Vase', 'Watch', 'WateringCan', 'WineBottle']

        self.alfworld_receptacles = [
                'BathtubBasin',
                'Bowl',
                'Cup',
                'Drawer',
                'Mug',
                'Plate',
                'Shelf',
                'SinkBasin',
                'Box',
                'Cabinet',
                'CoffeeMachine',
                'CounterTop',
                'Fridge',
                'GarbageCan',
                'HandTowelHolder',
                'Microwave',
                'PaintingHanger',
                'Pan',
                'Pot',
                'StoveBurner',
                'DiningTable',
                'CoffeeTable',
                'SideTable',
                'ToiletPaperHanger',
                'TowelHolder',
                'Safe',
                'BathtubBasin',
                'ArmChair',
                'Toilet',
                'Sofa',
                'Ottoman',
                'Dresser',
                'LaundryHamper',
                'Desk',
                'Bed',
                'Cart',
                'TVStand',
                'Toaster',
        ]

        # self.include_classes = list(set(self.object_detector_objs + self.alfworld_receptacles))

        # self.include_classes = [
        #     'ShowerDoor', 'Cabinet', 'CounterTop', 'Sink', 'Towel', 'HandTowel', 'TowelHolder', 'SoapBar', 
        #     'ToiletPaper', 'ToiletPaperHanger', 'HandTowelHolder', 'SoapBottle', 'GarbageCan', 'Candle', 'ScrubBrush', 
        #     'Plunger', 'SinkBasin', 'Cloth', 'SprayBottle', 'Toilet', 'Faucet', 'ShowerHead', 'Box', 'Bed', 'Book', 
        #     'DeskLamp', 'BasketBall', 'Pen', 'Pillow', 'Pencil', 'CellPhone', 'KeyChain', 'Painting', 'CreditCard', 
        #     'AlarmClock', 'CD', 'Laptop', 'Drawer', 'SideTable', 'Chair', 'Blinds', 'Desk', 'Curtains', 'Dresser', 
        #     'Watch', 'Television', 'WateringCan', 'Newspaper', 'FloorLamp', 'RemoteControl', 'HousePlant', 'Statue', 
        #     'Ottoman', 'ArmChair', 'Sofa', 'DogBed', 'BaseballBat', 'TennisRacket', 'VacuumCleaner', 'Mug', 'ShelvingUnit', 
        #     'Shelf', 'StoveBurner', 'Apple', 'Lettuce', 'Bottle', 'Egg', 'Microwave', 'CoffeeMachine', 'Fork', 'Fridge', 
        #     'WineBottle', 'Spatula', 'Bread', 'Tomato', 'Pan', 'Cup', 'Pot', 'SaltShaker', 'Potato', 'PepperShaker', 
        #     'ButterKnife', 'StoveKnob', 'Toaster', 'DishSponge', 'Spoon', 'Plate', 'Knife', 'DiningTable', 'Bowl', 
        #     'LaundryHamper', 'Vase', 'Stool', 'CoffeeTable', 'Poster', 'Bathtub', 'TissueBox', 'Footstool', 'BathtubBasin', 
        #     'ShowerCurtain', 'TVStand', 'Boots', 'RoomDecor', 'PaperTowelRoll', 'Ladle', 'Kettle', 'Safe', 'GarbageBag', 'TeddyBear', 
        #     'TableTopDecor', 'Dumbbell', 'Desktop', 'AluminumFoil', 'Window', 'LightSwitch', 'Lamp',
        #     'AppleSliced', 'BreadSliced', 'LettuceSliced', 'PotatoSliced', 'TomatoSliced']

        alfred_objects, object_mapping  = get_alfred_constants()
        self.OBJECTS_LOWER_TO_UPPER = {obj.lower(): obj for obj in self.include_classes}

        for obj in alfred_objects:
            if obj not in self.include_classes:
                self.include_classes.append(obj)

        self.include_classes.append('no_object') # ddetr has no object class

        self.z_params = navigation_calibration_params()

        self.name_to_id = {}
        self.id_to_name = {}
        self.instance_counter = {}
        idx = 0
        for name in self.include_classes:
            self.name_to_id[name] = idx
            self.id_to_name[idx] = name
            self.instance_counter[name] = 0
            idx += 1

        self.name_to_parsed_name = {
            "showerdoor": "shower door",
            "handtowel": "hand towel",
            "towelholder": "towel holder",
            "soapbar": "soap bar",
            "soapbottle": "soap bottle",
            "toiletpaper": "toilet paper",
            "toiletpaperhanger": "toilet paper hanger",
            "handtowelholder": "hand towel holder",
            "garbagecan": "garbage can",
            "scrubbrush": "brush",
            "sinkbasin": "sink",
            "spraybottle": "spray bottle",
            "showerhead": "shower head",
            "desklamp": "desk lamp",
            "keychain": "key chain",
            "creditcard": "credit card",
            "alarmclock": "alarm clock",
            "sidetable": "side table",
            "wateringcan": "watering can",
            "floorlamp": "floor lamp",
            "remotecontrol": "remote control",
            "houseplant": "house plant",
            "dogbed": "dog bed",
            "baseballbat": "baseball bat",
            "tennisracket": "tennis racket",
            "vacuumcleaner": "vacuum cleaner",
            "shelvingunit": "shelving unit",
            "stoveburner": "stove burner",
            "coffeemachine": "coffee machine",
            "winebottle": "wine bottle",
            "peppershaker": "pepper shaker",
            "stoveknob": "stove knob",
            "dishsponge": "dish sponge",
            "diningtable": "dining table",
            "laundryhamper": "laundry hamper",
            "butterknife": "butterknife",
            "coffeetable": "coffee table",
            "poster": "poster",
            "tissuebox": "tissue box",
            "bathtubbasin": "bathtub",
            "showercurtain": "shower curtain",
            "tvstand": "television stand",
            "roomdecor": "room decor",
            "papertowelroll": "paper towel roll",
            "garbagebag": "garbage bag",
            "teddybear": "teddy bear",
            "tabletopdecor": "table top decor",
            "aluminumfoil": "aluminum foil",
            "lightswitch": "light switch",
            "glassbottle": "glass bottle",
            "laundryhamperlid": "laundry hamper lid",
            "papertowel": "paper towel",
            "showerglass": "shower glass",
            "toiletpaperroll": "toilet paper roll",
            "applesliced": "sliced apple",
            "breadsliced": "sliced bread",
            "lettucesliced": "sliced lettuce",
            "potatosliced": "sliced potato",
            "tomatosliced": "slcied tomato",
            "no_object": "no object",
            'toggleon': "turn on",
            "toggleoff": "turn off",
            'pickup': 'pick up',
        }
        
        # self.name_to_mapped_name = {'AppleSliced':'Apple', 'BreadSliced':'Bread', 'LettuceSliced':'Lettuce', 'PotatoSliced':'Potato', 'TomatoSliced':'Tomato', 'DeskLamp':'FloorLamp'}
        self.name_to_mapped_name = {'DeskLamp':'FloorLamp', 'Lamp':'FloorLamp', 'SinkBasin':'Sink', 'BathtubBasin':'Bathtub'}
        # self.name_to_mapped_name = {'DeskLamp':'FloorLamp', 'SinkBasin':'Sink', 'BathtubBasin':'Bathtub'}
        self.id_to_mapped_id = {self.name_to_id[k]:self.name_to_id[v] for k, v in self.name_to_mapped_name.items()}
                
        self.set_name = args.set_name #'test'
        print("set name=", self.set_name)
        self.data_path = args.data_path  #f'.'
        self.checkpoint_root = self.data_path + '/checkpoints'
        if not os.path.exists(self.checkpoint_root):
            os.mkdir(self.checkpoint_root)
        # self.log_dir = '.' + '/tb' + '/' + self.set_name
        self.checkpoint_path = self.checkpoint_root + '/' + self.set_name        
        
        if self.set_name != 'test00':
            if not os.path.exists(self.checkpoint_path):
                if utils.dist.is_main_process():
                    os.mkdir(self.checkpoint_path)
            # elif args.distributed:
            #     pass
            # else:
            #     print(self.checkpoint_path)
            #     val = input("Path exists. Delete folder? [y/n]: ")
            #     if val == 'y':
            #         import shutil
            #         shutil.rmtree(self.checkpoint_path)
            #         os.mkdir(self.checkpoint_path)
            #     else:
            #         print("ENDING")
            #         assert(False)

            # if not os.path.exists(self.log_dir):
            #     if utils.dist.is_main_process():
            #         os.mkdir(self.log_dir)
            # elif args.distributed:
            #     pass
            # else:
            #     print(self.log_dir)
            #     val = input("Path exists. Delete folder? [y/n]: ")
            #     if val == 'y':
            #         for filename in os.listdir(self.log_dir):
            #             file_path = os.path.join(self.log_dir, filename)
            #             try:
            #                 if os.path.isfile(file_path) or os.path.islink(file_path):
            #                     os.unlink(file_path)
            #                 elif os.path.isdir(file_path):
            #                     shutil.rmtree(file_path)
            #             except Exception as e:
            #                 print('Failed to delete %s. Reason: %s' % (file_path, e))
            #     else:
            #         print("ENDING")
            #         assert(False)


        self.fov = args.fov

        actions = [
            'MoveAhead', 
            'RotateRight', 
            'RotateLeft', 
            'LookDown',
            'LookUp',
            'PickupObject', 
            'PutObject', 
            'OpenObject', 
            'CloseObject', 
            'SliceObject',
            'ToggleObjectOn',
            'ToggleObjectOff',
            'Done',
            ]

        subgoals = [
            'PickupObject', 
            'PutObject', 
            'OpenObject', 
            'CloseObject', 
            'SliceObject',
            'GotoLocation',
            'HeatObject',
            "ToggleObject",
            "CleanObject",
            "HeatObject",
            "CoolObject",
            ]

        
        self.actions = {i:actions[i] for i in range(len(actions))}
        self.actions2idx = {actions[i]:i for i in range(len(actions))}
        self.subgoals2idx = {subgoals[i]:i for i in range(len(subgoals))}
        self.idx2subgoals = {i:subgoals[i] for i in range(len(subgoals))}

        # self.action_mapping = {
        #     "Pass":0, 
        #     "MoveAhead":1, 
        #     # "MoveLeft":2, 
        #     # "MoveRight":3, 
        #     # "MoveBack":4, 
        #     "RotateRight":5, 
        #     "RotateLeft":6, 
        #     "LookUp":9, 
        #     "LookDown":10,
        #     }

        hfov = float(self.fov) * np.pi / 180.
        self.pix_T_camX = np.array([
            [(self.W/2.)*1 / np.tan(hfov / 2.), 0., 0., 0.],
            [0., (self.H/2.)*1 / np.tan(hfov / 2.), 0., 0.],
            [0., 0.,  1, 0],
            [0., 0., 0, 1]])
        self.pix_T_camX[0,2] = self.W/2.
        self.pix_T_camX[1,2] = self.H/2.

        self.agent_height = 1.5759992599487305

        # fix these
        self.STEP_SIZE = args.STEP_SIZE #0.25 # move step size
        self.DT = args.DT #45 # yaw rotation degrees
        self.HORIZON_DT = args.HORIZON_DT #30 # pitch rotation degrees

        self.obs = Namespace()
        self.obs.STEP_SIZE = self.STEP_SIZE
        self.obs.DT = self.DT
        self.obs.HORIZON_DT = self.HORIZON_DT

        self.visibilityDistance = args.visibilityDistance

        self.obs.camera_height = 1.5759992599487305 #event.metadata['cameraPosition']['y']
        # obs.image_list = [event.frame]
        # obs.depth_map_list = [event.depth_frame]
        self.obs.camera_aspect_ratio = [self.H, self.W]
        self.obs.camera_field_of_view = self.fov
        # obs.return_status = "SUCCESSFUL" if event.metadata['lastActionSuccess'] else "OBSTRUCTED"
        self.obs.head_tilt = 0.0 #event.metadata['agent']['cameraHorizon']
        self.obs.reward = 0
        self.obs.goal = Namespace(metadata={"category": 'cover'})
        
        # if utils.dist.is_main_process():
        #     self.writer = SummaryWriter(self.log_dir, max_queue=args.MAX_QUEUE, flush_secs=60)

        self.num_save_traj = 10    

        if not args.dont_use_controller:
            from ai2thor.controller import Controller
            print(f"Initializing controller with params:\n step size {self.STEP_SIZE}\n W: {self.W}\n H: {self.H}\n fov: {self.fov}\n DT: {self.DT}\n")
            self.controller, self.server_port = init_controller(
                self.STEP_SIZE,
                self.W,
                self.H,
                self.fov,
                self.DT,
                args.visibilityDistance,
                # use_startx=False,
                )
            print("Controller: ", self.controller)

def init_controller(
    STEP_SIZE,
    W,
    H,
    fov,
    DT,
    visibilityDistance,
    # use_startx=False,
    ):

    if args.start_startx:
        if not (args.mode=="rearrangement") and not ("alfred" in args.mode):
            server_port = startx()
            print("SERVER PORT=", server_port)
            from ai2thor.controller import Controller
            controller = Controller(
                    # scene=mapname, 
                    visibilityDistance=visibilityDistance,
                    gridSize=STEP_SIZE,
                    width=W,
                    height=H,
                    fieldOfView=fov,
                    renderObjectImage=True,
                    renderDepthImage=True,
                    renderInstanceSegmentation=True,
                    x_display=str(server_port),
                    snapToGrid=False,
                    rotateStepDegrees=DT,
                    )
        else:
            server_port = startx()
            args.server_port = server_port
            print("SERVER PORT=", server_port)
            return None, server_port
    elif not ("alfred" in args.mode):
        # server_port = 3
        if args.do_headless_rendering:
            from ai2thor.platform import CloudRendering
            controller = Controller(
                    # scene=mapname, 
                    visibilityDistance=visibilityDistance,
                    gridSize=STEP_SIZE,
                    width=W,
                    height=H,
                    fieldOfView=fov,
                    renderObjectImage=True,
                    renderDepthImage=True,
                    renderInstanceSegmentation=True,
                    # x_display=str(server_port),
                    snapToGrid=False,
                    rotateStepDegrees=DT,
                    platform=CloudRendering,
                    )
        else:
            from ai2thor.controller import Controller
            print(f"Server port = {args.server_port}")
            controller = Controller(
                    # scene=mapname, 
                    visibilityDistance=visibilityDistance,
                    gridSize=STEP_SIZE,
                    width=W,
                    height=H,
                    fieldOfView=fov,
                    renderObjectImage=True,
                    renderDepthImage=True,
                    renderInstanceSegmentation=True,
                    x_display=str(args.server_port),
                    snapToGrid=False,
                    rotateStepDegrees=DT,
                    )
    else:
        controller = None
        server_port = args.server_port

    
    return controller, server_port

def get_alfred_constants():

    OBJECTS = [
            'AlarmClock',
            'Apple',
            'ArmChair',
            'BaseballBat',
            'BasketBall',
            'Bathtub',
            'BathtubBasin',
            'Bed',
            'Blinds',
            'Book',
            'Boots',
            'Bowl',
            'Box',
            'Bread',
            'ButterKnife',
            'Cabinet',
            'Candle',
            'Cart',
            'CD',
            'CellPhone',
            'Chair',
            'Cloth',
            'CoffeeMachine',
            'CounterTop',
            'CreditCard',
            'Cup',
            'Curtains',
            'Desk',
            'DeskLamp',
            'DishSponge',
            'Drawer',
            'Dresser',
            'Egg',
            'FloorLamp',
            'Footstool',
            'Fork',
            'Fridge',
            'GarbageCan',
            'Glassbottle',
            'HandTowel',
            'HandTowelHolder',
            'HousePlant',
            'Kettle',
            'KeyChain',
            'Knife',
            'Ladle',
            'Laptop',
            'LaundryHamper',
            'LaundryHamperLid',
            'Lettuce',
            'LightSwitch',
            'Microwave',
            'Mirror',
            'Mug',
            'Newspaper',
            'Ottoman',
            'Painting',
            'Pan',
            'PaperTowel',
            'PaperTowelRoll',
            'Pen',
            'Pencil',
            'PepperShaker',
            'Pillow',
            'Plate',
            'Plunger',
            'Poster',
            'Pot',
            'Potato',
            'RemoteControl',
            'Safe',
            'SaltShaker',
            'ScrubBrush',
            'Shelf',
            'ShowerDoor',
            'ShowerGlass',
            'Sink',
            'SinkBasin',
            'SoapBar',
            'SoapBottle',
            'Sofa',
            'Spatula',
            'Spoon',
            'SprayBottle',
            'Statue',
            'StoveBurner',
            'StoveKnob',
            'DiningTable',
            'CoffeeTable',
            'SideTable',
            'TeddyBear',
            'Television',
            'TennisRacket',
            'TissueBox',
            'Toaster',
            'Toilet',
            'ToiletPaper',
            'ToiletPaperHanger',
            'ToiletPaperRoll',
            'Tomato',
            'Towel',
            'TowelHolder',
            'TVStand',
            'Vase',
            'Watch',
            'WateringCan',
            'Window',
            'WineBottle',
        ]

    SLICED = [
        'AppleSliced',
        'BreadSliced',
        'LettuceSliced',
        'PotatoSliced',
        'TomatoSliced'
    ]

    OBJECTS += SLICED

    # object parents
    OBJ_PARENTS = {obj: obj for obj in OBJECTS}
    OBJ_PARENTS['AppleSliced'] = 'Apple'
    OBJ_PARENTS['BreadSliced'] = 'Bread'
    OBJ_PARENTS['LettuceSliced'] = 'Lettuce'
    OBJ_PARENTS['PotatoSliced'] = 'Potato'
    OBJ_PARENTS['TomatoSliced'] = 'Tomato'

    return OBJECTS, OBJ_PARENTS

def get_rearrangement_categories():

    REARRANGE_SIM_OBJECTS = [
        # A
        "AlarmClock", "AluminumFoil", "Apple", "AppleSliced", "ArmChair",
        "BaseballBat", "BasketBall", "Bathtub", "BathtubBasin", "Bed", "Blinds", "Book", "Boots", "Bottle", "Bowl", "Box",
        # B
        "Bread", "BreadSliced", "ButterKnife",
        # C
        "Cabinet", "Candle", "CD", "CellPhone", "Chair", "Cloth", "CoffeeMachine", "CoffeeTable", "CounterTop", "CreditCard",
        "Cup", "Curtains",
        # D
        "Desk", "DeskLamp", "Desktop", "DiningTable", "DishSponge", "DogBed", "Drawer", "Dresser", "Dumbbell",
        # E
        "Egg", "EggCracked",
        # F
        "Faucet", "Floor", "FloorLamp", "Footstool", "Fork", "Fridge",
        # G
        "GarbageBag", "GarbageCan",
        # H
        "HandTowel", "HandTowelHolder", "HousePlant", "Kettle", "KeyChain", "Knife",
        # L
        "Ladle", "Laptop", "LaundryHamper", "Lettuce", "LettuceSliced", "LightSwitch",
        # M
        "Microwave", "Mirror", "Mug",
        # N
        "Newspaper",
        # O
        "Ottoman",
        # P
        "Painting", "Pan", "PaperTowel", "Pen", "Pencil", "PepperShaker", "Pillow", "Plate", "Plunger", "Poster", "Pot",
        "Potato", "PotatoSliced",
        # R
        "RemoteControl", "RoomDecor",
        # S
        "Safe", "SaltShaker", "ScrubBrush", "Shelf", "ShelvingUnit", "ShowerCurtain", "ShowerDoor", "ShowerGlass",
        "ShowerHead", "SideTable", "Sink", "SinkBasin", "SoapBar", "SoapBottle", "Sofa", "Spatula", "Spoon", "SprayBottle",
        "Statue", "Stool", "StoveBurner", "StoveKnob",
        # T
        "TableTopDecor", "TargetCircle", "TeddyBear", "Television", "TennisRacket", "TissueBox", "Toaster", "Toilet",
        "ToiletPaper", "ToiletPaperHanger", "Tomato", "TomatoSliced", "Towel", "TowelHolder", "TVStand",
        # V
        "VacuumCleaner", "Vase",
        # W
        "Watch", "WateringCan", "Window", "WineBottle",
    ]
    # fmt: on


    # fmt: off
    OBJECT_TYPES_WITH_PROPERTIES = {
        "StoveBurner": {"openable": False, "receptacle": True, "pickupable": False},
        "Drawer": {"openable": True, "receptacle": True, "pickupable": False},
        "CounterTop": {"openable": False, "receptacle": True, "pickupable": False},
        "Cabinet": {"openable": True, "receptacle": True, "pickupable": False},
        "StoveKnob": {"openable": False, "receptacle": False, "pickupable": False},
        "Window": {"openable": False, "receptacle": False, "pickupable": False},
        "Sink": {"openable": False, "receptacle": True, "pickupable": False},
        "Floor": {"openable": False, "receptacle": True, "pickupable": False},
        "Book": {"openable": True, "receptacle": False, "pickupable": True},
        "Bottle": {"openable": False, "receptacle": False, "pickupable": True},
        "Knife": {"openable": False, "receptacle": False, "pickupable": True},
        "Microwave": {"openable": True, "receptacle": True, "pickupable": False},
        "Bread": {"openable": False, "receptacle": False, "pickupable": True},
        "Fork": {"openable": False, "receptacle": False, "pickupable": True},
        "Shelf": {"openable": False, "receptacle": True, "pickupable": False},
        "Potato": {"openable": False, "receptacle": False, "pickupable": True},
        "HousePlant": {"openable": False, "receptacle": False, "pickupable": False},
        "Toaster": {"openable": False, "receptacle": True, "pickupable": False},
        "SoapBottle": {"openable": False, "receptacle": False, "pickupable": True},
        "Kettle": {"openable": True, "receptacle": False, "pickupable": True},
        "Pan": {"openable": False, "receptacle": True, "pickupable": True},
        "Plate": {"openable": False, "receptacle": True, "pickupable": True},
        "Tomato": {"openable": False, "receptacle": False, "pickupable": True},
        "Vase": {"openable": False, "receptacle": False, "pickupable": True},
        "GarbageCan": {"openable": False, "receptacle": True, "pickupable": False},
        "Egg": {"openable": False, "receptacle": False, "pickupable": True},
        "CreditCard": {"openable": False, "receptacle": False, "pickupable": True},
        "WineBottle": {"openable": False, "receptacle": False, "pickupable": True},
        "Pot": {"openable": False, "receptacle": True, "pickupable": True},
        "Spatula": {"openable": False, "receptacle": False, "pickupable": True},
        "PaperTowelRoll": {"openable": False, "receptacle": False, "pickupable": True},
        "Cup": {"openable": False, "receptacle": True, "pickupable": True},
        "Fridge": {"openable": True, "receptacle": True, "pickupable": False},
        "CoffeeMachine": {"openable": False, "receptacle": True, "pickupable": False},
        "Bowl": {"openable": False, "receptacle": True, "pickupable": True},
        "SinkBasin": {"openable": False, "receptacle": True, "pickupable": False},
        "SaltShaker": {"openable": False, "receptacle": False, "pickupable": True},
        "PepperShaker": {"openable": False, "receptacle": False, "pickupable": True},
        "Lettuce": {"openable": False, "receptacle": False, "pickupable": True},
        "ButterKnife": {"openable": False, "receptacle": False, "pickupable": True},
        "Apple": {"openable": False, "receptacle": False, "pickupable": True},
        "DishSponge": {"openable": False, "receptacle": False, "pickupable": True},
        "Spoon": {"openable": False, "receptacle": False, "pickupable": True},
        "LightSwitch": {"openable": False, "receptacle": False, "pickupable": False},
        "Mug": {"openable": False, "receptacle": True, "pickupable": True},
        "ShelvingUnit": {"openable": False, "receptacle": True, "pickupable": False},
        "Statue": {"openable": False, "receptacle": False, "pickupable": True},
        "Stool": {"openable": False, "receptacle": True, "pickupable": False},
        "Faucet": {"openable": False, "receptacle": False, "pickupable": False},
        "Ladle": {"openable": False, "receptacle": False, "pickupable": True},
        "CellPhone": {"openable": False, "receptacle": False, "pickupable": True},
        "Chair": {"openable": False, "receptacle": True, "pickupable": False},
        "SideTable": {"openable": False, "receptacle": True, "pickupable": False},
        "DiningTable": {"openable": False, "receptacle": True, "pickupable": False},
        "Pen": {"openable": False, "receptacle": False, "pickupable": True},
        "SprayBottle": {"openable": False, "receptacle": False, "pickupable": True},
        "Curtains": {"openable": False, "receptacle": False, "pickupable": False},
        "Pencil": {"openable": False, "receptacle": False, "pickupable": True},
        "Blinds": {"openable": True, "receptacle": False, "pickupable": False},
        "GarbageBag": {"openable": False, "receptacle": False, "pickupable": False},
        "Safe": {"openable": True, "receptacle": True, "pickupable": False},
        "Painting": {"openable": False, "receptacle": False, "pickupable": False},
        "Box": {"openable": True, "receptacle": True, "pickupable": True},
        "Laptop": {"openable": True, "receptacle": False, "pickupable": True},
        "Television": {"openable": False, "receptacle": False, "pickupable": False},
        "TissueBox": {"openable": False, "receptacle": False, "pickupable": True},
        "KeyChain": {"openable": False, "receptacle": False, "pickupable": True},
        "FloorLamp": {"openable": False, "receptacle": False, "pickupable": False},
        "DeskLamp": {"openable": False, "receptacle": False, "pickupable": False},
        "Pillow": {"openable": False, "receptacle": False, "pickupable": True},
        "RemoteControl": {"openable": False, "receptacle": False, "pickupable": True},
        "Watch": {"openable": False, "receptacle": False, "pickupable": True},
        "Newspaper": {"openable": False, "receptacle": False, "pickupable": True},
        "ArmChair": {"openable": False, "receptacle": True, "pickupable": False},
        "CoffeeTable": {"openable": False, "receptacle": True, "pickupable": False},
        "TVStand": {"openable": False, "receptacle": True, "pickupable": False},
        "Sofa": {"openable": False, "receptacle": True, "pickupable": False},
        "WateringCan": {"openable": False, "receptacle": False, "pickupable": True},
        "Boots": {"openable": False, "receptacle": False, "pickupable": True},
        "Ottoman": {"openable": False, "receptacle": True, "pickupable": False},
        "Desk": {"openable": False, "receptacle": True, "pickupable": False},
        "Dresser": {"openable": False, "receptacle": True, "pickupable": False},
        "Mirror": {"openable": False, "receptacle": False, "pickupable": False},
        "DogBed": {"openable": False, "receptacle": True, "pickupable": False},
        "Candle": {"openable": False, "receptacle": False, "pickupable": True},
        "RoomDecor": {"openable": False, "receptacle": False, "pickupable": False},
        "Bed": {"openable": False, "receptacle": True, "pickupable": False},
        "BaseballBat": {"openable": False, "receptacle": False, "pickupable": True},
        "BasketBall": {"openable": False, "receptacle": False, "pickupable": True},
        "AlarmClock": {"openable": False, "receptacle": False, "pickupable": True},
        "CD": {"openable": False, "receptacle": False, "pickupable": True},
        "TennisRacket": {"openable": False, "receptacle": False, "pickupable": True},
        "TeddyBear": {"openable": False, "receptacle": False, "pickupable": True},
        "Poster": {"openable": False, "receptacle": False, "pickupable": False},
        "Cloth": {"openable": False, "receptacle": False, "pickupable": True},
        "Dumbbell": {"openable": False, "receptacle": False, "pickupable": True},
        "LaundryHamper": {"openable": True, "receptacle": True, "pickupable": False},
        "TableTopDecor": {"openable": False, "receptacle": False, "pickupable": True},
        "Desktop": {"openable": False, "receptacle": False, "pickupable": False},
        "Footstool": {"openable": False, "receptacle": True, "pickupable": True},
        "BathtubBasin": {"openable": False, "receptacle": True, "pickupable": False},
        "ShowerCurtain": {"openable": True, "receptacle": False, "pickupable": False},
        "ShowerHead": {"openable": False, "receptacle": False, "pickupable": False},
        "Bathtub": {"openable": False, "receptacle": True, "pickupable": False},
        "Towel": {"openable": False, "receptacle": False, "pickupable": True},
        "HandTowel": {"openable": False, "receptacle": False, "pickupable": True},
        "Plunger": {"openable": False, "receptacle": False, "pickupable": True},
        "TowelHolder": {"openable": False, "receptacle": True, "pickupable": False},
        "ToiletPaperHanger": {"openable": False, "receptacle": True, "pickupable": False},
        "SoapBar": {"openable": False, "receptacle": False, "pickupable": True},
        "ToiletPaper": {"openable": False, "receptacle": False, "pickupable": True},
        "HandTowelHolder": {"openable": False, "receptacle": True, "pickupable": False},
        "ScrubBrush": {"openable": False, "receptacle": False, "pickupable": True},
        "Toilet": {"openable": True, "receptacle": True, "pickupable": False},
        "ShowerGlass": {"openable": False, "receptacle": False, "pickupable": False},
        "ShowerDoor": {"openable": True, "receptacle": False, "pickupable": False},
        "AluminumFoil": {"openable": False, "receptacle": False, "pickupable": True},
        "VacuumCleaner": {"openable": False, "receptacle": False, "pickupable": False}
    }
    # fmt: on

    PICKUPABLE_OBJECTS = list(
        sorted(
            [
                object_type
                for object_type, properties in OBJECT_TYPES_WITH_PROPERTIES.items()
                if properties["pickupable"]
            ]
        )
    )

    OPENABLE_OBJECTS = list(
        sorted(
            [
                object_type
                for object_type, properties in OBJECT_TYPES_WITH_PROPERTIES.items()
                if properties["openable"] and not properties["pickupable"]
            ]
        )
    )

    RECEPTACLE_OBJECTS = list(
        sorted(
            [
                object_type
                for object_type, properties in OBJECT_TYPES_WITH_PROPERTIES.items()
                if properties["receptacle"]
            ]
        )
    )

    return REARRANGE_SIM_OBJECTS, OBJECT_TYPES_WITH_PROPERTIES, PICKUPABLE_OBJECTS, OPENABLE_OBJECTS, RECEPTACLE_OBJECTS


def navigation_calibration_params():

    z_params = {
        # 0s
        'FloorPlan1':[0.05, 2.0],
        'FloorPlan2':[0.05, 2.0],
        'FloorPlan3':[0.05, 2.0], # this one is broken?
        'FloorPlan4':[0.05, 1.7],
        'FloorPlan5':[0.05, 1.9],
        'FloorPlan6':[0.05, 1.9],
        'FloorPlan7':[0.05, 2.5],
        'FloorPlan8':[0.05, 2.5],
        'FloorPlan9':[0.05, 2.0],
        'FloorPlan10':[0.05, 2.2],
        'FloorPlan11':[0.05, 2.0],
        'FloorPlan12':[0.05, 2.1],
        'FloorPlan13':[0.05, 2.7],
        'FloorPlan14':[0.05, 2.2],
        'FloorPlan15':[0.05, 1.8],
        'FloorPlan16':[0.05, 2.2],
        'FloorPlan17':[0.05, 2.0],
        'FloorPlan18':[0.05, 2.0],
        'FloorPlan19':[0.05, 2.0],
        'FloorPlan20':[0.05, 2.3],
        'FloorPlan21':[0.05, 1.7],
        'FloorPlan22':[0.05, 2.2],
        'FloorPlan23':[0.05, 2.2],
        'FloorPlan24':[0.05, 2.0],
        'FloorPlan25':[0.05, 2.0],
        'FloorPlan26':[0.05, 2.2],
        'FloorPlan27':[0.05, 2.2],
        'FloorPlan28':[0.05, 1.9],
        'FloorPlan29':[0.05, 1.9],
        'FloorPlan30':[0.05, 2.0],
        # 200s
        'FloorPlan201':[0.05, 1.8],
        'FloorPlan202':[0.05, 2.4],
        'FloorPlan203':[0.05, 2.0],
        'FloorPlan204':[0.05, 2.4],
        'FloorPlan205':[0.05, 2.0],
        'FloorPlan206':[0.05, 2.0],
        'FloorPlan207':[0.05, 2.0],
        'FloorPlan208':[0.05, 2.4],
        'FloorPlan209':[0.05, 2.5],
        'FloorPlan210':[0.05, 2.4],
        'FloorPlan211':[0.05, 2.0],
        'FloorPlan212':[0.05, 1.7],
        'FloorPlan213':[0.05, 2.1],
        'FloorPlan214':[0.05, 2.5],
        'FloorPlan215':[0.05, 2.2],
        'FloorPlan216':[0.05, 2.1],
        'FloorPlan217':[0.05, 2.1],
        'FloorPlan218':[0.05, 2.0],
        'FloorPlan219':[0.05, 2.0],
        'FloorPlan220':[0.05, 2.0],
        'FloorPlan221':[0.05, 2.0], # broken
        'FloorPlan221':[0.05, 2.1],
        'FloorPlan222':[0.05, 2.1],
        'FloorPlan223':[0.05, 2.2],
        'FloorPlan224':[0.05, 2.2],
        'FloorPlan225':[0.05, 2.0],
        'FloorPlan226':[0.05, 2.0],
        'FloorPlan227':[0.05, 2.5],
        'FloorPlan228':[0.05, 2.5],
        'FloorPlan229':[0.05, 2.4],
        'FloorPlan230':[0.05, 2.5], # needs finetuning
        # 300s
        'FloorPlan301':[0.05, 2.0],
        'FloorPlan302':[0.05, 2.1],
        'FloorPlan303':[0.05, 2.0],
        'FloorPlan304':[0.05, 2.4],
        'FloorPlan305':[0.05, 2.0],
        'FloorPlan306':[0.05, 2.1],
        'FloorPlan307':[0.05, 2.2],
        'FloorPlan308':[0.05, 2.0],
        'FloorPlan309':[0.05, 2.0],
        'FloorPlan310':[0.05, 2.0],
        'FloorPlan311':[0.05, 2.0],
        'FloorPlan312':[0.05, 2.0],
        'FloorPlan313':[0.05, 2.0],
        'FloorPlan314':[0.05, 1.8],
        'FloorPlan315':[0.05, 1.9],
        'FloorPlan316':[0.05, 2.0],
        'FloorPlan317':[0.05, 2.2],
        'FloorPlan318':[0.05, 2.2],
        'FloorPlan319':[0.05, 2.3],
        'FloorPlan320':[0.05, 2.0],
        'FloorPlan321':[0.05, 1.9],
        'FloorPlan322':[0.05, 2.1],
        'FloorPlan323':[0.05, 2.0],
        'FloorPlan324':[0.05, 2.0],
        'FloorPlan325':[0.05, 2.0], # needs finetuning/is broken
        'FloorPlan326':[0.05, 2.2],
        'FloorPlan327':[0.05, 2.0],
        'FloorPlan328':[0.05, 2.0],
        'FloorPlan329':[0.05, 2.0],
        'FloorPlan330':[0.05, 2.0],
        # 400s
        'FloorPlan401':[0.05, 2.0],
        'FloorPlan402':[0.05, 1.9],
        'FloorPlan403':[0.05, 2.1],
        'FloorPlan404':[0.05, 2.0],
        'FloorPlan405':[0.05, 2.0],
        'FloorPlan406':[0.05, 2.0],
        'FloorPlan407':[0.05, 2.0],
        'FloorPlan408':[0.05, 2.0],
        'FloorPlan409':[0.05, 2.0],
        'FloorPlan410':[0.05, 2.0],
        'FloorPlan411':[0.05, 2.0],
        'FloorPlan412':[0.05, 2.0],
        'FloorPlan413':[0.05, 2.0],
        'FloorPlan414':[0.05, 2.0],
        'FloorPlan415':[0.05, 2.5],
        'FloorPlan416':[0.05, 2.5],
        'FloorPlan417':[0.05, 2.0],
        'FloorPlan418':[0.05, 2.0],
        'FloorPlan419':[0.05, 2.0],
        'FloorPlan420':[0.05, 2.0],
        'FloorPlan421':[0.05, 2.0],
        'FloorPlan422':[0.05, 2.5],
        'FloorPlan423':[0.05, 2.0],
        'FloorPlan424':[0.05, 2.4],
        'FloorPlan425':[0.05, 2.0],
        'FloorPlan426':[0.05, 2.0],
        'FloorPlan427':[0.05, 2.0],
        'FloorPlan428':[0.05, 2.1],
        'FloorPlan429':[0.05, 1.9],
        'FloorPlan430':[0.05, 2.5],
    }

    return z_params

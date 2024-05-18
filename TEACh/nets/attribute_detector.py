import numpy as np
import utils.aithor
import torch
from PIL import Image
import ipdb
st = ipdb.set_trace
from utils.ddetr_utils import check_for_detections
import numpy as np
import utils.aithor
import utils.geom
import torch
from PIL import Image
from arguments import args
import sys
from backend import saverloader
import os
import cv2
import matplotlib.pyplot as plt
import ipdb
st = ipdb.set_trace
from utils.ddetr_utils import check_for_detections
from scipy.spatial import distance
import skimage

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class AttributeDetectorVLM():

    def __init__(
        self, 
        W, H, 
        default_attributes,
        keep_attribute_detector_cpu=False,
        score_threshold=0.6,
        ): 
        '''
        score_threshold: if below this threshold, model is "unsure" and default value will be used
        '''

        from nets.cogVLM import COGVLM
        self.mllm = COGVLM()

        self.W, self.H = W, H
        self.default_attributes = default_attributes
        self.score_threshold = score_threshold
        self.keep_attribute_detector_cpu = keep_attribute_detector_cpu

        # self.model = build_state_estimator(args.attribute_detector_checkpoint, device).to(device) #StateEstimator()
        # self.enlarge_ratio = self.model.enlarge_ratio
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

        self.opposite = {
            'cooked':'raw',
            'dirty':'clean',
            'filled with liquid':'empty',
            'held':'not held',
            'filled with water':'filled with coffee',
            'opened':'closed',
            'sliced':'whole',
            'toggled on':'toggled off',
        }
        self.word_mapping = {
            'cooked':'cooked',
            'dirty':'dirty',
            'filled':'filled with liquid',
            'holding':'held',
            'fillLiquid':'filled with water',
            'open':'opened',
            'sliced':'sliced',
            'toggled':'toggled on',
        }
        self.reverse_word_mapping = {v:k for (k,v) in self.word_mapping.items()}

        if self.keep_attribute_detector_cpu:
            self.mllm.model.cpu()

    def get_prompt(self, attribute, category):
        
        category_ = category.lower()
        if category_ in self.name_to_parsed_name.keys():
            category_ = self.name_to_parsed_name[category_]
        word1 = self.word_mapping[attribute]
        word2 = self.opposite[word1]
        query_text = f'Is this {category_} {word1} or {word2}? Provide only your answer, either "{word1}" or "{word2}", or "unsure" if you do not know.'
        return query_text, word1, word2

    @torch.no_grad()
    def get_attribute(self, cropped, attribute, category, score_threshold=60):
        
        if self.keep_attribute_detector_cpu:
            self.mllm.model.to(device)

        query_text, word1, word2 = self.get_prompt(attribute, category)

        # plt.figure()
        # plt.imshow(cropped)
        # plt.savefig('output/test.png')
        response, scores_percent = self.mllm.run_cogvlm([Image.fromarray(cropped)], query_text, return_scores=True)
        
        if np.min(scores_percent) < self.score_threshold:
            # if score is low, model is uncertain. Use default value.
            val = self.default_attributes[attribute]
        else:
            if response==word1:
                val = True
            elif response==word2:
                val = False
            else:
                val = self.default_attributes[attribute]

        attribute_dict = {
            attribute: val
        }

        if attribute=="filled":
            if val:
                attribute_dict.update(self.get_attribute(cropped, "fillLiquid", category))
            else:
                attribute_dict.update({"fillLiquid": None})

        if self.keep_attribute_detector_cpu:
            self.mllm.model.cpu()

        return attribute_dict

class AttributeDetector():

    def __init__(
        self, 
        W, H, 
        default_attributes,
        ): 
        from nets.state_estimator import StateEstimator, build_state_estimator, name_to_interested_states, preds_to_state_dict

        from nets.state_estimator import (
            THING_NAMES,
            STUFF_NAMES,
            get_object_affordance,
            ObjectClass,
        )

        self.W, self.H = W, H
        self.default_attributes = default_attributes

        self.model = build_state_estimator(args.attribute_detector_checkpoint, device).to(device) #StateEstimator()
        self.enlarge_ratio = self.model.enlarge_ratio

    def mask_to_box(self, mask): #, padding = 20):

        padding = int(self.W * self.enlarge_ratio)

        segmentation = np.where(mask == True)

        # Bounding Box
        bbox = np.asarray([0, 0, 0, 0])
        if len(segmentation) != 0 and len(segmentation[1]) != 0 and len(segmentation[0]) != 0:
            x_min = int(np.min(segmentation[1]))
            x_max = int(np.max(segmentation[1]))
            y_min = int(np.min(segmentation[0]))
            y_max = int(np.max(segmentation[0]))

            bbox = np.asarray([max(0, x_min-padding), min(x_max+padding, self.W-1), max(0, y_min-padding), min(y_max+padding, self.H-1)])

        return bbox

    @torch.no_grad()
    def get_attributes(self, rgb, mask, object_category):

        box = self.mask_to_box(mask)
        x_min, x_max, y_min, y_max = box

        # cropped = rgb[x_min:x_max, y_min:y_max]

        cropped = rgb[y_min:y_max, x_min:x_max]

        # plt.figure()
        # plt.imshow(cropped)
        # plt.savefig('output/images/test.png')

        seagull_idx = ObjectClass[object_category].value
        with torch.no_grad():
            attributes = self.model([cropped], torch.tensor([seagull_idx], device=device).long())

        interested_states = name_to_interested_states(object_category)
        states_for_objects = preds_to_state_dict(attributes, [interested_states])[0]

        # This CLIP model gives isDirty, isFilledWithWater, and isToggled. The rest are set via default
        if 'isDirty' in states_for_objects.keys():
            clean = True if states_for_objects['isDirty']<0.5 else False
        else:
            clean = self.default_attributes["clean"]

        if 'isFilledWithWater' in states_for_objects.keys():
            filled = True if states_for_objects['isFilledWithWater']>0.5 else False
        else:
            filled = self.default_attributes["filled"]

        if 'isToggled' in states_for_objects.keys():
            toggled = True if states_for_objects['isToggled']>0.5 else False
        else:
            toggled = self.default_attributes["toggled"]

        attributes = {
            "clean":sliced,
            "filled":toasted,
            "toggled":clean,
        }

        return attributes

class AttributeDetectorALT():

    def __init__(
        self, 
        W, H, 
        ): 

        self.W, self.H = W, H

        from nets.clip import ALIGN
        self.model = ALIGN()

        self.slicable_objects = ['Apple', 'Bread', 'Lettuce', 'Potato', 'Tomato']

        self.cookable_objects = ['Apple', 'Lettuce', 'Potato', 'Tomato', 'AppleSliced', 'LettuceSliced', 'PotatoSliced', 'TomatoSliced']

        self.toastable_objects = ['Bread', 'BreadSliced']

        self.cleanable_objects = ["Bowl", "Cup", "Mug", "Plate", "Pot", "Pan"]

    def mask_to_box(self, mask, padding = 20):

        segmentation = np.where(mask == True)

        # Bounding Box
        bbox = np.asarray([0, 0, 0, 0])
        if len(segmentation) != 0 and len(segmentation[1]) != 0 and len(segmentation[0]) != 0:
            x_min = int(np.min(segmentation[1]))
            x_max = int(np.max(segmentation[1]))
            y_min = int(np.min(segmentation[0]))
            y_max = int(np.max(segmentation[0]))

            bbox = np.asarray([max(0, x_min-padding), min(x_max+padding, self.W-1), max(0, y_min-padding), min(y_max+padding, self.H-1)])

        return bbox

    def get_attributes(self, rgb, mask, object_category):

        box = self.mask_to_box(mask)
        x_min, x_max, y_min, y_max = box

        # cropped = rgb[x_min:x_max, y_min:y_max]

        cropped = rgb[y_min:y_max, x_min:x_max]

        probs = None

        if object_category in self.slicable_objects:
            lines = [f"The {object_category} is sliced", f"The {object_category} is not sliced"]
            probs = self.model.score(cropped, lines)[0]
            sliced = True if probs[0]>probs[1] else False
            print(f'{object_category} sliced? {sliced}')
        else:
            sliced = False

        if object_category in self.cookable_objects:
            lines = [f"The {object_category} is cooked", f"The {object_category} is not cooked"]
            probs = self.model.score(cropped, lines)[0]
            cooked = True if probs[0]>probs[1] else False
            print(f'{object_category} cooked? {cooked}')
        else:
            cooked = False

        if object_category in self.toastable_objects:
            lines = [f"The {object_category} is toasted", f"The {object_category} is not toasted"]
            probs = self.model.score(cropped, lines)[0]
            toasted = True if probs[0]>probs[1] else False
            print(f'{object_category} toasted? {toasted}')
        else:
            toasted = False

        if object_category in self.cleanable_objects:
            lines = [f"The {object_category} is clean", f"The {object_category} is dirty"]
            probs = self.model.score(cropped, lines)[0]
            clean = True if probs[0]>probs[1] else False
            print(f'{object_category} clean? {clean}')
        else:
            clean = False

        # if probs is not None:
        #     plt.figure()
        #     plt.imshow(rgb)
        #     plt.savefig('output/images/test1.png')
        #     plt.figure()
        #     plt.imshow(cropped)
        #     plt.savefig('output/images/test.png')
        #     print(sliced, cooked, toasted, clean)
        #     st()

        attributes = {
            "sliced":sliced,
            "toasted":toasted,
            "clean":clean,
            "cooked":cooked,
        }

        return attributes





#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Multi input models."""

import inspect
import random
import heapq
from torch.nn.init import xavier_uniform_
import collections
import torch
from torch.distributions.categorical import Categorical
from functools import reduce
import torch.nn.functional as F
import math
import copy
from einops import rearrange
import torch.nn as nn

from functools import reduce
from operator import mul
from .head_helper import MultiTaskHead, MultiTaskMViTHead
from .video_model_builder import SlowFast, _POOL1, MViT
from .build import MODEL_REGISTRY

import asyncio
import logging
import os
import random
import time
from typing import Any
import json

# import aiolimiter
import openai
from openai import OpenAI
from PIL import Image

import ipdb
st = ipdb.set_trace
import numpy as np

import base64
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Dict, TypedDict, Union

import logging
import sys
logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)
openai.log='info'

use_azure = True
if use_azure:
    from openai import AzureOpenAI
    client = AzureOpenAI(
        api_key = os.getenv("AZURE_OPENAI_KEY"),  
        api_version = "2023-05-15",
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    )
else:
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

def pil_to_b64(img: Image.Image) -> str:
    with BytesIO() as image_buffer:
        img.save(image_buffer, format="PNG")
        byte_data = image_buffer.getvalue()
        img_b64 = base64.b64encode(byte_data).decode("utf-8")
        img_b64 = "data:image/png;base64," + img_b64
    return img_b64

# from .SoM import SoM_inference
# from .deva import run_deva
from scipy.stats import expon


seed = 32
torch.manual_seed(seed)
np.random.seed(seed)

device = "cuda" if torch.cuda.is_available() else "cpu"

class CLIP:
    def __init__(self):
        from transformers import CLIPProcessor, CLIPModel

        # self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        # self.preprocess = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        clip_model = "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"

        self.model = CLIPModel.from_pretrained(clip_model).to(device).eval()
        self.preprocess = CLIPProcessor.from_pretrained(clip_model)
        print(clip_model)

        self.cos_sim = torch.nn.CosineSimilarity(dim=1)

    @torch.no_grad()
    def score(self, image=None, texts=None):

        if isinstance(texts, str):
            texts = [texts]

        inputs = self.preprocess(text=texts, images=image, return_tensors="pt", padding=True).to(device)

        outputs = self.model(**inputs)
        logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
        probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities

        return probs

    @torch.no_grad()
    def score_images(self, image_query=None, images=None):

        input_query = self.preprocess(text=None, images=image_query, return_tensors="pt", padding=True).to(device)
        image_features_query = self.model.get_image_features(**input_query)

        if isinstance(images, torch.Tensor):
            image_features = images.to(device)
        else:
            inputs = self.preprocess(text=None, images=images, return_tensors="pt", padding=True).to(device)
            image_features= self.model.get_image_features(**inputs)

        probs = self.cos_sim(image_features_query, image_features)

        return probs

    @torch.no_grad()
    def encode_images(self, images):
        inputs = self.preprocess(text=None, images=images, return_tensors="pt", padding=True).to(device)
        image_features = self.model.get_image_features(**inputs)

        return image_features

    @torch.no_grad()
    def encode_text(self, text):
        inputs = self.preprocess(text=text, images=None, return_tensors="pt", padding=True, truncation=True).to(device)
        text_features = self.model.get_text_features(**inputs)

        return text_features

@MODEL_REGISTRY.register()
class GPT4v():
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.build_model()

        if self.cfg.DO_ICAL_PROMPT:
            with open('ego4d_forecasting/models/prompts/prompt_idm_ical.txt') as f:
                prompt_idm = f.read()
            with open('ego4d_forecasting/models/prompts/prompt_ltp_ical.txt') as f:
                prompt_ltp = f.read()
        else:
            with open('ego4d_forecasting/models/prompts/prompt_idm.txt') as f:
                prompt_idm = f.read()
            with open('ego4d_forecasting/models/prompts/prompt_ltp.txt') as f:
                prompt_ltp = f.read()
        with open(os.path.join(self.cfg.DATA.PATH_TO_DATA_DIR, 'fho_lta_taxonomy.json')) as f:
            self.verbs_and_nouns = f.read()
        prompt_idm = prompt_idm.replace('{ACCEPTABLE_VERBS_NOUNS}', self.verbs_and_nouns)
        prompt_ltp = prompt_ltp.replace('{ACCEPTABLE_VERBS_NOUNS}', self.verbs_and_nouns)
        self.intro_idm = prompt_idm
        self.intro_ltp = prompt_ltp
        std = self.cfg.DATA.STD
        mean = self.cfg.DATA.MEAN
        self.image_mean = torch.from_numpy(np.array([mean]).reshape(3,1,1,1)).cuda()
        self.image_std = torch.from_numpy(np.array([std]).reshape(3,1,1,1)).cuda()
        with open('ego4d_forecasting/models/prompts/prompt_get_closest_word.txt') as f:
            self.prompt_closest_word = f.read()
        # self.ltp_pred_prompt = "Predicted video actions:\n\n{video_actions}"
        
        # self.examples_forecasting = 'ego4d_forecasting/models/prompts/examples/forecasting/examples.json'
        # with open(self.examples_forecasting) as json_data:
        #     self.examples_forecasting = json.load(json_data)
        # self.examples_recognition = 'ego4d_forecasting/models/prompts/examples/recognition/examples.json'
        # with open(self.examples_recognition) as json_data:
        #     self.examples_recognition = json.load(json_data)

        self.get_example_embeddings()

        if os.path.exists(os.path.join(self.cfg.DATA.PATH_TO_DATA_DIR, 'fho_lta_taxonomy.json')):
            with open(os.path.join(self.cfg.DATA.PATH_TO_DATA_DIR, 'fho_lta_taxonomy.json')) as json_data:
                self.taxonomy = json.load(json_data)

            self.verbs2idx = {self.taxonomy['verbs'][idx]:idx for idx in range(len(self.taxonomy['verbs']))}
            self.idx2verbs = {v:k for k,v in self.verbs2idx.items()}
            self.nouns2idx = {self.taxonomy['nouns'][idx]:idx for idx in range(len(self.taxonomy['nouns']))}
            self.idx2nouns = {v:k for k,v in self.nouns2idx.items()}

        from .deva import DEVA
        self.deva = DEVA()

        if self.cfg.DO_ICAL_PROMPT or self.cfg.DO_ABSTRACTION:
            self.clip = CLIP()

    def get_example_embeddings(
        self,
        example_json=None,
    ):

        if example_json is not None:
            examples = example_json
        else:
            with open(self.cfg.EXAMPLE_PATH) as json_data:
                examples = json.load(json_data)
        self.examples = list(examples.values())
        example_embeddings_images = []
        example_embeddings_summary = []
        example_embeddings_state = []
        for example in self.examples:
            embedding = np.load(example[3])
            example_embeddings_images.append(embedding)
            embedding = np.load(example[3].replace('embeddings_image', 'embeddings_summary'))
            example_embeddings_summary.append(embedding)
            embedding = np.load(example[3].replace('embeddings_image', 'embeddings_state'))
            example_embeddings_state.append(embedding)
        self.example_embeddings_images = torch.from_numpy(np.asarray(example_embeddings_images))
        self.example_embeddings_summary = torch.from_numpy(np.asarray(example_embeddings_summary))
        self.example_embeddings_state = torch.from_numpy(np.asarray(example_embeddings_state))

    def retrieve_topk(
        self,
        images,
        topk=3,
        sample_exponential=False,
        example_multipiler=None,
    ):

        cos_sim = torch.nn.CosineSimilarity(dim=1)

        # example_embeddings = self.example_embeddings

        image_encodings = self.clip.encode_images(images)
        image_encodings_mean = torch.from_numpy(image_encodings.mean(0).cpu().numpy()[None])

        sim_image = cos_sim(image_encodings_mean, self.example_embeddings_images)
        sim_instruction = cos_sim(image_encodings_mean, self.example_embeddings_summary)
        sim_state = cos_sim(image_encodings_mean, self.example_embeddings_state)

        sim = self.cfg.ALPHA_INSTRUCTION * sim_instruction + self.cfg.ALPHA_IMAGE * sim_image + self.cfg.ALPHA_STATE * sim_state

        sims_argsort = torch.argsort(sim, descending=True).cpu().numpy()

        if example_multipiler is not None:
            sims_argsort = sims_argsort[topk*example_multipiler:]
        elif sample_exponential:
            # sample examples according to an exponential
            num_examples_considered = len(sims_argsort)
            x = np.linspace(expon.ppf(0.01), expon.ppf(0.99), num_examples_considered + 1)
            y = expon.pdf(x)
            y_norm = (y - y.min()) / (y - y.min()).sum()
            y_norm = y_norm[:-1]
            selected_example_idxs = np.random.choice(
                np.arange(num_examples_considered), 
                size=len(sims_argsort),
                replace=False, 
                p=y_norm,
                )
            selected_inds = sims_argsort[selected_example_idxs]
        else:
            selected_inds = sims_argsort

        # images[1].save('output/test.png')

        examples_sorted = [self.examples[e_i] for e_i in list(sims_argsort)]
        examples_sorted_topk = examples_sorted[:topk]
        return examples_sorted_topk

    # to encode frames into a set of {cfg.FORECASTING.NUM_INPUT_CLIPS} clips
    def build_model(self):
        pass

    def forward(self, images, k=1, examples=[], return_images_only=False, with_SoM=True):
        if k not in [1,5]:
            assert False # k other than 1 and 5 not supported
        print(f"k is {k}")
        B, S, C, H, W = images[0][0].transpose(1,2).shape
        images_samples = images[0][0].transpose(1,2).flatten(0,1).transpose(0,1) #images[0][0]. #images[0][0,0] #.unbind(1)
        self.image_std, self.image_mean = self.image_std.to(images_samples.device), self.image_mean.to(images_samples.device)
        images_samples = images_samples * self.image_std + self.image_mean
        images_samples = images_samples.unbind(1)
        newsize = (768, 768)
        images_samples = [Image.fromarray((video_image.permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)).resize(newsize) for video_image in images_samples]
        if with_SoM:
            images_samples_SoM = self.deva.run_deva(images_samples)
        images_samples_ = copy.deepcopy(images_samples_SoM)
        
        forecast_forecast_idxs = []
        recognition_idxs = []
        response_idms = []
        response_ltps = []
        examples_retrieved = []
        for k_idx in range(k):
            # retrieve examples
            if self.cfg.DO_ICAL_PROMPT:
                # topk_examples = self.retrieve_topk(images_samples, sample_exponential=k_idx!=0)
                topk_examples = self.retrieve_topk(images_samples, example_multipiler=k_idx, topk=3)
            else:
                topk_examples = self.examples #[:2]

            examples_retrieved.append(topk_examples)

            if not self.cfg.ONLY_DO_FORECASTING:
                '''
                Infer the actions in the video first
                '''
                prompt_idm = self.get_prompt_inverse_dynamics(
                    images_samples_SoM, 
                    B, S, C, newsize[0], newsize[1],
                    topk_examples
                    )
                if return_images_only:
                    forecast_forecast_idxs = torch.tensor([0, 0]).unsqueeze(0).unsqueeze(0).repeat(1,k,1,1).unbind(-1)
                    return forecast_forecast_idxs, forecast_forecast_idxs, '', '', images_samples_, examples_retrieved
                response_idm = self.generate_from_openai_completion(prompt_idm)
                response_idms.append(response_idm)
                video_actions = response_idm.split('Video Actions:\n')[-1]
                recognition_idxs_ = self.split_actions(video_actions)
            else:
                video_actions = ""
                response_idms.append("")
                recognition_idxs_ = []

            verb, noun = 0, 0
            recognition_idxs_ = recognition_idxs_[:B]
            for line_idx in range(B - len(recognition_idxs_)):
                # append if predicted too few
                recognition_idxs_.append([verb, noun])
            recognition_idxs_ = torch.tensor(recognition_idxs_)
            recognition_idxs.append(recognition_idxs_)
            use_examples = not self.cfg.DO_ZERO_SHOT
            prompt_ltp = self.get_prompt_predict_actions(
                images_samples_SoM, 
                video_actions, 
                B, S, C, newsize[0], newsize[1],
                topk_examples,
                use_examples=use_examples,
                )
            response_ltp = self.generate_from_openai_completion(prompt_ltp)
            response_ltps.append(response_ltp)
            forecast_list = response_ltp.split('Future Actions:\n')[-1]
            forecast_forecast_idxs_ = self.split_actions(forecast_list)
            forecast_forecast_idxs_ = torch.tensor(forecast_forecast_idxs_)
            forecast_forecast_idxs.append(forecast_forecast_idxs_)
        if k==1:
            forecast_forecast_idxs = torch.stack(forecast_forecast_idxs, dim=0).unsqueeze(0).repeat(1,5,1,1).unbind(-1)
            recognition_idxs = torch.stack(recognition_idxs, dim=0).unsqueeze(0).repeat(1,5,1,1).unbind(-1)
        else:
            forecast_forecast_idxs = torch.stack(forecast_forecast_idxs, dim=0).unsqueeze(0).unbind(-1)
            recognition_idxs = torch.stack(recognition_idxs, dim=0).unsqueeze(0).unbind(-1)
        return forecast_forecast_idxs, recognition_idxs, response_idms, response_ltps, images_samples_, examples_retrieved

    def split_actions(
        self,
        actions_text
    ):
        recognition_idxs_ = []
        verb, noun = 0, 0
        for line in actions_text.split('\n'):
            try:
                line1 = line.split(' ')[1]
            except:
                line1 = ""
            try:
                line2 = line.split(' ')[2]
            except:
                line2 = ""
            if line1 in self.verbs2idx.keys():
                verb = self.verbs2idx[line1]
            elif line1=="":
                verb = 0
            else:
                prompt_closest_word_ = self.prompt_closest_word.replace('{LIST}', f"{self.taxonomy['verbs']}")
                line1 = self.prompt_gpt_text(prompt_closest_word_.replace('{WORD}', line1))
                if line1 in self.verbs2idx.keys():
                    verb = self.verbs2idx[line1]
                else:
                    verb = 0
            if line2 in self.nouns2idx.keys():
                noun = self.nouns2idx[line2]
            elif line2=="":
                noun = 0
            else:
                prompt_closest_word_ = self.prompt_closest_word.replace('{LIST}', f"{self.taxonomy['nouns']}")
                line2 = self.prompt_gpt_text(prompt_closest_word_.replace('{WORD}', line2))
                if line2 in self.nouns2idx.keys():
                    verb = self.nouns2idx[line2]
                else:
                    verb = 0
            recognition_idxs_.append([verb, noun])
        return recognition_idxs_


    def get_prompt_inverse_dynamics(
        self,
        images,
        B, S, C, H, W,
        topk_examples,
        use_examples=True,
        single_image=False,
        multivideo=True,
        with_SoM=True,
    ):
        # if len(images[0])>1:
        #     assert(False)
        message = [
            {
                "role": "system",
                "content": [{"type": "text", "text": self.intro_idm}],
            }
        ]
        if use_examples:
            for (a,b,c,d,e,f,g,h,i) in list(topk_examples):
                c = [Image.open(video_image) for video_image in c]
                c = np.concatenate([np.asarray(video_image) for video_image in c], axis=1)
                c = [Image.fromarray(c)]
                alphabet = 'abcdefghijklmnopqrstuvwxyz'
                example_content = []
                for image_i, image_f in enumerate(c):
                    if type(image_f)==str:
                        example_img = Image.open(image_f)
                    else:
                        example_img = image_f
                    example_content.extend(
                        [
                            {
                                "type": "text",
                                "text": f"Egocentric video clip for action #{image_i}:",
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": pil_to_b64(example_img),
                                    "detail": "high"
                                },
                            },
                        ]
                    )
                inputs_example = f"Inputs:"
                example_content = [{"type": "text", "text": inputs_example}] + example_content
                message.append({"role": "system", "name": "example_user", "content": example_content})
                if self.cfg.DO_ICAL_PROMPT:
                    output_example = f"Outputs:\n\nSummary: {e}\n\nAbstracted State:\n{f}\n\nStep-by-step Reasoning: {g}\n\nPredicted State Change: {h}\n\nAbstraction Comments:\n{i}\n\nVideo Actions:\n{a}"
                elif self.cfg.DO_RAW_DEMOS:
                    output_example = f"Outputs:\n\nVideo Actions:\n{a}"
                else:
                    output_example = f"Outputs:\n\nReasoning: {g}\n\nVideo Actions:\n{b}"
                message.append(
                        {
                            "role": "system",
                            "name": "example_assistant",
                            "content": [{"type": "text", "text": output_example}],
                        }
                    )
        if single_image:
            images_samples = np.concatenate([np.asarray(video_image) for video_image in images], axis=1)
            images_samples = [Image.fromarray(images_samples)]
        elif multivideo:
            images_samples = np.stack([np.asarray(video_image) for video_image in images], axis=0)
            images_samples = images_samples.reshape(B, S, H, W, C)
            images_samples = [Image.fromarray(np.concatenate([images_sample_ for images_sample_ in images_sample], axis=1)) for images_sample in images_samples]
        else:
            images_samples = images
        content = []
        print(f'Number of images: {len(images_samples)}')
        alphabet = 'abcdefghijklmnopqrstuvwxyz'
        for image_i, image in enumerate(images_samples):
            content.extend(
                [
                    {
                        "type": "text",
                        "text": f"Egocentric video clip for action #{image_i}:",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": pil_to_b64(image), "detail": "high"},
                    },
                ]
            )
        message.append({"role": "user", "content": content})
        return message

    def get_prompt_predict_actions(
        self,
        images,
        video_actions,
        B, S, C, H, W,
        topk_examples,
        use_examples=True,
        single_image=False,
        multivideo=True,
        with_SoM=True,
    ):
        message = [
            {
                "role": "system",
                "content": [{"type": "text", "text": self.intro_ltp}],
            }
        ]
        if use_examples:
            for (a,b,c,d,e,f,g,h,i) in list(topk_examples):
                if (1):
                    c = [Image.open(video_image).convert('RGB') for video_image in c]
                    c = np.concatenate([np.asarray(video_image) for video_image in c], axis=1)
                    c = [Image.fromarray(c).convert('RGB')]
                else:
                    c = [Image.open(video_image).convert('RGB') for video_image in c]
                    # c = np.concatenate([np.asarray(video_image) for video_image in c], axis=1)
                    # c = [Image.fromarray(c)]
                alphabet = 'abcdefghijklmnopqrstuvwxyz'
                example_content = []
                for image_i, image_f in enumerate(c):
                    if type(image_f)==str:
                        example_img = Image.open(image_f)
                    else:
                        example_img = image_f
                    print(f"Example video frame {image_i}:")
                    example_content.extend(
                        [
                            {
                                "type": "text",
                                "text": f"Egocentric video clip for action #{image_i}:",
                            },
                            
                            # {
                            #     "type": "text",
                            #     "text": f"Video frame {image_i}:",
                            # },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": pil_to_b64(example_img),
                                    "detail": "high"
                                },
                            },
                        ]
                    )
                if (1):
                    inputs_example = f"Inputs:"
                else:
                    inputs_example = f"Inputs:\n\nVideo Actions:\n{b}"
                example_content = [{"type": "text", "text": inputs_example}] + example_content
                message.append({"role": "system", "name": "example_user", "content": example_content})
                if self.cfg.DO_ICAL_PROMPT:
                    output_example = f"Outputs:\n\nSummary: {e}\n\nAbstracted State:\n{f}\n\nStep-by-step Reasoning: {g}\n\nPredicted State Change: {h}\n\nAbstraction Comments:\n{i}\n\nFuture Actions:\n{a}"
                elif self.cfg.DO_RAW_DEMOS:
                    output_example = f"Outputs:\n\nFuture Actions:\n{a}"
                else:
                    output_example = f"Outputs:\n\nReasoning: {g}\n\nFuture Actions:\n{a}"
                message.append(
                        {
                            "role": "system",
                            "name": "example_assistant",
                            "content": [{"type": "text", "text": output_example}],
                        }
                    )
        if (0):
            idx1 = np.random.randint(0, 3)
            idx2 = np.random.randint(3, 6)
            idx3 = np.random.randint(6, 9)
            idx4 = np.random.randint(9, 12)
            images_samples = [images[idx] for idx in [idx1, idx2, idx3, idx4]]
        elif single_image:
            images_samples = np.concatenate([np.asarray(video_image) for video_image in images], axis=1)
            images_samples = [Image.fromarray(images_samples)]
        elif multivideo:
            images_samples = np.stack([np.asarray(video_image) for video_image in images], axis=0)
            images_samples = images_samples.reshape(B, S, H, W, C)
            images_samples = [Image.fromarray(np.concatenate([images_sample_ for images_sample_ in images_sample], axis=1)) for images_sample in images_samples]
        else:
            images_samples = images
        content = []
        print(f'Number of images: {len(images_samples)}')
        alphabet = 'abcdefghijklmnopqrstuvwxyz'
        for image_i, image in enumerate(images_samples):
            print(f"Video frame {image_i}:")
            content.extend(
                [
                    {
                        "type": "text",
                        "text": f"Egocentric video clip for action #{image_i}:",
                    },
                    # {
                    #     "type": "text",
                    #     "text": f"Video frame {image_i}:",
                    # },
                    {
                        "type": "image_url",
                        "image_url": {"url": pil_to_b64(image), "detail": "high"},
                    },
                ]
            )
        if (1):
            current_prompt = f"Inputs:"
        else:
            current_prompt = f"Inputs:\n\nVideo Actions:\n{video_actions}"
        content = [{"type": "text", "text": current_prompt}] + content
        message.append({"role": "user", "content": content})
        return message

    def get_SoM_input(
        self,
        images,
        model_name="semantic-sam",
    ):

        SoM = SoM_inference(images)

        # if model_name=="semantic-sam":
        #     model = model_semsam
        #     output, mask_, label = inference_semsam_m2m_auto(model, image['image'], level, text, text_part, text_thresh, text_size, hole_scale, island_scale, semantic, label_mode=label_mode, alpha=alpha, anno_mode=anno_mode, label=label, return_label=True)
        # elif model_name=="seem":
        #     model = model_seem
        #     output, mask_, label = inference_seem_pano(model, image['image'], text_size, label_mode, alpha, anno_mode, label=label, return_label=True)
        # elif model_name=="sam":
        #     model = model_sam
        #     output, mask_, label = inference_sam_m2m_auto(model, image['image'], text_size, label_mode, alpha, anno_mode, label=label, return_label=True)

    def prompt_gpt_text(
        self,
        prompt,
        temperature: float = 0,
        max_tokens: int = 4096,
        stop_token = None,
        seed = 32,
    ):
        logging.basicConfig()
        logging.getLogger().setLevel(logging.INFO)
        openai.log='info'
        if use_azure:
            print("Running gpt-3.5 turbo...")
            messages = [
                        {"role": "user", "content": prompt},
                        ]
            response = client.chat.completions.create(
                model="gpt-35-turbo-1106",
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                seed=seed,
            )
            print(response.choices[0].message.content)
            answer = response.choices[0].message.content
        else:
            assert(False) # no support currently
            if "OPENAI_API_KEY" not in os.environ:
                raise ValueError(
                    "OPENAI_API_KEY environment variable must be set when using OpenAI API."
                )

            response = client.completions.create(
                prompt=prompt,
                engine=engine,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                stop=[stop_token],
            )
            answer: str = response["choices"][0]["text"]
        return answer


    def generate_from_openai_completion(
        self,
        messages,
        temperature: float = 0,
        max_tokens: int = 4096,
        stop_token = None,
        seed = 32,
    ) -> str:
        logging.basicConfig()
        logging.getLogger().setLevel(logging.INFO)
        openai.log='info'
        if use_azure:
            print("Running gpt-4-vision-preview...")
            response = client.chat.completions.create(
                model="gpt-4-vision-preview",
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                seed=seed,
            )
            print(response.choices[0].message.content)
            answer = response.choices[0].message.content
        else:
            assert(False) # no support currently
            if "OPENAI_API_KEY" not in os.environ:
                raise ValueError(
                    "OPENAI_API_KEY environment variable must be set when using OpenAI API."
                )

            response = client.completions.create(
                prompt=prompt,
                engine=engine,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                stop=[stop_token],
            )
            answer: str = response["choices"][0]["text"]
        return answer

    def generate(self, x, k=1, return_images_only=False):
        x = self.forward(x, k, return_images_only=return_images_only)
        # results = []
        # for head_x in x:
        #     if k>1:
        #         preds_dist = Categorical(logits=head_x)
        #         preds = [preds_dist.sample() for _ in range(k)]
        #     elif k==1:
        #         preds = [head_x.argmax(2)]
        #     head_x = torch.stack(preds, dim=1)
        #     results.append(head_x)

        return x
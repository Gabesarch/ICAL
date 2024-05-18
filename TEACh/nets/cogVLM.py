from PIL import Image
import numpy as np
import ipdb
st = ipdb.set_trace
import torch
from arguments import args
import PIL
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
import sys

from huggingface_hub import hf_hub_download
import torch
import matplotlib.pyplot as plt

import argparse
import torch

sys.path.append('cogVLM')

import torch
import requests
from PIL import Image
from transformers import AutoModelForCausalLM, LlamaTokenizer

from PIL import Image

import requests
from PIL import Image
from io import BytesIO
import re

device = "cuda" if torch.cuda.is_available() else "cpu"

class COGVLM:
    def __init__(self):

        self.tokenizer = LlamaTokenizer.from_pretrained('lmsys/vicuna-7b-v1.5')
        self.model = AutoModelForCausalLM.from_pretrained(
            'THUDM/cogvlm-chat-hf',
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        ).to(device).eval()

    @torch.no_grad()
    def run_cogvlm(self, images, text, return_scores=False):
        '''
        images is list, assumed that last image is query image
        '''

        """
        Step 2: Preprocessing images
        Details: For OpenFlamingo, we expect the image to be a torch tensor of shape 
        batch_size x num_media x num_frames x channels x height x width. 
        In this case batch_size = 1, num_media = 3, num_frames = 1,
        channels = 3, height = 224, width = 224.
        """
        inputs = self.model.build_conversation_input_ids(self.tokenizer, query=text, history=[], images=images, template_version='vqa')   # vqa mode
        inputs = {
            'input_ids': inputs['input_ids'].unsqueeze(0).to('cuda'),
            'token_type_ids': inputs['token_type_ids'].unsqueeze(0).to('cuda'),
            'attention_mask': inputs['attention_mask'].unsqueeze(0).to('cuda'),
            'images': [[inputs['images'][0].to('cuda').to(torch.bfloat16)]],
        }
        gen_kwargs = {"max_length": 2048, "do_sample": False}

        if return_scores:
            gen_kwargs.update({"return_dict_in_generate": True, "output_scores": True})

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)
            if return_scores:
                # import torch.nn as nn
                # print(nn.Softmax()(torch.sort(outputs['scores'][1]).values[0])[-20:])
                transition_scores = self.model.compute_transition_scores(
                    outputs.sequences, outputs.scores, normalize_logits=True
                )
                input_length = inputs['input_ids'].shape[1]
                generated_tokens = outputs['sequences'][:, input_length:]
                scores_percent = np.exp(transition_scores[0][:-1].cpu().numpy())
                # for tok, score in zip(generated_tokens[0], transition_scores[0]):
                #     # | token | token string | logits | probability
                #     print(f"| {tok:5d} | {self.tokenizer.decode(tok):8s} | {score.cpu().numpy():.3f} | {np.exp(score.cpu().numpy()):.2%}")
            if "return_dict_in_generate" in gen_kwargs:
                outputs = outputs['sequences']
            # outputs = self.model.compute_transition_scores(**inputs, **gen_kwargs)
            outputs = outputs[:, inputs['input_ids'].shape[1]:]
        output = self.tokenizer.decode(outputs[0]).strip()
        output = output.replace('</s>', '')

        visualize = False
        if visualize:

            print(text)
            plt.figure()
            plt.imshow(images[0])
            plt.savefig('output/test.png')
            st()

        if return_scores:
            return output, scores_percent
        else:
            return output


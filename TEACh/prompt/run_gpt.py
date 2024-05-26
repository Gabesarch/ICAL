import os
import openai
import tiktoken
import ipdb
st = ipdb.set_trace
from tqdm import tqdm
import glob
import json
import numpy as np
import copy
from arguments import args
import time
azure = not args.use_openai
if azure:
    openai.api_type = "azure"
    openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT") 
    openai.api_version = "2023-05-15"
    openai.api_key = os.getenv("AZURE_OPENAI_KEY")
else:
    try:
        home_directory = os.path.expanduser( '~' )
        with open(os.path.join(home_directory,".openai/openai.key"), 'r') as f:
            org_key = f.readlines()
            openai.api_key = org_key[1].strip()
            openai.organization = org_key[0].strip()
            msft_interal_key = org_key[2].strip()
    except:
        openai.api_key = os.getenv("OPENAI_API_KEY")
import logging
import sys
logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)
openai.log='info'

class LLMPlanner:
    '''
    LLM Planner for going from teach dialogue to executable program
    '''
    def __init__(
        self, 
        gpt_embedding_dir='', 
        nn_examples=True, 
        fillable_classes=[],
        openable_classes=[],
        include_classes=[],
        clean_classes=[],
        example_mode='',
        ):
        self.nn_examples = nn_examples
        self.fillable_classes = fillable_classes
        self.openable_classes = openable_classes
        self.include_classes = include_classes
        self.clean_classes = clean_classes
        self.examples = ''
        self.example_mode = example_mode

        with open('prompt/api_primitives_nodefinitions.py') as f:
            self.api = f.read()

        with open('prompt/api_corrective.py') as f:
            self.api_corrective = f.read()

        if args.gpt_model=="gpt-3.5-turbo-1106-ft" and not args.ft_with_retrieval and not args.zero_shot:
            with open('prompt/prompt_plan_input_zeroshot.txt') as f:
                self.prompt_plan = f.read()
        else:
            if not args.use_gt_attributes or not args.use_gt_metadata:
                with open('prompt/prompt_plan_estimated.txt') as f:
                    self.prompt_plan = f.read()
            else:
                with open('prompt/prompt_plan.txt') as f:
                    self.prompt_plan = f.read()

        with open('prompt/prompt_replan.txt') as f:
            self.prompt_replan = f.read()

        self.azure = azure
        self.model = args.gpt_model
        self.max_token_length = args.max_token_length

        self.command = None

        self.seed = args.seed

    def get_examples_planning(self, topk):
        if args.ablate_example_retrieval:
            print("Fixing examples!")
            distance_argsort_topk = np.arange(topk)
        else:
            if self.azure:
                embedding = openai.Embedding.create(
                        # engine="kateftextembeddingada002",
                        engine="text-embedding-ada-002",
                        input=self.command,
                        )['data'][0]['embedding']
            else:
                embedding = openai.Embedding.create(
                        model="text-embedding-ada-002",
                        input=self.command,
                        )['data'][0]['embedding']
            embedding = np.asarray(embedding)
            # nearest neighbor
            distance = np.linalg.norm(self.embeddings - embedding[None,:], axis=1)
            distance_argsort_topk = np.argsort(distance)[:topk]
        example_text = "Here are a few examples of typical inputs and outputs (only for in-context reference):\n"
        example_number = 1
        for idx in list(distance_argsort_topk):
            example_text += f'Example #{example_number}:\n'
            with open(f'{self.file_order[idx]}') as f:
                example = f.read()
            example_text += example
            example_text += '\n\n'
            example_number += 1
        print(f"most relevant examples are: {[self.file_order[idx] for idx in list(distance_argsort_topk)]}")
        self.examples = example_text

    def get_prompt_proportion(self, prompt):
        if self.model=="gpt-3.5-turbo":
            enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
            prompt_token_length = len(enc.encode(prompt))
            prompt_len_percent = prompt_token_length/min(self.max_token_length, 4096)
        elif self.model=="gpt-3.5-turbo-1106":
            enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
            prompt_token_length = len(enc.encode(prompt))
            prompt_len_percent = prompt_token_length/min(self.max_token_length,16384)
        elif self.model=="gpt-3.5-turbo-1106-ft":
            enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
            prompt_token_length = len(enc.encode(prompt))
            prompt_len_percent = prompt_token_length/min(self.max_token_length,16384)
        elif self.model=="gpt-3.5-turbo-16k":
            enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
            prompt_token_length = len(enc.encode(prompt))
            prompt_len_percent = prompt_token_length/min(self.max_token_length,16384)
        elif self.model=="gpt-3.5-turbo-instruct":
            enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
            prompt_token_length = len(enc.encode(prompt))
            prompt_len_percent = prompt_token_length/min(self.max_token_length,4096)
        elif self.model=="gpt-4":
            enc = tiktoken.encoding_for_model("gpt-4")
            prompt_token_length = len(enc.encode(prompt))
            prompt_len_percent = prompt_token_length/min(self.max_token_length,8192)
        elif self.model=="gpt-4-32k":
            enc = tiktoken.encoding_for_model("gpt-4")
            prompt_token_length = len(enc.encode(prompt))
            prompt_len_percent = prompt_token_length/min(self.max_token_length,32768)
        elif self.model=="gpt-4-1106-Preview":
            enc = tiktoken.encoding_for_model("gpt-4")
            prompt_token_length = len(enc.encode(prompt))
            prompt_len_percent = prompt_token_length/min(self.max_token_length,16384)
        elif self.model=="text-davinci-003":
            enc = tiktoken.encoding_for_model("text-davinci-003")
            prompt_token_length = len(enc.encode(prompt))
            prompt_len_percent = prompt_token_length/min(self.max_token_length,4097)
        elif self.model=="text-davinci-002":
            enc = tiktoken.encoding_for_model("text-davinci-002")
            prompt_token_length = len(enc.encode(prompt))
            prompt_len_percent = prompt_token_length/min(self.max_token_length,4097)
        elif self.model=="code-davinci-002":
            enc = tiktoken.encoding_for_model("code-davinci-002")
            prompt_token_length = len(enc.encode(prompt))
            prompt_len_percent = prompt_token_length/min(self.max_token_length,4097)
        else:
            assert(False) # what model is this? 
        return prompt_len_percent

    def run_gpt(self, prompt, log_plan=True, temperature=0, seed=None):
        if self.azure:
            if self.model=="gpt-3.5-turbo":
                for _ in range(5):
                    try:
                        print("RUNNING GPT 3.5")
                        messages = [
                        {"role": "user", "content": prompt},
                        ]
                        response = openai.ChatCompletion.create(
                            # engine="katefgpt35turbo",
                            engine = "gpt-35-turbo",
                            messages=messages,
                            temperature=temperature,
                            seed=seed if seed is not None else self.seed,
                            )
                        response = response["choices"][0]["message"]["content"]
                        break
                    except:
                        time.sleep(1)
            elif self.model=="gpt-3.5-turbo-1106":
                for _ in range(5):
                    try:
                        print("RUNNING GPT 3.5 1106")
                        messages = [
                        {"role": "user", "content": prompt},
                        ]
                        response = openai.ChatCompletion.create(
                            # engine="katefgpt35turbo",
                            engine = "gpt-35-turbo-1106",
                            messages=messages,
                            temperature=temperature,
                            seed=seed if seed is not None else self.seed,
                            )
                        response = response["choices"][0]["message"]["content"]
                        break
                    except Exception as e:
                        if "The response was filtered" in str(e):
                            response = ""
                            break
                        time.sleep(1)
            elif self.model=="gpt-3.5-turbo-1106-ft":
                for _ in range(5):
                    try:
                        print("RUNNING GPT 3.5 1106 FT")
                        messages = [
                        {"role": "user", "content": prompt},
                        ]
                        response = openai.ChatCompletion.create(
                            # engine="katefgpt35turbo",
                            engine = "gpt-35-turbo-1106-ft",
                            messages=messages,
                            temperature=temperature,
                            seed=seed if seed is not None else self.seed,
                            )
                        response = response["choices"][0]["message"]["content"]
                        break
                    except Exception as e:
                        if "The response was filtered" in str(e):
                            response = ""
                            break
                        time.sleep(1)
            elif self.model=="gpt-3.5-turbo-instruct":
                for _ in range(5):
                    try:
                        print("RUNNING GPT gpt-3.5-turbo-instruct")
                        enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
                        prompt_token_length = len(enc.encode(prompt))
                        print(f"Max tokens = {4096 - prompt_token_length}")
                        response = openai.Completion.create(
                            engine = "gpt-35-turbo-instruct",
                            prompt=prompt,
                            temperature=temperature,
                            max_tokens=4096 - prompt_token_length,
                            )
                        response = response['choices'][0]['text'] #response["choices"][0]["message"]["content"]
                        break
                    except Exception as e:
                        time.sleep(1)
            elif self.model=="gpt-4":
                for _ in range(5):
                    try:
                        print("RUNNING GPT 4")
                        messages = [
                        {"role": "user", "content": prompt},
                        ]
                        response = openai.ChatCompletion.create(
                            # engine="katefgpt35turbo",
                            engine = "gpt-4",
                            messages=messages,
                            temperature=temperature,
                            )
                        response = response["choices"][0]["message"]["content"]
                        break
                    except:
                        time.sleep(1)
            elif self.model=="gpt-4-1106-Preview":
                for _ in range(5):
                    try:
                        print("RUNNING GPT 4 1106 Preview")
                        messages = [
                        {"role": "user", "content": prompt},
                        ]
                        response = openai.ChatCompletion.create(
                            engine = "gpt-4-1106-Preview",
                            messages=messages,
                            temperature=temperature,
                            )
                        response = response["choices"][0]["message"]["content"]
                        break
                    except:
                        if "The response was filtered" in str(e):
                            response = ""
                            break
                        time.sleep(1)
            elif self.model=="gpt-4-32k":
                for _ in range(5):
                    try:
                        print("RUNNING GPT 4 32k")
                        messages = [
                        {"role": "user", "content": prompt},
                        ]
                        response = openai.ChatCompletion.create(
                            engine = "gpt-4-32k",
                            messages=messages,
                            temperature=temperature,
                            )
                        response = response["choices"][0]["message"]["content"]
                        break
                    except:
                        time.sleep(1)
            elif self.model=="text-davinci-003":
                for _ in range(5):
                    try:
                        enc = tiktoken.encoding_for_model("text-davinci-003")
                        prompt_token_length = len(enc.encode(prompt))
                        print(f"Max tokens = {4097 - prompt_token_length}")
                        response = openai.Completion.create(
                            engine="kateftextdavinci003",
                            prompt=prompt,
                            temperature=temperature,
                            max_tokens=4097 - prompt_token_length,
                            )
                        response = response['choices'][0]['text']
                        break
                    except:
                        time.sleep(1)
            elif self.model=="text-davinci-002":
                enc = tiktoken.encoding_for_model("text-davinci-002")
                prompt_token_length = len(enc.encode(prompt))
                response = openai.Completion.create(
                    engine="kateftextdavinci002",
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=4097 - prompt_token_length,
                    )
                response = response['choices'][0]['text']
                print(f"Max tokens = {4097 - prompt_token_length}")
            elif self.model=="code-davinci-002":
                enc = tiktoken.encoding_for_model("code-davinci-002")
                prompt_token_length = len(enc.encode(prompt))
                response = openai.Completion.create(
                    engine="katefcodedavinci002",
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=8001 - prompt_token_length,
                    )
                response = response['choices'][0]['text']
                print(f"Max tokens = {8001 - prompt_token_length}")
            else:
                assert(False) # what model is this? 

        else:
            if self.model=="gpt-3.5-turbo":
                messages = [
                {"role": "system", "content": prompt},
                ]
                response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                # model="gpt-4",
                messages=messages,
                temperature=temperature,
                )["choices"][0]["message"]["content"]
            elif self.model=="gpt-3.5-turbo-instruct":
                messages = [
                {"role": "system", "content": prompt},
                ]
                response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-instruct",
                # model="gpt-4",
                messages=messages,
                temperature=temperature,
                )["choices"][0]["message"]["content"]
            elif self.model=="gpt-3.5-turbo-16k":
                messages = [
                {"role": "system", "content": prompt},
                ]
                response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-16k-0613",
                # model="gpt-4",
                messages=messages,
                temperature=temperature,
                )["choices"][0]["message"]["content"]
            elif self.model=="gpt-4":
                messages = [
                        {"role": "system", "content": prompt},
                        ]
                while True:
                    try:
                        response = openai.ChatCompletion.create(
                        model="gpt-4-0613",
                        messages=messages,
                        temperature=temperature,
                        )["choices"][0]["message"]["content"]
                        break
                    except:
                        time.sleep(0.1)
            elif self.model=="text-davinci-003":
                enc = tiktoken.encoding_for_model("text-davinci-003")
                prompt_token_length = len(enc.encode(prompt))
                response = openai.Completion.create(
                    model="text-davinci-003",
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=4097 - prompt_token_length,
                    ) 
                response = response['choices'][0]['text']
                print(f"Max tokens = {4097 - prompt_token_length}")
            else:
                assert(False) # what model is this? 
        if self.command is not None:
            print(f"\n\ncommand is {self.command}\n\n")
        print(response)
        if log_plan:
            self.plan = response
        return response

    def get_embedding(self, text_to_embed):
        if self.azure:
            embedding = openai.Embedding.create(
                    # engine="kateftextembeddingada002",
                    engine="text-embedding-ada-002",
                    input=text_to_embed,
                    )['data'][0]['embedding']
        else:
            embedding = openai.Embedding.create(
                    model="text-embedding-ada-002",
                    input=text_to_embed,
                    )['data'][0]['embedding']
        embedding = np.asarray(embedding)
        return embedding

    def parse_object(self, line, subgoals, objects, object_mapping, search_dict):
        exec(line)
        var_name = line.split(' = InteractionObject')[0]
        object_cat = eval(f'{var_name}.object_class')
        search_obj = eval(f'{var_name}.landmark')
        attributes = eval(f'{var_name}.attributes').copy()
        map_ = var_name #line.split(' = ')[0]
        if object_cat not in self.include_classes:
            object_cat = self.get_closest_category_to_word(object_cat)
        if object_cat in self.clean_classes:
            attributes.append("clean")
        object_mapping[map_] = object_cat
        if search_obj is not None:
            if search_obj not in self.include_classes:
                search_obj = self.get_closest_category_to_word(search_obj)
            if search_obj in self.include_classes:
                if object_cat not in search_dict.keys():
                    search_dict[object_cat] = []
                search_dict[object_cat].append(search_obj)
        for attribute in attributes:
            if attribute=="toasted":
                subgoals.append("Toast")
                objects.append(object_cat)
            elif attribute=="clean":
                subgoals.append("Clean")
                objects.append(object_cat)
            elif attribute=="cooked":
                subgoals.append("Cook")
                objects.append(object_cat)
        return subgoals, objects, object_mapping, search_dict
    
    def response_to_subgoals(self, response, remove_objects=True):
        '''
        Map from output code to Teach subgoals
        '''

        subgoals = []
        objects = []
        search_dict = {}
        code_lines = response.split('\n')
        comment_block = False
        object_mapping = {}
        agent_mapping = {}
        subgoal_mapping = {
            "go_to": "Navigate", 
            "pickup_and_place":"pickup_and_place", 
            "pickup":"Pickup", 
            "place":"Place", 
            "slice":"Slice", 
            "toggle_on":"ToggleOn", 
            "toggle_off":"ToggleOff", 
            "open":"Open", 
            "close":"Close", 
            "clean":"Clean", 
            "put_down":"PutDown",
             "pour":"Pour", 
             "fill_up":"FillUp",
             "empty":"Empty",
             "toast":"Toast",
             "cook":"Cook",
             "move_back":"MoveBack",
             "move_closer":"MoveCloser",
             "move_alternate_viewpoint":"MoveAlternate"
             }
        for line_i in range(len(code_lines)):
            line = code_lines[line_i]
            if line[:4]=='    ':
                line = line[4:] # remove tab
            if comment_block:
                if line[:3]=='"""':
                    comment_block = False # out of comment block
                continue
            elif len(line)==0:
                continue # nothing
            elif line[0]=="#":
                continue # comment
            elif line[:5]=='print':
                continue # print
            elif line[:3]=='"""':
                comment_block = True
                continue # comment block
            elif line[:4]=='def ':
                continue # function statement
            elif line[:2]=='- ':
                continue # bullet
            elif line[:5]=="Plan:":
                continue # start of plan
            elif 'InteractionObject' in line:
                try:
                    subgoals, objects, object_mapping, search_dict = self.parse_object(line, subgoals, objects, object_mapping, search_dict)
                except:
                    continue
            elif ' = AgentCorrective' in line:
                map_ = line.split(' = ')[0]
                agent_mapping[map_] = "agent"
            else:
                # log subgoal
                map_ = line.split('.')[0]

                if map_ in agent_mapping.keys():
                    sub_ = line.split('.')[1].split('(')[0]
                    if sub_ not in subgoal_mapping.keys():
                        continue
                    subgoals.append(subgoal_mapping[sub_])
                    objects.append("Agent")
                    continue

                # check for bad output by LLM
                if map_ not in object_mapping.keys():
                    try:
                        if '"' not in map_:
                            line_ = f'{map_} = InteractionObject("{map_}")'
                        else:
                            line_ = f'{map_} = InteractionObject({map_})'
                        subgoals, objects, object_mapping, search_dict = self.parse_object(line_, subgoals, objects, object_mapping, search_dict)
                    except:
                        continue
                    if map_ not in object_mapping.keys():
                        continue
                    # continue
                object_cat = object_mapping[map_]
                if object_cat not in self.include_classes:
                    continue
                try:
                    sub_ = line.split('.')[1].split('(')[0]
                except:
                    continue
                if sub_ not in subgoal_mapping.keys():
                    continue
                    
                subgoal = subgoal_mapping[sub_]
                if subgoal in ["Place", "Pour","pickup_and_place"]:

                    # get placing category
                    if 'InteractionObject' in line:
                        object_cat_ = line.split('InteractionObject("')[-1].split('"')[0]
                    else:
                        if subgoal in ["Place", "pickup_and_place"]:
                            map2_ = line.split('place(')[-1].split(')')[0]
                            if map2_ not in object_mapping.keys():
                                line_ = f'{map2_} = InteractionObject("{map2_}")'
                                subgoals, objects, object_mapping, search_dict = self.parse_object(line_, subgoals, objects, object_mapping, search_dict)
                                if map2_ not in object_mapping.keys():
                                    continue
                                # continue
                            object_cat_ = object_mapping[map2_]
                        elif subgoal=="Pour":
                            map2_ = line.split('pour(')[-1].split(')')[0]
                            if map2_ not in object_mapping.keys():
                                line_ = f'{map2_} = InteractionObject("{map2_}")'
                                subgoals, objects, object_mapping, search_dict = self.parse_object(line_, subgoals, objects, object_mapping, search_dict)
                                if map2_ not in object_mapping.keys():
                                    continue
                                # continue
                            object_cat_ = object_mapping[map2_]
                    if object_cat_ not in self.include_classes:
                        continue

                    if subgoal=="pickup_and_place":
                        subgoals.extend(["Navigate", "Pickup", "Navigate", "Place"])
                        objects.extend([object_cat, object_cat, object_cat_, object_cat_])
                    else:
                        if len(subgoals)<2 or (subgoal=="Place" and subgoals[-2]!="Pickup"):
                            # need to pickup before placing
                            subgoals.append("Navigate")
                            objects.append(object_cat)
                            subgoals.append("Pickup")
                            objects.append(object_cat)
                            subgoals.append("Navigate")
                            objects.append(object_cat_)
                        subgoals.append(subgoal)
                        objects.append(object_cat_)

                    if remove_objects:
                        # check if object is used after this, and if not, remove from list of interactable objects
                        # necessary for "do X with all Y"
                        object_used_after = False
                        for line_ in code_lines[line_i+1:]:
                            if map_ in line_:
                                object_used_after = True
                                break
                        if not object_used_after and (object_cat not in ["Knife"]) and (subgoal not in ["Pour"]):
                            subgoals.append("ObjectDone")
                            objects.append(object_cat)
                elif subgoal=="Clean":
                    subgoals.append("Clean")
                    objects.append(object_cat)
                elif subgoal=="FillUp":
                    subgoals.extend(["Navigate", "Place", "ToggleOn", "ToggleOff", "Pickup"])
                    objects.extend(["Sink", "Sink", "Faucet", "Faucet", object_cat])
                elif subgoal=="PutDown":
                    subgoals.extend(["PutDown"])
                    objects.extend(["PutDown"])
                elif subgoal=="Toast":
                    subgoals.append("Toast")
                    objects.append(object_cat)
                elif subgoal=="Cook":
                    subgoals.append("Cook")
                    objects.append(object_cat)
                elif subgoal in ["Open", "Close"]:
                    if object_cat in self.openable_classes:
                        subgoals.append(subgoal)
                        objects.append(object_cat)
                else:
                    subgoals.append(subgoal)
                    objects.append(object_cat)

        self.object_mapping = object_mapping
        self.search_dict = search_dict
        
        return subgoals, objects, search_dict

    def get_closest_category_to_word(self, word, grounding_phrase=None):
        '''
        Function for getting closest teach category by querying LLM
        '''
        with open('prompt/prompt_closest_category.txt') as f:
            prompt = f.read()
        prompt = prompt.replace('{word}', f'{word}')
        prompt = prompt.replace('{grounding_phrase}', f'{grounding_phrase}')
        response = self.run_gpt(prompt, log_plan=False)
        response = response.replace('output:', '')
        response = response.replace('Output:', '')
        response = response.replace(' ', '')
        response = response.replace('\n', '')
        response = response.replace('.', '')
        response = response.replace(',', '')
        return response

    def get_get_search_categories(self, target_category):
        '''
        Function for getting most likely search objects from LLM
        '''
        with open('prompt/prompt_search.txt') as f:
            prompt = f.read()
        prompt = prompt.replace('{target}', f'{target_category}')
        prompt = prompt.replace('{dialogue}', f'{self.command}')
        response = self.run_gpt(prompt, log_plan=False)
        response = response.replace('answer:', '')
        response = response.replace(' ', '')
        response = response.replace('\n', '')
        response = response.split(',')
        # make sure all are valid categories
        response_ = []
        for r in response:
            if r in self.include_classes:
                response_.append(r)
        response = response_

        return response

    def subgoals_to_program(self, subgoals, held_obj=None, initial_states=None):
        subgoals = copy.deepcopy(subgoals)
        subgoal_mapping = {
            "go_to": "Navigate", 
            "pickup_and_place":"pickup_and_place", 
            "pickup":"Pickup", 
            "place":"Place", 
            "slice":"Slice", 
            "toggle_on":"ToggleOn", 
            "toggle_off":"ToggleOff", 
            "open":"Open", 
            "close":"Close", 
            "clean":"Clean", 
            "put_down":"PutDown",
            "pour":"Pour", 
            "fill_up":"FillUp",
            "clean":"Clean",
            "empty":"Empty",
            "toast":"Toast",
            "cook":"Cook",
            "change_state":"StateChange",
            }
        subgoal_mapping_r = {v:k for k,v in subgoal_mapping.items()}
        objects = {}
        sliced_dict = {}
        objects_tmp = []
        obj_count = {}
        search_dict = copy.deepcopy(self.search_dict)
        program = ''
        first_subgoal=True
        if held_obj is not None:
            obj = held_obj
            if obj not in objects_tmp:
                objects[obj] = f'target_{obj.lower()}'
                if obj in obj_count.keys():
                    objects[obj] = objects[obj] + str(obj_count[obj])
                if obj in self.search_dict.keys():
                    program += f'{objects[obj]} = InteractionObject("{obj}", landmark = "{search_dict[obj][0]}")\n'
                    search_dict[obj].pop(0)
                    if len(search_dict[obj])==0:
                        del search_dict[obj]
                else:
                    program += f'{objects[obj]} = InteractionObject("{obj}")\n'
                objects_tmp.append(obj)
            held_obj = objects[obj]
            
        while len(subgoals)>0:

            subgoal = subgoals.pop(0)
            obj = subgoal[1]
            sub = subgoal[0]
            if type(obj) in [list, tuple]:
                obj = obj[0]

            if sub in ["MoveBack", "MoveCloser", "MoveAlternate"]:
                continue

            obj_type = obj.split('_')[0]
            if obj=="Sink" and sub=="Navigate" and [s[0] for s in subgoals[:4]]==["Place", "ToggleOn", "ToggleOff", "Pickup"]:
                # clean subgoal
                obj = subgoals[3][1]
                obj_type = obj.split('_')[0]
                if len(subgoals)>4 and subgoals[4][0]=="Pour":
                    sub = "Clean"
                    subgoals = subgoals[5:]
                elif "Pour" in [s[0] for s in subgoals]:
                    # if pour in future, then likely fillup subgoal
                    sub = "FillUp"
                    subgoals = subgoals[4:]
                else:
                    sub = "Clean"
                    subgoals = subgoals[4:]
            elif sub=="Navigate" and [s[0] for s in subgoals[:3]]==["Pickup", "Navigate", "Place"]:
                if (len(subgoals)>6 and [s[0] for s in subgoals[2:6]]==["Place", "ToggleOn", "ToggleOff", "Pickup"]):
                    # clean subgoals so skip
                    pass
                else:
                    # pickup and place subgoal
                    obj = subgoals[0][1]
                    obj_type = obj.split('_')[0]
                    if obj not in objects_tmp:
                        objects[obj] = f'target_{obj.lower()}'
                        if obj in obj_count.keys():
                            objects[obj] = objects[obj] + str(obj_count[obj])
                        if obj in self.search_dict.keys():
                            program += f'{objects[obj]} = InteractionObject("{obj_type}", landmark = "{search_dict[obj_type][0]}")\n'
                            st()
                            if initial_states is not None:
                                if "Sliced" in obj_type and obj_type.replace("Sliced", "") in sliced_dict.keys():
                                    program = f'{program[:-2]}, object_instance = None, parent_object = "{sliced_dict[obj_type.replace("Sliced", "")]}") # Initialize new sliced object from sliced parent\n'
                                else:
                                    program = f'{program[:-2]}, object_instance = "{obj}")\n'
                            search_dict[obj].pop(0)
                            if len(search_dict[obj])==0:
                                del search_dict[obj]
                        else:
                            program += f'{objects[obj]} = InteractionObject("{obj_type}")\n'
                            if initial_states is not None:
                                if "Sliced" in obj_type and obj_type.replace("Sliced", "") in sliced_dict.keys():
                                    program = f'{program[:-2]}, object_instance = None, parent_object = "{sliced_dict[obj_type.replace("Sliced", "")]}") # Initialize new sliced object from sliced parent\n'
                                else:
                                    program = f'{program[:-2]}, object_instance = "{obj}")\n'
                        objects_tmp.append(obj)
                    held_obj = objects[obj]
                    obj = subgoals[2][1]
                    obj_type = obj.split('_')[0]
                    sub = "pickup_and_place"
                    subgoals = subgoals[4:]

            if obj not in objects_tmp and obj is not None:
                objects[obj] = f'target_{obj.lower()}'
                if obj in obj_count.keys():
                    objects[obj] = objects[obj] + str(obj_count[obj])
                if obj in self.search_dict.keys():
                    program += f'{objects[obj]} = InteractionObject("{obj_type}", landmark = "{search_dict[obj_type][0]}"))\n'
                    if initial_states is not None:
                        if "Sliced" in obj_type and obj_type.replace("Sliced", "") in sliced_dict.keys():
                            program = f'{program[:-2]}, object_instance = None, parent_object = "{sliced_dict[obj_type.replace("Sliced", "")]}") # Initialize new sliced object from sliced parent\n'
                        else:
                            program = f'{program[:-2]}, object_instance = "{obj}")\n'
                    search_dict[obj].pop(0)
                    if len(search_dict[obj])==0:
                        del search_dict[obj]
                else:
                    program += f'{objects[obj]} = InteractionObject("{obj_type}")\n'
                    if initial_states is not None:
                        if "Sliced" in obj_type and obj_type.replace("Sliced", "") in sliced_dict.keys():
                            program = f'{program[:-2]}, object_instance = None, parent_object = "{sliced_dict[obj_type.replace("Sliced", "")]}") # Initialize new sliced object from sliced parent\n'
                        else:
                            program = f'{program[:-2]}, object_instance = "{obj}")\n'
                objects_tmp.append(obj)
            if sub=='ObjectDone':
                objects_tmp.remove(obj)
                if obj not in obj_count.keys():
                    obj_count[obj] = 1
                else:
                    obj_count[obj] += 1
                continue
            
            # add subgoal
            subgoal_text = subgoal_mapping_r[sub]
            if subgoal_text in ["go_to", "pickup", "slice", "toggle_on", "toggle_off", "open", "close", "clean", "fill_up", "empty", "cook", "toast"]:
                program += f'{objects[obj]}.{subgoal_text}()\n'
                if subgoal_text=="pickup":
                    held_obj = objects[obj]
                if subgoal_text=="slice":
                    sliced_dict[obj_type] = obj
            elif subgoal_text in ["change_state"]:
                program += f'{objects[obj]}.{subgoal_text}("{subgoal[1][1]}", {subgoal[1][2]})\n'
            elif subgoal_text in ["put_down"]:
                program += f'{held_obj}.{subgoal_text}()\n'
            elif subgoal_text in ["place", "pour", "pickup_and_place"]:
                program += f'{held_obj}.{subgoal_text}({objects[obj]})\n'
            else:
                st()
                assert(False) # what subgoal is this? 

            if first_subgoal:
                first_subgoal = False

        print(program)
        return program
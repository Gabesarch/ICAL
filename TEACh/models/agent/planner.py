import sys
import json

import ipdb
st = ipdb.set_trace
from arguments import args

import time
from time import sleep
from typing import List
import matplotlib.pyplot as plt
from ai2thor.controller import Controller

from task_base.object_tracker import ObjectTrack
from task_base.navigation import Navigation
from task_base.animation_util import Animation
from task_base.teach_base import TeachTask
from backend import saverloader
import pickle

import numpy as np
import os

import cv2

import csv
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from teach.dataset.dataset import Dataset
from teach.dataset.definitions import Definitions
from teach.logger import create_logger
from teach.simulators import simulator_factory
from teach.utils import get_state_changes, reduce_float_precision
import torch
import utils
import utils.geom
import logging

from teach.utils import (
    create_task_thor_from_state_diff,
    load_images,
    save_dict_as_json,
    with_retry,
    load_json
)
from teach.eval.compute_metrics import create_new_traj_metrics, evaluate_traj
from prompt.run_gpt import LLMPlanner
import copy
import traceback

from task_base.aithor_base import get_rearrangement_categories

from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s %(levelname)s %(message)s',
                filename='./subgoalcontroller.log',
                filemode='w'
            )

from IPython.core.debugger import set_trace
from PIL import Image
import wandb

from .simulate import SIMULATE
from scipy.stats import expon

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.manual_seed(args.seed)
np.random.seed(args.seed)


class PlannerController(SIMULATE):
    def __init__(self):
        '''
        subgoal planning controller
        inherited by models.SubGoalController
        '''
        super(PlannerController, self).__init__()    

    def run_llm(self, task_dict, log_tag=''):

        dialogue_history = task_dict['dialog_history_cleaned']
        command = ''
        for dialogue in dialogue_history:
            command += f'<{dialogue[0]}> {dialogue[1]}'
            if command[-1] not in ['.', '!', '?']:
                command += '. '
            else:
                command += ' '
        self.llm.command = command
        self.command = command

        

        if args.do_tree_of_thought:
            '''
            Tree of thought + simulation + LLM eval
            '''
            # cache navigation module
            self.navigation_cache = {}
            self.navigation_cache["mapper"] = copy.deepcopy(self.navigation.explorer.mapper)
            self.navigation_cache["obstructed_states"] = copy.deepcopy(self.navigation.explorer.obstructed_states)
            self.navigation_cache["step_count"] = copy.deepcopy(self.navigation.explorer.step_count)
            self.navigation_cache["image_list"] = copy.deepcopy(self.navigation.obs.image_list)
            self.navigation_cache["depth_map_list"] = copy.deepcopy(self.navigation.obs.depth_map_list)
            self.navigation_cache["return_status"] = copy.deepcopy(self.navigation.obs.return_status)
            # self.navigation_cache["return_status"] = copy.deepcopy(self.navigation.explorer.return_status)
            self.navigation_cache["prev_act_id"] = copy.deepcopy(self.navigation.explorer.prev_act_id)
            self.navigation_cache["position"] = copy.deepcopy(self.navigation.explorer.position)
            self.navigation_cache["rotation"] = copy.deepcopy(self.navigation.explorer.rotation)
            self.navigation_cache["head_tilt"] = copy.deepcopy(self.navigation.explorer.head_tilt)

            self.track_dict_initial = copy.deepcopy(self.object_tracker.objects_track_dict)
            # self.navigation_initial = copy.deepcopy(self.navigation)
            executable_code = self.do_tree_of_thought(
                self.llm.prompt_plan,
            )
            self.object_tracker.objects_track_dict = copy.deepcopy(self.track_dict_initial)
        else:
            '''
            Chain of thought
            '''
            executable_code = self.do_chain_of_thought()
        
        return executable_code

    def do_tree_of_thought(
        self,
        prompt_template
        ):
        
        prompt = prompt_template.replace('{API}', self.llm.api)

        prompt = prompt.replace('{command}', self.command)

        sorted_examples, sorted_states = self.example_retrieval(self.command)
        self.skill_functions = '\n\n'.join(sorted_examples)
        self.sorted_functions = sorted_examples

        sorted_examples = sorted_examples[:args.num_nodes*args.topk_mem_examples]

        tree_nodes = {}
        for node_idx in range(args.num_nodes):
            
            if node_idx==0:
                # take top k for first node
                sorted_examples_ = sorted_examples[:args.topk_mem_examples]
                sorted_states_ = sorted_states[:args.topk_mem_examples]
            else:
                # sample examples according to an exponential
                num_examples_considered = min(args.num_nodes*args.topk_mem_examples*2, len(sorted_examples))
                x = np.linspace(expon.ppf(0.01), expon.ppf(0.99), num_examples_considered)
                y = expon.pdf(x)
                y_norm = (y - y.min()) / (y - y.min()).sum()
                selected_example_idxs = np.random.choice(
                    np.arange(num_examples_considered), 
                    size=min(len(sorted_examples), args.topk_mem_examples),
                    replace=False, 
                    p=y_norm,
                    )
                sorted_examples_ = [sorted_examples[idx] for idx in list(selected_example_idxs)]
                sorted_states_ = [sorted_states[idx] for idx in list(selected_example_idxs)]

            '''
            State abstraction: filter the relevant objects to give to the LLM
            '''
            if args.do_state_abstraction:
                self.relevant_objects = self.get_relevant_objects(sorted_states_)
                print(f"Relevant objects are: {self.relevant_objects}")
                if not args.use_gt_attributes:
                    self.infer_relevant_attributes(self.relevant_objects)
            else:
                self.relevant_objects = None
            current_state = self.get_current_state(
                self.object_tracker.objects_track_dict,
                relevant_objects=self.relevant_objects,
                metadata=self.controller.last_event.metadata, # note this is only used if ground truth is on
                )
            self.initial_state = current_state

            prompt_ = prompt.replace('{STATE}', current_state)

            retrieved_examples = self.get_skills_to_prompt_length(
                prompt_, 
                sorted_examples_,
                max_len_percent=0.8,
                )
            prompt_ = prompt_.replace('{RETRIEVED_SKILLS}', retrieved_examples)

            if node_idx==0:
                temperature = 0
            else:
                temperature = 0

            skill = self.llm.run_gpt(prompt_, log_plan=False, temperature=temperature, seed=node_idx)

            try:
                skill_summary = skill.split("Plan:\n")[0].split('\n\n')[0]
                if skill_summary[0]==' ':
                    skill_summary = skill_summary[1:]
                skill_function = skill.split("```python\n")[1].split("```")[0]
                plan = skill.split("Plan:\n")[-1].split('\n\n')[0]
            except:
                continue # wrong output skip

            executable_code = skill_function
            if args.run_modelbased_refinement and not args.load_script_from_tmp:
                skill_function, skill_summary, plan, success, end_object_state = self.run_modelbased_refinement_check(
                    skill_function, 
                    skill_summary, 
                    plan,
                    executable_code,
                    num_refinements=2,
                )
            else:
                success = True
                end_object_state = None

            if self.log:
                text_to_output = f'TAG: {self.folder_tag}\n\nPROMPT:\n\n{prompt_}\n\n\n\nOUTPUT:\n{skill}'
                output_folder = self.output_folder 
                os.makedirs(os.path.join(output_folder, 'logging', self.folder_tag, 'tree_search'), exist_ok=True)
                with open(os.path.join(output_folder, 'logging', self.folder_tag, 'tree_search', f'tree_search_node{node_idx}.txt'), 'wb') as f:
                    f.write(text_to_output.encode('utf-8'))
            
            # instantiate tree nodes
            tree_nodes[node_idx] = {}
            tree_nodes[node_idx]["skill_function"] = skill_function
            tree_nodes[node_idx]["skill_summary"] = skill_summary
            tree_nodes[node_idx]["plan"] = plan
            tree_nodes[node_idx]["success"] = success
            tree_nodes[node_idx]["end_object_state"] = end_object_state
            tree_nodes[node_idx]["initial_state"] = current_state

        print(tree_nodes)

        option_text = ''
        for node_idx in list(tree_nodes.keys()):
            if node_idx>0:
                option_text += '\n\n'
            option_num = node_idx+1
            tree_node_text = f'Option #{option_num}:\n\nPython Program option #{option_num}:\n```python\n{tree_nodes[node_idx]["skill_function"]}\n```\n\nRun Success option #{option_num}: {tree_nodes[node_idx]["success"]}\n\nFinal Environment State option #{option_num}:\n{tree_nodes[node_idx]["end_object_state"]}'
            option_text += tree_node_text

        with open('prompt/prompt_tree_search_critic.txt') as f:
            prompt_template = f.read()

        if len(tree_nodes.keys())==0:
            # problem with tree of thought
            return ''
        elif len(tree_nodes.keys())==1:
            decision = str(list(tree_nodes.keys())[0]+1)
            selected_tree_node = list(tree_nodes.keys())[0]
        else:
            # Get decision from LLM critic
            prompt = prompt_template
            prompt = prompt.replace('{API}', self.llm.api)
            prompt = prompt.replace('{command}', f"{self.command}")
            prompt = prompt.replace('{OPTIONS}', option_text)
            decision = self.llm.run_gpt(prompt, log_plan=False)
            try:
                selected_tree_node = int(decision.split('Decision: ')[-1].split('\n')[0]) - 1
            except:
                decision_text = decision.split('Decision:')[-1]
                decision = '1'
                selected_tree_node = 0
                for node_idx in range(args.num_nodes):
                    if str(node_idx+1) in decision_text:
                        decision = str(node_idx+1)
                        selected_tree_node = node_idx

        skill_function = tree_nodes[selected_tree_node]["skill_function"]
        skill_summary = tree_nodes[selected_tree_node]["skill_summary"]
        plan = tree_nodes[selected_tree_node]["plan"]
        self.initial_state = tree_nodes[selected_tree_node]["initial_state"]

        if self.log:
            text_to_output = f'TAG: {self.folder_tag}\n\nPROMPT:\n\n{prompt}\n\n\n\nLLM CRITIC\n\n{decision}\n\nPLAN:\n{plan}\n\nSUMMARY:\n{skill_summary}\n\nPYTHON PROGRAM:\n\n{skill_function}'
            output_folder = self.output_folder #f'output/skill_logging'
            os.makedirs(output_folder, exist_ok=True)
            with open(os.path.join(output_folder, 'logging', self.folder_tag, f'original_skill.txt'), 'wb') as f:
                f.write(text_to_output.encode('utf-8'))
            with open(os.path.join(output_folder, 'logging', self.folder_tag, f'tree_search.txt'), 'wb') as f:
                f.write(str(tree_nodes).encode('utf-8'))

        return skill_function

    def do_chain_of_thought(
        self,
        ):

        # sorted_functions, sorted_states = self.skill_retrieval(command)
        sorted_functions, sorted_states = self.example_retrieval(self.command)
        if args.do_state_abstraction:
            self.relevant_objects = self.get_relevant_objects(sorted_states)
            print(f"Relevant objects are: {self.relevant_objects}")
            if not args.use_gt_attributes:
                self.infer_relevant_attributes(self.relevant_objects)
        else:
            self.relevant_objects = None
        current_state = self.get_current_state(
            self.object_tracker.objects_track_dict,
            relevant_objects=self.relevant_objects,
            metadata=self.controller.last_event.metadata,
            )
        self.initial_state = current_state
        
        prompt = self.llm.prompt_plan
        prompt = prompt.replace('{API}', self.llm.api)
        prompt = prompt.replace('{STATE}', current_state)
        prompt = prompt.replace('{command}', self.command)
        
        retrieved_examples = self.get_skills_to_prompt_length(prompt, sorted_functions)
        self.skill_functions = '\n\n'.join(sorted_functions)
        self.sorted_functions = sorted_functions
        if (not self.llm.model=="gpt-3.5-turbo-1106-ft" or args.ft_with_retrieval) and not args.zero_shot:
            prompt = prompt.replace('{RETRIEVED_SKILLS}', retrieved_examples)

        print(f"Prompt percent: {self.llm.get_prompt_proportion(prompt)}")
        print(f"Retrieved Examples:\n\n{retrieved_examples}")
        print(f"Command: {self.command}")

        cache_f = os.path.join('output', 'cache', args.split, self.llm.model)
        os.makedirs(cache_f, exist_ok=True)
        cache_f = os.path.join(cache_f, f'{self.tag}.txt')
        if args.use_saved_program_output and os.path.exists(cache_f):
            with open(cache_f) as f:
                program = f.read()
        else:
            program = self.llm.run_gpt(prompt)
            if args.use_saved_program_output:
                with open(cache_f, "a") as myfile:
                    myfile.write(program)
        try:
            executable_code = program.split("```python\n")[1].split("```")[0]
        except:
            executable_code = ""

        tbl = wandb.Table(columns=["Dialogue", "LLM output", "subgoals", "full_prompt"])
        tbl.add_data(self.command, program, executable_code, prompt)
        wandb.log({f"LLM_plan/{self.tag}": tbl})

        if self.log:
            # text_to_log = f'CoT\n------\n\nDialogue:\n\n{self.command}\n\nSummary: {program}'
            text_to_log = f'TAG: {self.folder_tag}\n\nPROMPT:\n\n{prompt}\n\n\n\n\n\nSummary: {program}'
            output_folder = self.output_folder 
            os.makedirs(os.path.join(output_folder, 'logging', self.folder_tag, 'CoT'), exist_ok=True)
            with open(os.path.join(output_folder, 'logging', self.folder_tag, 'CoT', f'CoT.txt'), 'wb') as f:
                f.write(text_to_log.encode('utf-8'))

        return executable_code


    def get_relevant_objects(self, object_states, topk=5, use_llm=True):
        if use_llm:
            try:
                with open('prompt/prompt_get_relevant_objects.txt') as f:
                    prompt_relevant_objects = f.read()
                prompt = prompt_relevant_objects
                prompt = prompt.replace('{command}', self.command)
                output = self.llm.run_gpt(prompt)
                relevant_objects = output.split('List: ')[-1].split(', ')
            except:
                print("Getting relevant objects from LLM failed!")
                relevant_objects = []
        else:
            relevant_objects = []
        topk_object_states = object_states[:topk]
        for object_state in topk_object_states:
            object_state = '{' + object_state + '}'
            object_state = object_state.replace('\n', ',')
            object_state = eval(object_state)
            relevant_objects.extend([object_state[k]["label"] for k in object_state.keys()])
        return set(relevant_objects)

    def infer_relevant_attributes(self, relevant_objects):
        start = time.time()
        for k in self.object_tracker.objects_track_dict.keys():
            obj_label = self.object_tracker.objects_track_dict[k]["label"]
            if obj_label in relevant_objects:
                current_dict = self.object_tracker.objects_track_dict[k]
                crop = current_dict["crop"]
                attribute_to_check = []
                # if current_dict["label"] in self.FILLABLE_CLASSES:
                #     attribute_to_check.append("filled")
                # if current_dict["label"] in self.OPENABLE_CLASS_LIST:
                #     attribute_to_check.append("open")
                # if current_dict["label"] in self.SLICEABLE:
                #     attribute_to_check.append("sliced")
                # if current_dict["label"] in self.TOGGLEABLE:
                #     attribute_to_check.append("toggled")
                if current_dict["label"] in self.DIRTYABLE:
                    attribute_to_check.append("dirty")
                if current_dict["label"] in self.COOKABLE:
                    attribute_to_check.append("cooked")
                attribute_predicted = {}
                for attribute in attribute_to_check:
                    attribute_predicted.update(self.object_tracker.attribute_detector.get_attribute(crop, attribute, current_dict["label"]))
                if attribute_predicted:
                    self.object_tracker.objects_track_dict[k].update(attribute_predicted)
        end = time.time()
        print(f"Time to infer relevant attributes: {end - start}")

    def get_reduced_state(self, state_string):
        object_state = '{' + state_string + '}'
        object_state = object_state.replace('\n', ',')
        object_state = eval(object_state)
        keys_iter = list(object_state.keys())
        for k in keys_iter:
            keys_iter_ = list(object_state[k].keys())
            for k_ in keys_iter_:
                if k_ not in self.object_tracker.attributes.keys():
                    del object_state[k][k_]
        object_state = str(object_state)
        object_state = object_state.replace('}, ', '}\n')
        object_state = object_state[1:]
        object_state = object_state[:-1]
        return object_state

    def example_retrieval(self, command):
        '''
        Retrieve examples of revising demonstration script
        command: (str) input command dialogue
        output is sorted by embedding distance to command
        '''

        skill_function_folder = os.path.join(args.skill_folder, 'successful_skill_functions')
        skill_summary_folder = os.path.join(args.skill_folder, 'successful_skill_summary')
        skill_plan_folder = os.path.join(args.skill_folder, 'successful_skill_plan')
        skill_state_filtered_folder = os.path.join(args.skill_folder, 'successful_skill_states_filtered')
        skill_commands_folder = os.path.join(args.skill_folder, 'successful_skill_commands')
        skill_embedding_folder = os.path.join(args.skill_folder, 'successful_skill_embedding')
        skill_function_demo_folder = os.path.join(args.skill_folder, 'successful_skill_functions_demo')
        skill_states_filtered_demo_folder = os.path.join(args.skill_folder, 'successful_skill_states_filtered_demo')
        skill_states_filtered_folder = os.path.join(args.skill_folder, 'successful_skill_states_filtered')
        skill_explanation_folder = os.path.join(args.skill_folder, 'successful_skill_explanation')
        skill_visual_embedding_folder = os.path.join(args.skill_folder, 'successful_skill_visual_embedding')
        def read(filename):
            f = open(filename, 'r', encoding='utf-8')
            output = f.read()
            f.close()
            return output
        if os.path.exists(skill_function_folder):
            success_func_files = os.listdir(skill_function_folder)
            skill_idxs = sorted([int(f.split('skill_func_')[-1].split('.txt')[0]) for f in success_func_files])
            if args.use_first_X_examples_learned is not None:
                skill_idxs = skill_idxs[:args.use_first_X_examples_learned]
        else:
            skill_idxs = []
        success_skill_functions = [read(os.path.join(skill_function_folder, f'skill_func_{success_skill_number}.txt')) for success_skill_number in skill_idxs]
        success_skill_func_names = [t.split('def ')[-1].split('(')[0] for t in success_skill_functions]
        success_skill_summaries = [read(os.path.join(skill_summary_folder, f'skill_summ_{success_skill_number}.txt')) for success_skill_number in skill_idxs]
        success_skill_plans = [read(os.path.join(skill_plan_folder, f'skill_plan_{success_skill_number}.txt')) for success_skill_number in skill_idxs]
        success_skill_commands = [read(os.path.join(skill_commands_folder, f'skill_command_{success_skill_number}.txt')) for success_skill_number in skill_idxs]
        success_skill_embeds = [np.load(os.path.join(skill_embedding_folder, f'skill_embed_{success_skill_number}.npy')) for success_skill_number in skill_idxs]
        success_skill_visual_embeds = [np.load(os.path.join(skill_visual_embedding_folder, f'skill_visual_embed_{success_skill_number}.npy')) for success_skill_number in skill_idxs if os.path.exists(os.path.join(skill_visual_embedding_folder, f'skill_visual_embed_{success_skill_number}.npy'))]
        success_skill_function_demo = [read(os.path.join(skill_function_demo_folder, f'skill_func_demo_{success_skill_number}.txt')) for success_skill_number in skill_idxs if os.path.exists(os.path.join(skill_function_demo_folder, f'skill_func_demo_{success_skill_number}.txt'))]
        success_skill_states_filtered_demo = [read(os.path.join(skill_states_filtered_demo_folder, f'skill_state_demo_{success_skill_number}.txt')) for success_skill_number in skill_idxs if os.path.exists(os.path.join(skill_states_filtered_demo_folder, f'skill_state_demo_{success_skill_number}.txt'))]
        success_skill_states_filtered = [read(os.path.join(skill_states_filtered_folder, f'skill_state_filtered_{success_skill_number}.txt')) for success_skill_number in skill_idxs]
        success_skill_explanation = [read(os.path.join(skill_explanation_folder, f'skill_explanation_{success_skill_number}.txt')) for success_skill_number in skill_idxs if os.path.exists(os.path.join(skill_explanation_folder, f'skill_explanation_{success_skill_number}.txt'))]
        if args.skill_folder2 is not None:
            # option for second folder to read from
            skill_function_folder2 = os.path.join(args.skill_folder2, 'successful_skill_functions')
            skill_plan_folder2 = os.path.join(args.skill_folder2, 'successful_skill_plan')
            skill_states_folder2 = os.path.join(args.skill_folder2, 'successful_skill_states_filtered')
            skill_commands_folder2 = os.path.join(args.skill_folder2, 'successful_skill_commands')
            skill_embedding_folder2 = os.path.join(args.skill_folder2, 'successful_skill_embedding')

            skill_function_folder2 = os.path.join(args.skill_folder2, 'successful_skill_functions')
            skill_summary_folder2 = os.path.join(args.skill_folder2, 'successful_skill_summary')
            skill_plan_folder2 = os.path.join(args.skill_folder2, 'successful_skill_plan')
            skill_state_filtered_folder2 = os.path.join(args.skill_folder2, 'successful_skill_states_filtered')
            skill_commands_folder2 = os.path.join(args.skill_folder2, 'successful_skill_commands')
            skill_embedding_folder2 = os.path.join(args.skill_folder2, 'successful_skill_embedding')
            skill_function_demo_folder2 = os.path.join(args.skill_folder2, 'successful_skill_functions_demo')
            skill_states_filtered_demo_folder2 = os.path.join(args.skill_folder2, 'successful_skill_states_filtered_demo')
            skill_explanation_folder2 = os.path.join(args.skill_folder2, 'successful_skill_explanation')
            skill_visual_embedding_folder2 = os.path.join(args.skill_folder2, 'successful_skill_visual_embedding')
            skill_states_filtered_folder2 = os.path.join(args.skill_folder2, 'successful_skill_states_filtered')

            success_func_files = os.listdir(skill_function_folder2)
            skill_idxs = sorted([int(f.split('skill_func_')[-1].split('.txt')[0]) for f in success_func_files])
            success_skill_functions += [read(os.path.join(skill_function_folder2, f'skill_func_{success_skill_number}.txt')) for success_skill_number in skill_idxs]
            success_skill_func_names += [t.split('def ')[-1].split('(')[0] for t in success_skill_functions]
            success_skill_summaries += [read(os.path.join(skill_summary_folder2, f'skill_summ_{success_skill_number}.txt')) for success_skill_number in skill_idxs]
            success_skill_plans += [read(os.path.join(skill_plan_folder2, f'skill_plan_{success_skill_number}.txt')) for success_skill_number in skill_idxs]
            success_skill_commands += [read(os.path.join(skill_commands_folder2, f'skill_command_{success_skill_number}.txt')) for success_skill_number in skill_idxs]
            success_skill_function_demo += [read(os.path.join(skill_function_demo_folder2, f'skill_func_demo_{success_skill_number}.txt')) for success_skill_number in skill_idxs]
            success_skill_states_filtered_demo += [read(os.path.join(skill_states_filtered_demo_folder2, f'skill_state_demo_{success_skill_number}.txt')) for success_skill_number in skill_idxs]
            success_skill_states_filtered += [read(os.path.join(skill_states_filtered_folder2, f'skill_state_filtered_{success_skill_number}.txt')) for success_skill_number in skill_idxs]
            success_skill_explanation += [read(os.path.join(skill_explanation_folder2, f'skill_explanation_{success_skill_number}.txt')) for success_skill_number in skill_idxs]

            success_skill_embeds += [np.load(os.path.join(skill_embedding_folder2, f'skill_embed_{success_skill_number}.npy')) for success_skill_number in skill_idxs]
            success_skill_visual_embeds += [np.load(os.path.join(skill_visual_embedding_folder2, f'skill_visual_embed_{success_skill_number}.npy')) for success_skill_number in skill_idxs if os.path.exists(os.path.join(skill_visual_embedding_folder2, f'skill_visual_embed_{success_skill_number}.npy'))]

        if args.skill_folder3 is not None:
            # option for second folder to read from
            skill_function_folder2 = os.path.join(args.skill_folder3, 'successful_skill_functions')
            skill_plan_folder2 = os.path.join(args.skill_folder3, 'successful_skill_plan')
            skill_states_folder2 = os.path.join(args.skill_folder3, 'successful_skill_states_filtered')
            skill_commands_folder2 = os.path.join(args.skill_folder3, 'successful_skill_commands')
            skill_embedding_folder2 = os.path.join(args.skill_folder3, 'successful_skill_embedding')

            skill_function_folder2 = os.path.join(args.skill_folder3, 'successful_skill_functions')
            skill_summary_folder2 = os.path.join(args.skill_folder3, 'successful_skill_summary')
            skill_plan_folder2 = os.path.join(args.skill_folder3, 'successful_skill_plan')
            skill_state_filtered_folder2 = os.path.join(args.skill_folder3, 'successful_skill_states_filtered')
            skill_commands_folder2 = os.path.join(args.skill_folder3, 'successful_skill_commands')
            skill_embedding_folder2 = os.path.join(args.skill_folder3, 'successful_skill_embedding')
            skill_function_demo_folder2 = os.path.join(args.skill_folder3, 'successful_skill_functions_demo')
            skill_states_filtered_demo_folder2 = os.path.join(args.skill_folder3, 'successful_skill_states_filtered_demo')
            skill_explanation_folder2 = os.path.join(args.skill_folder3, 'successful_skill_explanation')
            skill_visual_embedding_folder2 = os.path.join(args.skill_folder3, 'successful_skill_visual_embedding')
            skill_states_filtered_folder2 = os.path.join(args.skill_folder3, 'successful_skill_states_filtered')

            success_func_files = os.listdir(skill_function_folder2)
            skill_idxs = sorted([int(f.split('skill_func_')[-1].split('.txt')[0]) for f in success_func_files])
            success_skill_functions += [read(os.path.join(skill_function_folder2, f'skill_func_{success_skill_number}.txt')) for success_skill_number in skill_idxs]
            success_skill_func_names += [t.split('def ')[-1].split('(')[0] for t in success_skill_functions]
            success_skill_summaries += [read(os.path.join(skill_summary_folder2, f'skill_summ_{success_skill_number}.txt')) for success_skill_number in skill_idxs]
            success_skill_plans += [read(os.path.join(skill_plan_folder2, f'skill_plan_{success_skill_number}.txt')) for success_skill_number in skill_idxs]
            success_skill_commands += [read(os.path.join(skill_commands_folder2, f'skill_command_{success_skill_number}.txt')) for success_skill_number in skill_idxs]
            success_skill_function_demo += [read(os.path.join(skill_function_demo_folder2, f'skill_func_demo_{success_skill_number}.txt')) for success_skill_number in skill_idxs]
            success_skill_states_filtered_demo += [read(os.path.join(skill_states_filtered_demo_folder2, f'skill_state_demo_{success_skill_number}.txt')) for success_skill_number in skill_idxs]
            success_skill_explanation += [read(os.path.join(skill_explanation_folder2, f'skill_explanation_{success_skill_number}.txt')) for success_skill_number in skill_idxs]
            success_skill_states_filtered += [read(os.path.join(skill_states_filtered_folder2, f'skill_state_filtered_{success_skill_number}.txt')) for success_skill_number in skill_idxs]

            success_skill_embeds += [np.load(os.path.join(skill_embedding_folder2, f'skill_embed_{success_skill_number}.npy')) for success_skill_number in skill_idxs]
            success_skill_visual_embeds += [np.load(os.path.join(skill_visual_embedding_folder2, f'skill_visual_embed_{success_skill_number}.npy')) for success_skill_number in skill_idxs if os.path.exists(os.path.join(skill_visual_embedding_folder3, f'skill_visual_embed_{success_skill_number}.npy'))]

        model_before = self.llm.model
        self.llm.model = "text-embedding-ada-002"

        state_before = self.get_current_state(
            self.object_tracker.objects_track_dict, 
            )
        
        to_embed_instruction = f"Instruction: {self.command}"
        instruction_embedding = np.asarray(self.llm.get_embedding(to_embed_instruction))

        to_embed_state = f"Initial Object State:\n{state_before}"
        state_embedding = np.asarray(self.llm.get_embedding(to_embed_state))

        images = self.navigation.mapping_images if len(self.navigation.mapping_images)>0 else [self.controller.last_event.frame]
        images = [Image.fromarray(im) for im in images]
        visual_feature_embedding = torch.mean(self.clip.encode_images(images), axis=0).squeeze().cpu().numpy()

        if len(success_skill_visual_embeds)==len(success_skill_embeds):
            '''
            Weight distance by:
            1. distance to instruction embedding
            2. distance to state embedding
            3. distance to visual embedding
            '''
            distance = \
                args.instruct_lambda * cosine_similarity(success_skill_embeds, instruction_embedding[None,:])[:,0] + \
                args.state_lambda * cosine_similarity(success_skill_embeds, state_embedding[None,:])[:,0] + \
                args.visual_lambda * cosine_similarity(success_skill_visual_embeds, visual_feature_embedding[None,:])[:,0]
        else:
            distance = \
                args.instruct_lambda * cosine_similarity(success_skill_embeds, instruction_embedding[None,:])[:,0] + \
                args.state_lambda * cosine_similarity(success_skill_embeds, state_embedding[None,:])[:,0]

        # distance = np.linalg.norm(success_skill_embeds - scripts_embedding[None,:], axis=1)
        distance_argsort_topk = np.argsort(-distance) #[:topk]
        
        sorted_commands = [success_skill_commands[idx] for idx in distance_argsort_topk]
        # sorted_explanations = [success_skill_explanation[idx] for idx in distance_argsort_topk]
        if args.use_raw_demos:
            sorted_function_demo = [success_skill_function_demo[idx].replace('```python\n', '').replace('\n```', '') for idx in distance_argsort_topk]
            sorted_states_filtered_demo = [success_skill_states_filtered_demo[idx] for idx in distance_argsort_topk]
            sorted_states = sorted_states_filtered_demo
            sorted_examples = [f"For example, given these inputs:\n\nCurrent State:\n{a}\n\nDialogue:\n{b}\n\nA good output would be:\n\nSummary: (summary here)\n\nPlan:\n(plan here)\n\nPython Script:\n```python\n{c}\n```" for example_idx, (a,b,c) in enumerate(zip(sorted_states_filtered_demo, sorted_commands, sorted_function_demo))]
        else:
            sorted_plans = [success_skill_plans[idx] for idx in distance_argsort_topk]
            sorted_summaries = [success_skill_summaries[idx] for idx in distance_argsort_topk]
            sorted_functions = [success_skill_functions[idx] for idx in distance_argsort_topk]
            sorted_states_filtered = [success_skill_states_filtered[idx] for idx in distance_argsort_topk]
            sorted_states = sorted_states_filtered
            sorted_examples = [f"For example, given these inputs:\n\nCurrent State:\n{a}\n\nDialogue:\n{b}\n\nA good output would be:\n\nSummary: {c}\n\nPlan:\n{d}\n\nPython Script:\n```python\n{e}\n```" for example_idx, (a,b,c,d,e) in enumerate(zip(sorted_states_filtered, sorted_commands, sorted_summaries, sorted_plans, sorted_functions))]
        self.llm.model = model_before
        return sorted_examples, sorted_states

    def get_current_state(
        self,
        objects_track_dict,
        include_location=False,
        metadata=None,
        relevant_objects=None,
    ):
        current_state = ''
        for obj_id in objects_track_dict.keys():
            obj_label = objects_track_dict[obj_id]['label']
            if relevant_objects is not None and obj_label not in relevant_objects:
                continue
            obj_name = f"{obj_label}_{objects_track_dict[obj_id]['ID']}"
            current_dict = copy.deepcopy(objects_track_dict[obj_id])
            if not current_dict["can_use"]:
                continue
            del current_dict["scores"], current_dict["ID"], current_dict["can_use"]
            if include_location:
                if current_dict["locs"] is None:
                    current_dict["object_xyz_location"] = None
                else:
                    current_dict["object_xyz_location"] = [round(v, 1) for v in list(current_dict["locs"])]
                del current_dict["locs"]
            else:
                del current_dict["locs"]

            if "crop" in current_dict:
                del current_dict["crop"]
            
            if metadata is not None:
                obj_metadata_IDs = {}
                for obj in metadata["objects"]:
                    obj_metadata_IDs[obj["objectId"]] = obj

            if current_dict["label"] not in self.FILLABLE_CLASSES:
                del current_dict["fillLiquid"], current_dict["filled"]
            # if current_dict["label"] not in self.RECEPTACLE_OBJECTS:
            #     del current_dict["full"]
            if current_dict["label"] not in self.OPENABLE_CLASS_LIST:
                del current_dict["open"]
            if current_dict["label"] not in self.SLICEABLE:
                del current_dict["sliced"]
            # if current_dict["label"] not in self.TOASTABLE:
            #     del current_dict["toasted"]
            if current_dict["label"] not in self.TOGGLEABLE:
                del current_dict["toggled"]
            if current_dict["label"] not in self.DIRTYABLE:
                del current_dict["dirty"]
            if current_dict["label"] not in self.COOKABLE:
                del current_dict["cooked"]
            if current_dict["label"] not in self.PICKUPABLE_OBJECTS:
                del current_dict["holding"]
            # if "supported_by" in current_dict.keys() and current_dict["supported_by"] is None:
            #     del current_dict["supported_by"]
            if current_dict["label"] in self.PICKUPABLE_OBJECTS:
                if metadata is not None and "metaID" in current_dict.keys():
                    if obj_metadata_IDs[current_dict["metaID"]]["parentReceptacles"] is not None:
                        current_dict["supported_by"] = [o.split('|')[0] for o in obj_metadata_IDs[current_dict["metaID"]]["parentReceptacles"]]
            elif "supported_by" in current_dict.keys():
                del current_dict["supported_by"]

            if current_dict["label"] in self.EMPTYABLE:
                if metadata is not None and "metaID" in current_dict.keys():
                    if obj_metadata_IDs[current_dict["metaID"]]["receptacleObjectIds"] is not None:
                        current_dict["supporting"] = [o.split('|')[0] for o in obj_metadata_IDs[current_dict["metaID"]]["receptacleObjectIds"]]
            elif "supporting" in current_dict.keys():
                del current_dict["supporting"]
            
            if metadata is not None and "metaID" in current_dict.keys():
                for k in current_dict.keys():
                    if k in ["label", "metaID", "supported_by", "supporting", "emptied"]:
                        continue
                    obj = obj_metadata_IDs[current_dict["metaID"]]
                    current_dict[k] = obj[self.object_tracker.attributes_to_metadata[k]]

            if "metaID" in current_dict.keys():
                del current_dict["metaID"]
            current_state += f'"{obj_name}": {str(current_dict)}\n'.replace("'", '"')
        return current_state

    def get_skills_to_prompt_length(
        self, 
        prompt_template, 
        text_list,
        max_len_percent=0.8,
        ):

        scripts = ''
        example_count = 0
        for text in text_list:
            prompt = prompt_template
            scripts_ = scripts
            # scripts_ += f'\n\n{text}'
            scripts_ += f'\n\nExample #{example_count} (use as an in-context example):\n\n{text}'
            prompt = prompt.replace('{RETRIEVED_SKILLS}', f'{scripts_}')
            prompt_len_percent = self.llm.get_prompt_proportion(prompt)
            example_count += 1
            if prompt_len_percent>max_len_percent or example_count>args.max_examples:
                break
            scripts = scripts_
        scripts = scripts[2:] 
        return scripts

    def refine_skill(
        self,
        skill_function,
        skill_summary,
        execution_error,
        code_finished,
        ):

        with open('prompt/prompt_skill_refine_v3.py') as f:
            prompt_template = f.read()

        with open('prompt/api_primitives_nodefinitions.py') as f:
            api = f.read()

        prompt = prompt_template
        prompt = prompt.replace('{API}', self.llm.api)
        prompt = prompt.replace('{SCRIPT}', skill_function)
        prompt = prompt.replace('{SCRIPT_PREVIOUS_ROUND}', skill_function)
        # prompt = prompt.replace('{CODE_COMPLETED}', code_finished)
        prompt = prompt.replace('{STATE}', self.initial_state)
        prompt = prompt.replace('{EXECUTION_ERROR}', execution_error)
        prompt = prompt.replace('{command}', self.command)

        # self.llm.model = "gpt-4-1106-Preview"

        skill = self.llm.run_gpt(prompt, log_plan=False)
        skill_summary = skill.split("Summary: ")[1].split("\n")[0]
        if skill_summary[0]==' ':
            skill_summary = skill_summary[1:]
        skill_function = skill.split("```python\n")[1].split("```")[0]
        explanation = skill.split("Explanation: ")[-1].split('\n\n')[0]
        plan = skill.split("Plan:\n")[-1].split('\n\n')[0]

        return skill_function, skill_summary, explanation, plan

    def run_llm_replan(
        self, 
        execution_error, 
        code_remaining,
        code_finished,
        ):

        # current_state = self.get_current_state(self.object_tracker.objects_track_dict)
        current_state = self.get_current_state(
            self.object_tracker.objects_track_dict,
            relevant_objects=self.relevant_objects,
            )
        prompt = self.llm.prompt_replan
        prompt = prompt.replace('{API}', self.llm.api)
        prompt = prompt.replace('{API_CORRECTIVE}', self.llm.api_corrective)
        # prompt = prompt.replace('{RETRIEVED_SKILLS}', self.retrieved_skills)
        prompt = prompt.replace('{PYTHON_SCRIPT}', code_remaining)
        prompt = prompt.replace('{CODE_COMPLETED}', code_finished)
        prompt = prompt.replace('{STATE}', current_state)
        prompt = prompt.replace('{EXECUTION_ERROR}', execution_error)
        prompt = prompt.replace('{command}', self.command)

        retrieved_skills = self.get_skills_to_prompt_length(prompt, self.sorted_functions)
        prompt = prompt.replace('{RETRIEVED_SKILLS}', retrieved_skills)
        
        program = self.llm.run_gpt(prompt, log_plan=False)
        executable_code = program.split("```python\n")[1].split("```")[0]

        # text = '' #'-----DIALOGUE----\n'
        # for line in task_dict['dialog_history_cleaned']:
        #     text += f'<{line[0]}> {line[1]}\n'
        tbl = wandb.Table(columns=["Dialogue", "LLM output", "subgoals", "full_prompt"])
        tbl.add_data(self.command, program, executable_code, prompt)
        wandb.log({f"LLM_replan/{self.tag}_replan#{self.replan_number}": tbl})
        self.llm_log = {"Dialogue":self.command, "LLM output":program, "code":executable_code, "full_prompt":prompt}

        self.replan_number += 1

        return executable_code

    def get_search_objects(self, object_name):
        search_objects = self.llm.get_get_search_categories(object_name)
        if object_name not in self.search_dict.keys():
            self.search_dict[object_name] = []
        self.search_dict[object_name].extend(search_objects)
        self.search_dict[object_name] = list(dict.fromkeys(self.search_dict[object_name]))[:3] # top 3
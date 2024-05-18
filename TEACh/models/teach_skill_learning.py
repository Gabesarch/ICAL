import sys
import json

import ipdb
st = ipdb.set_trace
from arguments import args

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
import random

import numpy as np
import os
import glob
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
from teach.replay.episode_replay import EpisodeReplay
if args.mode in ["teach_eval_tfd", "teach_eval_custom", "teach_eval_continual", "teach_skill_learning"]:
    from teach.inference.tfd_inference_runner import TfdInferenceRunner as InferenceRunner
elif args.mode=="teach_eval_edh":
    from teach.inference.edh_inference_runner import EdhInferenceRunner as InferenceRunner
else:
    assert(False) # what mode is this? 
from teach.inference.edh_inference_runner import InferenceRunnerConfig
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

import openai

logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s %(levelname)s %(message)s',
                filename='./subgoalcontroller.log',
                filemode='w'
            )

from IPython.core.debugger import set_trace
from PIL import Image
import wandb
import re
from tqdm import tqdm

from .teach_eval_embodied_llm import SubGoalController
from .agent.api_primitives_executable import InteractionObject, CustomError
from deepdiff import DeepDiff
import random

from .expert_program_generation import ProgramGen

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

random.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)

class ContinualSubGoalController(SubGoalController, ProgramGen):
    def __init__(
        self, 
        task_type: str,
        data_dir: str, 
        output_dir: str, 
        images_dir: str, 
        edh_instance: str = None, 
        task_files: list = [],
        instance_dir: str = '',
        max_init_tries: int =5, 
        replay_timeout: int = 500, 
        num_processes: int = 1, 
        task_file_iteration: int = 1, 
        iteration=0,
        er=None,
        embeddings_continual=None,
        file_order_continual=None,
        depth_network=None,
        segmentation_network=None,
        inverse_dynamics_model=None,
        global_iteration=0,
        ) -> None:

        self.task_type = task_type
        self.task_files = task_files
        self.instance_dir = instance_dir
        self.task_file_iteration = task_file_iteration
        self.global_iteration = global_iteration
        self.folder_tag = f'{self.task_type}_{os.path.split(edh_instance)[-1].split(".tfd")[0]}'

        self.data_dir = data_dir
        self.output_dir = output_dir
        self.images_dir = images_dir
        self.edh_instance_file = edh_instance
        self.max_init_tries = max_init_tries
        self.replay_timeout = replay_timeout
        self.num_processes = num_processes
        self.iteration = iteration
        self.er = er
        self.depth_network = depth_network
        self.segmentation_network = segmentation_network
        self.environment_index = 0
        self.change_check_lag = 1
        self.total_env_steps = 0
        self.total_env_resets = 1
        self.demo_folder = args.demo_folder

        tag = "allfiles"
        self.task_name_to_descs = load_json(f'./data/task_name_to_descs_{tag}.json')

        super(ContinualSubGoalController, self).__init__(
            data_dir, 
            output_dir, 
            images_dir, 
            edh_instance, 
            max_init_tries, 
            replay_timeout, 
            num_processes, 
            iteration, 
            er, 
            depth_network=depth_network, 
            segmentation_network=segmentation_network
            )
        

        if args.get_skills_demos:
            self.run_abstraction_phase()
            assert(False) # the end
        elif args.get_expert_program:
            if self.init_success:
                self.get_expert_program()
        elif args.get_expert_program_idm:
            if inverse_dynamics_model is None:
                self.init_idm()
            if self.init_success:
                self.get_expert_program_idm()

        if args.save_memory_images:
            from nets.clip import CLIP
            self.clip = CLIP()

    def run_ICAL_learning(
        self,
        num_environments=3,
        ):
        '''
        Run online example learning. Will refine the script based on abstraction phase + human in the loop feedback.
        If successful script, will save out all relevant information.
        '''

        if num_environments!=1:
            assert(False) # memory expansion only supports a single environment at a time.

        self.search_dict = {}
        camX0_T_camXs = self.map_and_explore()

        if args.use_gt_attributes:
            # used to compare object tracker dictionary for feedback to skill refiner
            self.track_dict_before = copy.deepcopy(self.object_tracker.objects_track_dict)
            self.track_dict_initial = copy.deepcopy(self.object_tracker.objects_track_dict)

        if args.run_from_code_file:
            with open('output/executable_code_file_summary.txt') as f:
                skill_summary = f.read()
            with open('output/executable_code_file_plan.txt') as f:
                plan = f.read()
            self.get_command()
            with open('output/executable_code_file.py') as f:
                skill_function = f.read()
        else:
            skill_summary, skill_function, plan = self.run_abstraction_phase(
                # task_folder_name=self.task_type,
                return_skills=True,
            )

            if not skill_function:
                return self.teach_task.metrics, self.er

        env_successes = []
        executable_code = skill_function 
        for env_idx in range(num_environments): # For learning scripts, we do do per environment   num_environments):
            self.environment_index = env_idx

            self.state_before_execution = copy.deepcopy(self.object_tracker.objects_track_dict) #self.get_current_state(include_location=True)
            self.metadata_before_execution = copy.deepcopy(self.controller.last_event.metadata)

            if args.episode_in_try_except:
                try:
                    executable_code = skill_function 
                    skill_function, skill_summary, plan, success = self.run_human_in_the_loop_phase(
                        skill_function, 
                        skill_summary, 
                        plan,
                        num_refinements=args.num_refinements_skills,
                        )
                    env_successes.append(success)
                except Exception as e:
                    print(e)
                    print(traceback.format_exc())
                    env_successes.append(False)
                    success = False
            else:
                executable_code = skill_function
                skill_function, skill_summary, plan, success = self.run_human_in_the_loop_phase(
                    skill_function, 
                    skill_summary, 
                    plan,
                    num_refinements=args.num_refinements_skills,
                    )
                env_successes.append(success)

            if self.log and self.vis is not None:
                os.makedirs(os.path.join(self.output_folder, 'logging', self.folder_tag, 'movies'), exist_ok=True)
                self.vis.render_movie(os.path.join(self.output_folder, 'logging', self.folder_tag, 'movies'), 0, tag=f"movie_FINAL_success?{env_successes[-1]}_{self.tag}")
        
        # state before execution unfiltered
        state_before = self.get_current_state(self.state_before_execution, include_location=False, metadata=self.metadata_before_execution)
        
        # state before execution based on demo filtered
        successful_skill = success

        relevent_objects = self.get_state_change(
            self.state_before_execution,
            self.object_tracker.objects_track_dict,
            f'{executable_code}\n\n{self.initial_demo_script}',
        )
        state_before_execution_filtered_demo = {k:v for k,v in self.state_before_execution.items() if k in relevent_objects}
        state_before_demo = self.get_current_state(state_before_execution_filtered_demo, include_location=False, metadata=self.metadata_before_execution)

        # state before execution based on revised demo filtered
        relevent_objects = self.get_state_change(
            self.state_before_execution,
            self.object_tracker.objects_track_dict,
            executable_code,
        )
        state_before_execution_filtered = {k:v for k,v in self.state_before_execution.items() if k in relevent_objects}
        state_before_filtered = self.get_current_state(state_before_execution_filtered, include_location=False, metadata=self.metadata_before_execution)

        track_dict_before = copy.deepcopy(self.track_dict_initial)
        self.object_tracker.update_attributes_from_metadata()
        track_dict_after = copy.deepcopy(self.object_tracker.objects_track_dict)
        difference = DeepDiff(track_dict_before, track_dict_after)
        change_attributes_dict = {}
        if len(difference)>0:
            count = 0
            if 'values_changed' in difference.keys():
                for key_change in difference['values_changed']:
                    if 'locs' in key_change:
                        continue
                    key_object = int(key_change.split("root[")[-1].split("]")[0])
                    object_label = self.object_tracker.objects_track_dict[key_object]["label"]
                    key_changed = key_change.split("[")[-1].split("]")[0].replace("'", "")
                    if key_changed in ['scores']:
                        continue
                    change_attributes_dict[count] = {"id": key_object, "label": object_label, "attribute": key_changed, "before": difference["values_changed"][key_change]["old_value"], "after": difference["values_changed"][key_change]["new_value"]}
                    count += 1
            if 'type_changes' in difference.keys():
                for key_change in difference['type_changes']:
                    if 'locs' in key_change:
                        continue
                    key_object = int(key_change.split("root[")[-1].split("]")[0])
                    object_label = self.object_tracker.objects_track_dict[key_object]["label"]
                    key_changed = key_change.split("[")[-1].split("]")[0].replace("'", "")
                    if key_changed in ['scores']:
                        continue
                    change_attributes_dict[count] = {"id": key_object, "label": object_label, "attribute": key_changed, "before": difference["type_changes"][key_change]["old_value"], "after": difference["type_changes"][key_change]["new_value"]}
                    count += 1

        text = f"\n\nDIALOGUE:\n{self.command}\n\nSUMMARY:\n{skill_summary}\n\nOBJECT STATE:\n{state_before_demo}\n\nSUCCESSES:\n{env_successes}\n\nFUNCTION:\n{skill_function}\n\n"
        output_folder = self.output_folder 
        os.makedirs(output_folder, exist_ok=True)

        stats = {'tag':self.folder_tag, 'task_type': f'{self.task_type}', 'successful_skill': bool(successful_skill), 'total_env_steps': self.total_env_steps, 'total_env_resets': self.total_env_resets, 'global_iteration':self.global_iteration}
        if os.path.exists(os.path.join(output_folder, 'stats.json')):
            stats_dict = load_json(os.path.join(output_folder, 'stats.json'))
            stats_dict[os.path.split(self.edh_instance_file)[-1]] = stats
            save_dict_as_json(stats_dict, os.path.join(output_folder, 'stats.json'))
        else:
            stats_dict = {}
            stats_dict[os.path.split(self.edh_instance_file)[-1]] = stats
            save_dict_as_json(stats_dict, os.path.join(output_folder, 'stats.json'))

        if successful_skill:
            with open(os.path.join(output_folder, 'successful_skills.txt'), "a") as myfile:
                myfile.write(text)
        else:
            with open(os.path.join(output_folder, 'failed_skills.txt'), "a") as myfile:
                myfile.write(text)

        # visualize trajectory images
        # images_concat = np.concatenate(self.teach_task.interaction_images, axis=1)
        # image_PIL = Image.fromarray(images_concat)
        # image_PIL.save(f'output/interaction_video_{self.tag}.png')

        if successful_skill:
            '''
            Save out successful example
            '''
            print("Example episode successful! Adding it to memory...")
            skill_function_folder = os.path.join(self.output_folder, 'successful_skill_functions')
            skill_summary_folder = os.path.join(self.output_folder, 'successful_skill_summary')
            skill_plan_folder = os.path.join(self.output_folder, 'successful_skill_plan')
            skill_state_folder = os.path.join(self.output_folder, 'successful_skill_states')
            skill_state_filtered_folder = os.path.join(self.output_folder, 'successful_skill_states_filtered')
            skill_state_attribute_change_folder = os.path.join(self.output_folder, 'successful_skill_attribute_change')
            skill_stats_instance_folder = os.path.join(self.output_folder, 'successful_skill_stats_instance')
            skill_commands_folder = os.path.join(self.output_folder, 'successful_skill_commands')
            skill_embedding_folder = os.path.join(self.output_folder, 'successful_skill_embedding')
            skill_visual_embedding_folder = os.path.join(self.output_folder, 'successful_skill_visual_embedding')
            skill_function_demo_folder = os.path.join(self.output_folder, 'successful_skill_functions_demo')
            skill_states_filtered_demo_folder = os.path.join(self.output_folder, 'successful_skill_states_filtered_demo')
            skill_explanation_folder = os.path.join(self.output_folder, 'successful_skill_explanation')
            os.makedirs(skill_function_folder, exist_ok=True)
            os.makedirs(skill_summary_folder, exist_ok=True)
            os.makedirs(skill_plan_folder, exist_ok=True)
            os.makedirs(skill_state_folder, exist_ok=True)
            os.makedirs(skill_state_filtered_folder, exist_ok=True)
            os.makedirs(skill_state_attribute_change_folder, exist_ok=True)
            os.makedirs(skill_stats_instance_folder, exist_ok=True)
            os.makedirs(skill_commands_folder, exist_ok=True)
            os.makedirs(skill_embedding_folder, exist_ok=True)
            os.makedirs(skill_visual_embedding_folder, exist_ok=True)
            os.makedirs(skill_function_demo_folder, exist_ok=True)
            os.makedirs(skill_states_filtered_demo_folder, exist_ok=True)
            os.makedirs(skill_explanation_folder, exist_ok=True)
            success_skill_number = len(os.listdir(skill_function_folder))
            with open(os.path.join(skill_function_folder, f'skill_func_{success_skill_number}.txt'), "w") as myfile:
                myfile.write(skill_function)
            with open(os.path.join(skill_summary_folder, f'skill_summ_{success_skill_number}.txt'), "w") as myfile:
                myfile.write(skill_summary)
            with open(os.path.join(skill_plan_folder, f'skill_plan_{success_skill_number}.txt'), "w") as myfile:
                myfile.write(plan)
            with open(os.path.join(skill_state_folder, f'skill_state_{success_skill_number}.txt'), "w") as myfile:
                myfile.write(state_before)
            with open(os.path.join(skill_state_filtered_folder, f'skill_state_filtered_{success_skill_number}.txt'), "w") as myfile:
                myfile.write(state_before_filtered)
            with open(os.path.join(skill_commands_folder, f'skill_command_{success_skill_number}.txt'), "w") as myfile:
                myfile.write(self.command)
            # initial scripts
            with open(os.path.join(skill_function_demo_folder, f'skill_func_demo_{success_skill_number}.txt'), "w") as myfile:
                myfile.write(self.initial_demo_script)
            with open(os.path.join(skill_states_filtered_demo_folder, f'skill_state_demo_{success_skill_number}.txt'), "w") as myfile:
                myfile.write(state_before_demo)
            save_dict_as_json(change_attributes_dict, os.path.join(skill_state_attribute_change_folder, f'skill_state_attribute_change_{success_skill_number}.json'))
            save_dict_as_json(stats, os.path.join(skill_stats_instance_folder, f'stats_instance_{success_skill_number}.json'))
            to_embed = f"Instruction: {self.command}\n\nSummary: {skill_summary}\n\nInitial Object State:\n{state_before}"
            skill_embedding = self.llm.get_embedding(to_embed)
            np.save(os.path.join(skill_embedding_folder, f'skill_embed_{success_skill_number}.npy'), skill_embedding)
            
            if args.save_memory_images:
                images = self.teach_task.interaction_images
                if len(images)==0:
                    images = [self.controller.last_event.frame]
                images = [Image.fromarray(im) for im in images]
                visual_feature_embeddings = self.clip.encode_images(images)
                visual_feature_embeddings_mean = torch.mean(visual_feature_embeddings, axis=0).cpu().numpy()
                np.save(os.path.join(skill_visual_embedding_folder, f'skill_visual_embed_{success_skill_number}.npy'), visual_feature_embeddings_mean)
            
            try:
                explanation = self.get_explanation(
                    self.command,
                    state_before_demo,
                    self.initial_demo_script,
                    skill_function,
                )
            except:
                explanation = self.explanation # original explanation if this one fails
            with open(os.path.join(skill_explanation_folder, f'skill_explanation_{success_skill_number}.txt'), "w") as myfile:
                myfile.write(explanation)

        return self.teach_task.metrics, self.er

    def run_abstraction_phase(
        self,
        return_skills=False,
        do_state_abstraction=True,
        max_state_demos=3,
    ):
        '''
        Runs abstraction phase to refine a demo and add abstractions using an LLM
        task_folder_name: (str) identifies exact task folder to retrieve demo from
        return_skills: (bool) Return the skill info
        do_state_abstraction: (bool) Only include relevant state in the LLM prompt
        '''

        self.llm = LLMPlanner(
                args.gpt_embedding_dir, 
                fillable_classes=self.FILLABLE_CLASSES, 
                openable_classes=self.OPENABLE_CLASS_LIST,
                include_classes=self.include_classes,
                clean_classes=self.clean_classes,
                example_mode=args.mode,
                )

        if args.demos_from_idm:
            with open('prompt/prompt_getnextskill_retrieval_idm.txt') as f:
                prompt_template = f.read()
        else:
            with open('prompt/prompt_getnextskill_retrieval.txt') as f:
                prompt_template = f.read()

        with open('prompt/api_primitives_nodefinitions.py') as f:
            api = f.read()
        
        prompt_template = prompt_template.replace('{API}', f'{api}')

        command = self.get_command()
        print(f"Dialogue: {command}")
        prompt_template = prompt_template.replace('{command}', command)

        prompt = prompt_template
        
        scripts, demo_state = self.get_demo_script()

        if not scripts:
            return "", "", ""

        prompt = prompt.replace('{SCRIPTS}', f'{scripts}')

        current_state = self.get_current_state(self.object_tracker.objects_track_dict, metadata=self.controller.last_event.metadata)
        self.initial_state = current_state

        sorted_examples, sorted_states = self.example_retrieval(command)

        sorted_examples = sorted_examples[:args.num_nodes*args.topk_mem_examples]
        sorted_states = sorted_states[:args.num_nodes*args.topk_mem_examples]

        if do_state_abstraction:
            demo_state = self.get_reduced_state(demo_state)
            relevant_objects = self.get_relevant_objects(sorted_states[:max_state_demos] + [demo_state], use_llm=args.use_llm_state_abstraction)
            current_state = self.get_current_state(
                self.object_tracker.objects_track_dict,
                relevant_objects=relevant_objects,
                metadata=self.controller.last_event.metadata,
                )
        else:
            current_state = self.get_current_state(self.object_tracker.objects_track_dict, metadata=self.controller.last_event.metadata)
        
        self.initial_state = current_state
        self.initial_demo_script = scripts

        prompt = prompt.replace('{STATE}', current_state)

        tree_nodes = {}
        for node_idx in range(args.num_nodes):
            '''
            Optionally sample multiple LLM outputs and have LLM choose one (re-ranking)
            '''
            
            if node_idx==0:
                sorted_examples_ = sorted_examples[:args.topk_mem_examples]
            else:
                sorted_examples_ = random.sample(sorted_examples, min(len(sorted_examples), args.topk_mem_examples))
            example_count = 1
            example_text = ''
            for example in sorted_examples_:
                if example_count>1:
                    example_text += '\n\n'
                example_text += f'Example #{example_count} (use as an in-context example only):\n\n{example}'
                example_count += 1
            prompt_ = prompt.replace('{EXAMPLES}', f'{example_text}')

            if node_idx==0:
                temperature = 0
            else:
                temperature = 0

            skill = self.llm.run_gpt(prompt_, log_plan=False, temperature=temperature, seed=node_idx)

            try:
                explanation = skill.split("\n\nSummary:")[0]
                skill_summary = skill.split("Summary: ")[1].split("\n")[0]
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
                )
            else:
                success = True
                end_object_state = None

            if self.log:
                text_to_output = f'TAG: {self.folder_tag}\n\nTASK: {self.task_type}\n\nPROMPT:\n\n{prompt}\n\n\n\nOUTPUT:\n{skill}'
                output_folder = self.output_folder 
                os.makedirs(os.path.join(output_folder, 'logging', self.folder_tag, 'tree_search'), exist_ok=True)
                with open(os.path.join(output_folder, 'logging', self.folder_tag, 'tree_search', f'tree_search_node{node_idx}.txt'), 'w') as f:
                    f.write(text_to_output)
            
            # instantiate tree nodes
            tree_nodes[node_idx] = {}
            tree_nodes[node_idx]["skill_function"] = skill_function
            tree_nodes[node_idx]["skill_summary"] = skill_summary
            tree_nodes[node_idx]["skill_explanation"] = explanation
            tree_nodes[node_idx]["plan"] = plan
            tree_nodes[node_idx]["success"] = success
            tree_nodes[node_idx]["end_object_state"] = end_object_state

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
            if return_skills:
                if args.ablate_offline:
                    scripts = scripts.split("```python\n")[1].split("```")[0]
                    print(scripts)
                    return "", scripts, ""
                else:
                    return "", "", ""
        elif len(tree_nodes.keys())==1:
            decision = str(list(tree_nodes.keys())[0]+1)
            selected_tree_node = list(tree_nodes.keys())[0]
        else:
            # Get decision from LLM critic
            prompt = prompt_template
            prompt = prompt.replace('{API}', api)
            prompt = prompt.replace('{command}', f"{self.command}")
            prompt = prompt.replace('{OPTIONS}', option_text)
            decision = self.llm.run_gpt(prompt, log_plan=False)
            selected_tree_node = int(decision.split('Decision: ')[-1].split('\n')[0]) - 1

        skill_function = tree_nodes[selected_tree_node]["skill_function"]
        skill_summary = tree_nodes[selected_tree_node]["skill_summary"]
        self.explanation = tree_nodes[selected_tree_node]["skill_explanation"]
        plan = tree_nodes[selected_tree_node]["plan"]

        if self.log:
            text_to_output = f'TAG: {self.folder_tag}\n\nTASK: {self.task_type}\n\nPROMPT:\n\n{prompt}\n\n\n\nLLM CRITIC\n\n{decision}\n\nPLAN:\n{plan}\n\nSUMMARY:\n{skill_summary}\n\nPYTHON PROGRAM:\n\n{skill_function}'
            output_folder = self.output_folder #f'output/skill_logging'
            os.makedirs(output_folder, exist_ok=True)
            with open(os.path.join(output_folder, 'logging', self.folder_tag, f'original_skill.txt'), 'w') as f:
                f.write(text_to_output)
            with open(os.path.join(output_folder, 'logging', self.folder_tag, f'tree_search.txt'), 'w') as f:
                f.write(str(tree_nodes))

        if args.load_script_from_tmp:
            # load custom program
            with open('tmp.py') as f:
                skill_function = f.read()
        
        if return_skills:
            print(f"Current State:\n{current_state}")
            return skill_summary, skill_function, plan

    def run_human_in_the_loop_phase(
        self, 
        skill_function, 
        skill_summary,
        plan,
        num_refinements=3,
        return_execution=False,
        return_end_state=False,
        ):
        '''
        Human in the loop learning.
        Runs script and refines it if an execution failure occurs based on feedback.
        '''

        with open('prompt/run_script_template.txt') as f:
            python_script_template = f.read()

        text = ''
        success = False
        failed_code = None
        failed_code_prev = ''
        execution_error_prev = ''
        explanation = ''
        skill_function_previous = skill_function
        no_refine = False
        if num_refinements==0:
            num_refinements = 1
            no_refine = True
        for refinement_attempt in range(num_refinements):
            print(f"\n\n\n\n\n-----------------")
            print(f"Skill summary: {skill_summary}")
            print(f"Skill script: {skill_function}")
            
            execution_error = None
            self.refinement_attempt = refinement_attempt

            func_name = skill_function.split('def ')[-1].split('(')[0]

            executable_code = skill_function

            executable_code_ = executable_code
            code_finished = ''
            
            try:
                self.state_before_execution = copy.deepcopy(self.object_tracker.objects_track_dict) #self.get_current_state(include_location=True)
                self.metadata_before_execution = copy.deepcopy(self.controller.last_event.metadata)

                executable_code = re.sub(r'(InteractionObject\()', r'\1self, ', executable_code_)
                executable_code = re.sub(r'(AgentCorrective\()', r'\1self, ', executable_code)
                self.executable_code = executable_code
                exec(executable_code)
            except CustomError as e:
                print(traceback.format_exc())
                execution_error = str(traceback.format_exc())
                failed_line = int(re.sub("[^0-9]", "", execution_error.split('exec(executable_code)\n  File "<string>", ')[1].split(',')[0].replace('l','').replace('i', '').replace('n','').replace('e','').replace(' ',''))) - 1
                failed_code = executable_code_.split('\n')[failed_line]
                if failed_line<len(executable_code_.split('\n')):
                    code_finished += '\n'.join(executable_code_.split('\n')[:failed_line])
                    code_remaining = '\n'.join(executable_code_.split('\n')[failed_line:])
                else:
                    code_finished = executable_code_
                    code_remaining = ""
                only_error = execution_error.split(', in <module>\n')[-1]
                execution_error = f"Code failed when executing line {failed_code} in the Python Script. {only_error}"
                print(f"Execution Error: {execution_error}")
                print(f"Failed code:\n{failed_code}")
                print(f"Code already run:\n{code_finished}")
            except Exception as e:
                print(traceback.format_exc())
                execution_error = str(traceback.format_exc())
                failed_line = int(re.sub("[^0-9]", "", execution_error.split('exec(executable_code)\n  File "<string>", ')[1].split(',')[0].replace('l','').replace('i', '').replace('n','').replace('e','').replace(' ',''))) - 1
                failed_code = executable_code_.split('\n')[failed_line]
                if failed_line<len(executable_code_.split('\n')):
                    code_finished += '\n'.join(executable_code_.split('\n')[:failed_line])
                    code_remaining = '\n'.join(executable_code_.split('\n')[failed_line:])
                else:
                    code_finished = executable_code_
                    code_remaining = ""
                only_error = execution_error.split(', in <module>\n')[-1]
                execution_error = f"Code failed when executing line {failed_code} in the Python Script: {only_error}"
                print(f"Execution Error: {execution_error}")
                print(f"Failed code:\n{failed_code}")
                print(f"Code already run:\n{code_finished}")

            if not args.simulate_actions:
                self.total_env_steps += self.teach_task.steps

            if execution_error is not None and 'No revisions are necessary.' in execution_error:
                execution_error = "No error. Nothing to change in the code."

            end_object_state = copy.deepcopy(self.object_tracker.objects_track_dict)

            if return_end_state and (refinement_attempt==num_refinements-1 or execution_error is None):
                if execution_error is None:
                    success = True
                break
            
            if execution_error is not None and ("No error. Nothing to change in the code." in execution_error or (execution_error_prev==execution_error and failed_code_prev==failed_code)):
                '''
                Nothing to change or revision did nothing indicates it is hopeless..
                '''
                text += f"--------REFINEMENT ITERATION {refinement_attempt}--------\n\n\n\nPrevious function:\n\n{skill_function_previous}\n\nExecution error:\n\n{execution_error}\n\nRevised function:\n\n{skill_function}"
                print("\n\n\n\n")
                print(f"Previous function:\n\n{skill_function_previous}")
                print(f"Execution error:\n\n{execution_error}")
                print(f"Revised function:\n\n{skill_function}")
                if self.log:
                    output_folder = self.output_folder 
                    os.makedirs(output_folder, exist_ok=True)
                    with open(os.path.join(output_folder, 'logging', self.folder_tag, f'environment{self.environment_index}_refinement{self.refinement_attempt}_online_phase.txt'), 'w') as f:
                        f.write(text)
                break

            if execution_error is None or no_refine: 
                if args.use_critic_gt:
                    success, critique = self.critic_gt()
                else:
                    raise NotImplementedError # need to alter prompt for memory expansion
                    success, critique = self.critic(skill_summary, skill_function, executable_code)
                print("Skill Completed!")
                if success or no_refine:
                    code_finished = skill_function
                    break
                else:
                    execution_error = critique
                    code_finished = skill_function

            skill_function_previous = skill_function

            skill_function, skill_summary, explanation, plan = self.refine_skill(
                skill_function,
                skill_summary,
                execution_error,
                code_finished,
            )

            text += f"--------REFINEMENT ITERATION {refinement_attempt}--------\n\n\n\nPrevious function:\n\n{skill_function_previous}\n\nExecution error:\n\n{execution_error}\n\nExplanation:\n\n{explanation}\n\nRevised function:\n\n{skill_function}"
            print("\n\n\n\n")
            print(f"Previous function:\n\n{skill_function_previous}")
            print(f"Execution error:\n\n{execution_error}")
            print(f"Explanation: {explanation}")
            print(f"Revised function:\n\n{skill_function}")
            print("\n\n\n\n")

            if self.log and self.vis is not None:
                os.makedirs(os.path.join(self.output_folder, 'logging', self.folder_tag, 'movies'), exist_ok=True)
                self.vis.render_movie(os.path.join(self.output_folder, 'logging', self.folder_tag, 'movies'), 0, tag=f"movie_environment{self.environment_index}_refinement{self.refinement_attempt}_{self.tag}")

            if not refinement_attempt==num_refinements-1:
                self.reset()

            if self.log:
                output_folder = self.output_folder 
                os.makedirs(output_folder, exist_ok=True)
                with open(os.path.join(output_folder, 'logging', self.folder_tag, f'environment{self.environment_index}_refinement{self.refinement_attempt}_online_phase.txt'), 'w') as f:
                    f.write(text)
            
            execution_error_prev = execution_error
            failed_code_prev = failed_code

        if args.relabel_unsuccessful and not args.simulate_actions and not success:     
            if not code_finished:
                code_finished = skill_function
            instruction_, plan_, summary, success = self.relabel_unsuccessful(
                code_finished
            )
            if success:
                skill_function = code_finished
                skill_summary = summary
                plan = plan_
                self.command = instruction_

        if return_end_state:
            return skill_function, skill_summary, plan, success, end_object_state
        else:
            return skill_function, skill_summary, plan, success  

    def example_retrieval(self, command):
        '''
        Retrieve examples of revising demonstration script
        command: (str) input command dialogue
        output is sorted by embedding distance to command
        '''

        skill_function_folder = os.path.join(self.output_folder, 'successful_skill_functions')
        skill_summary_folder = os.path.join(self.output_folder, 'successful_skill_summary')
        skill_plan_folder = os.path.join(self.output_folder, 'successful_skill_plan')
        skill_state_filtered_folder = os.path.join(self.output_folder, 'successful_skill_states_filtered')
        skill_commands_folder = os.path.join(self.output_folder, 'successful_skill_commands')
        skill_embedding_folder = os.path.join(self.output_folder, 'successful_skill_embedding')
        skill_function_demo_folder = os.path.join(self.output_folder, 'successful_skill_functions_demo')
        skill_states_filtered_demo_folder = os.path.join(self.output_folder, 'successful_skill_states_filtered_demo')
        skill_explanation_folder = os.path.join(self.output_folder, 'successful_skill_explanation')
        def read(filename):
            f = open(filename, 'r')
            output = f.read()
            f.close()
            return output
        if os.path.exists(skill_function_folder):
            success_func_files = os.listdir(skill_function_folder)
            skill_idxs = sorted([int(f.split('skill_func_')[-1].split('.txt')[0]) for f in success_func_files])
        else:
            skill_idxs = []
        success_skill_functions = [read(os.path.join(skill_function_folder, f'skill_func_{success_skill_number}.txt')) for success_skill_number in skill_idxs]
        success_skill_func_names = [t.split('def ')[-1].split('(')[0] for t in success_skill_functions]
        success_skill_summaries = [read(os.path.join(skill_summary_folder, f'skill_summ_{success_skill_number}.txt')) for success_skill_number in skill_idxs]
        success_skill_plans = [read(os.path.join(skill_plan_folder, f'skill_plan_{success_skill_number}.txt')) for success_skill_number in skill_idxs]
        success_skill_commands = [read(os.path.join(skill_commands_folder, f'skill_command_{success_skill_number}.txt')) for success_skill_number in skill_idxs]
        success_skill_embeds = [np.load(os.path.join(skill_embedding_folder, f'skill_embed_{success_skill_number}.npy')) for success_skill_number in skill_idxs]
        success_skill_function_demo = [read(os.path.join(skill_function_demo_folder, f'skill_func_demo_{success_skill_number}.txt')) for success_skill_number in skill_idxs]
        success_skill_states_filtered_demo = [read(os.path.join(skill_states_filtered_demo_folder, f'skill_state_demo_{success_skill_number}.txt')) for success_skill_number in skill_idxs]
        success_skill_explanation = [read(os.path.join(skill_explanation_folder, f'skill_explanation_{success_skill_number}.txt')) for success_skill_number in skill_idxs]
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
            
            if (0):
                model_before = self.llm.model
                self.llm.model = "text-embedding-ada-002"
                count = 0
                for command, initial_state in zip(success_skill_commands, success_skill_states_filtered_demo):
                    to_embed = f"Instruction: {command}\n\nInitial Object State:\n{initial_state}"
                    scripts_embedding = np.asarray(self.llm.get_embedding(to_embed))
                    np.save(os.path.join(skill_embedding_folder2, f'skill_embed_{count}.npy'), scripts_embedding)
                    count += 1
                self.llm.model = model_before

            success_skill_embeds += [np.load(os.path.join(skill_embedding_folder2, f'skill_embed_{success_skill_number}.npy')) for success_skill_number in skill_idxs]

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
            
            if (0):
                model_before = self.llm.model
                self.llm.model = "text-embedding-ada-002"
                count = 0
                for command, initial_state in zip(success_skill_commands, success_skill_states_filtered_demo):
                    to_embed = f"Instruction: {command}\n\nInitial Object State:\n{initial_state}"
                    scripts_embedding = np.asarray(self.llm.get_embedding(to_embed))
                    np.save(os.path.join(skill_embedding_folder2, f'skill_embed_{count}.npy'), scripts_embedding)
                    count += 1
                self.llm.model = model_before

            success_skill_embeds += [np.load(os.path.join(skill_embedding_folder2, f'skill_embed_{success_skill_number}.npy')) for success_skill_number in skill_idxs]

        model_before = self.llm.model
        self.llm.model = "text-embedding-ada-002"

        state_before = self.get_current_state(
            self.object_tracker.objects_track_dict, 
            )
        
        to_embed = f"Instruction: {self.command}" #\n\nInitial Object State:\n{self.initial_state}"
        scripts_embedding = np.asarray(self.llm.get_embedding(to_embed))

        distance = np.linalg.norm(success_skill_embeds - scripts_embedding[None,:], axis=1)
        distance_argsort_topk = np.argsort(distance) #[:topk]
        sorted_functions = [success_skill_functions[idx] for idx in distance_argsort_topk]
        sorted_plans = [success_skill_plans[idx] for idx in distance_argsort_topk]
        sorted_commands = [success_skill_commands[idx] for idx in distance_argsort_topk]
        sorted_summaries = [success_skill_summaries[idx] for idx in distance_argsort_topk]
        sorted_function_demo = [success_skill_function_demo[idx].replace('```python\n', '').replace('\n```', '') for idx in distance_argsort_topk]
        sorted_states_filtered_demo = [success_skill_states_filtered_demo[idx] for idx in distance_argsort_topk]
        sorted_explanations = [success_skill_explanation[idx] for idx in distance_argsort_topk]
        sorted_examples = [f"For example, given these inputs:\n\nCurrent State:\n{a}\n\nDialogue:\n{b}\n\nDemonstration Script:\n```python\n{c}\n```\n\nA good output would be:\n\nExplanation: {d}\n\nSummary: {e}\n\nPlan:\n{f}\n\nPython Script:\n```python\n{g}\n```" for example_idx, (a,b,c,d,e,f,g) in enumerate(zip(sorted_states_filtered_demo, sorted_commands, sorted_function_demo, sorted_explanations, sorted_summaries, sorted_plans, sorted_functions))]
        self.llm.model = model_before
        return sorted_examples, sorted_states_filtered_demo

    def get_reduced_state(self, state_string):
        '''
        Given state as string, reduce it to keys only in object dict 
        '''
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

    def get_full_steps(self):
        '''
        User feedback
        '''
        check_task = InferenceRunner._get_check_task(self.edh_instance, self.runner_config)
        progress_check_output = check_task.check_episode_progress(self.er.simulator.get_objects(self.controller.last_event), self.er.simulator)
        steps = []
        for subgoal in progress_check_output['subgoals']:
            if subgoal['description']:
                steps.append(subgoal['description'])
            elif subgoal['steps']:
                steps.append(' '.join([s['desc'] for s in subgoal['steps']]))
        return steps

    def get_command(self):
        '''
        Get formatted dialogue
        '''
        dialogue_history = self.edh_instance['dialog_history_cleaned']
        command = ''
        for dialogue in dialogue_history:
            command += f'<{dialogue[0]}> {dialogue[1]}'
            if command[-1] not in ['.', '!', '?']:
                command += '. '
            else:
                command += ' '
        self.command = command
        return command

    def get_demo_script(
        self,
    ):
        '''
        Get expert demonstration script
        '''
        demo_file = f'{args.demo_folder}/{self.task_type}/{self.tag.split(".tfd")[0]}.txt'
        if not os.path.exists(demo_file):
            print(f"Demo script does not exist.. returning...")
            return "", ""
        with open(demo_file) as f:
            demo = f.read()
        script = demo.split('Demonstration Script:\n')[-1]
        object_state = demo.split('Initial Object State:\n')[-1].split('\n\nDemonstration Script:')[0]
        return script, object_state

    def get_scripts_to_prompt_length(
        self, 
        prompt_template, 
        task_type,
        # root=f'output/expert_programs',
        shuffle=True,
        return_dialogues=False,
        prompt_prop=0.80,
        ):

        root = args.demo_folder
        if task_type in self.task_name_to_descs.keys():
            t_folders = self.task_name_to_descs[task_type]
            task_files = []
            for f in t_folders:
                if os.path.exists(os.path.join(root, f)):
                    task_files += [os.path.join(root, f, l) for l in os.listdir(os.path.join(root, f))]
        else:
            task_files = [os.path.join(root, task_type, l) for l in os.listdir(os.path.join(root, task_type))]
        
        scripts = ''
        example_idx = 1
        if shuffle:
            random.shuffle(task_files)
        dialogues = ''
        for file in task_files:
            with open(file) as f:
                example = f.read()

            dialogue = example.split('Dialogue Instruction:\n')[-1].split('Initial Object State:')[0]
            prompt = prompt_template
            scripts_ = scripts
            scripts_ += f'\n\nScript #{example_idx}:"""\n{example}"""'
            prompt = prompt.replace('{SCRIPTS}', f'{scripts_}')
            prompt_len_percent = self.llm.get_prompt_proportion(prompt)
            if prompt_len_percent>prompt_prop:
                break
            scripts = scripts_
            dialogues += dialogue
            example_idx += 1
        if return_dialogues:
            return scripts, dialogues
        else:
            return scripts

    def get_current_state(
        self,
        objects_track_dict,
        include_location=False,
        metadata=None,
        relevant_objects=None,
        include_supporting=False,
    ):
        '''
        Gets current state in textual format
        '''
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
            
            if metadata is not None:
                obj_metadata_IDs = {}
                for obj in metadata["objects"]:
                    obj_metadata_IDs[obj["objectId"]] = obj

            if current_dict["label"] not in self.FILLABLE_CLASSES:
                del current_dict["fillLiquid"], current_dict["filled"]
            if current_dict["label"] not in self.OPENABLE_CLASS_LIST:
                del current_dict["open"]
            if current_dict["label"] not in self.SLICEABLE:
                del current_dict["sliced"]
            if current_dict["label"] not in self.TOGGLEABLE:
                del current_dict["toggled"]
            if current_dict["label"] not in self.DIRTYABLE:
                del current_dict["dirty"]
            if current_dict["label"] not in self.COOKABLE:
                del current_dict["cooked"]
            if current_dict["label"] not in self.PICKUPABLE_OBJECTS:
                del current_dict["holding"]
            if "crop" in current_dict:
                del current_dict["crop"]
            if "emptied" in current_dict.keys():
                del current_dict["emptied"]
            if current_dict["label"] in self.PICKUPABLE_OBJECTS:
                if metadata is not None and "metaID" in current_dict.keys():
                    if obj_metadata_IDs[current_dict["metaID"]]["parentReceptacles"] is not None:
                        current_dict["supported_by"] = [o.split('|')[0] for o in obj_metadata_IDs[current_dict["metaID"]]["parentReceptacles"]]
            else:
                del current_dict["supported_by"]

            if current_dict["label"] in self.EMPTYABLE:
                if metadata is not None and "metaID" in current_dict.keys():
                    if obj_metadata_IDs[current_dict["metaID"]]["receptacleObjectIds"] is not None:
                        current_dict["supporting"] = [o.split('|')[0] for o in obj_metadata_IDs[current_dict["metaID"]]["receptacleObjectIds"]]
            else:
                del current_dict["supporting"]

            if "supporting" in current_dict.keys() and not include_supporting:
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

    def get_state_change(
        self,
        state1,
        state2,
        executable_code=None,
    ):

        diff_objects = []
        for k in list(state1.keys())+list(state2.keys()):
            if k in state1.keys():
                obj_name = f"{state1[k]['label']}_{state1[k]['ID']}"
                label = state1[k]['label']
            else:
                obj_name = f"{state2[k]['label']}_{state2[k]['ID']}"
                label = state2[k]['label']
            if executable_code is not None and (obj_name in executable_code or label in executable_code):
                diff_objects.append(k)
                continue
            if k not in state1.keys() or k not in state2.keys():
                diff = True
            else:
                diff = not all([np.array_equal(state1[k][k2],state2[k][k2]) for k2 in state1[k].keys()])
            if diff:
                diff_objects.append(k)
        return set(diff_objects)

    def get_explanation(
        self,
        dialogue,
        state,
        demo,
        revised_demo,
    ):
        '''
        gets explanation for the difference between original demo and revised demo.
        dalogue: (str) dialogue input
        state: (str) object state at beginning of execution
        demo: (str) original demo script
        revised_demo (str) revised demo script

        output:
        explanation: (str) explanation of the difference
        '''

        with open('prompt/prompt_get_code_difference.txt') as f:
            prompt_template = f.read()

        prompt = prompt_template
        prompt = prompt.replace('{DIALOGUE}', dialogue)
        prompt = prompt.replace('{STATE}', state)
        prompt = prompt.replace('{DEMO}', demo)
        prompt = prompt.replace('{REVISED_DEMO}', revised_demo)

        response = self.llm.run_gpt(prompt, log_plan=False)
        explanation = response
        return explanation

    def run_modelbased_refinement_check(
        self,
        skill_function,
        skill_summary,
        plan,
        executable_code,
    ):
        '''
        Simulates actions for simulated check
        '''

        # simulate 
        args.simulate_actions = True
        _error_on_action_fail = args.error_on_action_fail
        args.error_on_action_fail = False
        self.reset()
        skill_function, skill_summary, plan, success, end_object_state = self.run_human_in_the_loop_phase(
                        skill_function, 
                        skill_summary, 
                        plan,
                        num_refinements=2,
                        return_execution=True,
                        return_end_state=True,
                        )
        args.simulate_actions = False
        args.error_on_action_fail = _error_on_action_fail

        relevent_objects = self.get_state_change(
            self.state_before_execution,
            end_object_state,
            skill_function,
        )
        
        state_after_filtered = {k:v for k,v in end_object_state.items() if k in relevent_objects}
        state_after = self.get_current_state(state_after_filtered, include_location=False)
        self.reset(add_env_reset=False)
        # self.total_env_resets -= 1 # adjust for resetting when nothing was changed

        return skill_function, skill_summary, plan, success, state_after

    def get_gt_state_mismatch(
        self,
    ):
        '''
        gets mismatch between predicted state and actual state attributes
        '''

        attributes_check = {
            "label",
            "holding",
            "sliced",
            "dirty",
            "cooked",
            "filled",
            "fillLiquid",
            "toggled",
            "open",
            }

        track_dict_before = copy.deepcopy(self.object_tracker.objects_track_dict)
        self.object_tracker.update_attributes_from_metadata()
        track_dict_after = copy.deepcopy(self.object_tracker.objects_track_dict)
        difference = DeepDiff(track_dict_before, track_dict_after)

        objs = []
        for obj in self.controller.last_event.metadata['objects']:
            if "Sliced" in obj["objectType"]:
                objs.append(obj)

        if len(difference)>0:
            text = f'An object state mismatch was detected between the actual state of the environment and the predicted state changes (from the change_state() calls) after code execution:\n'
            count = 0
            if 'values_changed' in difference.keys():
                for key_change in difference['values_changed']:
                    key_object = int(key_change.split("root[")[-1].split("]")[0])
                    object_label = self.object_tracker.objects_track_dict[key_object]["label"]
                    key_changed = key_change.split("[")[-1].split("]")[0].replace("'", "")
                    if key_changed not in attributes_check:
                        continue
                    if key_changed in ["holding", "supported_by", "supporting", "toggled", "open", "crop"]:
                        continue
                    count += 1
                    if key_changed=="dirty" and difference["values_changed"][key_change]["new_value"]==True:
                        text_to_add = f'{count}. Object {object_label}_{key_object} was changed to attribute "dirty" being False, but the object is still "dirty" = True in actuality. This means the object was not cleaned properly before the object state was changed in the code. The object needs to be placed in the sink basin, and the faucet toggled on to clean the object, and only then should the change_state("dirty", False) be called to change the state of the object. Please verify that the code follows these steps to clean the object.'
                    else:
                        text_to_add = f'{count}. Object {object_label}_{key_object} attribute state for "{key_changed}" is set to {difference["values_changed"][key_change]["old_value"]} at the end of code execution, but the actual state value of this attribute is {difference["values_changed"][key_change]["new_value"]}. Thus, this change was not properly carried out by calling change_state() in the function code to properly keep track of the object state. You should change the code to include change_state("{key_changed}", {difference["values_changed"][key_change]["new_value"]}) for object "{object_label}_{key_object}" in the proper location in the code to ensure the state tracker is consistent with the actual object states. Make sure you add the change in the appropriate part of the code.\n'
                        if object_label in ["Potato", "PotatoSliced"] and key_changed in ["cooked"] and difference["values_changed"][key_change]["new_value"]==True:
                            text_to_add += f'The {object_label} was likely cooked when put on a boiling pot or hot pan that is on a stove burner that is turned on, so this is the reason that the object was cooked. The change_state("cooked", True) should be added after 1) putting the {object_label} in a pot or pan (if its not already in a pot or pan), and 2) putting the pot or pan on a stove burner (if its not already on the stove burner), and 3) the stove burner is turned on (if its not on already).\n'
                        elif object_label in ["Potato", "PotatoSliced"] and key_changed in ["cooked"] and difference["values_changed"][key_change]["new_value"]==False:
                            # text_to_add = f"The {object_label} was likely changed to be cooked pre-maturely. {object_label}s are cooked when put on a boiling pot or hot pan that is on a stove burner that is turned on. The change_state() should be ONLY be added after 1) putting the {object_label} in a pot or pan, and 2) putting the pot or pan on a stove burner, and 3) the stove burner is turned on. Only after these three conditions are met should the change_state be called.\n"
                            text_to_add = f'The {object_label} was likely changed to be cooked pre-maturely. {object_label}s are cooked when put on a boiling pot or hot pan that is on a stove burner that is turned on. The change_state() should ONLY be added after 1) putting the {object_label} in a pot or pan (if its not already in a pot or pan), and 2) putting the pot or pan on a stove burner (if its not already on the stove burner), and 3) the stove burner is turned on (if its not on already).\n'
                    text += text_to_add
            if 'type_changes' in difference.keys():
                for key_change in difference['type_changes']:
                    key_object = int(key_change.split("root[")[-1].split("]")[0])
                    object_label = self.object_tracker.objects_track_dict[key_object]["label"]
                    key_changed = key_change.split("[")[-1].split("]")[0].replace("'", "")
                    if key_changed not in attributes_check:
                        continue
                    if key_changed in ["holding", "supported_by", "supporting", "toggled", "open", "crop"]:
                        continue
                    count += 1
                    if key_changed=="dirty" and difference["type_changes"][key_change]["new_value"]==True:
                        text_to_add = f'{count}. Object {object_label}_{key_object} was changed to attribute "dirty" being False, but the object is still "dirty" = True in actuality. This means the object was not cleaned properly before the object state was changed in the code. The object needs to be placed in the sink basin, and the faucet toggled on to clean the object, and only then should the change_state("dirty", False) be called to change the state of the object. Please verify that the code follows these steps to clean the object.'
                    else:
                        text_to_add = f'{count}. Object {object_label}_{key_object} attribute state for "{key_changed}" is set to {difference["type_changes"][key_change]["old_value"]} at the end of code execution, but the actual state value of this attribute is {difference["type_changes"][key_change]["new_value"]}. Thus, this change was not properly carried out by calling change_state() in the function code to properly keep track of the object state. You should change the code to include change_state("{key_changed}", {difference["type_changes"][key_change]["new_value"]}) for object "{object_label}_{key_object}" in the proper location in the code to ensure the state tracker is consistent with the actual object states. Make sure you add the change in the appropriate part of the code.\n'
                        # text_to_add = f'{count}. Object {object_label}_{key_object} changed attribute state in the environment for "{key_changed}" from {difference["type_changes"][key_change]["old_value"]} to {difference["type_changes"][key_change]["new_value"]}, but this change was not carried out by calling change_state() in the function code to properly keep track of the object state. You should change the code to include change_state("{key_changed}", {difference["type_changes"][key_change]["new_value"]}) for object "{object_label}_{key_object}" in the proper location in the code to ensure the state tracker is consistent with the actual object states. Make sure you add the change in the appropriate part of the code.\n'
                        if object_label in ["Potato", "PotatoSliced"] and key_changed in ["cooked"] and difference["type_changes"][key_change]["new_value"]==True:
                            text_to_add += f'The {object_label} was likely cooked when put on a boiling pot or hot pan that is on a stove burner that is turned on, so this is the reason that the object was cooked. The change_state("cooked", True) should be added after 1) putting the {object_label} in a pot or pan (if its not already in a pot or pan), and 2) putting the pot or pan on a stove burner (if its not already on the stove burner), and 3) the stove burner is turned on (if its not on already).\n'
                        elif object_label in ["Potato", "PotatoSliced"] and key_changed in ["cooked"] and difference["type_changes"][key_change]["new_value"]==False:
                            # text_to_add = f"The {object_label} was likely changed to be cooked pre-maturely. {object_label}s are cooked when put on a boiling pot or hot pan that is on a stove burner that is turned on. The change_state() should be ONLY be added after 1) putting the {object_label} in a pot or pan, and 2) putting the pot or pan on a stove burner, and 3) the stove burner is turned on. Only after these three conditions are met should the change_state be called.\n"
                            text_to_add = f'The {object_label} was likely changed to be cooked pre-maturely. {object_label}s are cooked when put on a boiling pot or hot pan that is on a stove burner that is turned on. The change_state() should ONLY be added after 1) putting the {object_label} in a pot or pan (if its not already in a pot or pan), and 2) putting the pot or pan on a stove burner (if its not already on the stove burner), and 3) the stove burner is turned on (if its not on already).\n'
                    text += text_to_add
            if count>0:
                return text
                # self.agent.err_message = text
                # self.agent.help_message = ''
                # raise CustomError(text)
            else:
                return None
        else:
            return None

    def critic_gt(
        self,
    ):
        '''
        Checks for episode success, and formulates human feedback if not success
        User feedback from metadata
        '''

        check_task = InferenceRunner._get_check_task(self.edh_instance, self.runner_config)
        progress_check_output = check_task.check_episode_progress(self.er.simulator.get_objects(self.controller.last_event), self.er.simulator)
        success = progress_check_output['success']

        critique = ""
        if success:
            critique = ""
        else:
            critique = "The code fails to do the following tasks necessary to carry out the instruction: "
            for subgoal in progress_check_output['subgoals']:
                if not subgoal['success']:
                    subgoal_description = subgoal['description']
                    subgoal_step_description = []
                    objects_mentioned = []
                    idx_to_phrase = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh"]
                    for step in subgoal['steps']:
                        if not step['success']:
                            if step['objectId'] not in objects_mentioned:
                                objects_mentioned.append(step['objectId'])
                            object_idx = objects_mentioned.index(step['objectId'])
                            if step['desc'] in ['All salad components need to be on a plate.', "All sandwich components need to be on a plate."]:
                                step['desc'] = f'The {step["objectType"]} components need to be on a clean plate. Ensure proper steps are taken to clean the plate and bring the {step["objectType"]} components on the clean plate.'
                            subgoal_step_description.append(f"{step['objectType']}: {step['desc']}")
                    if subgoal['description']:
                        subgoal_failed_text = f"The code did not fully complete the step: {subgoal['description']} "
                    else:
                        subgoal_failed_text = ""
                    if " x Cook a slice of" in subgoal_description:
                        number_objects = subgoal_description.split(" x Cook a slice of")[0]
                        subgoal_failed_text += f"Ensure the correct number of {step['objectType']} are cooked in the code ({number_objects} of them) and all necessary steps are taken to properly cook each one. Cooking occurs by either 1) placing the object in a pan or pot that is on a turned on stove, or 2) placing it in the microwave, closing the door, and toggling it on and then off. Be sure to also check to see if the number of cooked objects requested ({number_objects}) matches the number actually cooked."
                    for step_desc in subgoal_step_description:
                        subgoal_failed_text += f'{step_desc} '
                    critique += subgoal_failed_text

        if success:
            # check for predicted state mismatch
            critique_ = self.get_gt_state_mismatch()
            if critique_ is not None:
                success = False
                critique = critique_
        
        return success, critique

    def critic(
        self,
        skill_summary,
        skill_function,
        executable_code,
    ):
        '''
        Optionally query LLM for success.
        '''

        with open('prompt/prompt_evaluate.txt') as f:
            prompt_template = f.read()

        with open('prompt/api_primitives_nodefinitions.py') as f:
            api = f.read()

        relevent_objects = self.get_state_change(
            self.state_before_execution,
            self.object_tracker.objects_track_dict,
            executable_code,
        )

        state_before_execution_filtered = {k:v for k,v in self.state_before_execution.items() if k in relevent_objects}
        state_before = self.get_current_state(state_before_execution_filtered, include_location=False, metadata=self.metadata_before_execution)
        self.metadata_after_execution = copy.deepcopy(self.controller.last_event.metadata)
        state_after_filtered = {k:v for k,v in self.object_tracker.objects_track_dict.items() if k in relevent_objects}
        state_after = self.get_current_state(state_after_filtered, include_location=False, metadata=self.metadata_after_execution)

        prompt = prompt_template
        prompt = prompt.replace('{API}', api)
        prompt = prompt.replace('{RETRIEVED_SKILLS}', f"{self.retrieved_functions}")
        prompt = prompt.replace('{STATE_BEFORE}', state_before)
        prompt = prompt.replace('{STATE_AFTER}', state_after)
        prompt = prompt.replace('{SKILL_SUMMARY}', skill_summary)
        prompt = prompt.replace('{SKILL_FUNCTION}', skill_function)
        prompt = prompt.replace('{EXECUTED_SCRIPT}', executable_code)
        scripts = self.get_scripts_to_prompt_length(prompt, self.task_type)
        prompt_len_percent = self.llm.get_prompt_proportion(prompt)
        prompt = prompt.replace('{SCRIPTS}', scripts)

        response = self.llm.run_gpt(prompt, log_plan=False)

        print(response)

        success = eval(response.split('Success:')[-1].split('\n')[0])
        critique = response.split('Critique:')[-1].split('\n')[0]

        text = f"Execution code:\n\n{executable_code}\n\nState before:\n\n{state_before}\n\nState after:\n\n{state_after}\n\nCritique:\n\n{critique}\n\nRevised function:\n\n{skill_function}"
        if self.log:
            output_folder = self.output_folder
            os.makedirs(output_folder, exist_ok=True)
            with open(os.path.join(output_folder, 'logging', self.folder_tag, f'CRITIC_success?{success}_environment{self.environment_index}_refinement{self.refinement_attempt}_critic.txt'), 'w') as f:
                f.write(text)

        return success, critique

    def reset(self, add_env_reset=True):
        '''
        Reset environment
        '''
        print("Resetting environment...")
        if args.simulate_actions:
            self.object_tracker.objects_track_dict = copy.deepcopy(self.track_dict_initial)
            return

        super(ContinualSubGoalController, self).__init__(
            self.data_dir, 
            self.output_dir, 
            self.images_dir, 
            self.edh_instance_file, 
            self.max_init_tries, 
            self.replay_timeout, 
            self.num_processes, 
            self.iteration, 
            self.er, 
            depth_network=self.depth_network, 
            segmentation_network=self.segmentation_network
            )
        self.search_dict = {}
        camX0_T_camXs = self.map_and_explore()

        if args.use_gt_attributes:
            # used to compare object tracker dictionary for feedback to skill refiner
            self.track_dict_before = copy.deepcopy(self.object_tracker.objects_track_dict)

        if add_env_reset:
            self.total_env_resets += 1  

    def relabel_unsuccessful(
        self,
        code_finished,
    ):
        with open('prompt/prompt_relabel.txt') as f:
            prompt_template = f.read()

        with open('prompt/api_primitives_nodefinitions.py') as f:
            api = f.read()

        relevent_objects = self.get_state_change(
            self.state_before_execution,
            self.object_tracker.objects_track_dict,
            f'{code_finished}\n\n{self.initial_demo_script}',
        )
        state_before_execution_filtered_demo = {k:v for k,v in self.state_before_execution.items() if k in relevent_objects}
        state_before_demo = self.get_current_state(state_before_execution_filtered_demo, include_location=False, metadata=self.metadata_before_execution)

        prompt = prompt_template
        prompt = prompt.replace('{API}', api)
        prompt = prompt.replace('{SCRIPT}', code_finished)
        prompt = prompt.replace('{STATE}', state_before_demo)

        response = self.llm.run_gpt(prompt, log_plan=False)

        try:
            instruction = response.split('Instruction: ')[-1].split('\n')[0]
            plan = response.split('Plan:\n')[-1].split('\n\nSummary: ')[0]
            summary = response.split('Summary: ')[-1]
            success = True
        except:
            instruction = ''
            plan = ''
            summary = ''
            success = False
        return instruction, plan, summary, success

    def refine_skill(
        self,
        skill_function,
        skill_summary,
        execution_error,
        code_finished,
        ):

        with open('prompt/prompt_skill_refine.py') as f:
            prompt_template = f.read()

        with open('prompt/api_primitives_nodefinitions.py') as f:
            api = f.read()

        prompt = prompt_template
        prompt = prompt.replace('{API}', self.llm.api)
        prompt = prompt.replace('{SCRIPT}', self.initial_demo_script)
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

    def get_object_state_dict(self, objects, object_names):
        attribute_dict = {k:{} for k in list(object_names)}
        attribute_mapping = {
            "toggleable":"isToggled",
            "breakable":"isBroken", 
            "canFillWithLiquid":"isFilledWithLiquid", 
            "dirtyable":"isDirty", 
            "canBeUsedUp":"isUsedUp", 
            "cookable":"isCooked", 
            "sliceable":"isSliced", 
            "openable":"isOpen", 
            "pickupable":"isPickedUp"
            }
        for object_ in objects: 
            if object_["name"] in object_names:
                print(object_)
                for k in object_.keys():
                    if k in attribute_mapping.keys():
                        if object_[k]:
                            k_ = attribute_mapping[k]
                            attribute_dict[object_["name"]][k_] = object_[k_]
                            if k_=="isFilledWithLiquid":
                                attribute_dict[object_["name"]]['fillLiquid'] = object_['fillLiquid']
                            elif k=="pickupable":
                                if object_["isPickedUp"]:
                                    attribute_dict[object_["name"]]['parentReceptacles'] = "in_hand"
                                else:
                                    attribute_dict[object_["name"]]['parentReceptacles'] = object_['parentReceptacles']
        
        attribute_dict_new = {}
        for k in attribute_dict.keys():
            attribute_dict_new[k] = {} 
            for k_ in attribute_dict[k].keys():
                if k_=='parentReceptacles':
                    k_new = "location_in_room"
                else:
                    k_new = re.sub( '(?<!^)(?=[A-Z])', '_', k_ ).lower()
                attribute_dict_new[k][k_new] = attribute_dict[k][k_]
        attribute_dict = attribute_dict_new
        return attribute_dict

def run_teach():
    save_metrics = True
    split_ = args.split
    data_dir = args.teach_data_dir 
    output_dir = "./plots/subgoal_output"
    images_dir = "./plots/subgoal_output"
    instance_dir = os.path.join(data_dir, f"tfd_instances/{split_}")
    output_folder = f'output/{args.mode}_{args.set_name}'

    if args.online_skill_learning:
        tag = "skills"
        task_demo_files = glob.glob(f'{args.demo_folder}/*/*.txt')
        files = [os.path.join(instance_dir, os.path.split(f)[-1].split('.txt')[0]+'.tfd.json') for f in task_demo_files]
    else:
        files = os.listdir(instance_dir) # sample every other
        tag = "allfiles"
    
    if not os.path.exists(f'./data/sorted_task_files_{split_}_{tag}.json'):
        sorted_task_files_objects = {}
        sorted_task_files = {}
        task_name_to_descs = {}
        runner_config = InferenceRunnerConfig(
            data_dir=data_dir,
            split=split_,
            output_dir=output_dir,
            images_dir=images_dir,
            model_class=None,
            model_args=None,
            num_processes=1,
            max_init_tries=5,
            replay_timeout=500,
            max_api_fails=args.max_api_fails,
            max_traj_steps=args.max_traj_steps,
        )
        for file in tqdm(files):
            task_instance = os.path.join(instance_dir, file)
            instance = load_json(task_instance)
            game_file = InferenceRunner.get_game_file(instance, runner_config)
            game = load_json(game_file)
            task_description = game["tasks"][0]['task_name'].replace('.', '').replace(' ', '_')
            task_description_object = game["tasks"][0]['desc'].replace('.', '').replace(' ', '_')
            if task_description not in sorted_task_files.keys():
                sorted_task_files[task_description] = []
                task_name_to_descs[task_description] = set()
            sorted_task_files[task_description].append(file)
            task_name_to_descs[task_description].add(task_description_object)
            if task_description_object not in sorted_task_files_objects.keys():
                sorted_task_files_objects[task_description_object] = []
            sorted_task_files_objects[task_description_object].append(file)
        save_dict_as_json(sorted_task_files, f'./data/sorted_task_files_{split_}_{tag}.json')
        save_dict_as_json(sorted_task_files_objects, f'./data/sorted_task_files_{split_}_{tag}_taskparams.json')
        task_name_to_descs = {k:list(v) for k,v in task_name_to_descs.items()}
        save_dict_as_json(task_name_to_descs, f'./data/task_name_to_descs_{split_}_{tag}.json')
    else:
        if args.get_expert_program or args.get_expert_program_idm:
            sorted_task_files = load_json(f'./data/sorted_task_files_{split_}_{tag}_taskparams.json')
        else:
            if args.use_instance_programs:
                sorted_task_files = load_json(f'./data/sorted_task_files_{split_}_{tag}_taskparams.json')
            elif args.use_task_to_instance_programs:
                # import random
                def shuffled_iter(lst):
                    shuffled = lst.copy()  # Copy the list to not modify the original
                    random.shuffle(shuffled)
                    for item in shuffled:
                        yield item
                sorted_task_files = load_json(f'./data/sorted_task_files_{split_}_{tag}.json')
                sorted_task_files_ = load_json(f'./data/sorted_task_files_{split_}_{tag}_taskparams.json')
                task_to_instance_dict = load_json(f'./data/task_name_to_descs_{split_}_{tag}.json')
                task_to_instance_dict_iter = {k:iter(shuffled_iter(v)) for k,v in task_to_instance_dict.items()}
            else:
                sorted_task_files = load_json(f'./data/sorted_task_files_{split_}_{tag}.json')

    # initialize wandb
    if args.set_name=="test00" or args.online_skill_learning:
        wandb.init(mode="disabled")
    else:
        wandb.init(project="embodied-llm-memory-learning", name=args.set_name, group=args.group, config=args, dir=args.wandb_directory)

    metrics = {}
    metrics_file = os.path.join(args.metrics_dir, f'{args.mode}_metrics_{split_}.txt')
    if os.path.exists(metrics_file) and args.skip_if_exists:
        metrics = load_json(metrics_file)

    if args.online_skill_learning:
        metrics_file = os.path.join(args.metrics_dir, f'{args.mode}_metrics_{split_}.txt')
        if os.path.exists(metrics_file) and args.skip_if_exists:
            metrics = load_json(metrics_file)

    task_type = None
    task_type = args.task_type
    # ['Put_All_X_On_Y', 'Salad', 'Clean_All_X', 'Plate_Of_Toast', 'N_Slices_Of_X_In_Y', 'Coffee', 'Breakfast', 'Water_Plant', 'Put_All_X_In_One_Y', 'Sandwich', 'Boil_X', 'N_Cooked_Slices_Of_X_In_Y']
    # task_type = 'Coffee'

    # ['Put_all_Newspaper_on_any_Sofa', 'Put_all_Fork_in_any_Sink', 'Make_a_salad', 'Clean_all_the_Cloths', 'Make_a_plate_of_toast', 'Serve_1_slice(s)_of_Lettuce_in_a_Bowl', 'Prepare_coffee_in_a_clean_mug', 'Prepare_breakfast', 'Serve_1_slice(s)_of_Tomato_on_a_Plate', 'Serve_1_slice(s)_of_Tomato_in_a_Bowl', 'Water_the_plant', 'Put_all_Mug_in_any_Sink', 'Put_all_SmallHandheldObjects_on_one_CoffeeTable', 'Make_a_sandwich', 'Boil_Potato', 'Put_all_Cup_in_one_Cabinet', 'Clean_all_the_Pots', 'Cook_2_slice(s)_of_Potato_and_serve_in_a_Bowl', 'Put_all_Watch_on_one_Tables', 'Serve_2_slice(s)_of_Tomato_on_a_Plate', 'Serve_1_slice(s)_of_Lettuce_on_a_Plate', 'Put_all_RemoteControl_on_one_Sofa', 'Clean_all_the_Bowls', 'Put_all_Watch_on_one_ArmChair', 'Clean_all_the_Plates', 'Put_all_SaltShaker_on_any_DiningTable', 'Put_all_Silverware_in_any_Sink', 'Put_all_Newspaper_on_one_Sofa', 'Clean_all_the_Pans', 'Cook_1_slice(s)_of_Potato_and_serve_in_a_Bowl', 'Clean_all_the_Cups', 'Put_all_PepperShaker_in_any_Cabinet', 'Put_all_Mug_in_one_Cabinet', 'Clean_all_the_Drinkwares', 'Put_all_TissueBox_on_one_Tables', 'Put_all_Kettle_on_any_DiningTable', 'Put_all_Lettuce_in_any_Fridge', 'Put_all_Apple_in_one_Cabinet', 'Put_all_Watch_on_any_SideTable', 'Put_all_Dishware_on_any_DiningTable', 'Put_all_SmallHandheldObjects_on_one_Tables', 'Put_all_Pen_on_any_Bed', 'Put_all_RemoteControl_on_one_Chairs', 'Clean_all_the_Mugs', 'Put_all_Lettuce_on_any_DiningTable', 'Put_all_Candle_on_any_CoffeeTable', 'Put_all_Spoon_in_any_Sink', 'Cook_1_slice(s)_of_Potato_and_serve_on_a_Plate', 'Put_all_TissueBox_on_any_Tables', 'Put_all_TissueBox_on_one_CoffeeTable', 'Put_all_Bread_in_any_Cabinet', 'Put_all_Watch_on_one_Sofa', 'Cook_3_slice(s)_of_Potato_and_serve_in_a_Bowl', 'Cook_5_slice(s)_of_Potato_and_serve_in_a_Bowl', 'Cook_3_slice(s)_of_Potato_and_serve_on_a_Plate', 'Put_all_AlarmClock_on_any_Bed', 'Put_all_Newspaper_on_any_ArmChair', 'Put_all_Pillow_on_any_Chairs', 'Put_all_RemoteControl_on_any_Chairs', 'Put_all_Cloth_in_any_Bathtub', 'Put_all_RemoteControl_on_one_Tables', 'Cook_2_slice(s)_of_Potato_and_serve_on_a_Plate', 'Put_all_SmallHandheldObjects_on_any_Chairs', 'Put_all_RemoteControl_on_one_ArmChair', 'Cook_4_slice(s)_of_Potato_and_serve_on_a_Plate', 'Put_all_Newspaper_on_one_SideTable', 'Put_all_Condiments_in_any_Cabinet', 'Put_all_RemoteControl_on_one_Dresser', 'Put_all_Book_on_any_Desk', 'Put_all_RemoteControl_on_any_Tables', 'Serve_2_slice(s)_of_Lettuce_on_a_Plate', 'Clean_all_the_Dishwares', 'Put_all_RemoteControl_on_any_Sofa', 'Put_all_RemoteControl_on_one_Ottoman', 'Put_all_Egg_in_one_Cabinet', 'Put_all_Spatula_in_any_Cabinet', 'Clean_all_the_Tablewares', 'Put_all_Bread_on_any_DiningTable', 'Cook_5_slice(s)_of_Potato_and_serve_on_a_Plate', 'Put_all_CreditCard_on_any_Bed', 'Put_all_Drinkware_on_any_DiningTable', 'Put_all_SmallHandheldObjects_on_one_ArmChair', 'Put_all_Newspaper_on_one_Dresser', 'Put_all_Newspaper_on_one_Ottoman', 'Put_all_RemoteControl_on_one_SideTable', 'Put_all_Newspaper_on_one_Chairs', 'Put_all_Watch_on_one_Furniture', 'Put_all_Egg_on_any_DiningTable', 'Put_all_CreditCard_on_any_Desk', 'Put_all_Newspaper_in_one_Box', 'Put_all_Newspaper_on_one_Furniture', 'Put_all_SaltShaker_in_one_Cabinet', 'Put_all_Bowl_on_any_DiningTable', 'Serve_3_slice(s)_of_Tomato_on_a_Plate', 'Serve_3_slice(s)_of_Lettuce_on_a_Plate', 'Put_all_RemoteControl_on_any_Dresser', 'Put_all_Watch_in_one_Box', 'Put_all_Condiments_in_one_Cabinet', 'Put_all_RemoteControl_on_one_CoffeeTable', 'Put_all_Candle_on_one_CoffeeTable', 'Put_all_RemoteControl_on_one_Furniture', 'Put_all_Fork_on_any_DiningTable', 'Put_all_SmallHandheldObjects_on_one_Sofa', 'Put_all_SmallHandheldObjects_on_one_Furniture', 'Put_all_Watch_on_one_Chairs', 'Put_all_Watch_on_any_Tables', 'Put_all_DishSponge_in_any_Sink', 'Put_all_Potato_in_any_Cabinet', 'Put_all_Mug_on_any_DiningTable', 'Put_all_Apple_in_any_Cabinet', 'Put_all_Bottle_in_one_Cabinet', 'Put_all_Watch_on_any_CoffeeTable', 'Put_all_RemoteControl_in_one_Box', 'Put_all_Pillow_on_any_Sofa', 'Put_all_RemoteControl_on_any_ArmChair', 'Put_all_Plate_on_any_DiningTable', 'Put_all_RemoteControl_on_any_SideTable', 'Put_all_Tomato_in_any_Fridge', 'Put_all_TissueBox_on_one_SideTable', 'Put_all_Newspaper_on_one_Tables', 'Put_all_Newspaper_on_any_SideTable', 'Put_all_Bowl_in_any_Sink', 'Put_all_Newspaper_on_one_ArmChair', 'Put_all_Ladle_in_one_Cabinet', 'Clean_all_the_Cookwares', 'Put_all_Spoon_in_one_Drawer', 'Put_all_Apple_in_any_Fridge', 'Put_all_Cup_in_any_Cabinet', 'Put_all_Ladle_in_any_Sink', 'Put_all_Tomato_in_one_Cabinet', 'Put_all_Spatula_in_one_Drawer', 'Put_all_SaltShaker_in_any_Cabinet', 'Put_all_SoapBar_on_any_CounterTop', 'Put_all_Fruit_in_any_Fridge', 'Put_all_SportsEquipment_on_any_Bed', 'Put_all_Spatula_in_any_Sink', 'Put_all_Candle_on_one_Tables', 'Put_all_Mug_in_any_Cabinet', 'Put_all_PepperShaker_in_one_Cabinet', 'Put_all_Laptop_on_any_Bed', 'Put_all_Ladle_in_one_Drawer', 'Put_all_Newspaper_on_one_CoffeeTable', 'Put_all_Plate_in_any_Cabinet', 'Serve_4_slice(s)_of_Tomato_on_a_Plate', 'Put_all_ScrubBrush_on_any_CounterTop', 'Put_all_Cup_in_any_Sink', 'Put_all_Cup_on_any_DiningTable', 'Put_all_Watch_on_one_Ottoman', 'Put_all_Ladle_in_any_Cabinet', 'Put_all_SmallHandheldObjects_on_one_Chairs', 'Put_all_Watch_on_one_SideTable', 'Put_all_Tomato_in_any_Cabinet', 'Put_all_CreditCard_on_any_Furniture', 'Put_all_RemoteControl_on_any_Furniture', 'Put_all_Pillow_on_any_ArmChair', 'Cook_4_slice(s)_of_Potato_and_serve_in_a_Bowl', 'Serve_5_slice(s)_of_Tomato_in_a_Bowl', 'Put_all_RemoteControl_on_any_TVStand', 'Serve_3_slice(s)_of_Lettuce_in_a_Bowl', 'Put_all_Fork_in_any_Drawer', 'Serve_2_slice(s)_of_Tomato_in_a_Bowl', 'Put_all_Lettuce_in_any_Cabinet', 'Put_all_AlarmClock_on_any_Furniture', 'Put_all_Fork_in_one_Drawer', 'Put_all_Plate_in_one_Cabinet', 'Put_all_Watch_on_any_Sofa', 'Put_all_Fruit_on_any_DiningTable', 'Put_all_Pen_on_any_Desk', 'Put_all_Spatula_in_one_Cabinet', 'Put_all_Candle_on_any_Tables', 'Put_all_Spoon_in_any_Drawer', 'Serve_3_slice(s)_of_Tomato_in_a_Bowl', 'Put_all_Book_on_any_Furniture', 'Put_all_Bowl_in_any_Cabinet', 'Put_all_Book_on_any_Bed', 'Put_all_Apple_on_any_DiningTable', 'Put_all_RemoteControl_in_any_Box', 'Put_all_Newspaper_on_any_CoffeeTable']
    # task_type = 'Put_all_Fork_in_any_Sink'
    # task_type = 'Salad'
    # task_type = 'Plate_Of_Toast'
    # task_type = 'Boil_X'
    # task_type = 'N_Cooked_Slices_Of_X_In_Y'
    # task_type = 'Coffee'
    # task_type = 'Prepare_coffee_in_a_clean_mug'
    # task_type = 'Breakfast'
    # task_type = ['Salad', 'Breakfast', 'Sandwich', 'N_Cooked_Slices_Of_X_In_Y', 'Plate_Of_Toast']
    if task_type is not None:
        if type(task_type)==list:
            sorted_task_files = {tt_:sorted_task_files[tt_] for tt_ in task_type}
        else:
            sorted_task_files = {task_type:sorted_task_files[task_type]}

    if args.episode_file is not None:
        # for debugging
        file_to_task_type_dict = {}
        for k in sorted_task_files.keys():
            task_files = sorted_task_files[k]
            for t in task_files:
                file_to_task_type_dict[os.path.split(t)[-1]] = k
        task_type = file_to_task_type_dict[args.episode_file]
        sorted_task_files = {task_type:sorted_task_files[task_type]}
    
    er = None
    depth_estimation_network = None
    segmentation_network = None
    task_idx = -1
    
    if args.online_skill_learning:
        num_iterations = args.num_online_learning_iterations
    else:
        num_iterations = 1

    global_iteration = -1
    if args.skip_if_exists:
        if args.online_skill_learning:
            # output_folder = f'output/{args.mode}_{args.set_name}' #f'output/skill_logging_{args.set_name}'
            stats_path = os.path.join(output_folder, 'stats.json')
            if os.path.exists(stats_path):
                stats = load_json(stats_path)
                global_iters = [stats[k]['global_iteration'] for k in stats.keys()]
                global_iteration = max(global_iters)

    # if (1):
    #     if split_ in ["valid_seen", "valid_unseen"]:
    #         num_iterations = 0

    #     files_train = []
    #     for task_type in list(sorted_task_files.keys()):
    #         files_train.extend(sorted_task_files[task_type][num_iterations:])
    #         print(task_type, len(sorted_task_files[task_type][num_iterations:]))
        
    #     st()
        
    #     files_train_ = []
    #     for f_t in files_train:
    #         files_train_.append(os.path.split(f_t)[-1].split('.tfd.json')[0])

    #     with open(f'data/teach_idm_{split_}.p', 'wb') as f:
    #         pickle.dump(files_train_, f)

    #     st()

    if args.get_expert_program_idm or args.get_expert_program or args.online_skill_learning:
        with open(f'./data/teach_idm_{split_}.p', 'rb') as f:
            file_list = pickle.load(f)
        for task_type in list(sorted_task_files.keys()):
            sorted_task_files[task_type] = [f for f in sorted_task_files[task_type] if f.replace('.tfd.json', '') not in file_list]

    if args.online_skill_learning:
        with open(f'./data/teach_idm_{split_}.p', 'rb') as f:
            file_list = pickle.load(f)
        # print(len(sum(list(sorted_task_files.values()), [])))
        for task_type in list(sorted_task_files.keys()):
            sorted_task_files[task_type] = [f for f in sorted_task_files[task_type] if os.path.split(f)[-1].replace('.tfd.json', '') not in file_list]
        # print(len(sum(list(sorted_task_files.values()), [])))

    for task_file_iteration in range(num_iterations):
        for task_type_ in list(sorted_task_files.keys()):
            
            task_type = task_type_
            if args.use_task_to_instance_programs:
                try:
                    task_type = next(task_to_instance_dict_iter[task_type])
                    files_ = sorted_task_files_[task_type]
                except StopIteration:
                    task_to_instance_dict_iter[task_type] = iter(task_to_instance_dict[task_type])
                    task_type = next(task_to_instance_dict_iter[task_type])
                    files_ = sorted_task_files_[task_type]
            else:
                files_ = sorted_task_files[task_type]

            task_idx += 1

            print(f"Task type={task_type}")

            if args.shuffle:
                random.shuffle(files_)

            if args.sample_every_other:
                files_ = files_[::2]

            if args.episode_file is not None:
                # for debugging
                files_ = [f for f in files_ if args.episode_file in f]
            
            iter_ = 0
            successful_episodes = 0
            for file in files_:

                print("Running ", file)
                print(f"File iteration {iter_+1}/{len(files_)}")
                print(f"Task file iteration: {task_file_iteration}")
                
                if args.skip_if_exists:
                    if args.online_skill_learning:
                        # output_folder = f'output/{args.mode}_{args.set_name}' #f'output/skill_logging_{args.set_name}'
                        stats_path = os.path.join(output_folder, 'stats.json')
                        if os.path.exists(stats_path):
                            stats = load_json(stats_path)
                            if len(stats.keys())>=args.max_memory_episodes:
                                assert(False) # max episodes reached.. ending memory learning
                            if os.path.split(file)[-1] in stats.keys():
                                print(f"File already in metrics... skipping...")
                                # sorted_task_files_[task_type].remove(file)
                                # if len(sorted_task_files_[task_type])==0:
                                #     task_to_instance_dict[task_type_].remove(task_type)
                                iter_ += 1
                                continue
                    elif args.get_expert_program or args.get_expert_program_idm:

                        # output_folder = f'output/{args.mode}_{args.set_name}' #f'output/skill_logging_{args.set_name}'
                        stats_path = os.path.join(args.demo_folder, task_type, file.replace('.tfd.json', '.txt'))
                        if os.path.exists(stats_path):
                            successful_episodes += 1
                            print(f"File {stats_path} already exists... skipping...")
                            iter_ += 1
                            continue 
                    else:
                        if file in metrics.keys():
                            print(f"File already in metrics... skipping...")
                            iter_ += 1
                            continue

                global_iteration += 1

                task_instance = os.path.join(instance_dir, file)
                subgoalcontroller = ContinualSubGoalController(
                        task_type,
                        data_dir, 
                        output_dir, 
                        images_dir, 
                        task_instance, 
                        files_,
                        instance_dir,
                        iteration=iter_, 
                        er=er, 
                        depth_network=depth_estimation_network, 
                        segmentation_network=segmentation_network,
                        task_file_iteration=task_file_iteration,
                        global_iteration=global_iteration,
                        )
                if subgoalcontroller.init_success and not (args.get_expert_program or args.get_expert_program_idm):
                    if not (args.get_skills_demos or args.online_skill_learning):
                        metrics_instance, er = subgoalcontroller.run()
                    if args.online_skill_learning:
                        if args.episode_in_try_except:
                            try:
                                metrics_instance, er = subgoalcontroller.run_ICAL_learning(num_environments=args.num_environments_skills)
                            except KeyboardInterrupt:
                                sys.exit(0)
                            except Exception as e:
                                print(e)
                                print(traceback.format_exc())
                        else:
                            metrics_instance, er = subgoalcontroller.run_ICAL_learning(num_environments=args.num_environments_skills)
                        # # remove file from list
                        # sorted_task_files_[task_type].remove(file)
                        # if len(sorted_task_files_[task_type])==0:
                        #     task_to_instance_dict[task_type_].remove(task_type)
                        break
                    if segmentation_network is None:
                        segmentation_network = subgoalcontroller.object_tracker.ddetr
                    if depth_estimation_network is None:
                        depth_estimation_network = subgoalcontroller.navigation.depth_estimator
                else:
                    metrics_instance, er = subgoalcontroller.teach_task.metrics, subgoalcontroller.er

                if args.get_expert_program or args.get_expert_program_idm:
                    if args.max_episodes is not None:
                        if successful_episodes>=args.max_episodes:
                            break
                
                if subgoalcontroller.init_success:
                    successful_episodes += 1

                iter_ += 1

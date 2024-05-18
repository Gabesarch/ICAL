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

from models.agent.api_primitives_executable import InteractionObject, CustomError
from deepdiff import DeepDiff
import random

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

random.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)

class SIMULATE:
    def __init__(self, **kwargs):
        '''
        Functions for simulating, interacting, and refinement
        '''
        pass
    
    def run_modelbased_refinement_check(
        self,
        skill_function,
        skill_summary,
        plan,
        executable_code,
        num_refinements=2,
    ):
        '''
        Simulates actions for simulated check
        '''

        # simulate 
        args.simulate_actions = True
        _error_on_action_fail = args.error_on_action_fail
        args.error_on_action_fail = False
        self.reset()
        skill_function, skill_summary, plan, success, end_object_state = self.run_skill_refinement(
                        skill_function, 
                        skill_summary, 
                        plan,
                        num_refinements=num_refinements,
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
        self.reset()
        # self.total_env_resets -= 1 # adjust for resetting when nothing was changed

        return skill_function, skill_summary, plan, success, state_after

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
    
    def run_skill_refinement(
        self, 
        skill_function, 
        skill_summary,
        plan,
        num_refinements=3,
        return_execution=False,
        return_end_state=False,
        ):
        '''
        Runs script and refines it if an execution failure occurs
        '''

        with open('prompt/run_script_template.txt') as f:
            python_script_template = f.read()

        text = ''
        success = False
        failed_code = None
        failed_code_prev = ''
        execution_error_prev = ''
        skill_function_previous = skill_function
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
                code_finished += '\n'.join(executable_code_.split('\n')[:failed_line])
                code_remaining = '\n'.join(executable_code_.split('\n')[failed_line:])
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
                code_finished += '\n'.join(executable_code_.split('\n')[:failed_line])
                code_remaining = '\n'.join(executable_code_.split('\n')[failed_line:])
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

            if execution_error is None: 
                if execution_error is None:
                    if args.use_critic_gt:
                        success, critique = self.critic_gt()
                    else:
                        raise NotImplementedError # need to alter prompt for memory expansion
                        success, critique = self.critic(skill_summary, skill_function, executable_code)
                    print("Skill Completed!")
                    if success:
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

        if return_end_state:
            return skill_function, skill_summary, plan, success, end_object_state
        else:
            return skill_function, skill_summary, plan, success 

    def reset(self):
        print("Resetting object tracker...")
        self.object_tracker.objects_track_dict = copy.deepcopy(self.track_dict_initial)
        # self.navigation = copy.deepcopy(self.navigation_initial)
        navigation_cache = copy.deepcopy(self.navigation_cache)
        self.navigation.explorer.mapper = navigation_cache["mapper"]
        self.navigation.explorer.obstructed_states = navigation_cache["obstructed_states"] 
        self.navigation.explorer.step_count = navigation_cache["step_count"]
        self.navigation.obs.image_list = navigation_cache["image_list"]
        self.navigation.obs.depth_map_list = navigation_cache["depth_map_list"]
        self.navigation.obs.return_status = navigation_cache["return_status"]
        # self.navigation.explorer.return_status = navigation_cache["return_status"]
        self.navigation.explorer.prev_act_id = navigation_cache["prev_act_id"]
        self.navigation.explorer.position = navigation_cache["position"]
        self.navigation.explorer.rotation = navigation_cache["rotation"]
        self.navigation.explorer.head_tilt = navigation_cache["head_tilt"]
        return
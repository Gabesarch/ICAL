# real_time_agent_runner.py

import os
import base64
from io import BytesIO
from typing import Optional, Tuple, List, Union, Dict

from PIL import Image
import uuid
import json
import numpy as np

# Browser environment / agent imports
from browser_env import ScriptBrowserEnv
from browser_env.actions import ActionTypes, Action
from agent import construct_agent, PromptAgent
from browser_env.actions import action2str
import argparse
from datetime import datetime
from llms.providers.openai_utils import run_embedding_model
class RealTimeAgentRunner:
    """
    A class managing the Playwright environment and agent in synchronous mode,
    with optional human-in-the-loop feedback.
    """

    def __init__(
            self, 
            human_feedback_enabled: bool = False,
            model: str = "gpt4o"
            ):

        assert model in ["gpt4o", "qwen2vl"], "Invalid model"

        self.model = model
        self.env: Optional[ScriptBrowserEnv] = None
        self.agent: Optional[PromptAgent] = None

        self.json_data = []
        self.retrieval_data = []
        self.correction_abstractions = []
        self.episode_tag = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4()}"
        self.save_dir = f"experiments/{self.episode_tag}"
        self.do_save = True

        self.trajectory: list = []
        self.action_list: list = []
        self.current_obs: Optional[dict] = None

        self.user_intent: Optional[str] = None
        self.images: list = []
        self.meta_data: Dict[str, list] = {"action_history": ["None"]}
        self.config_file: Optional[str] = None
        self.is_running: bool = False

        # Additional for human-in-the-loop
        self.human_feedback_enabled = human_feedback_enabled
        # Store feedback info as in your original handle_human_in_the_loop code
        # We'll store it as a list of lists of tuples, same as your code
        self.human_feedback_list: List[Union[None, List[Tuple[int, str, str, Action, str, Action]]]] = []

        self.count: int = 0  # step counter
        args = {
            "render": False,
            "debug": False,
            "slow_mo": 0,
            "action_set_tag": "id_accessibility_tree",
            "observation_type": "accessibility_tree",
            "current_viewport_only": False,
            "viewport_width": 1280,
            "viewport_height": 2048,
            "save_trace_enabled": False,
            "sleep_after_execution": 0.0,
            "max_steps": 30,
            "agent_type": "prompt",
            "instruction_path": "agent/prompts/ical_examples/learned_examples/human_demos_with_abstractions/planning_examples.json",
            "feedback_instruction_path": "agent/prompts/jsons/p_multimodal_humanfeedback.json",
            "parsing_failure_th": 3,
            "repeating_action_failure_th": 5,
            "episode_in_try_except": False,
            "skip_if_finished": False,
            "skip_if_exists": False,
            "test_config_base_dir": None,
            "eval_captioning_model_device": "cpu",
            "eval_captioning_model": "Salesforce/blip2-flan-t5-xl",
            "captioning_model": "Salesforce/blip2-flan-t5-xl",
            "instruction_jsons": ["agent/prompts/ical_examples/learned_examples/merged_classifieds_shopping_reddit_V2/planning_examples.json", "agent/prompts/ical_examples/learned_examples/human_demos_with_abstractions/planning_examples.json"],
            "feedback_jsons": [],
            "save_examples_memory": False,
            "continually_add_saved_examples": False,
            "provider": "openai",
            "model": "gpt-3.5-turbo-0613",
            "mode": "chat",
            "temperature": 0.2,
            "top_p": 0.1,
            "context_length": 0,
            "max_tokens": 4096,
            "stop_token": None,
            "max_retry": 1,
            "max_obs_length": 0,
            "experiment_name": "run0",
            "wandb_directory": ".",
            "shopping": None,
            "reddit": None,
            "classifieds": None,
            "pick_random_subset": None,
            "seed": 32,
            "topk": 3,
            "w_obs": 0.3,
            "w_act": 0.2,
            "w_task": 0.5,
            "w_vis": 0.0,
            "using_full_actions_examples": False,
            "shuffle": False,
            "no_add_abstractions_system": False,
            "ablate_image_context": False,
            "test_start_idx": 0,
            "test_end_idx": 910,
            "result_dir": "",
            "evalMode": "regular",
        }

        if self.human_feedback_enabled:
            if self.model == "qwen2vl":
                # Create agent
                changed_values = {
                    'model': 'qwen2vl',
                    'temperature': 0.,
                    'action_set_tag': 'som',
                    'save_examples_memory': False,
                    'provider': 'vllm',
                    'captioning_model': None,
                    'eval_captioning_model': None,
                    'evalMode': 'human_in_the_loop',
                    'instruction_path': "agent/prompts/jsons/p_som_qwen2vl_humanfeedback.json",
                }
            elif self.model == "gpt4o":
                # Create agent
                changed_values = {
                    'model': 'gpt-4o',
                    'evalMode': 'human_in_the_loop',
                    'temperature': 0.2,
                    'top_p': 0.9,
                    'action_set_tag': 'som',
                    'save_examples_memory': False,
                    'provider': 'openai',
                    'captioning_model': None,
                    'eval_captioning_model': None,
                    'feedback_jsons': ["agent/prompts/jsons/p_multimodal_humanfeedback.json"],
                    # 'instruction_jsons': [""],
                }
        else:
            if self.model == "qwen2vl":
                # Create agent
                changed_values = {
                    'model': 'qwen2vl',
                    'temperature': 0.,
                    'action_set_tag': 'som',
                    'save_examples_memory': False,
                    'provider': 'vllm',
                    # 'captioning_model': None,
                    # 'eval_captioning_model': None,
                    'instruction_path': "agent/prompts/jsons/p_som_qwen2vl.json",
                }
            elif self.model == "gpt4o":
                # Create agent
                changed_values = {
                    'model': 'gpt-4o',
                    'temperature': 0.2,
                    'top_p': 0.9,
                    'action_set_tag': 'som',
                    'save_examples_memory': False,
                    'provider': 'openai',
                    'captioning_model': None,
                    'eval_captioning_model': None,
                }
        print(changed_values)
        args.update(changed_values)
        # convert to argparse
        args = argparse.Namespace(**args)

        self.args = args

        

    def setup(
            self, 
            start_url: str, 
            user_intent: str,
            config_file: str = "config_files/demo/demo.json"
            ):
        """
        Initialize environment + agent. If there's an existing session, close it first.
        """
        self.close()

        # 1. Create environment (sync)
        self.env = ScriptBrowserEnv(
            headless=False,             # or True for headless
            slow_mo=0,
            observation_type="image_som",
            save_trace_enabled=True,
        )

        with open(config_file, "r") as f:
            config = json.load(f)
        config["start_url"] = start_url
        config["intent"] = user_intent
        config["intent_template"] = user_intent
        # save config_file
        with open(config_file, "w") as f:
            json.dump(config, f, indent=4)

        # 2. Create agent
        self.agent = construct_agent(self.args)

        # 3. Reset environment
        obs, info = self.env.reset(options={"config_file": config_file})
        self.current_obs = obs
        self.trajectory = [{"observation": obs, "info": info}]
        self.action_list = []
        self.images = []
        self.meta_data = {"action_history": ["None"]}

        self.config_file = config_file
        self.user_intent = user_intent
        self.start_url = start_url
        self.is_running = True
        self.count = 0

        # If we are in human feedback mode, also track it
        if self.human_feedback_enabled:
            self.human_feedback_list = []
            # For each step, we might append a None entry for storing feedback
        else:
            self.human_feedback_list = []

    def propose_action(self) -> Tuple[str, str]:
        """
        If human_feedback_enabled = True:
          - The agent proposes an action (no step).
          - Return the action string, no updated screenshot yet.
        If human_feedback_enabled = False:
          - The agent proposes an action and we immediately step the environment.
          - Return the action string + new screenshot.
        """
        if not self.env or not self.agent or not self.is_running:
            return ("No environment running.", "", "")

        # If in feedback mode, create a None entry for storing feedback
        if self.human_feedback_enabled:
            self.human_feedback_list.append(None)

        # Agent picks next action
        action = self.agent.next_action(
            trajectory=self.trajectory,
            intent=self.user_intent,
            images=self.images,
            meta_data=self.meta_data,
            config_file=self.config_file,
        )

        def get_element_line(state: str, element_id: int) -> str:
            lines = state.split("\n")
            element_str = f"[{element_id}]"
            for line in lines:
                if element_str in line:
                    return line
            return ""

        element_id_line = get_element_line(self.current_obs["text"], action["element_id"])

        def get_action_history_format(
            raw_action_str: str, element_id: int, element_line: str
        ) -> str:
            element_str = f"[{element_id}]"
            replaced = element_line.replace(element_str, "")
            if replaced and replaced.startswith(" "):
                replaced = replaced[1:]
            return raw_action_str.replace(element_str, replaced)

        # Remove extra info from action string
        action_str_nums = action2str(action, self.args.action_set_tag)
        action_str_nums = action_str_nums.split(" where")[0].replace("[A] ", "")

        action_str_history = get_action_history_format(
            action_str_nums, action["element_id"], element_id_line
        )

        action_str_display_to_user = f"{action_str_history} (element {action['element_id']})"

        if self.human_feedback_enabled:
            # Just store it, no step yet
            self.proposed_action = action
            return (action_str_display_to_user, "", action['raw_prediction'])
        else:
            # If feedback is disabled, automatically commit this action
            self.proposed_action = action
            return self._auto_commit()

    def apply_feedback(self, feedback: str) -> str:
        """
        If human feedback is enabled, revise the proposed action 
        before environment step.
        """
        if not self.human_feedback_enabled:
            return "Human feedback is disabled.", ""
        if not self.env or not self.agent or not self.is_running:
            return "No environment or agent available to apply feedback.", ""
        if self.proposed_action is None:
            return "No proposed action to revise.", ""

        prev_action = self.proposed_action
        observation_text = self.trajectory[-1]["observation"].get("text", "")

        # Revise
        if self.args.model == "qwen2vl":
            revised_action = self.agent.next_action(
                trajectory=self.trajectory,
                intent=self.user_intent,
                images=self.images,
                meta_data=self.meta_data,
                humanFeedback=feedback,
                prev_action=prev_action,
            )
        else:
            revised_action = self.agent.next_action_humanFeedback(
                trajectory=self.trajectory,
                intent=self.user_intent,
                images=self.images,
                meta_data=self.meta_data,
                humanFeedback=feedback,
                prev_action=prev_action,
            )

        # Store feedback info
        feedback_tuple = (
            self.count,
            self.user_intent,
            observation_text,
            prev_action,
            feedback,
            revised_action,
        )
        if self.human_feedback_list and self.human_feedback_list[-1] is None:
            self.human_feedback_list[-1] = [feedback_tuple]
        else:
            self.human_feedback_list.append([feedback_tuple])

        self.proposed_action = revised_action

        def get_element_line(state: str, element_id: int) -> str:
            lines = state.split("\n")
            element_str = f"[{element_id}]"
            for line in lines:
                if element_str in line:
                    return line
            return ""

        element_id_line = get_element_line(self.current_obs["text"], revised_action["element_id"])

        def get_action_history_format(
            raw_action_str: str, element_id: int, element_line: str
        ) -> str:
            element_str = f"[{element_id}]"
            replaced = element_line.replace(element_str, "")
            if replaced and replaced.startswith(" "):
                replaced = replaced[1:]
            return raw_action_str.replace(element_str, replaced)

        # Remove extra info from action string
        action_str_nums = action2str(revised_action, self.args.action_set_tag)
        action_str_nums = action_str_nums.split(" where")[0].replace("[A] ", "")

        action_str_history = get_action_history_format(
            action_str_nums, revised_action["element_id"], element_id_line
        )

        action_str_display_to_user = f"{action_str_history} (element {revised_action['element_id']})"

        return f"Revised action: {action_str_display_to_user}", revised_action['raw_prediction']

    def commit_action(self) -> Tuple[str, str, str]:
        """
        If feedback is enabled, user calls this to step the environment with the current proposed action.
        Returns (final_action_str, screenshot, final_action_output).
        """
        if not self.human_feedback_enabled:
            return ("Feedback disabled, no commit needed.", "")
        if not self.env or not self.agent or not self.is_running:
            return ("No environment running.", "")
        if self.proposed_action is None:
            return ("No proposed action to commit.", "")

        # Step environment with the proposed action
        self.proposed_action, action_str, screenshot_b64, action_output = self._execute_action(self.proposed_action)
        return (action_str, screenshot_b64, action_output)

    def stop(self) -> str:
        """
        Stop the environment session.
        """
        self.is_running = False
        self.close()
        return "Stopped"

    def get_screenshot_b64(self) -> str:
        """
        Return current screenshot as base64.
        """
        if not self.current_obs or "image" not in self.current_obs:
            return ""
        return self._image_to_b64(self.current_obs["image"])

    def close(self):
        if self.env:
            self.env.close()
            self.env = None
        self.agent = None
        self.is_running = False
        self.proposed_action = None

    # -----------------------------
    # Internal Helpers
    # -----------------------------
    def _auto_commit(self) -> Tuple[str, str, str]:
        """
        For the no-feedback scenario, we automatically step the environment 
        after the agent proposes an action.
        """
        if self.proposed_action is None:
            return ("No proposed action to commit automatically.", "")

        self.proposed_action, action_str, screenshot_b64, action_output = self._execute_action(self.proposed_action)
        return (action_str, screenshot_b64, action_output)

    def _execute_action(self, action: Action) -> Tuple[Optional[Action], str, str, str]:
        """
        Actually step the environment with the provided action.
        Return (None if STOP, action_str, screenshot_b64, action_output).
        """
        
        if not action["action_type"] == ActionTypes.STOP:
            obs, _, terminated, _, info = self.env.step(action)
            self.trajectory.append({"observation": obs, "info": info})
            self.action_list.append(action)
        
        self.count += 1

        if self.do_save:
            
            if self.human_feedback_list and self.human_feedback_list[-1] is not None:
                print(self.human_feedback_list[-1])
                human_feedback = {
                    "revision": self.human_feedback_list[-1][-1][-1]['raw_prediction'],
                    "feedback": self.human_feedback_list[-1][-1][-2],
                }
                try:
                    correction_abstraction = self.human_feedback_list[-1][-1][-1]['raw_prediction'].split("Correction Abstraction:")[1].split("\n\nPlan")[0]
                    self.correction_abstractions.append(correction_abstraction)
                except:
                    pass
                # action['raw_prediction'] = "Plan:"+action['raw_prediction'].split("\n\nPlan:")[-1].replace("Revised Action:", "Action:")
            else:
                human_feedback = None

            def format_action_history(
                action_history
            ):
                if len(action_history)==1:
                    return 'None'
                else:
                    action_text = ''
                    count = 1
                    for action in action_history[1:]:
                        action_text += f'\n{count}. {action}'
                        count += 1
                    return action_text
                
            template = "OBJECTIVE: {objective}\n\nOBSERVATION:\n{observation}\n\nURL: {url}\n\nPREVIOUS ACTIONS: {previous_action}"
            previous_action = format_action_history(self.meta_data["action_history"])
            template = template.format(
                objective=self.user_intent,
                observation=self.current_obs["text"],
                url=self.start_url,
                previous_action=previous_action
            )

            im_path = f"{self.save_dir}/images/{self.episode_tag}/screenshot_{self.count}.png"
            os.makedirs(os.path.split(im_path)[0], exist_ok=True)

            self.retrieval_data.append([
                template,
                action['raw_prediction'],
                im_path,
                self.correction_abstractions,
                1
                ])
            
            with open(self.args.instruction_path, "r") as f:
                instruction = json.load(f)
            instruction["examples"] = self.retrieval_data

            json_data = {
                "user_intent": self.user_intent,
                "input": self.current_obs["text"],
                "output": action['raw_prediction'],
                # "screenshot": screenshot_b64,
                "action_history": self.meta_data["action_history"],
                "human_feedback": human_feedback,
                "step": self.count,
                "config_file": self.config_file,
            }
            self.json_data.append(json_data)

            # save json_data and PIL image to file
            os.makedirs(self.save_dir, exist_ok=True)
            screenshot_pil = Image.fromarray(self.current_obs["image"][:, :, :3])
            
            screenshot_pil.save(im_path)
            with open(f"{self.save_dir}/trajectory_data.json", "w") as f:
                json.dump(self.json_data, f, indent=4)
            with open(f"{self.save_dir}/planning_examples.json", "w") as f:
                json.dump(instruction, f, indent=4)
            embed_path_visual = im_path.replace('images', 'embeddings_visual').replace('.png', '.npy')
            os.makedirs(os.path.split(embed_path_visual)[0], exist_ok=True)
            if not os.path.exists(embed_path_visual):
                visual_feature_embedding = self.agent.prompt_constructor.clip_model.encode_images([screenshot_pil]).squeeze().cpu().numpy()
                np.save(embed_path_visual, visual_feature_embedding)
            
            embed_path_task = im_path.replace('images', 'embeddings_task').replace('.png', '.npy')
            os.makedirs(os.path.split(embed_path_task)[0], exist_ok=True)
            if not os.path.exists(embed_path_task):
                embedding_task = run_embedding_model(self.user_intent)
                np.save(embed_path_task, embedding_task)

            embed_path_obs = im_path.replace('images', 'embeddings_obs').replace('.png', '.npy')
            os.makedirs(os.path.split(embed_path_obs)[0], exist_ok=True)
            if not os.path.exists(embed_path_obs):
                embedding_obs = run_embedding_model(self.current_obs["text"])
                np.save(embed_path_obs, embedding_obs)

            embed_path_act = im_path.replace('images', 'embeddings_act').replace('.png', '.npy')
            os.makedirs(os.path.split(embed_path_act)[0], exist_ok=True)
            if not os.path.exists(embed_path_act):
                embedding_act = run_embedding_model(previous_action)
                np.save(embed_path_act, embedding_act)

        if action["action_type"] == ActionTypes.STOP:
            self.is_running = False
            return (None, "STOP", "", "")
        
        def get_element_line(state: str, element_id: int) -> str:
            lines = state.split("\n")
            element_str = f"[{element_id}]"
            for line in lines:
                if element_str in line:
                    return line
            return ""

        element_id_line = get_element_line(self.current_obs["text"], action["element_id"])

        def get_action_history_format(
            raw_action_str: str, element_id: int, element_line: str
        ) -> str:
            element_str = f"[{element_id}]"
            replaced = element_line.replace(element_str, "")
            if replaced and replaced.startswith(" "):
                replaced = replaced[1:]
            return raw_action_str.replace(element_str, replaced)

        # Remove extra info from action string
        action_str_nums = action2str(action, self.args.action_set_tag)
        action_str_nums = action_str_nums.split(" where")[0].replace("[A] ", "")

        action_str_history = get_action_history_format(
            action_str_nums, action["element_id"], element_id_line
        )

        self.meta_data["action_history"].append(action_str_history)

        action_str_display_to_user = f"{action_str_history} (element {action['element_id']})"
        
        self.current_obs = obs
        screenshot_b64 = self._image_to_b64(self.current_obs["image"])

        if terminated:
            self.is_running = False
            return (None, f"{action_str_display_to_user} (Environment terminated)", screenshot_b64)

        return (None, action_str_display_to_user, screenshot_b64, action['raw_prediction'])

    def _image_to_b64(self, image_ndarray) -> str:
        """
        Convert a screenshot (numpy array) to base64.
        """
        screenshot_pil = Image.fromarray(image_ndarray[:, :, :3])
        buf = BytesIO()
        screenshot_pil.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def stop(self) -> str:
        """
        Stop the environment session.
        """
        self.is_running = False
        self.close()
        return "Stopped"

    def close(self):
        """
        Close the environment if open.
        """
        if self.env:
            self.env.close()
            self.env = None
        self.agent = None
        self.is_running = False

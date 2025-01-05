# real_time_agent_runner.py

import os
import base64
from io import BytesIO
from typing import Optional, Tuple, List, Union, Dict

from PIL import Image

# Browser environment / agent imports
from browser_env import ScriptBrowserEnv
from browser_env.actions import ActionTypes, Action
from agent import construct_agent, PromptAgent
from browser_env.actions import action2str
import argparse

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
            "instruction_path": "learned_examples/human_demos_with_abstractions/planning_examples.json",
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
            "instruction_jsons": ["learned_examples/merged_classifieds_shopping_reddit_V2/planning_examples.json", "learned_examples/human_demos_with_abstractions/planning_examples.json"],
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
            "w_obs": 0.025,
            "w_act": 0.05,
            "w_task": 0.9,
            "w_vis": 0.025,
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
                "feedback_jsons": ["agent/prompts/jsons/p_multimodal_humanfeedback.json"],
            }
        else:
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
        args.update(changed_values)
        # convert to argparse
        args = argparse.Namespace(**args)

        self.args = args

        

    def setup(self, config_file: str, user_intent: str):
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
            return ("No environment running.", "")

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
        self.meta_data["action_history"].append(action_str_history)

        action_str_display_to_user = f"{action_str_history} (element {action['element_id']})"

        if self.human_feedback_enabled:
            # Just store it, no step yet
            self.proposed_action = action
            return (action_str_display_to_user, "")
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
            return "Human feedback is disabled."
        if not self.env or not self.agent or not self.is_running:
            return "No environment or agent available to apply feedback."
        if self.proposed_action is None:
            return "No proposed action to revise."

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
        self.meta_data["action_history"].append(action_str_history)

        action_str_display_to_user = f"{action_str_history} (element {revised_action['element_id']})"

        return f"Revised action: {action_str_display_to_user}"

    def commit_action(self) -> Tuple[str, str]:
        """
        If feedback is enabled, user calls this to step the environment with the current proposed action.
        Returns (final_action_str, screenshot).
        """
        if not self.human_feedback_enabled:
            return ("Feedback disabled, no commit needed.", "")
        if not self.env or not self.agent or not self.is_running:
            return ("No environment running.", "")
        if self.proposed_action is None:
            return ("No proposed action to commit.", "")

        # Step environment with the proposed action
        self.proposed_action, action_str, screenshot_b64 = self._execute_action(self.proposed_action)
        return (action_str, screenshot_b64)

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
    def _auto_commit(self) -> Tuple[str, str]:
        """
        For the no-feedback scenario, we automatically step the environment 
        after the agent proposes an action.
        """
        if self.proposed_action is None:
            return ("No proposed action to commit automatically.", "")

        self.proposed_action, action_str, screenshot_b64 = self._execute_action(self.proposed_action)
        return (action_str, screenshot_b64)

    def _execute_action(self, action: Action) -> Tuple[Optional[Action], str, str]:
        """
        Actually step the environment with the provided action.
        Return (None if STOP, action_str, screenshot_b64).
        """
        if action["action_type"] == ActionTypes.STOP:
            self.is_running = False
            return (None, "STOP", "")

        obs, _, terminated, _, info = self.env.step(action)
        self.current_obs = obs
        self.trajectory.append({"observation": obs, "info": info})
        self.action_list.append(action)

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

        screenshot_b64 = self._image_to_b64(obs["image"])
        self.count += 1

        if terminated:
            self.is_running = False
            return (None, f"{action_str_display_to_user} (Environment terminated)", screenshot_b64)

        return (None, action_str_display_to_user, screenshot_b64)

    def _image_to_b64(self, image_ndarray) -> str:
        """
        Convert a screenshot (numpy array) to base64.
        """
        screenshot_pil = Image.fromarray(image_ndarray[:, :, :3])
        buf = BytesIO()
        screenshot_pil.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    # def propose_action(self) -> str:
    #     """
    #     1. The agent proposes an action (next_action),
    #     2. We do NOT step the environment yet.
    #     3. Return the action string so the user can see it.
    #     """
    #     if not self.env or not self.agent or not self.is_running:
    #         return "No environment running."

    #     # If in feedback mode, we create a None entry for storing feedback (like your original code).
    #     if self.human_feedback_enabled:
    #         self.human_feedback_list.append(None)

    #     # Ask agent for next action
    #     action = self.agent.next_action(
    #         trajectory=self.trajectory,
    #         intent=self.user_intent,
    #         images=self.images,
    #         meta_data=self.meta_data,
    #         config_file=self.config_file,
    #     )

    #     # If the agent wants to STOP, we won't commit that automatically.
    #     # We just store it as a proposed action; user can choose to commit or not.
    #     self.proposed_action = action

    #     def get_element_line(state: str, element_id: int) -> str:
    #         lines = state.split("\n")
    #         element_str = f"[{element_id}]"
    #         for line in lines:
    #             if element_str in line:
    #                 return line
    #         return ""

    #     element_id_line = get_element_line(self.current_obs["text"], action["element_id"])

    #     def get_action_history_format(
    #         raw_action_str: str, element_id: int, element_line: str
    #     ) -> str:
    #         element_str = f"[{element_id}]"
    #         replaced = element_line.replace(element_str, "")
    #         if replaced and replaced.startswith(" "):
    #             replaced = replaced[1:]
    #         return raw_action_str.replace(element_str, replaced)

    #     # Remove extra info from action string
    #     action_str_nums = action2str(action, self.args.action_set_tag)
    #     action_str_nums = action_str_nums.split(" where")[0].replace("[A] ", "")

    #     action_str_history = get_action_history_format(
    #         action_str_nums, action["element_id"], element_id_line
    #     )
    #     self.meta_data["action_history"].append(action_str_history)

    #     action_str_display_to_user = f"{action_str_history} (element {action['element_id']})"

    #     if self.human_feedback_enabled:
    #         # Just store it, no step yet
    #         self.proposed_action = action
    #         return (action_str_display_to_user, "")
    #     else:
    #         # If feedback is disabled, automatically commit this action
    #         self.proposed_action = action
    #         return self.commit_action()

    #     # action_str = f"{action['action_type']} on element {action['element_id']}"
    #     # return action_str_display_to_user

    # def apply_feedback(self, feedback: str) -> str:
    #     """
    #     The user is unhappy with the current proposed_action, so they provide feedback.
    #     We refine the action using next_action_humanFeedback (or agent.next_action).
    #     """
    #     if not self.human_feedback_enabled:
    #         return "Human feedback not enabled."
    #     if not self.env or not self.agent or not self.is_running:
    #         return "No environment or agent available to apply feedback."
    #     if self.proposed_action is None:
    #         return "No proposed action to revise."

    #     # The agent might have different methods for feedback:
    #     # next_action_humanFeedback or next_action with a feedback param.
    #     # We'll replicate your handle_human_in_the_loop logic.

    #     prev_action = self.proposed_action
    #     observation_text = self.trajectory[-1]["observation"].get("text", "")

    #     if self.args.model == "qwen2vl":
    #         revised_action = self.agent.next_action(
    #             trajectory=self.trajectory,
    #             intent=self.user_intent,
    #             images=self.images,
    #             meta_data=self.meta_data,
    #             humanFeedback=feedback,
    #             prev_action=prev_action,
    #         )
    #     else:
    #         revised_action = self.agent.next_action_humanFeedback(
    #             trajectory=self.trajectory,
    #             intent=self.user_intent,
    #             images=self.images,
    #             meta_data=self.meta_data,
    #             humanFeedback=feedback,
    #             prev_action=prev_action,
    #         )

    #     # Log the feedback
    #     feedback_tuple = (
    #         self.count,
    #         self.user_intent,
    #         observation_text,
    #         prev_action,
    #         feedback,
    #         revised_action,
    #     )
    #     if self.human_feedback_list and self.human_feedback_list[-1] is None:
    #         self.human_feedback_list[-1] = [feedback_tuple]
    #     else:
    #         self.human_feedback_list.append([feedback_tuple])

    #     self.proposed_action = revised_action

    #     def get_element_line(state: str, element_id: int) -> str:
    #         lines = state.split("\n")
    #         element_str = f"[{element_id}]"
    #         for line in lines:
    #             if element_str in line:
    #                 return line
    #         return ""

    #     element_id_line = get_element_line(self.current_obs["text"], action["element_id"])

    #     def get_action_history_format(
    #         raw_action_str: str, element_id: int, element_line: str
    #     ) -> str:
    #         element_str = f"[{element_id}]"
    #         replaced = element_line.replace(element_str, "")
    #         if replaced and replaced.startswith(" "):
    #             replaced = replaced[1:]
    #         return raw_action_str.replace(element_str, replaced)

    #     # Remove extra info from action string
    #     action_str_nums = action2str(action, self.args.action_set_tag)
    #     action_str_nums = action_str_nums.split(" where")[0].replace("[A] ", "")

    #     action_str_history = get_action_history_format(
    #         action_str_nums, action["element_id"], element_id_line
    #     )
    #     self.meta_data["action_history"].append(action_str_history)

    #     action_str_display_to_user = f"{action_str_history} (element {action['element_id']})"
    #     return f"Revised action: {action_str_display_to_user}"

    # def commit_action(self) -> str:
    #     """
    #     Once the user is satisfied with the proposed action, we commit it:
    #     we step the environment with that action, return the screenshot.
    #     """
    #     if not self.env or not self.agent or not self.is_running:
    #         return "No environment running."
    #     if self.proposed_action is None:
    #         return "No proposed action to commit."

    #     action = self.proposed_action
    #     self.proposed_action = None  # Clear after committing

    #     # If the agent's action is STOP, we finalize
    #     if action["action_type"] == ActionTypes.STOP:
    #         self.is_running = False
    #         return "STOP"

    #     obs, _, terminated, _, info = self.env.step(action)
    #     self.current_obs = obs
    #     self.trajectory.append({"observation": obs, "info": info})
    #     self.action_list.append(action)
    #     def get_element_line(state: str, element_id: int) -> str:
    #         lines = state.split("\n")
    #         element_str = f"[{element_id}]"
    #         for line in lines:
    #             if element_str in line:
    #                 return line
    #         return ""

    #     element_id_line = get_element_line(obs["text"], action["element_id"])

    #     def get_action_history_format(
    #         raw_action_str: str, element_id: int, element_line: str
    #     ) -> str:
    #         element_str = f"[{element_id}]"
    #         replaced = element_line.replace(element_str, "")
    #         if replaced and replaced.startswith(" "):
    #             replaced = replaced[1:]
    #         return raw_action_str.replace(element_str, replaced)

    #     # Remove extra info from action string
    #     action_str_nums = action2str(action, self.args.action_set_tag)
    #     action_str_nums = action_str_nums.split(" where")[0].replace("[A] ", "")

    #     action_str_history = get_action_history_format(
    #         action_str_nums, action["element_id"], element_id_line
    #     )
    #     self.meta_data["action_history"].append(action_str_history)

    #     action_str_display_to_user = f"{action_str_history} (element {action['element_id']})"

    #     self.count += 1
    #     if terminated:
    #         self.is_running = False
    #         return f"{action_str_display_to_user} | Environment terminated."

    #     return action_str_display_to_user

    # def step(self) -> Tuple[str, str]:
    #     """
    #     Execute one step in the environment and return (action_str, screenshot_base64).
    #     """
    #     if not self.env or not self.agent or not self.is_running:
    #         return ("No environment running.", "")

    #     # If in human feedback mode, we store a None placeholder for feedback
    #     if self.human_feedback_enabled:
    #         self.human_feedback_list.append(None)

    #     # 1. next_action
    #     action = self.agent.next_action(
    #         trajectory=self.trajectory,
    #         intent=self.user_intent,
    #         images=self.images,
    #         meta_data=self.meta_data,
    #         config_file=self.config_file,
    #     )

    #     # 2. Check STOP
    #     if action["action_type"] == ActionTypes.STOP:
    #         self.is_running = False
    #         return ("STOP", "")

    #     # 3. Env step
    #     obs, _, terminated, _, info = self.env.step(action)
    #     self.current_obs = obs

    #     self.trajectory.append({"observation": obs, "info": info})
    #     self.action_list.append(action)

    #     def get_element_line(state: str, element_id: int) -> str:
    #         lines = state.split("\n")
    #         element_str = f"[{element_id}]"
    #         for line in lines:
    #             if element_str in line:
    #                 return line
    #         return ""

    #     element_id_line = get_element_line(obs["text"], action["element_id"])

    #     def get_action_history_format(
    #         raw_action_str: str, element_id: int, element_line: str
    #     ) -> str:
    #         element_str = f"[{element_id}]"
    #         replaced = element_line.replace(element_str, "")
    #         if replaced and replaced.startswith(" "):
    #             replaced = replaced[1:]
    #         return raw_action_str.replace(element_str, replaced)

    #     # Remove extra info from action string
    #     action_str_nums = action2str(action, self.args.action_set_tag)
    #     action_str_nums = action_str_nums.split(" where")[0].replace("[A] ", "")

    #     action_str_history = get_action_history_format(
    #         action_str_nums, action["element_id"], element_id_line
    #     )
    #     self.meta_data["action_history"].append(action_str_history)

    #     print(action["element_id"])
    #     action_str_display_to_user = f"{action_str_history} (element {action['element_id']})"

    #     # self.images.append(obs["image"])
    #     # action_str = f"{action['action_type']} on element {action['element_id']}"
    #     # self.meta_data["action_history"].append(action_str)

    #     # Convert screenshot to base64
    #     screenshot_pil = Image.fromarray(obs["image"][:, :, :3])
    #     buf = BytesIO()
    #     screenshot_pil.save(buf, format="PNG")
    #     screenshot_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    #     self.count += 1
    #     if terminated:
    #         self.is_running = False

    #     return (action_str_display_to_user, screenshot_b64)

    # def apply_feedback(self, feedback: str) -> Tuple[str, str]:
    #     """
    #     If the user says "no" or wants the agent to adjust its action, 
    #     we apply the feedback to the agent and produce a new action + screenshot.
    #     """
    #     if not self.env or not self.agent or not self.is_running:
    #         return ("No environment or agent available to apply feedback.", "")

    #     # The agent might have different methods for feedback, e.g. next_action_humanFeedback
    #     # We'll replicate the logic from your handle_human_in_the_loop.
    #     # For example, we can check if self.args.model == "qwen2vl" or not.

    #     # Let's get the last action that was taken
    #     prev_action = self.action_list[-1] if self.action_list else None

    #     if self.args.model == "qwen2vl":
    #         new_action = self.agent.next_action(
    #             trajectory=self.trajectory,
    #             intent=self.user_intent,
    #             images=self.images,
    #             meta_data=self.meta_data,
    #             humanFeedback=feedback,
    #             prev_action=prev_action,
    #         )
    #     else:
    #         # We'll assume the agent has a method next_action_humanFeedback
    #         # that specifically uses the feedback
    #         new_action = self.agent.next_action_humanFeedback(
    #             trajectory=self.trajectory,
    #             intent=self.user_intent,
    #             images=self.images,
    #             meta_data=self.meta_data,
    #             humanFeedback=feedback,
    #             prev_action=prev_action,
    #         )

    #     # Log the feedback details into self.human_feedback_list
    #     # We'll replicate your code that stored a tuple 
    #     # (count, intent, last_observation_text, old_action, human_feedback, new_action).
    #     observation_text = self.trajectory[-1]["observation"].get("text", "")
    #     feedback_tuple = (
    #         self.count,
    #         self.user_intent,
    #         observation_text,
    #         prev_action,
    #         feedback,
    #         new_action,
    #     )
    #     # If the last entry in self.human_feedback_list is None, store it there
    #     if self.human_feedback_list and self.human_feedback_list[-1] is None:
    #         self.human_feedback_list[-1] = [feedback_tuple]
    #     else:
    #         self.human_feedback_list.append([feedback_tuple])

    #     # Now let's "take" that new_action as if it was a step
    #     # If new_action is STOP, we simply stop. Otherwise, we do env.step.
    #     if new_action["action_type"] == ActionTypes.STOP:
    #         self.is_running = False
    #         return ("STOP", "")

    #     obs, _, terminated, _, info = self.env.step(new_action)
    #     self.current_obs = obs
    #     self.trajectory.append({"observation": obs, "info": info})
    #     self.action_list.append(new_action)

    #     def get_element_line(state: str, element_id: int) -> str:
    #         lines = state.split("\n")
    #         element_str = f"[{element_id}]"
    #         for line in lines:
    #             if element_str in line:
    #                 return line
    #         return ""

    #     element_id_line = get_element_line(obs["text"], new_action["element_id"])

    #     def get_action_history_format(
    #         raw_action_str: str, element_id: int, element_line: str
    #     ) -> str:
    #         element_str = f"[{element_id}]"
    #         replaced = element_line.replace(element_str, "")
    #         if replaced and replaced.startswith(" "):
    #             replaced = replaced[1:]
    #         return raw_action_str.replace(element_str, replaced)

    #     # Remove extra info from action string
    #     action_str_nums = action2str(new_action, self.args.action_set_tag)
    #     action_str_nums = action_str_nums.split(" where")[0].replace("[A] ", "")

    #     action_str_history = get_action_history_format(
    #         action_str_nums, new_action["element_id"], element_id_line
    #     )
    #     self.meta_data["action_history"].append(action_str_history)

    #     action_str_display_to_user = f"{action_str_history} (element {new_action['element_id']})"

    #     # Convert screenshot to base64
    #     screenshot_pil = Image.fromarray(obs["image"][:, :, :3])
    #     buf = BytesIO()
    #     screenshot_pil.save(buf, format="PNG")
    #     screenshot_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    #     self.count += 1
    #     if terminated:
    #         self.is_running = False

    #     return (action_str_display_to_user, screenshot_b64)

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

import json
import re
from pathlib import Path
from typing import Any, TypedDict
from PIL import ImageFile, Image
ImageFile.LOAD_TRUNCATED_IMAGES = True

from browser_env import Action, ActionParsingError, Trajectory
from browser_env.env_config import URL_MAPPINGS
from browser_env.utils import StateInfo, pil_to_b64, pil_to_vertex
from llms import lm_config
from llms.tokenizers import Tokenizer
from llms.utils import APIInput

import numpy as np
import scipy
import os
import wandb
import torch
import copy

class Instruction(TypedDict):
    """Instruction for constructing prompt"""

    intro: str
    examples: list[tuple[str, str]]
    template: str
    meta_data: dict[str, Any]


class PromptConstructor(object):
    def __init__(
        self,
        instruction_path: str | Path,
        lm_config: lm_config.LMConfig,
        tokenizer: Tokenizer,
    ):
        self.instruction_path = Path(instruction_path)
        self.obs_modality = "text"
        self.lm_config = lm_config
        instruction = json.load(open(self.instruction_path))
        instruction["examples"] = [tuple(e) for e in instruction["examples"]]
        self.instruction: Instruction = instruction
        self.tokenizer = tokenizer

    def get_lm_api_input(
        self, intro: str, examples: list[tuple[str, str]], current: str
    ) -> APIInput:

        """Return the require format for an API"""
        message: list[dict[str, str]] | str
        if "openai" in self.lm_config.provider:
            if self.lm_config.mode == "chat":
                message = [{"role": "system", "content": intro}]
                for (x, y) in examples:
                    message.append(
                        {
                            "role": "system",
                            "name": "example_user",
                            "content": x,
                        }
                    )
                    message.append(
                        {
                            "role": "system",
                            "name": "example_assistant",
                            "content": y,
                        }
                    )
                message.append({"role": "user", "content": current})
                return message
            elif self.lm_config.mode == "completion":
                message = f"{intro}\n\n"
                message += "Here are a few examples:\n"
                for example in examples:
                    message += f"Observation\n:{example[0]}\n\n"
                    message += f"Action: {example[1]}\n\n"
                message += "Now make prediction given the observation\n\n"
                message += f"Observation\n:{current}\n\n"
                message += "Action:"
                return message
            else:
                raise ValueError(
                    f"OpenAI models do not support mode {self.lm_config.mode}"
                )
        elif "huggingface" in self.lm_config.provider:
            # https://huggingface.co/blog/llama2#how-to-prompt-llama-2
            # https://github.com/facebookresearch/llama/blob/main/llama/generation.py#L320
            if "Llama-2" in self.lm_config.model:
                if self.lm_config.mode == "chat":
                    B_INST, E_INST = "[INST]", "[/INST]"
                    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
                    BOS, EOS = "<s>", "</s>"
                    # adding the system message to be the starting of the first example
                    examples = [
                        (
                            B_SYS + intro + E_SYS + examples[0][0],
                            examples[0][1],
                        )
                    ] + examples[1:]
                    message = "".join(
                        [
                            f"{BOS}{B_INST} {x.strip()} {E_INST} {y.strip()} {EOS}"
                            for (x, y) in examples
                        ]
                    )
                    # add the current observation
                    message += f"{BOS}{B_INST} {current.strip()} {E_INST} {self.instruction['meta_data'].get('force_prefix', '')}"

                    return message
                else:
                    raise ValueError("Only chat mode is supported for Llama-2")
            else:
                raise ValueError(
                    f"Huggingface models do not support model_tag {self.lm_config.gen_config['model_tag']}"
                )
        else:
            raise NotImplementedError(
                f"Provider {self.lm_config.provider} not implemented"
            )

    def construct(
        self,
        trajectory: Trajectory,
        intent: str,
        meta_data: dict[str, Any] = {},
    ) -> APIInput:
        raise NotImplementedError

    def map_url_to_real(self, url: str) -> str:
        """Map the urls to their real world counterparts"""
        for i, j in URL_MAPPINGS.items():
            if i in url:
                url = url.replace(i, j)
        return url

    def map_url_to_local(self, url: str) -> str:
        """Map the urls to their local counterparts"""
        for i, j in URL_MAPPINGS.items():
            if j in url:
                url = url.replace(j, i)
            # https
            if j.replace("http", "https") in url:
                url = url.replace(j.replace("http", "https"), i)
        return url

    def _extract_action(self, response: str) -> str:
        raise NotImplementedError

    def extract_action(self, response: str) -> str:
        response = self._extract_action(response)
        response = self.map_url_to_local(response)
        return response


class DirectPromptConstructor(PromptConstructor):
    """The agent will direct predict the action"""

    def __init__(
        self,
        instruction_path: str | Path,
        lm_config: lm_config.LMConfig,
        tokenizer: Tokenizer,
    ):
        super().__init__(instruction_path, lm_config, tokenizer)

    def construct(
        self,
        trajectory: Trajectory,
        intent: str,
        meta_data: dict[str, Any] = {},
    ) -> APIInput:
        """Construct prompt given the trajectory"""
        intro = self.instruction["intro"]
        examples = self.instruction["examples"]
        template = self.instruction["template"]
        keywords = self.instruction["meta_data"]["keywords"]
        state_info: StateInfo = trajectory[-1]  # type: ignore[assignment]

        obs = state_info["observation"][self.obs_modality]
        max_obs_length = self.lm_config.gen_config["max_obs_length"]
        if max_obs_length:
            obs = self.tokenizer.decode(self.tokenizer.encode(obs)[:max_obs_length])  # type: ignore[arg-type]

        page = state_info["info"]["page"]
        url = page.url
        previous_action_str = meta_data["action_history"][-1]

        # input x
        current = template.format(
            objective=intent,
            url=self.map_url_to_real(url),
            observation=obs,
            previous_action=previous_action_str,
        )

        # make sure all keywords are replaced
        assert all([f"{{k}}" not in current for k in keywords])
        prompt = self.get_lm_api_input(intro, examples, current)
        return prompt

    def _extract_action(self, response: str) -> str:
        action_splitter = self.instruction["meta_data"]["action_splitter"]
        pattern = rf"{action_splitter}((.|\n)*?){action_splitter}"
        match = re.search(pattern, response)
        if match:
            return match.group(1).strip()
        else:
            raise ActionParsingError(
                f"Cannot parse action from response {response}"
            )


class CoTPromptConstructor(PromptConstructor):
    """The agent will perform step-by-step reasoning before the answer"""

    def __init__(
        self,
        instruction_path: str | Path,
        lm_config: lm_config.LMConfig,
        tokenizer: Tokenizer,
    ):
        super().__init__(instruction_path, lm_config, tokenizer)
        self.answer_phrase = self.instruction["meta_data"]["answer_phrase"]

    def construct(
        self,
        trajectory: Trajectory,
        intent: str,
        meta_data: dict[str, Any] = {},
    ) -> APIInput:
        intro = self.instruction["intro"]
        examples = self.instruction["examples"]
        template = self.instruction["template"]
        keywords = self.instruction["meta_data"]["keywords"]
        state_info: StateInfo = trajectory[-1]  # type: ignore[assignment]

        obs = state_info["observation"][self.obs_modality]
        max_obs_length = self.lm_config.gen_config["max_obs_length"]
        if max_obs_length:
            obs = self.tokenizer.decode(self.tokenizer.encode(obs)[:max_obs_length])  # type: ignore[arg-type]

        page = state_info["info"]["page"]
        url = page.url
        previous_action_str = meta_data["action_history"][-1]
        current = template.format(
            objective=intent,
            url=self.map_url_to_real(url),
            observation=obs,
            previous_action=previous_action_str,
        )

        assert all([f"{{k}}" not in current for k in keywords])

        prompt = self.get_lm_api_input(intro, examples, current)
        return prompt

    def _extract_action(self, response: str) -> str:
        # find the first occurence of action
        action_splitter = self.instruction["meta_data"]["action_splitter"]
        pattern = rf"{action_splitter}((.|\n)*?){action_splitter}"
        # match = re.search(pattern, response)
        match = re.findall(pattern, response)
        if match:
            # return match.group(1).strip()
            return match[-1][0].strip()
        else:
            raise ActionParsingError(
                f'Cannot find the answer phrase "{self.answer_phrase}" in "{response}"'
            )

class MultimodalCoTPromptConstructor(CoTPromptConstructor):
    """The agent will perform step-by-step reasoning before the answer"""

    def __init__(
        self,
        instruction_path: str | Path,
        lm_config: lm_config.LMConfig,
        tokenizer: Tokenizer,
    ):
        super().__init__(instruction_path, lm_config, tokenizer)
        self.answer_phrase = self.instruction["meta_data"]["answer_phrase"]

        self.prompt_log = {
            "current_prompt":[],
            "intro":[],
        }
        for idx in range(3):
            self.prompt_log[f"example{idx}_input"] = []
            self.prompt_log[f"example{idx}_gt_action"] = []

    def construct(
        self,
        trajectory: Trajectory,
        intent: str,
        page_screenshot_img: Image.Image,
        images: list[Image.Image],
        meta_data: dict[str, Any] = {},
        humanFeedback: str = None,
        prev_action: str = None,
    ) -> APIInput:
        intro = self.instruction["intro"]
        examples = self.instruction["examples"]
        template = self.instruction["template"]
        keywords = self.instruction["meta_data"]["keywords"]
        state_info: StateInfo = trajectory[-1]  # type: ignore[assignment]

        obs = state_info["observation"][self.obs_modality]
        max_obs_length = self.lm_config.gen_config["max_obs_length"]
        if max_obs_length:
            if self.lm_config.provider == "google":
                print("NOTE: This is a Gemini model, so we use characters instead of tokens for max_obs_length.")
                obs = obs[:max_obs_length]
            else:
                obs = self.tokenizer.decode(self.tokenizer.encode(obs)[:max_obs_length])  # type: ignore[arg-type]

        page = state_info["info"]["page"]
        url = page.url

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
        
        previous_action_str = format_action_history(meta_data["action_history"])
        
        current = template.format(
            objective=intent,
            url=self.map_url_to_real(url),
            observation=obs,
            previous_action=previous_action_str,
        )

        if humanFeedback is not None:
            from browser_env.helper_functions import get_action_description
            wrong_action_str = get_action_description(
                prev_action, 
                state_info["info"]["observation_metadata"],
                action_set_tag="som",
                prompt_constructor=self
            )
            wrong_action_str = wrong_action_str.split(' where')[0].replace('[A] ', '')
            feedback_text = f"\n\nFAILED PREVIOUS ACTION: {wrong_action_str}\n\nHUMAN FEEDBACK: {humanFeedback}"
            current = current.split('\n\nIMAGE:')[0] + feedback_text + '\n\nIMAGE:'

        assert all([f"{{k}}" not in current for k in keywords])

        prompt = self.get_lm_api_input(
            intro, examples, current, page_screenshot_img, images
        )
        return prompt

    def get_lm_api_input(
        self,
        intro: str,
        examples: list[tuple[str, str, str]],
        current: str,
        page_screenshot_img: Image.Image,
        images: list[Image.Image],
    ) -> APIInput:
        """Return the require format for an API"""
        message: list[dict[str, str]] | str | list[str | Image.Image]
        if "openai" in self.lm_config.provider:
            if self.lm_config.mode == "chat":
                message = [
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": intro}],
                    }
                ]
                for (x, y, z) in examples:
                    example_img = Image.open(z)
                    message.append(
                        {
                            "role": "system",
                            "name": "example_user",
                            "content": [
                                {"type": "text", "text": x},
                                {
                                    "type": "text",
                                    "text": "IMAGES: (1) current page screenshot",
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": pil_to_b64(example_img)
                                    },
                                },
                            ],
                        }
                    )
                    message.append(
                        {
                            "role": "system",
                            "name": "example_assistant",
                            "content": [{"type": "text", "text": y}],
                        }
                    )

                # Encode images and page_screenshot_img as base64 strings.
                current_prompt = current
                content = [
                    {
                        "type": "text",
                        "text": "IMAGES: (1) current page screenshot",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": pil_to_b64(page_screenshot_img)},
                    },
                ]
                for image_i, image in enumerate(images):
                    content.extend(
                        [
                            {
                                "type": "text",
                                "text": f"({image_i+2}) input image {image_i+1}",
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": pil_to_b64(image)},
                            },
                        ]
                    )
                content = [{"type": "text", "text": current_prompt}] + content

                message.append({"role": "user", "content": content})

                self.prompt_log["current_prompt"] = current_prompt
                self.prompt_log["intro"] = intro
                for idx in range(len(examples)):
                    x,y,z = examples[idx]
                    self.prompt_log[f"example{idx}_input"] = str(x)
                    self.prompt_log[f"example{idx}_gt_action"] = str(y)

                return message, self.prompt_log
            else:
                raise ValueError(
                    f"GPT-4V models do not support mode {self.lm_config.mode}"
                )
        elif "vllm" in self.lm_config.provider:
            if self.lm_config.mode == "chat":
                message = [
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": intro}],
                    }
                ]
                for (x, y, z) in examples:
                    raise NotImplementedError("QWEN2VL does not currently support examples")
                    message.append(
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": x},
                            ],
                        }
                    )
                    message.append(
                        {
                            "role": "assistant",
                            "content": [{"type": "text", "text": y}],
                        }
                    )
                # Encode images and page_screenshot_img as base64 strings.
                current_prompt = current
                content = [
                    {
                        "type": "image_url",
                        "image_url": {"url": pil_to_b64(page_screenshot_img)},
                    },
                ]

                content = [{"type": "text", "text": current_prompt}] + content
                message.append({"role": "user", "content": content})

                self.prompt_log["current_prompt"] = current_prompt
                self.prompt_log["intro"] = intro

                return message, self.prompt_log
        elif "google" in self.lm_config.provider:
            if self.lm_config.mode == "completion":
                message = [
                    intro,
                    "Here are a few examples:",
                ]
                for (x, y, z) in examples:
                    example_img = Image.open(z)
                    message.append(f"Observation\n:{x}\n")
                    message.extend(
                        [
                            "IMAGES:",
                            "(1) current page screenshot:",
                            pil_to_vertex(example_img),
                        ]
                    )
                    message.append(f"Action: {y}")
                message.append("Now make prediction given the observation")
                message.append(f"Observation\n:{current}\n")
                message.extend(
                    [
                        "IMAGES:",
                        "(1) current page screenshot:",
                        pil_to_vertex(page_screenshot_img),
                    ]
                )
                for image_i, image in enumerate(images):
                    message.extend(
                        [
                            f"({image_i+2}) input image {image_i+1}",
                            pil_to_vertex(image),
                        ]
                    )
                message.append("Action:")
                return message
            else:
                raise ValueError(
                    f"Gemini models do not support mode {self.lm_config.mode}"
                )
        else:
            raise NotImplementedError(
                f"Provider {self.lm_config.provider} not implemented"
            )


class MultimodalCoTPromptConstructorMemoryAugmented(CoTPromptConstructor):
    """The agent will perform step-by-step reasoning before the answer"""

    def __init__(
        self,
        instruction_path: str | Path,
        lm_config: lm_config.LMConfig,
        tokenizer: Tokenizer,
        cache=True,
    ):  
        from llms.providers.openai_utils import run_embedding_model
        super().__init__(instruction_path, lm_config, tokenizer)
        self.answer_phrase = self.instruction["meta_data"]["answer_phrase"]

        self.topk = self.lm_config.gen_config["topk"]

        self.no_add_abstractions_system = self.lm_config.gen_config["no_add_abstractions_system"]
        self.ablate_image_context = self.lm_config.gen_config["ablate_image_context"]

        self.experiment_name = self.lm_config.gen_config["experiment_name"]
        self.using_full_actions_examples = self.lm_config.gen_config["using_full_actions_examples"]

        if self.using_full_actions_examples:
            with open('agent/prompts/intro_full_actions.txt') as f:
                intro = f.read()
        else:
            with open('agent/prompts/intro.txt') as f:
                intro = f.read()
        self.instruction["intro"] = intro

        self.embedding_model = run_embedding_model
        self.examples = self.instruction["examples"]

        from clip import CLIP 
        self.clip_model = CLIP()

        self.cos_sim = torch.nn.CosineSimilarity(dim=1)

        self.prompt_log = {
            "current_prompt":[],
            "current_observation":[],
            "current_intent":[],
            "current_prev_actions":[],
            "knowledge":[],
        }
        for idx in range(self.topk):
            self.prompt_log[f"example{idx}_input"] = []
            self.prompt_log[f"example{idx}_gt_action"] = []

        if not self.no_add_abstractions_system:
            del self.prompt_log["knowledge"]

        w_obs, w_act, w_task, w_vis = self.lm_config.gen_config["w_obs"], self.lm_config.gen_config["w_act"], self.lm_config.gen_config["w_task"], self.lm_config.gen_config["w_vis"] #0.2, 0.2, 0.5, 0.1
        self.w_obs, self.w_act, self.w_task, self.w_vis = w_obs, w_act, w_task, w_vis

        self.refresh_examples()

    def refresh_examples(
        self,
    ):

        with open('agent/prompts/intro.txt') as f:
            intro = f.read()
        instruction_jsons = self.lm_config.gen_config["instruction_jsons"]
        instruction = json.load(open(self.instruction_path))
        instruction["examples"] = []
        for instruction_json in instruction_jsons:
            if not os.path.exists(instruction_json):
                continue
            instruction_ = json.load(open(instruction_json))
            instruction_["examples"] = [tuple(e) for e in instruction_["examples"]]
            instruction["examples"].extend(instruction_["examples"])
        instruction["intro"] = intro
        self.instruction = instruction
        self.example_embeddings_task, self.example_embeddings_obs, self.example_embeddings_act, self.example_embeddings_visual = self.get_example_embeddings(self.instruction["examples"])
        self.examples = self.instruction["examples"]

        self.same_episode_dict = {}
        for example_idx in range(len(self.examples)):
            example = self.examples[example_idx]
            im_file = example[-3]
            if type(im_file)==list:
                im_file = im_file[0]
            config_file = os.path.split(os.path.split(im_file)[0])[-1]
            if config_file not in self.same_episode_dict.keys():
                self.same_episode_dict[config_file] = []
            self.same_episode_dict[config_file].append(example_idx)
        
        if self.lm_config.gen_config["eval_mode"]=="human_in_the_loop":
            with open('agent/prompts/prompt_llm_human_feedback.txt') as f:
                intro_human_feedback = f.read()
            feedback_jsons = self.lm_config.gen_config["feedback_jsons"]
            instruction_humanfeedback = json.load(open(feedback_jsons[0]))
            instruction_humanfeedback["examples"] = []
            for instruction_json in feedback_jsons:
                if not os.path.exists(instruction_json):
                    continue
                instruction_humanfeedback_ = json.load(open(instruction_json))
                instruction_humanfeedback_["examples"] = [tuple(e) for e in instruction_humanfeedback_["examples"]]
                instruction_humanfeedback["examples"].extend(instruction_humanfeedback_["examples"])
            instruction_humanfeedback["intro"] = intro_human_feedback
            self.instruction_humanfeedback = instruction_humanfeedback
            self.feedback_embeddings_task, self.feedback_embeddings_obs, self.feedback_embeddings_act, self.feedback_embeddings_visual = self.get_example_embeddings(self.instruction_humanfeedback["examples"])
            self.feedback_examples = self.instruction_humanfeedback["examples"]

    def get_example_embeddings(
        self,
        examples,
    ):

        example_embeddings_task = []
        example_embeddings_obs = []
        example_embeddings_act = []
        example_embeddings_visual = []
        for example in examples:
            im_path = example[2]
            if type(im_path)==list:
                im_path = im_path[-1]
            embed_path_visual = im_path.replace('images', 'embeddings_visual').replace('.png', '.npy')
            embed_path_task = im_path.replace('images', 'embeddings_task').replace('.png', '.npy')
            embed_path_obs = im_path.replace('images', 'embeddings_obs').replace('.png', '.npy')
            embed_path_act = im_path.replace('images', 'embeddings_act').replace('.png', '.npy')
            embed_vis = np.load(embed_path_visual)
            embed_task = np.load(embed_path_task)
            embed_obs = np.load(embed_path_obs)
            embed_act = np.load(embed_path_act)
            example_embeddings_task.append(embed_task)
            example_embeddings_obs.append(embed_obs)
            example_embeddings_act.append(embed_act)
            example_embeddings_visual.append(embed_vis)
        example_embeddings_task = torch.from_numpy(np.asarray(example_embeddings_task))
        example_embeddings_obs = torch.from_numpy(np.asarray(example_embeddings_obs))
        example_embeddings_act = torch.from_numpy(np.asarray(example_embeddings_act))
        example_embeddings_visual = torch.from_numpy(np.asarray(example_embeddings_visual))
        return example_embeddings_task, example_embeddings_obs, example_embeddings_act, example_embeddings_visual

    def construct(
        self,
        trajectory: Trajectory,
        intent: str,
        page_screenshot_img: Image.Image,
        images: list[Image.Image],
        meta_data: dict[str, Any] = {},
        humanFeedback: str = None,
        prev_action = None,
    ) -> APIInput:
        intro = self.instruction["intro"]
        template = self.instruction["template"]
        keywords = self.instruction["meta_data"]["keywords"]
        state_info: StateInfo = trajectory[-1]  # type: ignore[assignment]

        obs = state_info["observation"][self.obs_modality]
        max_obs_length = self.lm_config.gen_config["max_obs_length"]
        if max_obs_length:
            if self.lm_config.provider == "google":
                print("NOTE: This is a Gemini model, so we use characters instead of tokens for max_obs_length.")
                obs = obs[:max_obs_length]
            else:
                obs = self.tokenizer.decode(self.tokenizer.encode(obs)[:max_obs_length])  # type: ignore[arg-type]

        page = state_info["info"]["page"]
        url = page.url

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
        
        previous_action_str = format_action_history(meta_data["action_history"])
        
        current = template.format(
            objective=intent,
            url=self.map_url_to_real(url),
            observation=obs,
            previous_action=previous_action_str,
        )

        assert all([f"{{k}}" not in current for k in keywords])

        observation = current.split('\nOBSERVATION: ')[-1].split('PREVIOUS ACTION:')[0]
        previous_action_text = current.split('OBJECTIVE: ')[-1].split('\n')[-1]
        objective_text = current.split('OBJECTIVE: ')[-1].split('\n')[0]
        url = self.map_url_to_real(url)
        website = None
        if "onestopmarket" in url:
            website = "shopping"
        if "classifieds" in url:
            website = "classifieds"
        if "reddit" in url:
            website = "reddit"

        examples, knowledge = self.retrieve_examples(
            observation,
            previous_action_text,
            objective_text,
            page_screenshot_img,
            self.example_embeddings_task,
            self.example_embeddings_obs,
            self.example_embeddings_act,
            self.example_embeddings_visual,
            self.examples,
            topk=self.topk,
            website=website,
        )
        
        prompt = self.get_lm_api_input(
            intro, examples, current, page_screenshot_img, images, knowledge
        )
        return prompt

    def construct_humanfeedback(
        self,
        trajectory: Trajectory,
        intent: str,
        page_screenshot_img: Image.Image,
        images: list[Image.Image],
        meta_data: dict[str, Any] = {},
        humanFeedback: str = "",
        prev_action = None,
    ) -> APIInput:

        intro = self.instruction_humanfeedback["intro"]
        template = self.instruction_humanfeedback["template"]
        keywords = self.instruction_humanfeedback["meta_data"]["keywords"]
        state_info: StateInfo = trajectory[-1]  # type: ignore[assignment]

        obs = state_info["observation"][self.obs_modality]
        max_obs_length = self.lm_config.gen_config["max_obs_length"]
        if max_obs_length:
            if self.lm_config.provider == "google":
                print("NOTE: This is a Gemini model, so we use characters instead of tokens for max_obs_length.")
                obs = obs[:max_obs_length]
            else:
                obs = self.tokenizer.decode(self.tokenizer.encode(obs)[:max_obs_length])  # type: ignore[arg-type]

        page = state_info["info"]["page"]
        url = page.url

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
        
        previous_action_str = format_action_history(meta_data["action_history"])

        from browser_env.helper_functions import get_action_description
        wrong_action_str = get_action_description(
            prev_action, 
            state_info["info"]["observation_metadata"],
            action_set_tag="som",
            prompt_constructor=self
            )
        wrong_action_str = wrong_action_str.split(' where')[0].replace('[A] ', '')
        
        current = template.format(
            objective=intent,
            url=self.map_url_to_real(url),
            observation=obs,
            previous_action=previous_action_str,
            wrong_action=wrong_action_str,
            human_feedback=humanFeedback
        )

        assert all([f"{{k}}" not in current for k in keywords])

        observation = current.split('\nOBSERVATION: ')[-1].split('PREVIOUS ACTION:')[0]
        previous_action_text = current.split('OBJECTIVE: ')[-1].split('\n')[-1]
        objective_text = current.split('OBJECTIVE: ')[-1].split('\n')[0]

        examples, knowledge = self.retrieve_examples(
            observation,
            previous_action_text,
            objective_text,
            page_screenshot_img,
            self.feedback_embeddings_task,
            self.feedback_embeddings_obs,
            self.feedback_embeddings_act,
            self.feedback_embeddings_visual,
            self.feedback_examples,
            topk=self.topk,
        )
        
        prompt = self.get_lm_api_input(
            intro, examples, current, page_screenshot_img, images, knowledge
        )
        return prompt

    def retrieve_examples(
        self,
        observation,
        previous_action_text,
        objective_text,
        page_screenshot_img,
        example_embeddings_task,
        example_embeddings_obs,
        example_embeddings_act,
        example_embeddings_visual,
        examples,
        topk=5,
        website=None,
        topk_knowledge=15,
        remove_same_episode=True,
    ):
        
        if website is not None:
            objective_text = f"Website: {website}\nObjective: {objective_text}"
        embedding_task = torch.from_numpy(self.embedding_model(objective_text))
        embedding_obs = torch.from_numpy(self.embedding_model(observation))
        embedding_act = torch.from_numpy(self.embedding_model(previous_action_text))
        embedding_vis = self.clip_model.encode_images([page_screenshot_img]).squeeze().cpu() #.cpu().numpy()

        sim_task = self.cos_sim(embedding_task, example_embeddings_task)
        sim_obs = self.cos_sim(embedding_obs, example_embeddings_obs)
        sim_act = self.cos_sim(embedding_act, example_embeddings_act)
        sim_vis = self.cos_sim(embedding_vis, example_embeddings_visual)

        sims = self.w_obs * sim_obs + self.w_act * sim_act + self.w_task * sim_task + self.w_vis * sim_vis
        sims_argsort = torch.argsort(sims, descending=True)
        if remove_same_episode:
            # remove the same episode examples
            sims_argsort_ = copy.deepcopy(list(sims_argsort.cpu().numpy()))
            sims_argsort_topk = []
            while len(sims_argsort_topk)<topk and len(sims_argsort_)>0:
                sims_argsort_topk.append(sims_argsort_[0])
                example = examples[sims_argsort_[0]]
                im_file = example[-3]
                if type(im_file)==list:
                    im_file = im_file[0]
                config_file = os.path.split(os.path.split(im_file)[0])[-1]
                if config_file in self.same_episode_dict:
                    for idx_remove in self.same_episode_dict[config_file]:
                        sims_argsort_.remove(idx_remove)
        else:
            sims_argsort_topk = sims_argsort[:topk]
        examples_selected = []
        for idx_topk in list(sims_argsort_topk):
            examples_selected.append(examples[idx_topk])
        knowledge = set()
        for idx_topk in list(sims_argsort):
            for f_s in examples[idx_topk][3]:
                knowledge.add(f_s)
            if len(knowledge)>=topk_knowledge:
                break
        knowledge = list(knowledge)
        return examples_selected, knowledge

    def get_lm_api_input(
        self,
        intro: str,
        examples: list[tuple[str, str, str]],
        current: str,
        page_screenshot_img: Image.Image,
        images: list[Image.Image],
        knowledge: list,
    ) -> APIInput:
        """Return the require format for an API"""
        message: list[dict[str, str]] | str | list[str | Image.Image]
        if "openai" in self.lm_config.provider:
            if self.lm_config.mode == "chat":
                # add additional knowledge to system prompt
                if self.no_add_abstractions_system:
                    intro_ = intro.split('\n\n**Additional Knowledge**')[0]
                else:
                    knowledge_text = ""
                    count = 1
                    for f_s in knowledge:
                        knowledge_text += f'{count}. {f_s}\n'
                        count += 1
                    if knowledge_text:
                        intro_ = intro.replace('{knowledge}', knowledge_text)
                    else:
                        intro_ = intro.split('\n\n**Additional Knowledge**')[0]                    
                message = [
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": intro_}],
                    }
                ]

                for (x, y, z, f, s) in examples:
                    example_img = Image.open(z)
                    if not self.ablate_image_context:
                        message.append(
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": x},
                                    {
                                        "type": "text",
                                        "text": "IMAGES: (1) current page screenshot",
                                    },
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": pil_to_b64(example_img)
                                        },
                                    },
                                ],
                            }
                        )
                    else:
                        # remove image
                        message.append(
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": x},
                                ],
                            }
                        )
                    message.append(
                        {
                            "role": "assistant",
                            "content": [{"type": "text", "text": y}],
                        }
                    )

                # Encode images and page_screenshot_img as base64 strings.
                current_prompt = current
                content = [
                    {
                        "type": "text",
                        "text": "IMAGES: (1) current page screenshot",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": pil_to_b64(page_screenshot_img)},
                    },
                ]
                for image_i, image in enumerate(images):
                    content.extend(
                        [
                            {
                                "type": "text",
                                "text": f"({image_i+2}) input image {image_i+1}",
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": pil_to_b64(image)},
                            },
                        ]
                    )
                content = [{"type": "text", "text": current_prompt}] + content

                message.append({"role": "user", "content": content})

                observation = current.split('\nOBSERVATION:')[-1].split('PREVIOUS ACTIONS:')[0]
                prev_action = current.split('\nPREVIOUS ACTIONS: ')[-1]
                intent = current.split('OBJECTIVE: ')[-1].split('\n')[0]

                self.prompt_log["current_prompt"] = current_prompt
                self.prompt_log["current_observation"] = observation
                self.prompt_log["current_intent"] = intent
                self.prompt_log["current_prev_actions"] = prev_action
                if not self.no_add_abstractions_system:
                    self.prompt_log["knowledge"] = knowledge_text
                self.prompt_log["intro"] = intro_
                for idx in range(len(examples)):
                    x,y,z,f,s = examples[idx]
                    self.prompt_log[f"example{idx}_input"] = str(x)
                    self.prompt_log[f"example{idx}_gt_action"] = str(y)
                
                return message, self.prompt_log
            else:
                raise ValueError(
                    f"GPT-4V models do not support mode {self.lm_config.mode}"
                )
        elif "google" in self.lm_config.provider:
            if self.lm_config.mode == "completion":
                message = [
                    intro,
                    "Here are a few examples:",
                ]
                for (x, y, z) in examples:
                    example_img = Image.open(z)
                    message.append(f"Observation\n:{x}\n")
                    message.extend(
                        [
                            "IMAGES:",
                            "(1) current page screenshot:",
                            pil_to_vertex(example_img),
                        ]
                    )
                    message.append(f"Action: {y}")
                message.append("Now make prediction given the observation")
                message.append(f"Observation\n:{current}\n")
                message.extend(
                    [
                        "IMAGES:",
                        "(1) current page screenshot:",
                        pil_to_vertex(page_screenshot_img),
                    ]
                )
                for image_i, image in enumerate(images):
                    message.extend(
                        [
                            f"({image_i+2}) input image {image_i+1}",
                            pil_to_vertex(image),
                        ]
                    )
                message.append("Action:")
                return message
            else:
                raise ValueError(
                    f"Gemini models do not support mode {self.lm_config.mode}"
                )
        else:
            raise NotImplementedError(
                f"Provider {self.lm_config.provider} not implemented"
            )

class MultimodalCoTPromptConstructorMemoryAugmentedFULLTRAJECTORY(MultimodalCoTPromptConstructorMemoryAugmented):
    """The agent will perform step-by-step reasoning before the answer"""

    def __init__(
        self,
        instruction_path: str | Path,
        lm_config: lm_config.LMConfig,
        tokenizer: Tokenizer,
        cache=True,
    ):  
        from llms.providers.openai_utils import run_embedding_model
        super().__init__(instruction_path, lm_config, tokenizer)

        with open('agent/prompts/intro_full_trajectory.txt') as f:
            intro = f.read()
        self.instruction["intro"] = intro

        for idx in range(self.topk):
            del self.prompt_log[f"example{idx}_gt_action"]

    def get_lm_api_input(
        self,
        intro: str,
        examples: list[tuple[str, str, str]],
        current: str,
        page_screenshot_img: Image.Image,
        images: list[Image.Image],
        knowledge: list,
    ) -> APIInput:
        """Return the require format for an API"""
        message: list[dict[str, str]] | str | list[str | Image.Image]
        if "openai" in self.lm_config.provider:
            if self.lm_config.mode == "chat":
                knowledge_text = ''

                intro_ = intro

                example_text = ''
                example_number = 1
                for (x, y, z, f, s, l) in examples:
                    example_text += f"\n\n<start example {example_number}>:\n\n{y}\n\n<end example {example_number}>"
                    example_number += 1
                
                intro_ = intro_.replace('{RETRIEVED_EXAMPLES}', example_text)

                print(f"Intro token length: {len(self.tokenizer.encode(intro_))}")
                print(f"Example token length: {len(self.tokenizer.encode(example_text))}")
                print(f"Current prompt token length: {len(self.tokenizer.encode(current))}")

                message = [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": intro_}],
                    }
                ]

                # Encode images and page_screenshot_img as base64 strings.
                current_prompt = current
                content = [
                    {
                        "type": "text",
                        "text": "IMAGES: (1) current page screenshot",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": pil_to_b64(page_screenshot_img)},
                    },
                ]
                for image_i, image in enumerate(images):
                    content.extend(
                        [
                            {
                                "type": "text",
                                "text": f"({image_i+2}) input image {image_i+1}",
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": pil_to_b64(image)},
                            },
                        ]
                    )
                content = [{"type": "text", "text": current_prompt}] + content

                message.append({"role": "user", "content": content})

                observation = current.split('\nOBSERVATION: ')[-1].split('PREVIOUS ACTION:')[0]
                prev_action = current.split('\nPREVIOUS ACTION: ')[-1]
                intent = current.split('OBJECTIVE: ')[-1].split('\n')[0]

                self.prompt_log["current_prompt"] = current_prompt
                self.prompt_log["current_observation"] = observation
                self.prompt_log["current_intent"] = intent
                self.prompt_log["current_prev_actions"] = prev_action
                self.prompt_log["knowledge"] = knowledge_text
                self.prompt_log["intro"] = intro_
                for idx in range(len(examples)):
                    x, y, z, f, s, l = examples[idx]
                    self.prompt_log[f"example{idx}_input"] = str(x)

                return message, self.prompt_log
            else:
                raise ValueError(
                    f"GPT-4V models do not support mode {self.lm_config.mode}"
                )
        elif "google" in self.lm_config.provider:
            if self.lm_config.mode == "completion":
                message = [
                    intro,
                    "Here are a few examples:",
                ]
                for (x, y, z) in examples:
                    example_img = Image.open(z)
                    message.append(f"Observation\n:{x}\n")
                    message.extend(
                        [
                            "IMAGES:",
                            "(1) current page screenshot:",
                            pil_to_vertex(example_img),
                        ]
                    )
                    message.append(f"Action: {y}")
                message.append("Now make prediction given the observation")
                message.append(f"Observation\n:{current}\n")
                message.extend(
                    [
                        "IMAGES:",
                        "(1) current page screenshot:",
                        pil_to_vertex(page_screenshot_img),
                    ]
                )
                for image_i, image in enumerate(images):
                    message.extend(
                        [
                            f"({image_i+2}) input image {image_i+1}",
                            pil_to_vertex(image),
                        ]
                    )
                message.append("Action:")
                return message
            else:
                raise ValueError(
                    f"Gemini models do not support mode {self.lm_config.mode}"
                )
        else:
            raise NotImplementedError(
                f"Provider {self.lm_config.provider} not implemented"
            )

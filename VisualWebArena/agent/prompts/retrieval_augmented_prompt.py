import json
from pathlib import Path
from PIL import ImageFile, Image
ImageFile.LOAD_TRUNCATED_IMAGES = True

from browser_env import Action, Trajectory
from browser_env.utils import StateInfo, pil_to_b64, pil_to_vertex
from llms import lm_config
from llms.tokenizers import Tokenizer
from llms.utils import APIInput
from clip import CLIP

import numpy as np
import os
import torch
import copy
from agent.prompts.prompt_constructor import CoTPromptConstructor

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
            with open('agent/prompts/system_prompts/intro_full_actions.txt') as f:
                intro = f.read()
        else:
            with open('agent/prompts/system_prompts/intro.txt') as f:
                intro = f.read()
        self.instruction["intro"] = intro

        # text embedding model (openai)
        self.embedding_model = run_embedding_model

        # image embedding model (clip)
        self.clip_model = CLIP()

        self.examples = self.instruction["examples"]
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

        with open('agent/prompts/system_prompts/intro.txt') as f:
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
        if 'PREVIOUS ACTIONS: ' in previous_action_text:
            previous_action_text = previous_action_text.replace('PREVIOUS ACTIONS: ', '')
        if website is not None:
            objective_text = f"Website: {website}\nObjective: {objective_text}"
        embedding_task = torch.from_numpy(self.embedding_model(objective_text))
        observation = observation.split('OBSERVATION:\n')[1].split('\n\nURL')[0]
        embedding_obs = torch.from_numpy(self.embedding_model(observation))
        embedding_act = torch.from_numpy(self.embedding_model(previous_action_text))
        embedding_vis = self.clip_model.encode_images([page_screenshot_img]).squeeze().cpu() #.cpu().numpy()

        # cosine similarity: sim(x, y) = (x * y) / (||x|| * ||y||)
        sim_task = self.cos_sim(embedding_task, example_embeddings_task)
        sim_obs = self.cos_sim(embedding_obs, example_embeddings_obs)
        sim_act = self.cos_sim(embedding_act, example_embeddings_act)
        sim_vis = self.cos_sim(embedding_vis, example_embeddings_visual)

        sims = self.w_obs * sim_obs + self.w_act * sim_act + self.w_task * sim_task + self.w_vis * sim_vis
        sims_argsort = torch.argsort(sims, descending=True)
        print("sim_obs", sim_obs)
        print("sim_act", sim_act)
        print("sim_task", sim_task)
        print("sim_vis", sim_vis)
        print("sims", sims)
        if remove_same_episode and len(example_embeddings_task)>topk*2:
            # if True, ensure topk time steps are from different episodes
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
                        if idx_remove in sims_argsort_:
                            sims_argsort_.remove(idx_remove)
            if len(sims_argsort_topk)<topk and len(sims_argsort)>=topk:
                sims_argsort_ = copy.deepcopy(list(sims_argsort.cpu().numpy()))
                # remove sims_argsort_topk from sims_argsort_
                for idx_remove in sims_argsort_topk:
                    if idx_remove in sims_argsort_:
                        sims_argsort_.remove(idx_remove)
                sims_argsort_topk_to_add = sims_argsort_[:topk-len(sims_argsort_topk)]
                sims_argsort_topk = sims_argsort_topk + sims_argsort_topk_to_add
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
        return examples_selected[::-1], knowledge

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

                for idx, (x, y, z, f, s) in enumerate(examples):
                    example_img = Image.open(z)
                    if not self.ablate_image_context:
                        message.append(
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": f"## EXAMPLE INPUT {idx+1}:\n{x}"},
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
                                    {"type": "text", "text": f"## EXAMPLE INPUT {idx+1}:\n{x}"},
                                ],
                            }
                        )
                    message.append(
                        {
                            "role": "assistant",
                            "content": [{"type": "text", "text": f"## EXAMPLE OUTPUT {idx+1}:\n{y}"}],
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
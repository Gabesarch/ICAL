import json
import os
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
import torch
import copy

import faiss

from agent.prompts.prompt_constructor import CoTPromptConstructor


#################################
# Helper function to unify embeddings
# This function returns w_obs * e_obs + w_act * e_act + w_task * e_task + w_vis * e_vis
#################################
def unify_embeddings(
    e_task: np.ndarray,
    e_obs: np.ndarray,
    e_act: np.ndarray,
    e_vis: np.ndarray,
    w_obs: float,
    w_act: float,
    w_task: float,
    w_vis: float,
) -> np.ndarray:
    """
    Return a single unified embedding from the four embeddings (task, obs, act, visual).
    """
    # We assume all embeddings are 1D vectors of the same dimension.
    # Convert to float32 to ensure Faiss compatibility if needed.
    return (w_obs * e_obs 
            + w_act * e_act 
            + w_task * e_task 
            + w_vis * e_vis).astype(np.float32)


#################################
# Simple LRU or dictionary-based cache
# for text embeddings to prevent repeated calls.
#################################
EMBEDDING_CACHE = {}

def cached_text_embedding(text: str, embed_func):
    """
    If text is in cache, return cached embedding. Otherwise, compute and store.
    """
    if text in EMBEDDING_CACHE:
        return EMBEDDING_CACHE[text]
    emb = embed_func(text)
    EMBEDDING_CACHE[text] = emb
    return emb


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
        self.embedding_model = run_embedding_model  # function for text embeddings

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

        #################################
        # Store weights
        #################################
        w_obs, w_act, w_task, w_vis = (
            self.lm_config.gen_config["w_obs"], 
            self.lm_config.gen_config["w_act"], 
            self.lm_config.gen_config["w_task"], 
            self.lm_config.gen_config["w_vis"],
        )
        self.w_obs, self.w_act, self.w_task, self.w_vis = w_obs, w_act, w_task, w_vis

        #################################
        # We'll build Faiss indexes for examples and feedback examples
        # after loading them in refresh_examples.
        #################################
        self.example_faiss_index = None
        self.example_unified_embeddings = None

        self.feedback_faiss_index = None
        self.feedback_unified_embeddings = None

        self.refresh_examples()


    def refresh_examples(
        self,
    ):
        """
        Loads all examples from instruction_jsons, unifies embeddings, builds a Faiss index, etc.
        """
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

        #################################
        # Get example embeddings
        #################################
        (
            self.example_embeddings_task,
            self.example_embeddings_obs,
            self.example_embeddings_act,
            self.example_embeddings_visual
        ) = self.get_example_embeddings(self.instruction["examples"])
        self.examples = self.instruction["examples"]

        #################################
        # Unify embeddings for each example and build Faiss index
        #################################
        self.example_unified_embeddings = []
        for i in range(len(self.examples)):
            # w_obs * e_obs + w_act * e_act + w_task * e_task + w_vis * e_vis
            e = unify_embeddings(
                self.example_embeddings_task[i].numpy(),
                self.example_embeddings_obs[i].numpy(),
                self.example_embeddings_act[i].numpy(),
                self.example_embeddings_visual[i].numpy(),
                self.w_obs, self.w_act, self.w_task, self.w_vis
            )
            self.example_unified_embeddings.append(e)

        self.example_unified_embeddings = np.vstack(self.example_unified_embeddings)  # shape [N, D]
        
        # Build a Faiss index for these unified embeddings
        if self.example_unified_embeddings.shape[0] > 0:
            d = self.example_unified_embeddings.shape[1]
            self.example_faiss_index = faiss.IndexFlatIP(d)  # IP = Inner Product
            self.example_faiss_index.add(self.example_unified_embeddings)

        #################################
        # Prepare same_episode_dict for removing same-episode examples
        #################################
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
        
        #################################
        # If in human_in_the_loop mode, do the same for feedback examples
        #################################
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
                instruction_humanfeedback_["examples"] = [
                    tuple(e) for e in instruction_humanfeedback_["examples"]
                ]
                instruction_humanfeedback["examples"].extend(instruction_humanfeedback_["examples"])
            instruction_humanfeedback["intro"] = intro_human_feedback
            self.instruction_humanfeedback = instruction_humanfeedback

            (
                self.feedback_embeddings_task,
                self.feedback_embeddings_obs,
                self.feedback_embeddings_act,
                self.feedback_embeddings_visual
            ) = self.get_example_embeddings(self.instruction_humanfeedback["examples"])
            self.feedback_examples = self.instruction_humanfeedback["examples"]

            # Unify and build Faiss index
            self.feedback_unified_embeddings = []
            for i in range(len(self.feedback_examples)):
                e = unify_embeddings(
                    self.feedback_embeddings_task[i].numpy(),
                    self.feedback_embeddings_obs[i].numpy(),
                    self.feedback_embeddings_act[i].numpy(),
                    self.feedback_embeddings_visual[i].numpy(),
                    self.w_obs, self.w_act, self.w_task, self.w_vis
                )
                self.feedback_unified_embeddings.append(e)
            self.feedback_unified_embeddings = np.vstack(self.feedback_unified_embeddings)

            if self.feedback_unified_embeddings.shape[0] > 0:
                d_fb = self.feedback_unified_embeddings.shape[1]
                self.feedback_faiss_index = faiss.IndexFlatIP(d_fb)
                self.feedback_faiss_index.add(self.feedback_unified_embeddings)


    def get_example_embeddings(
        self,
        examples,
    ):
        """
        Loads example embeddings from .npy files for each input example.
        We store them as torch Tensors (N, D).
        """
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
        meta_data: dict[str, any] = {},
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

        def format_action_history(action_history):
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

        # Extract pieces of current for logging or further usage
        observation = current.split('\nOBSERVATION: ')[-1].split('PREVIOUS ACTION:')[0]
        previous_action_text = current.split('OBJECTIVE: ')[-1].split('\n')[-1]
        objective_text = current.split('OBJECTIVE: ')[-1].split('\n')[0]

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
            self.example_faiss_index,
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
        meta_data: dict[str, any] = {},
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

        def format_action_history(action_history):
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
            self.feedback_faiss_index,
            self.feedback_examples,
            topk=self.topk,
        )
        
        prompt = self.get_lm_api_input(
            intro, examples, current, page_screenshot_img, images, knowledge
        )
        return prompt


    def retrieve_examples(
        self,
        observation: str,
        previous_action_text: str,
        objective_text: str,
        page_screenshot_img: Image.Image,
        faiss_index: faiss.IndexFlatIP,
        examples,
        topk=5,
        website=None,
        topk_knowledge=15,
        remove_same_episode=True,
    ):
        """
        Now uses Faiss to retrieve top-k examples from a unified embedding index.
        """
        if faiss_index is None or len(examples) == 0:
            return [], []

        if 'PREVIOUS ACTIONS: ' in previous_action_text:
            previous_action_text = previous_action_text.replace('PREVIOUS ACTIONS: ', '')

        # Prepend website info if present
        if website is not None:
            objective_text = f"Website: {website}\nObjective: {objective_text}"

        # Cache text embeddings
        embedding_task = cached_text_embedding(objective_text, self.embedding_model)
        embedding_obs = cached_text_embedding(observation, self.embedding_model)
        embedding_act = cached_text_embedding(previous_action_text, self.embedding_model)

        # Clip image embedding
        embedding_vis = self.clip_model.encode_images([page_screenshot_img]).squeeze().cpu().numpy()

        # Now unify these four embeddings
        query_unified = unify_embeddings(
            embedding_task,
            embedding_obs,
            embedding_act,
            embedding_vis,
            self.w_obs, self.w_act, self.w_task, self.w_vis
        )
        # shape (1, D)
        query_unified = np.expand_dims(query_unified, axis=0)

        # Perform Faiss search for top matches across all examples
        # We search for all to handle remove_same_episode logic, then pick topk
        D, I = faiss_index.search(query_unified, len(examples))  
        # D and I are each of shape (1, N), we want the top ranks from I[0]
        candidates = I[0]
        # distances = D[0]  # If you want to see similarities

        # Implement remove_same_episode by skipping items in the same episode
        # We’ll still keep track in rank order
        final_example_indices = []
        candidates_list = list(candidates)
        if remove_same_episode and len(examples) > topk*2:
            while len(final_example_indices) < topk and len(candidates_list) > 0:
                idx_candidate = candidates_list.pop(0)
                example = examples[idx_candidate]
                im_file = example[-3]
                if type(im_file)==list:
                    im_file = im_file[0]
                config_file = os.path.split(os.path.split(im_file)[0])[-1]
                
                # only add if not from same episode
                if config_file in self.same_episode_dict:
                    # if it collides with an already chosen example, skip it
                    # but we only skip if the example was from the *same episode*
                    # that we already selected
                    # We'll assume here we skip it if we haven't selected that config_file yet
                    already_selected_configs = set()
                    for chosen_idx in final_example_indices:
                        chosen_file = examples[chosen_idx][-3]
                        if type(chosen_file)==list:
                            chosen_file = chosen_file[0]
                        chosen_config_file = os.path.split(os.path.split(chosen_file)[0])[-1]
                        already_selected_configs.add(chosen_config_file)

                    if config_file not in already_selected_configs:
                        final_example_indices.append(idx_candidate)
                else:
                    # if config_file not in dictionary, just add
                    final_example_indices.append(idx_candidate)
            
            # If we didn't get enough distinct episodes, we fill up from the top
            if len(final_example_indices) < topk and len(candidates) >= topk:
                # put back any we haven't used
                unused_candidates = [
                    c for c in I[0] if c not in final_example_indices
                ]
                # take the top (topk - len(final_example_indices)) from unused
                final_example_indices += list(unused_candidates[: (topk - len(final_example_indices))])
        else:
            # simpler path if we are not removing or don't have enough examples
            final_example_indices = candidates[:topk]

        # Reverse them if you want them in ascending order of similarity, etc.
        # For demonstration, we'll keep the natural order (most similar first).
        # You can do final_example_indices[::-1] if you want to invert them.
        examples_selected = [examples[idx] for idx in final_example_indices]

        # Collect knowledge from top example indices
        knowledge = set()
        # We'll just iterate in the order of top similarity
        for idx_topk in candidates:
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
        """
        Return the required format for an API. 
        This part hasn't changed much, except for references to the newly changed code.
        """
        if "openai" in self.lm_config.provider:
            if self.lm_config.mode == "chat":
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

                # Logging for debugging
                observation = current.split('\nOBSERVATION:')[-1].split('PREVIOUS ACTIONS:')[0]
                prev_action = current.split('\nPREVIOUS ACTIONS: ')[-1] if 'PREVIOUS ACTIONS: ' in current else ""
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

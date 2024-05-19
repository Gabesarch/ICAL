import pickle

import os
import json
import numpy as np
from browser_env.actions import action2str
from PIL import Image  
import PIL  
import ipdb
from browser_env.utils import StateInfo, pil_to_b64, pil_to_vertex
from tqdm import tqdm
import random

from browser_env import (
    Action,
    ActionTypes,
    ScriptBrowserEnv,
    StateInfo,
    Trajectory,
    create_stop_action,
)

from clip import CLIP 

import ipdb
st = ipdb.set_trace
do_state_abstraction = True

from llms.providers.openai_utils import generate_from_openai_chat_completion

output_default = "In summary, the next action I will perform is ```{ACTION}```"

def get_state_reduced(
    state,
    element_id,
    prob = 0.2,
):
    state_lines = state.split('\n')
    element_id_text = '[' + str(element_id) + ']'
    state_lines_reduced = []
    element_id_line = ''
    for s_line in state_lines:
        if element_id_text in s_line:
            state_lines_reduced.append(s_line)
            element_id_line = s_line
        elif np.random.uniform() < prob:
            state_lines_reduced.append(s_line)
    return '\n'.join(state_lines_reduced), element_id_line

def save_image(
    image,
    task_id,
    time_step,
    image_save_root,
):
    im = Image.fromarray(image)
    path_root = os.path.join(image_save_root, str(task_id)) #, f'{time_step}.png')
    os.makedirs(path_root, exist_ok=True)
    im_path = os.path.join(path_root, f'{time_step}.png')
    im.save(im_path)

    return im_path

def format_action_history(
    action_history
):
    if len(action_history)==0:
        return 'None'
    else:
        action_text = ''
        count = 1
        for action in action_history:
            action_text += f'\n{count}. {action}'
            count += 1
        return action_text

def get_action_history_format(
    action_str,
    element_id,
    element_id_line,
):
    element_id_text = '[' + str(element_id) + ']'
    element_id_replaced = element_id_line.replace(element_id_text, '')
    if element_id_replaced and element_id_replaced[0]==' ':
        element_id_replaced = element_id_replaced[1:]
    action_str_history = action_str.replace(element_id_text, element_id_replaced)
    return action_str_history

instruction_path = "agent/prompts/jsons/p_som_cot_id_actree_3s.json"
instruction = json.load(open(instruction_path))
instruction["examples"] = [tuple(e) for e in instruction["examples"]]

clip_model = CLIP() 

with open('agent/prompts/prompt_llm_cleanup.txt') as f:
    intro = f.read()

current_template = "OBJECTIVE: {objective}\n\nOBSERVATION: {observation}\n\nPREVIOUS ACTION: {previous_action}\n\nEXPERT ACTION: {expert_action}"

def get_prompt(
    objective,
    observation,
    previous_action,
    expert_action,
    page_screenshot_img_path,
):
    examples = instruction["examples"]

    message = [
        {
            "role": "system",
            "content": [{"type": "text", "text": intro}],
        }
    ]
    for (x, y, z) in examples:
        example_img = Image.open(z)
        expert_action_example = y.split(' In summary, the next action I will perform is ')[-1]
        explanation = y.split(' In summary, the next action I will perform is')[0]
        example_info = f'{x}\n\nEXPERT ACTION: {expert_action_example}'
        message.append(
            {
                "role": "system",
                "name": "example_user",
                "content": [
                    {"type": "text", "text": example_info},
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
                "content": [{"type": "text", "text": explanation}],
            }
        )

    # Encode images and page_screenshot_img as base64 strings.
    current = current_template.replace('{objective}', f'{objective}')
    current = current.replace('{observation}', f'{observation}')
    current = current.replace('{previous_action}', f'{previous_action}')
    current = current.replace('{expert_action}', f'{expert_action}')
    current_prompt = current
    page_screenshot_img = Image.open(page_screenshot_img_path)
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
    content = [{"type": "text", "text": current_prompt}] + content

    message.append({"role": "user", "content": content})
    return message

instruction_next_state_path = "agent/prompts/jsons/p_predict_next_state.json"
instruction_next_state = json.load(open(instruction_next_state_path))
instruction_next_state["examples"] = [tuple(e) for e in instruction_next_state["examples"]]
template_next_state = instruction_next_state['template']
with open('agent/prompts/prompt_predicted_state.txt') as f:
    intro_next_state = f.read()

def get_prompt_predicted_state(
    objective,
    observation_t,
    observation_t_plus_1,
    expert_action,
    page_screenshot_img_numpy_t,
    page_screenshot_img_numpy_t_plus_1,
):
    examples = instruction_next_state["examples"]

    message = [
        {
            "role": "system",
            "content": [{"type": "text", "text": intro_next_state}],
        }
    ]
    for (x, y, z, z2) in examples:
        example_img_t = Image.open(z)
        example_img_t_plus_1 = Image.open(z2)
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
                            "url": pil_to_b64(example_img_t)
                        },
                    },
                    {
                        "type": "text",
                        "text": "IMAGES: (1) next page screenshot",
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": pil_to_b64(example_img_t_plus_1)
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
    current = template_next_state.replace('{objective}', f'{objective}')
    current = current.replace('{observation_current}', f'{observation_t}')
    current = current.replace('{observation_next}', f'{observation_t_plus_1}')
    current = current.replace('{expert_action}', f'{expert_action}')
    current_prompt = current
    page_screenshot_img_t = Image.fromarray(page_screenshot_img_numpy_t)
    page_screenshot_img_t_plus_1 = Image.fromarray(page_screenshot_img_numpy_t_plus_1)
    content = [
        {
            "type": "text",
            "text": "IMAGES: (1) current page screenshot",
        },
        {
            "type": "image_url",
            "image_url": {
                "url": pil_to_b64(page_screenshot_img_t)
                },
        },
        {
            "type": "text",
            "text": "IMAGES: (1) next page screenshot",
        },
        {
            "type": "image_url",
            "image_url": {
                "url": pil_to_b64(page_screenshot_img_t_plus_1)
            },
        },
    ]
    content = [{"type": "text", "text": current_prompt}] + content

    message.append({"role": "user", "content": content})
    return message

template_state_abstraction = "OBJECTIVE: {objective}\n\nOBSERVATION:\n{observation_current}"
with open('agent/prompts/prompt_state_abstraction.txt') as f:
    intro_state_abstraction = f.read()

def get_prompt_abstracted_state(
    objective,
    observation_t,
    page_screenshot_img_numpy_t,
    element_id,
):

    message = [
        {
            "role": "system",
            "content": [{"type": "text", "text": intro_state_abstraction}],
        }
    ]

    state_lines = observation_t.split('\n')
    element_id_text = '[' + str(element_id) + ']'
    state_lines_formatted = []
    count = 1
    for s_line in state_lines:
        if element_id_text in s_line:
            element_id_count = count
        state_lines_formatted.append(f'{count}.) {s_line}')
        count += 1
    state_lines_formatted_str = '\n'.join(state_lines_formatted)

    # Encode images and page_screenshot_img as base64 strings.
    current = template_state_abstraction.replace('{objective}', f'{objective}')
    current = current.replace('{observation}', f'{observation_t}')
    current_prompt = current
    page_screenshot_img_t = Image.fromarray(page_screenshot_img_numpy_t)
    content = [
        {
            "type": "text",
            "text": "IMAGES: (1) current page screenshot",
        },
        {
            "type": "image_url",
            "image_url": {"url": pil_to_b64(page_screenshot_img_t)},
        },
    ]
    content = [{"type": "text", "text": current_prompt}] + content

    message.append({"role": "user", "content": content})
    return message, state_lines_formatted, element_id_count

def save_example(
    actions_example,
    states_example,
    images_example,
    humanFeedback_example,
    score,
    task_json_file,
    example_json_file,
    image_save_root,
    feedback_json_file,
    image_feedback_save_root,
    action_set_tag="som",
):

    with open(task_json_file) as json_file:
        task_json = json.load(json_file)
    
    with open("agent/prompts/jsons/p_multimodal_humanfeedback.json") as json_file:
        human_feedback_json = json.load(json_file)

    if os.path.exists(example_json_file):
        with open(example_json_file) as json_file:
            base_json = json.load(json_file)
    else:
        base_json = {}
        base_json["examples"] = []

    if os.path.exists(feedback_json_file):
        with open(feedback_json_file) as json_file:
            feedback_base_json = json.load(json_file)
    else:
        feedback_base_json = {}
        feedback_base_json["examples"] = []

    dataset_tag = task_json["sites"][0]

    action_history = []
    states_reduced = []
    for time_step in range(len(actions_example)):
        action_t = actions_example[time_step]
        state_t = states_example[time_step]
        image_t = images_example[time_step]
        human_feedback_t = humanFeedback_example[time_step]
        action_str = action2str(action_t, action_set_tag)
        action_str = action_str.split(' where')[0].replace('[A] ', '').replace(' [A]', '')
        if do_state_abstraction:
            try:
                prompt, state_lines_formatted, element_id_count = get_prompt_abstracted_state(
                    task_json["intent"],
                    state_t,
                    image_t,
                    action_t['element_id'],
                )
                response = generate_from_openai_chat_completion(
                    prompt,
                    "gpt-4-vision-preview",
                    0.2,
                    4000,
                    0.1,
                    8000,
                )
                numbers = eval(response.replace('Most Relevant State: ', ''))
                numbers.append(element_id_count)
                numbers = list(set(numbers))
                random.shuffle(numbers)
                state_reduced = []
                for number in numbers:
                    state_reduced.append(state_lines_formatted[number-1].split('.)')[-1])
                state_reduced = '\n'.join(state_reduced)
                element_id_line = state_lines_formatted[element_id_count-1].split('.)')[-1]
            except:
                state_reduced, element_id_line = get_state_reduced(state_t, action_t['element_id'])
        else:
            state_reduced, element_id_line = get_state_reduced(state_t, action_t['element_id'])
        action_str_history = get_action_history_format(action_str, action_t['element_id'], element_id_line)
        action_history.append(action_str_history)
        states_reduced.append(state_reduced)

    feedback_abstractions = []
    for time_step in tqdm(range(len(actions_example)), leave=False):
        if humanFeedback_example[time_step]:
            for humanFeedback in humanFeedback_example[time_step]:
                action_t = humanFeedback[-1]
                feedback_abstractions.append(action_t['raw_prediction'].split('Correction Abstraction: ')[-1].split('\n\nPlan')[0])
    
    for time_step in tqdm(range(len(actions_example)), leave=False):
        action_t = actions_example[time_step]
        state_t = states_example[time_step]
        image_t = images_example[time_step]
        feedback_t = humanFeedback_example[time_step]
        action_str = action2str(action_t, action_set_tag)
        action_str = action_str.split(' where')[0].replace('[A] ', '').replace(' [A]', '')
        action_formatted = output_default.replace('{ACTION}', action_str)
        im_path = save_image(image_t, f"{dataset_tag}_{task_json['task_id']}", time_step, image_save_root)
        action_history_formatted = format_action_history(action_history[:time_step])
        if time_step==len(actions_example)-1:
            if action_t["action_type"] == ActionTypes.STOP:
                next_state = "The episode will end after the stop action is issued and any information provided will be returned to the user."
            else:
                continue
        else:
            # given current & future frame - describe next state
            prompt = get_prompt_predicted_state(
                task_json["intent"],
                state_t,
                states_example[time_step+1],
                action_str,
                image_t,
                images_example[time_step+1],
            )
            response = generate_from_openai_chat_completion(
                prompt,
                "gpt-4-vision-preview",
                0.2,
                4000,
                0.1,
                8000,
            )
            next_state = response.replace('SUMMARY: ', '')
        state_reduced_t = states_reduced[time_step]
        state_example = f'OBJECTIVE: {task_json["intent"]}\n\nOBSERVATION:\n{state_reduced_t}\n\nPREVIOUS ACTIONS: {action_history_formatted}'
        plan = action_t['raw_prediction'].split('\n\nPlan: ')[-1].split('\n\nSummary: ')[0]
        summary = action_t['raw_prediction'].split('\n\nSummary: ')[-1].split('\n\nRevised Action:')[0]
        output_formatted = f"Plan: {plan}\n\nSummary: {summary}\n\nPredicted Next State: {next_state}\n\nAction: {action_formatted}"
        example_to_save = [state_example, output_formatted, im_path, feedback_abstractions, score] # last one is abstracted knowledge
        base_json['examples'].append(example_to_save)

        if feedback_t:
            im_path_feedback = save_image(image_t, f"{dataset_tag}_{task_json['task_id']}", time_step, image_feedback_save_root)

            feedback_t_last = feedback_t[-1]
            action_t_wrong = feedback_t_last[3]
            action_str_wrong = action2str(action_t_wrong, action_set_tag)
            action_str_wrong = action_str.split(' where')[0].replace('[A] ', '').replace(' [A]', '')

            state_feedback = human_feedback_json["template"]
            state_feedback = state_feedback.replace('{objective}', task_json["intent"])
            state_feedback = state_feedback.replace('{observation}', state_reduced_t)
            state_feedback = state_feedback.replace('\n\nURL: {url}', '')
            state_feedback = state_feedback.replace('{previous_action}', action_history_formatted)
            state_feedback = state_feedback.replace('{wrong_action}', action_str_wrong)
            state_feedback = state_feedback.replace('{human_feedback}', feedback_t_last[4])

            explanation = action_t['raw_prediction'].split('Explain: ')[-1].split('\n\nCorrection Abstraction:')[0]
            correction_abstraction = action_t['raw_prediction'].split('Correction Abstraction: ')[-1].split('\n\nPlan')[0]
            output_feedback = human_feedback_json["template_output"]
            output_feedback = output_feedback.replace('{explain}', explanation)
            output_feedback = output_feedback.replace('{correction_abstraction}', correction_abstraction)
            output_feedback = output_feedback.replace('{plan}', plan)
            output_feedback = output_feedback.replace('{summary}', summary)
            output_feedback = output_feedback.replace('{next_state}', next_state)
            output_feedback = output_feedback.replace('{new_action}', action_formatted)

            feedback_example_to_save = [state_feedback, output_feedback, im_path_feedback, feedback_abstractions, score] 
            feedback_base_json["examples"].append(feedback_example_to_save)

    # self.example_embeddings = []
    from llms.providers.openai_utils import run_embedding_model
    for example in tqdm(base_json['examples']):
        current = example[0]
        im_path = example[2]
        observation = current.split('OBSERVATION:\n')[-1].split('\n\nPREVIOUS ACTIONS:')[0]
        previous_action_text = current.split('OBJECTIVE: ')[-1].split('\n')[-1]
        objective_text = current.split('OBJECTIVE: ')[-1].split('\n')[0]
        # to_embed = f'Objective: {objective_text}\n\nObservation:\n{observation}\n\nPrevious actions: {previous_action_text}'

        images = [Image.open(im_path)]
        
        embed_path_visual = im_path.replace('images', 'embeddings_visual').replace('.png', '.npy')
        os.makedirs(os.path.split(embed_path_visual)[0], exist_ok=True)
        if not os.path.exists(embed_path_visual):
            visual_feature_embedding = clip_model.encode_images(images).squeeze().cpu().numpy()
            np.save(embed_path_visual, visual_feature_embedding)
        
        embed_path_task = im_path.replace('images', 'embeddings_task').replace('.png', '.npy')
        os.makedirs(os.path.split(embed_path_task)[0], exist_ok=True)
        if not os.path.exists(embed_path_task):
            embedding_task = run_embedding_model(objective_text)
            np.save(embed_path_task, embedding_task)

        embed_path_obs = im_path.replace('images', 'embeddings_obs').replace('.png', '.npy')
        os.makedirs(os.path.split(embed_path_obs)[0], exist_ok=True)
        if not os.path.exists(embed_path_obs):
            embedding_obs = run_embedding_model(observation)
            np.save(embed_path_obs, embedding_obs)

        embed_path_act = im_path.replace('images', 'embeddings_act').replace('.png', '.npy')
        os.makedirs(os.path.split(embed_path_act)[0], exist_ok=True)
        if not os.path.exists(embed_path_act):
            embedding_act = run_embedding_model(previous_action_text)
            np.save(embed_path_act, embedding_act)

    with open(example_json_file, "w") as outfile: 
        json.dump(base_json, outfile, indent=4, sort_keys=True)

    for example in tqdm(feedback_base_json['examples']):
        current = example[0]
        im_path = example[2]
        observation = current.split('OBSERVATION:\n')[-1].split('\n\nPREVIOUS ACTIONS:')[0]
        previous_action_text = current.split('OBJECTIVE: ')[-1].split('\n')[-1]
        objective_text = current.split('OBJECTIVE: ')[-1].split('\n')[0]
        # to_embed = f'Objective: {objective_text}\n\nObservation:\n{observation}\n\nPrevious actions: {previous_action_text}'

        images = [Image.open(im_path)]
        
        embed_path_visual = im_path.replace('images', 'embeddings_visual').replace('.png', '.npy')
        os.makedirs(os.path.split(embed_path_visual)[0], exist_ok=True)
        if not os.path.exists(embed_path_visual):
            visual_feature_embedding = clip_model.encode_images(images).squeeze().cpu().numpy()
            np.save(embed_path_visual, visual_feature_embedding)

        # embedding = w_obs * embedding_obs + w_act * embedding_act + w_task * embedding_task
        
        embed_path_task = im_path.replace('images', 'embeddings_task').replace('.png', '.npy')
        os.makedirs(os.path.split(embed_path_task)[0], exist_ok=True)
        if not os.path.exists(embed_path_task):
            embedding_task = run_embedding_model(objective_text)
            np.save(embed_path_task, embedding_task)

        embed_path_obs = im_path.replace('images', 'embeddings_obs').replace('.png', '.npy')
        os.makedirs(os.path.split(embed_path_obs)[0], exist_ok=True)
        if not os.path.exists(embed_path_obs):
            embedding_obs = run_embedding_model(observation)
            np.save(embed_path_obs, embedding_obs)

        embed_path_act = im_path.replace('images', 'embeddings_act').replace('.png', '.npy')
        os.makedirs(os.path.split(embed_path_act)[0], exist_ok=True)
        if not os.path.exists(embed_path_act):
            embedding_act = run_embedding_model(previous_action_text)
            np.save(embed_path_act, embedding_act)

    with open(feedback_json_file, "w") as outfile: 
        json.dump(feedback_base_json, outfile, indent=4, sort_keys=True)    
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
import sys
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import copy
import tiktoken

from clip import CLIP 

from browser_env import (
    Action,
    ActionTypes,
    ScriptBrowserEnv,
    StateInfo,
    Trajectory,
    create_stop_action,
)

st = ipdb.set_trace
pickle_path = 'data'
image_save_root = 'data/examples_redone/images'

new_folder = './data/memory_human_in_the_loop/merged_full_trajectory_state_abstraction_v2'

action_set_tag = "som"
do_state_abstraction = True
reset_examples = False
add_knowledge = True
clip_model = CLIP() 

os.makedirs(new_folder, exist_ok=True)

files_to_merge = [
    "JSON_FILES_WITH_DEMO_INFORMATION"
]

from llms.providers.openai_utils import generate_from_openai_chat_completion

with open(files_to_merge[0]) as json_file:
    base_json = json.load(json_file)

output_default = "In summary, the next action I will perform is ```{ACTION}```"

base_json['examples'] = []
for json_file_to_merge in files_to_merge:
    with open(json_file_to_merge) as json_file:
        json_to_merge = json.load(json_file)
    base_json['examples'].extend(json_to_merge["examples"])

with open('agent/prompts/prompt_trajectory_abstraction.txt') as f:
    prompt_state_abstraction = f.read()

with open('agent/prompts/prompt_add_knowledge.txt') as f:
    prompt_add_knowledge = f.read()

trajectory_dict = {}
for example in base_json['examples']:
    im_path = example[2]
    episode = os.path.split(os.path.split(im_path)[0])[-1]
    if episode not in trajectory_dict.keys():
        trajectory_dict[episode] = []
    trajectory_dict[episode].append(example)

base_json_trajectory = copy.deepcopy(base_json)
base_json_trajectory['examples'] = []
for episode in trajectory_dict.keys():
    trajectory = trajectory_dict[episode]
    actions = trajectory[-1][0].split('PREVIOUS ACTIONS: ')[-1]
    last_action = trajectory[-1][1].split('In summary, the next action I will perform is ```')[-1].split('```')[0]
    if actions=='None':
        last_count = 0
        actions = ''
    else:
        number_last_count = ''
        for s in actions.split('\n')[-1]:
            if s in '0123456789':
                number_last_count += s
            else:
                break
        last_count = int(number_last_count)
    full_actions = actions + f'\n{last_count+1}. {last_action}'
    objective = trajectory[-1][0].split('\n\nOBSERVATION:')[0].split('OBJECTIVE: ')[-1]

    traj_step = trajectory[0]
    if traj_step[3]:
        knowledge = ''
        count = 1
        for knowl in traj_step[3]:
            knowledge += f'\n{count}. {knowl}'
            count += 1
    else:
        knowledge = ' None'
    detailed_trajectory = f'###OBJECTIVE: {objective}\n\n###SUMMARY TRAJECTORY ACTIONS:\n{full_actions}\n\n###ABSTRACTION COMMENTS:{knowledge}\n\n###DETAILED TRAJECTORY INPUT/OUTPUT:'
    images = []
    for traj_idx in range(len(trajectory)):
        detailed_trajectory += '\n\n'
        traj_step = trajectory[traj_idx]
        int_action_traj_step = traj_step[1].split('In summary, the next action I will perform is ```')[-1].split('```')[0].split(' ') #.split('[')[-1].split(']')[0]

        if len(int_action_traj_step)==1:
            int_action_traj_step = int_action_traj_step[0]
        else:
            int_action_traj_step = int_action_traj_step[1].split('[')[-1].split(']')[0]

        obs_only = traj_step[0].split('OBSERVATION:\n')[-1].split('\n\nPREVIOUS ACTIONS')[0]
        obs_only_list = obs_only.split('\n')
        lines_rel_state = []
        if int_action_traj_step:
            for l_ in obs_only_list:
                if '[' + int_action_traj_step + ']' in l_:
                    lines_rel_state.append(l_)
                elif np.random.uniform()<0.1:
                    lines_rel_state.append(l_)
        if lines_rel_state:
            obs_reduced = '\n'.join(lines_rel_state)
        else:
            obs_reduced = ''
        obs = traj_step[0]
        output = traj_step[1]
        prev_actions = traj_step[0].split('\n\nPREVIOUS ACTIONS: ')[-1]
        traj_step_text = f'INPUT STEP {traj_idx+1}:\n\n"""\nOBJECTIVE: {objective}\n\nOBSERVATION:\n{obs_reduced}\n(additional state)\n\nPREVIOUS ACTIONS: {prev_actions}\n"""\n\nEXAMPLE OUTPUT STEP {traj_idx+1}:\n\n"""\n{output}\n"""'
        detailed_trajectory += traj_step_text
        images.append(traj_step[2])

    knowledge_added = None
    if add_knowledge:
        '''
        Causal and task abstractions
        '''
        if not trajectory[0][3]:
            prompt_abstract_add_ = prompt_add_knowledge
            prompt_abstract_add_ = prompt_abstract_add_.replace('{INPUTS}', f'{detailed_trajectory}')
            messages = [
                {"role": "user", "content": prompt_abstract_add_},
                ]
            response = generate_from_openai_chat_completion(
                    messages,
                    "gpt-4-1106-preview",
                    0.2,
                    4000,
                    0.1,
                    8000,
                )
            knowledge_added = response.replace('Abstractions:\n', '')
            knowledge_added = knowledge_added.replace('Abstractions:', '')
            trajectory[-1][3] = knowledge_added.split('\n')
            trajectory[0][3] = knowledge_added.split('\n')

    if do_state_abstraction:
        '''
        State abstraction
        '''
        if len(trajectory)>8:
            
            enc = tiktoken.encoding_for_model("gpt-4")
            prompt_token_length = len(enc.encode(detailed_trajectory))
            print(f"Prompt length before: {prompt_token_length}")

            prompt_state_abstraction_tmp = prompt_state_abstraction
            prompt_state_abstraction_tmp = prompt_state_abstraction_tmp.replace('{RETRIEVED_EXAMPLES}', detailed_trajectory)
            messages = [
                {"role": "user", "content": prompt_state_abstraction_tmp},
                ]
            response = generate_from_openai_chat_completion(
                    messages,
                    "gpt-4-1106-preview",
                    0.2,
                    4000,
                    0.1,
                    8000,
                )
            response_list = eval(response.split('Most Relevant Time Steps: ')[-1])
            response_list.append(last_count+1)
            response_list = list(set(response_list))
            response_list = list(np.asarray(response_list) - 1)
            traj_step = trajectory[0]
            if traj_step[3]:
                knowledge = ''
                count = 1
                for knowl in traj_step[3]:
                    knowledge += f'\n{count}. {knowl}'
                    count += 1
            else:
                knowledge = ' None'
            detailed_trajectory = f'###OBJECTIVE: {objective}\n\n###SUMMARY TRAJECTORY ACTIONS:{full_actions}\n\n###ABSTRACTION COMMENTS:{knowledge}\n\n###DETAILED TRAJECTORY INPUT/OUTPUT:'
            for traj_idx in response_list:
                detailed_trajectory += '\n\n'
                traj_step = trajectory[traj_idx]
                int_action_traj_step = traj_step[1].split('In summary, the next action I will perform is ```')[-1].split('```')[0].split(' ') #.split('[')[-1].split(']')[0]
                if len(int_action_traj_step)==1:
                    int_action_traj_step = int_action_traj_step[0]
                else:
                    int_action_traj_step = int_action_traj_step[1].split('[')[-1].split(']')[0]
                obs_only = traj_step[0].split('OBSERVATION:\n')[-1].split('\n\nPREVIOUS ACTIONS')[0]
                obs_only_list = obs_only.split('\n')
                lines_rel_state = []
                if int_action_traj_step:
                    for l_ in obs_only_list:
                        if '[' + int_action_traj_step + ']' in l_:
                            lines_rel_state.append(l_)
                        elif np.random.uniform()<0.1:
                            lines_rel_state.append(l_)
                if lines_rel_state:
                    obs_reduced = '\n' + '\n'.join(lines_rel_state)
                else:
                    obs_reduced = ''
                obs = traj_step[0]
                output = traj_step[1]
                prev_actions = traj_step[0].split('\n\nPREVIOUS ACTIONS: ')[-1]
                traj_step_text = f'INPUT STEP {traj_idx+1}:\n\n"""\nOBJECTIVE: {objective}\n\nOBSERVATION:{obs_reduced}\n(additional state)\n\nPREVIOUS ACTIONS: {prev_actions}\n"""\n\nOUTPUT STEP {traj_idx+1}:\n\n"""\n{output}\n"""'
                detailed_trajectory += traj_step_text

            enc = tiktoken.encoding_for_model("gpt-4")
            prompt_token_length = len(enc.encode(detailed_trajectory))
            print(f"Prompt length after: {prompt_token_length}")

    base_json_trajectory['examples'].append([objective, detailed_trajectory, images, trajectory[-1][3], trajectory[-1][4], full_actions])

from llms.providers.openai_utils import run_embedding_model
for example_idx in tqdm(range(len(base_json_trajectory['examples']))):
    example = base_json_trajectory['examples'][example_idx]
    im_path = example[2][-1]
    previous_action_text = example[-1]
    observation = example[1]
    objective_text = example[0] 
    if "classifieds" in im_path:
        objective_text = f"Website: classifieds\nObjective: {objective_text}"
    elif "shopping" in im_path:
        objective_text = f"Website: shopping\nObjective: {objective_text}"
    elif "reddit" in im_path:
        objective_text = f"Website: reddit\nObjective: {objective_text}"
    
    root, image_name = os.path.split(im_path)
    root, image_folder = os.path.split(root)
    im_path_new = os.path.join(new_folder, 'images', image_folder, image_name)

    os.makedirs(os.path.split(im_path_new)[0], exist_ok=True)
    im_PIL = Image.open(im_path)
    im_PIL.save(im_path_new)

    generic_text = "None"
    try:
        embedding_task = run_embedding_model(objective_text)
    except:
        print("fail")
        embedding_task = run_embedding_model(generic_text)
    try:
        embedding_obs = run_embedding_model(observation)
    except:
        print("fail")
        embedding_obs = run_embedding_model(objective_text)
    try:
        embedding_act = run_embedding_model(previous_action_text)
    except:
        print("fail")
        embedding_act = run_embedding_model(generic_text)


    images = []
    for im_path_traj_idx in range(len(example[2])):
        im_path_traj = example[2][im_path_traj_idx]
        images.append(Image.open(im_path_traj))
        # change path name
        root, image_name = os.path.split(im_path_traj)
        root, image_folder = os.path.split(root)
        im_path_new_traj = os.path.join(new_folder, 'images', image_folder, image_name)
        base_json_trajectory['examples'][example_idx][2][im_path_traj_idx] = im_path_new_traj

    visual_feature_embedding = clip_model.encode_images(images).mean(0).cpu().numpy()
    
    embed_path_visual = im_path_new.replace('images', 'embeddings_visual').replace('.png', '.npy')
    os.makedirs(os.path.split(embed_path_visual)[0], exist_ok=True)
    np.save(embed_path_visual, visual_feature_embedding)
    
    embed_path_task = im_path_new.replace('images', 'embeddings_task').replace('.png', '.npy')
    os.makedirs(os.path.split(embed_path_task)[0], exist_ok=True)
    np.save(embed_path_task, embedding_task)

    embed_path_obs = im_path_new.replace('images', 'embeddings_obs').replace('.png', '.npy')
    os.makedirs(os.path.split(embed_path_obs)[0], exist_ok=True)
    np.save(embed_path_obs, embedding_obs)

    embed_path_act = im_path_new.replace('images', 'embeddings_act').replace('.png', '.npy')
    os.makedirs(os.path.split(embed_path_act)[0], exist_ok=True)
    np.save(embed_path_act, embedding_act)

if "meta_data" in base_json_trajectory.keys():
    base_json_trajectory["meta_data"]["prompt_constructor"] = "MultimodalCoTPromptConstructorMemoryAugmentedFULLTRAJECTORY"

with open(os.path.join(new_folder, 'planning_examples.json'), "w") as outfile: 
    json.dump(base_json_trajectory, outfile, indent=4, sort_keys=True)
import os
import pickle
import pprint

import sys 
import json

from ego4d_forecasting.utils import logging
import numpy as np
import pytorch_lightning
import torch
from ego4d_forecasting.tasks.long_term_anticipation import MultiTaskClassificationTask, LongTermAnticipationTask, LongTermAnticipationTaskGPT4V
from ego4d_forecasting.utils.c2_model_loading import get_name_convert_func
from ego4d_forecasting.utils.misc import gpu_mem_usage
from ego4d_forecasting.utils.parser import load_config, parse_args
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.plugins import DDPPlugin

import copy 

logger = logging.get_logger(__name__)

import os
import pathlib
import shutil
import submitit
from PIL import Image

import base64
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Dict, TypedDict, Union

import ipdb
st = ipdb.set_trace

seed = 32
torch.manual_seed(seed)
np.random.seed(seed)


# Not sure why I can't import scripts.slurm?
# from scripts.slurm import copy_and_run_with_config
def init_and_run(run_fn, run_config):
    os.environ["RANK"] = os.environ["SLURM_LOCALID"]
    os.environ["LOCAL_RANK"] = os.environ["SLURM_LOCALID"]
    os.environ["NODE_RANK"] = os.environ["SLURM_LOCALID"]
    os.environ["WORLD_SIZE"] = os.environ["SLURM_NTASKS"]
    run_fn(run_config)


def copy_and_run_with_config(run_fn, run_config, directory, **cluster_config):
    working_directory = pathlib.Path(directory) / cluster_config["job_name"]
    copy_blacklist = [
        "cv",
        "data",
        "lightning_logs",
        "slurm",
        "logs",
        "pretrained_models",
        "Ego4D-Future-Hand-Prediction",
        "notebooks",
        "checkpoints",
        "experimental",
        ".git",
        "output",
    ]
    shutil.copytree(".", working_directory, ignore=lambda x, y: copy_blacklist)
    os.chdir(working_directory)
    print(f"Running at {working_directory}")

    executor = submitit.SlurmExecutor(folder=working_directory)
    executor.update_parameters(**cluster_config)
    job = executor.submit(init_and_run, run_fn, run_config)
    print(f"job_id: {job}")

def pil_to_b64(img: Image.Image) -> str:
    with BytesIO() as image_buffer:
        img.save(image_buffer, format="PNG")
        byte_data = image_buffer.getvalue()
        img_b64 = base64.b64encode(byte_data).decode("utf-8")
        img_b64 = "data:image/png;base64," + img_b64
    return img_b64

def get_prompt_abstraction_phase(
    intro,
    examples,
    images,
    future_actions,
    video_actions,
    deva,
    image_mean,
    image_std,
    use_examples=True,
    single_image=False,
    with_SoM=True,
):
    # if len(images[0])>1:
    #     assert(False)
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    message = [
        {
            "role": "system",
            "content": [{"type": "text", "text": intro}],
        }
    ]
    if use_examples:
        for (a,b,c,d,e,f,g,h,i) in list(examples):
            # a actions
            # b video_actions
            # c image_paths
            # d embed_npy
            # e summary
            # f abstracted_state
            # g reasoning
            # h state_change 
            # i abstraction_comments

            c = [Image.open(video_image) for video_image in c]
            # c = np.concatenate([np.asarray(video_image) for video_image in c], axis=1)
            # c = [Image.fromarray(c)]

            c = c[:2]
            
            example_content = []
            example_content.append(
                {
                    "type": "text",
                    "text": f"Egocentric video:",
                }
            )
            for image_i, image_f in enumerate(c):
                if type(image_f)==str:
                    example_img = Image.open(image_f)
                else:
                    example_img = image_f
                
                # current_prompt_example = self.ltp_pred_prompt.replace('{video_actions}', w)
                example_content.extend(
                    # {
                        # "role": "system",
                        # "name": "example_user",
                        # "content": 
                        [
                            # {"type": "text", "text": inputs_example},
                            {
                                "type": "text",
                                "text": f"Frame {image_i+1}:",
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": pil_to_b64(example_img),
                                    "detail": "high"
                                },
                            },
                        ]
                    # }
                )
            inputs_example = f"Inputs:\n\nVideo actions:\n{b}\n\nFuture actions:\n{a}"
            # print(inputs_example)
            example_content = [{"type": "text", "text": inputs_example}] + example_content
            message.append({"role": "system", "name": "example_user", "content": example_content})
            output_example = f"Outputs:\n\nSummary: {e}\n\nAbstracted State:\n{f}\n\nStep-by-step Reasoning: {g}\n\nPredicted State Change: {h}\n\nAbstraction Comments:\n{i}"
            # print(output_example)
            message.append(
                    {
                        "role": "system",
                        "name": "example_assistant",
                        "content": [{"type": "text", "text": output_example}],
                    }
                )
    # images_samples = images[0][0,0].unsqueeze(0) #.unbind(1)
    # image_std, image_mean = image_std.to(images_samples.device), image_mean.to(images_samples.device)
    # images_samples = images_samples * image_std + image_mean
    # images_samples = images_samples.unbind(1)
    newsize = (768, 768)
    if single_image:
        images_samples = [video_image.resize(newsize) for video_image in images]
        if with_SoM:
            images_samples = deva.run_deva(images_samples)
        images_samples_ = copy.deepcopy(images_samples)
        images_samples = np.concatenate([np.asarray(video_image) for video_image in images_samples], axis=1)
        images_samples = [Image.fromarray(images_samples)]
    else:
        images_samples = [video_image.resize(newsize) for video_image in images]
        if with_SoM:
            images_samples = deva.run_deva(images_samples)
        images_samples_ = copy.deepcopy(images_samples)
    content = []
    print(f'Number of images: {len(images_samples)}')
    # for image_i, image in enumerate(images_samples):
    content.append(
        {
            "type": "text",
            "text": f"Egocentric video:",
        }
    )
    for image_idx, image in enumerate(images_samples):
        content.extend(
            [
            {
                "type": "text",
                "text": f"Frame {image_idx+1}:",
            },
            {
                "type": "image_url",
                "image_url": {"url": pil_to_b64(image), "detail": "high"} ,
            }
            ]
        )
    inputs = f"Inputs:\n\nVideo actions:\n{video_actions}\n\nFuture actions:\n{future_actions}"
    # current_prompt = self.ltp_pred_prompt.replace('{video_actions}', video_actions)
    print(inputs)
    content = [{"type": "text", "text": inputs}] + content
    message.append({"role": "user", "content": content})
    return message, images_samples_

def save_examples_json(
    root,
    clip,
):
    path = os.path.join(root, 'actions')
    examples = {}
    files = os.listdir(path)
    for file_ in files:
        file_path = os.path.join(path, file_)
        with open(file_path) as f:
            actions = f.read()
        file_reasoning = file_path.replace('actions', 'reasoning')
        with open(file_reasoning) as f:
            reasoning = f.read()
        file_abstracted_state = file_path.replace('actions', 'abstracted_state')
        with open(file_abstracted_state) as f:
            abstracted_state = f.read()
        file_abstraction_comments = file_path.replace('actions', 'abstraction_comments')
        with open(file_abstraction_comments) as f:
            abstraction_comments = f.read()
        file_state_change = file_path.replace('actions', 'state_change')
        with open(file_state_change) as f:
            state_change = f.read()
        file_summary = file_path.replace('actions', 'summary')
        with open(file_summary) as f:
            summary = f.read()
        file_video_actions = file_path.replace('actions', 'video_actions')
        if os.path.exists(file_video_actions):
            with open(file_video_actions) as f:
                video_actions = f.read()
        else:
            video_actions = ''
        # SoM images
        image_paths = os.listdir(file_path.replace('actions', 'images_SoM').replace('.txt', ''))
        image_paths_idxs = np.argsort([int(im_path.replace('.png', '')) for im_path in image_paths])
        image_paths = [os.path.join(path.replace('actions', 'images_SoM'), file_.replace('.txt', ''), image_paths[im_path_idx]) for im_path_idx in list(image_paths_idxs)]
        # raw images
        # image_paths_raw = os.listdir(file_path.replace('actions', 'images_raw').replace('.txt', ''))
        
        images_raw = [Image.open(f.replace('images_SoM', 'images_raw')) for f in image_paths]
        image_encodings = clip.encode_images(images_raw)
        image_encodings_mean = image_encodings.mean(0).cpu().numpy()
        examples_embedding_dir = path.replace('actions', 'embeddings_image')
        os.makedirs(examples_embedding_dir, exist_ok=True)
        embed_npy = os.path.join(examples_embedding_dir, file_.replace('.txt', '.npy'))
        np.save(embed_npy, image_encodings_mean)
        embed_npy_image = embed_npy

        text_encodings = clip.encode_text(summary)
        text_encodings_mean = text_encodings.mean(0).cpu().numpy()
        examples_embedding_dir = path.replace('actions', 'embeddings_summary')
        os.makedirs(examples_embedding_dir, exist_ok=True)
        embed_npy = os.path.join(examples_embedding_dir, file_.replace('.txt', '.npy'))
        np.save(embed_npy, text_encodings_mean)

        text_encodings = clip.encode_text(abstracted_state)
        text_encodings_mean = text_encodings.mean(0).cpu().numpy()
        examples_embedding_dir = path.replace('actions', 'embeddings_state')
        os.makedirs(examples_embedding_dir, exist_ok=True)
        embed_npy = os.path.join(examples_embedding_dir, file_.replace('.txt', '.npy'))
        np.save(embed_npy, text_encodings_mean)

        example = [actions, video_actions, image_paths, embed_npy_image, summary, abstracted_state, reasoning, state_change, abstraction_comments]
        examples[file_.replace('.txt', '')] = example
    with open(os.path.join(path.replace('/actions', ''), 'examples.json'), "w") as outfile: 
        json.dump(examples, outfile, indent=4, sort_keys=True)

def retrieve_topk(
    examples_json,
    images,
    clip,
    topk=3,
):

    cos_sim = torch.nn.CosineSimilarity(dim=1)

    examples = list(examples_json.values())
    example_embeddings = []
    for example in examples:
        # example = examples_json[k]
        embedding = np.load(example[3])
        example_embeddings.append(embedding)
    example_embeddings = torch.from_numpy(np.asarray(example_embeddings))

    image_encodings = clip.encode_images(images)
    image_encodings_mean = torch.from_numpy(image_encodings.mean(0).cpu().numpy()[None])

    sim_task = cos_sim(image_encodings_mean, example_embeddings)
    sims_argsort = torch.argsort(sim_task, descending=True).cpu().numpy()
    examples_sorted = [examples[e_i] for e_i in list(sims_argsort)]
    examples_sorted_topk = examples_sorted[:topk]
    return examples_sorted_topk

def save_example(
    example_folder_name,
    images_samples_ltp,
    images,
    text_gt_forecast,
    text_gt_recognition,
    response,
    clip_id,
    image_std,
    image_mean,
):

    summary = response.split('Summary: ')[-1].split('\n')[0]
    abstracted_state = response.split('Abstracted State:\n')[-1].split('\n\nStep-by-step Reasoning:')[0]
    step_by_step = response.split('Step-by-step Reasoning: ')[-1].split('\n')[0]
    predicted_state_change = response.split('Predicted State Change: ')[-1].split('\n')[0]
    abstraction_comments = response.split('Abstraction Comments:\n')[-1]

    examples_image_dir = os.path.join('ego4d_forecasting', 'models', 'prompts', example_folder_name, 'forecasting', 'images_SoM')
    os.makedirs(examples_image_dir, exist_ok=True)
    examples_image_raw_dir = os.path.join('ego4d_forecasting', 'models', 'prompts', example_folder_name, 'forecasting', 'images_raw')
    os.makedirs(examples_image_raw_dir, exist_ok=True)
    examples_actions_dir = os.path.join('ego4d_forecasting', 'models', 'prompts', example_folder_name, 'forecasting', 'actions')
    os.makedirs(examples_actions_dir, exist_ok=True)
    examples_video_actions_dir = os.path.join('ego4d_forecasting', 'models', 'prompts', example_folder_name, 'forecasting', 'video_actions')
    os.makedirs(examples_video_actions_dir, exist_ok=True)
    examples_summary_dir = os.path.join('ego4d_forecasting', 'models', 'prompts', example_folder_name, 'forecasting', 'summary')
    os.makedirs(examples_summary_dir, exist_ok=True)
    examples_reasoning_dir = os.path.join('ego4d_forecasting', 'models', 'prompts', example_folder_name, 'forecasting', 'reasoning')
    os.makedirs(examples_reasoning_dir, exist_ok=True)
    examples_abstracted_state_dir = os.path.join('ego4d_forecasting', 'models', 'prompts', example_folder_name, 'forecasting', 'abstracted_state')
    os.makedirs(examples_abstracted_state_dir, exist_ok=True)
    examples_abstraction_comments_dir = os.path.join('ego4d_forecasting', 'models', 'prompts', example_folder_name, 'forecasting', 'abstraction_comments')
    os.makedirs(examples_abstraction_comments_dir, exist_ok=True)
    examples_state_change_dir = os.path.join('ego4d_forecasting', 'models', 'prompts', example_folder_name, 'forecasting', 'state_change')
    os.makedirs(examples_state_change_dir, exist_ok=True)
    for image_idx, image in enumerate(images_samples_ltp):
        video_image = image
        os.makedirs(os.path.join(examples_image_dir, f'{clip_id}'), exist_ok=True)
        video_image.save(os.path.join(examples_image_dir, f'{clip_id}', f'{image_idx}.png'))
    images_ = images[0][0,0].unbind(1)
    for image_idx, image in enumerate(images_):
        video_image = image.unsqueeze(0) * image_std + image_mean
        video_image = Image.fromarray((video_image.squeeze(0).permute(1,2,0).cpu().numpy() * 255).astype(np.uint8))
        os.makedirs(os.path.join(examples_image_raw_dir, f'{clip_id}'), exist_ok=True)
        video_image.save(os.path.join(examples_image_raw_dir, f'{clip_id}', f'{image_idx}.png'))
    
    # save text
    with open(os.path.join(examples_actions_dir, f'{clip_id}.txt'), 'w') as f:
        f.write(text_gt_forecast)
    with open(os.path.join(examples_video_actions_dir, f'{clip_id}.txt'), 'w') as f:
        f.write(text_gt_recognition)
    with open(os.path.join(examples_reasoning_dir, f'{clip_id}.txt'), 'w') as f:
        f.write(step_by_step)
    with open(os.path.join(examples_abstracted_state_dir, f'{clip_id}.txt'), 'w') as f:
        f.write(abstracted_state)
    with open(os.path.join(examples_abstraction_comments_dir, f'{clip_id}.txt'), 'w') as f:
        f.write(abstraction_comments)
    with open(os.path.join(examples_state_change_dir, f'{clip_id}.txt'), 'w') as f:
        f.write(predicted_state_change)
    with open(os.path.join(examples_summary_dir, f'{clip_id}.txt'), 'w') as f:
        f.write(summary)

    # examples_image_dir = os.path.join('ego4d_forecasting', 'models', 'prompts', 'examples', 'recognition', 'images')
    # os.makedirs(examples_image_dir, exist_ok=True)
    examples_image_dir = os.path.join('ego4d_forecasting', 'models', 'prompts', example_folder_name, 'recognition', 'images_SoM')
    os.makedirs(examples_image_dir, exist_ok=True)
    examples_image_raw_dir = os.path.join('ego4d_forecasting', 'models', 'prompts', example_folder_name, 'recognition', 'images_raw')
    os.makedirs(examples_image_raw_dir, exist_ok=True)
    examples_actions_dir = os.path.join('ego4d_forecasting', 'models', 'prompts', example_folder_name, 'recognition', 'actions')
    os.makedirs(examples_actions_dir, exist_ok=True)
    examples_summary_dir = os.path.join('ego4d_forecasting', 'models', 'prompts', example_folder_name, 'recognition', 'summary')
    os.makedirs(examples_summary_dir, exist_ok=True)
    examples_reasoning_dir = os.path.join('ego4d_forecasting', 'models', 'prompts', example_folder_name, 'recognition', 'reasoning')
    os.makedirs(examples_reasoning_dir, exist_ok=True)
    examples_abstracted_state_dir = os.path.join('ego4d_forecasting', 'models', 'prompts', example_folder_name, 'recognition', 'abstracted_state')
    os.makedirs(examples_abstracted_state_dir, exist_ok=True)
    examples_abstraction_comments_dir = os.path.join('ego4d_forecasting', 'models', 'prompts', example_folder_name, 'recognition', 'abstraction_comments')
    os.makedirs(examples_abstraction_comments_dir, exist_ok=True)
    examples_state_change_dir = os.path.join('ego4d_forecasting', 'models', 'prompts', example_folder_name, 'recognition', 'state_change')
    os.makedirs(examples_state_change_dir, exist_ok=True)
    for image_idx, image in enumerate(images_samples_ltp):
        video_image = image
        os.makedirs(os.path.join(examples_image_dir, f'{clip_id}'), exist_ok=True)
        video_image.save(os.path.join(examples_image_dir, f'{clip_id}', f'{image_idx}.png'))
    images_ = images[0][0,0].unbind(1)
    for image_idx, image in enumerate(images_):
        video_image = image.unsqueeze(0) * image_std + image_mean
        video_image = Image.fromarray((video_image.squeeze(0).permute(1,2,0).cpu().numpy() * 255).astype(np.uint8))
        os.makedirs(os.path.join(examples_image_raw_dir, f'{clip_id}'), exist_ok=True)
        video_image.save(os.path.join(examples_image_raw_dir, f'{clip_id}', f'{image_idx}.png'))

    # save text
    with open(os.path.join(examples_actions_dir, f'{clip_id}.txt'), 'w') as f:
        f.write(text_gt_recognition)
    with open(os.path.join(examples_reasoning_dir, f'{clip_id}.txt'), 'w') as f:
        f.write(step_by_step)
    with open(os.path.join(examples_abstracted_state_dir, f'{clip_id}.txt'), 'w') as f:
        f.write(abstracted_state)
    with open(os.path.join(examples_abstraction_comments_dir, f'{clip_id}.txt'), 'w') as f:
        f.write(abstraction_comments)
    with open(os.path.join(examples_state_change_dir, f'{clip_id}.txt'), 'w') as f:
        f.write(predicted_state_change)
    with open(os.path.join(examples_summary_dir, f'{clip_id}.txt'), 'w') as f:
        f.write(summary)

def main(cfg):
    seed_everything(cfg.RNG_SEED)

    logging.setup_logging(cfg.OUTPUT_DIR)
    logger.info("Run with config:")
    logger.info(pprint.pformat(cfg))

    # Choose task type based on config.
    # TODO: change this to TASK_REGISTRY.get(cfg.cfg.DATA.TASK)(cfg)
    if cfg.DATA.TASK == "detection":
        TaskType = DetectionTask
    elif cfg.DATA.TASK == "classification":
        TaskType = MultiTaskClassificationTask
    elif cfg.DATA.TASK == "long_term_anticipation":
        TaskType = LongTermAnticipationTask
    elif cfg.DATA.TASK == "long_term_anticipation_gpt4v":
        TaskType = LongTermAnticipationTaskGPT4V
    elif cfg.DATA.TASK == "short_term_anticipation":
        TaskType = ShortTermAnticipationTask

    task = TaskType(cfg)

    

    checkpoint_callback = ModelCheckpoint(
        monitor=task.checkpoint_metric, mode="min", save_last=True, save_top_k=1
    )
    if cfg.ENABLE_LOGGING:
        args = {"callbacks": [LearningRateMonitor(), checkpoint_callback]}
    else:
        args = {"logger": False, "callbacks": checkpoint_callback}

    do_pytorch_lightning = False
     
    if do_pytorch_lightning:
        trainer = Trainer(
            gpus=cfg.NUM_GPUS,
            num_nodes=cfg.NUM_SHARDS,
            accelerator=cfg.SOLVER.ACCELERATOR,
            max_epochs=cfg.SOLVER.MAX_EPOCH,
            num_sanity_val_steps=3,
            benchmark=True,
            log_gpu_memory="min_max",
            replace_sampler_ddp=False,
            fast_dev_run=cfg.FAST_DEV_RUN,
            default_root_dir=cfg.OUTPUT_DIR,
            plugins=DDPPlugin(find_unused_parameters=False),
            **args,
        )

        if cfg.TRAIN.ENABLE and cfg.TEST.ENABLE:
            trainer.fit(task)

            # Calling test without the lightning module arg automatically selects the best
            # model during training.
            return trainer.test()

        elif cfg.TRAIN.ENABLE:
            return trainer.fit(task)

        elif cfg.TEST.ENABLE:
            return trainer.test(task)

    else:
        std = cfg.DATA.STD
        mean = cfg.DATA.MEAN
        image_mean = torch.from_numpy(np.array([mean]).reshape(1,3,1,1))
        image_std = torch.from_numpy(np.array([std]).reshape(1,3,1,1))
        task.setup("")
        results = {}
        metric_dir = f'log/{cfg.EXPERIMENT_NAME}'
        os.makedirs(metric_dir, exist_ok=True)
        if os.path.exists(os.path.join(metric_dir, 'metrics.json')):
            with open(os.path.join(metric_dir, 'metrics.json')) as json_file:
                results = json.load(json_file)
        
        do_abstraction = cfg.DO_ABSTRACTION #False

        # save_examples_json(f'ego4d_forecasting/models/prompts/examples_abstraction_phase_only00/forecasting/', task.model.clip)
        # save_examples_json(f'ego4d_forecasting/models/prompts/examples_abstraction_phase_only00/recognition/', task.model.clip)
        # save_examples_json(f'ego4d_forecasting/models/prompts/examples/forecasting/', task.model.clip)
        # save_examples_json(f'ego4d_forecasting/models/prompts/examples/recognition/', task.model.clip)
        # st()
        examples_forecasting_json2 = None
        for step, batch in enumerate(task.val_loader):

            print(f"Step: {step} / {len(task.val_loader)}")

            clip_id = batch[3][0]

            if cfg.SKIP_IF_EXISTS:
                if do_abstraction:
                    examples_forecasting2 = f'ego4d_forecasting/models/prompts/{cfg.EXAMPLE_FOLDER_NAME}/forecasting/examples.json'
                    if os.path.exists(examples_forecasting2):
                        if examples_forecasting_json2 is None:
                            with open(examples_forecasting2) as json_data:
                                examples_forecasting_json2 = json.load(json_data)
                    if examples_forecasting_json2 is not None and clip_id in examples_forecasting_json2.keys():
                        print(f"{clip_id} exists... skipping...")
                        continue
                else:
                    if clip_id in results.keys():
                        print(f"{clip_id} exists... skipping...")
                        continue

            # if do_abstraction:
            #     if step<=2:
            #         continue
            # if step<=1:
            #     continue

            if cfg.IN_TRY_EXCEPT:
                try:
                    result, images_samples_ltp = task.validation_step(batch, step, return_images_only=do_abstraction)
                except:
                    print("failed... continuing...")
                    continue
            else:
                result, images_samples_ltp = task.validation_step(batch, step, return_images_only=do_abstraction)
            
            for k in list(result.keys()):
                if isinstance(result[k],np.ndarray):
                    result[k] = float(result[k])
            results[clip_id] = result

            video_image = np.concatenate([np.asarray(image_sample) for image_sample in images_samples_ltp], axis=1)
            video_image = Image.fromarray(video_image) #(image_sample.squeeze(0).permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)) 
            os.makedirs(os.path.join(metric_dir, 'videos'), exist_ok=True)
            video_image.save(os.path.join(metric_dir, 'videos', f'{clip_id}.png'))
            
            with open(os.path.join(metric_dir, 'metrics.json'), "w") as outfile: 
                json.dump(results, outfile, indent=4, sort_keys=True)

            images, forecast_labels, video_labels, _, _ = batch
            
            if do_abstraction:

                example_folder_name = cfg.EXAMPLE_FOLDER_NAME #'examples_abstraction_phase_only_multiimage'

                text_gt_idm = ''
                for l_idx in range(video_labels.shape[1]):
                    if l_idx>0:
                        text_gt_idm += f'\n'
                    text_gt_idm += f'{l_idx+1}. {task.idx2verbs[int(video_labels[0,l_idx,0].cpu().numpy())]} {task.idx2nouns[int(video_labels[0,l_idx,1].cpu().numpy())]}'
                text_gt_forecast = ''
                for l_idx in range(forecast_labels.shape[1]):
                    if l_idx>0:
                        text_gt_forecast += f'\n'
                    text_gt_forecast += f'{l_idx+1}. {task.idx2verbs[int(forecast_labels[0,l_idx,0].cpu().numpy())]} {task.idx2nouns[int(forecast_labels[0,l_idx,1].cpu().numpy())]}'

                images_ = images[0][0,0].unbind(1)
                current_images = []
                for image_idx, image in enumerate(images_):
                    video_image = image.unsqueeze(0) * image_std + image_mean
                    video_image = Image.fromarray((video_image.squeeze(0).permute(1,2,0).cpu().numpy() * 255).astype(np.uint8))
                    current_images.append(video_image)

                examples_forecasting = 'ego4d_forecasting/models/prompts/examples/forecasting/examples.json'
                with open(examples_forecasting) as json_data:
                    examples_forecasting_json = json.load(json_data)

                examples_forecasting2 = f'ego4d_forecasting/models/prompts/{example_folder_name}/forecasting/examples.json'
                if os.path.exists(examples_forecasting2):
                    with open(examples_forecasting2) as json_data:
                        examples_forecasting_json2 = json.load(json_data)
                    examples_forecasting_json.update(examples_forecasting_json2)
                
                # examples_forecasting = retrieve_topk(
                #     examples_forecasting_json,
                #     current_images,
                #     task.model.clip,
                # )
                update_examples = False
                if update_examples:
                    task.model.get_example_embeddings(example_json=examples_forecasting_json)
                    examples_forecasting = task.model.retrieve_topk(current_images)
                else:
                    examples_forecasting = task.model.examples

                with open('ego4d_forecasting/models/prompts/prompt_abstraction_phase.txt') as f:
                    intro_abstract_phase = f.read()

                try:
                    prompt, images_samples_prompt = get_prompt_abstraction_phase(
                        intro_abstract_phase,
                        examples_forecasting,
                        current_images,
                        text_gt_forecast,
                        text_gt_idm,
                        task.model.deva,
                        image_mean,
                        image_std,
                    )
                except:
                    continue

                try:
                    response = task.model.generate_from_openai_completion(prompt)
                except:
                    continue

                # st()
                # images_samples_prompt[2].save('output/test.png')

                # video_image = np.concatenate([np.asarray(image_sample) for image_sample in images_samples_ltp], axis=1)
                # video_image = Image.fromarray(video_image) #(image_sample.squeeze(0).permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)) 
                # video_image.save('images/video.png')

                save_example(
                    example_folder_name,
                    images_samples_ltp,
                    images,
                    text_gt_forecast,
                    text_gt_idm,
                    response,
                    clip_id,
                    image_std,
                    image_mean,
                )

                save_examples_json(f'ego4d_forecasting/models/prompts/{example_folder_name}/forecasting/', task.model.clip)
                save_examples_json(f'ego4d_forecasting/models/prompts/{example_folder_name}/recognition/', task.model.clip)

            log = False
            if log:
                log_dir = f'log/{cfg.EXPERIMENT_NAME}/{clip_id}'
                os.makedirs(log_dir, exist_ok=True)

                video_image = np.concatenate([np.asarray(image_sample) for image_sample in images_samples_ltp], axis=1)
                video_image = Image.fromarray(video_image) 
                video_image.save(os.path.join(log_dir, 'video.png'))

                vid_num = 0
                video_image = torch.cat(images[1][0,vid_num].unbind(1), dim=2).unsqueeze(0) * image_std + image_mean
                video_image = Image.fromarray((video_image.squeeze(0).permute(1,2,0).cpu().numpy() * 255).astype(np.uint8))
                video_image.save(os.path.join(log_dir, 'video_full.png'))

                text_gt_idm = 'GT Video Actions:'
                for l_idx in range(video_labels.shape[1]):
                    text_gt_idm += f'\n{l_idx+1}. {task.idx2verbs[int(video_labels[0,l_idx,0].cpu().numpy())]} {task.idx2nouns[int(video_labels[0,l_idx,1].cpu().numpy())]}'
                text_to_log_idm = f'------------------ACTION RECOGNITION------------------\n\nGPT4V PREDICTED:\n{result["response_idm"]}\n\nGROUND TRUTH:\n{text_gt_idm}'
                with open(os.path.join(log_dir, 'action_recognition.txt'), 'w') as f:
                    f.write(text_to_log_idm)

                text_gt_forecast = 'GT Video Actions:'
                for l_idx in range(forecast_labels.shape[1]):
                    text_gt_forecast += f'\n{l_idx+1}. {task.idx2verbs[int(forecast_labels[0,l_idx,0].cpu().numpy())]} {task.idx2nouns[int(forecast_labels[0,l_idx,1].cpu().numpy())]}'

                text_to_forecast = f'------------------FORECASTING------------------\n\nGPT4V PREDICTED:\n{result["response_ltp"]}\n\nGROUND TRUTH:\n{text_gt_forecast}'
                with open(os.path.join(log_dir, 'forecasting.txt'), 'w') as f:
                    f.write(text_to_forecast)
                
            save_examples = False
            if save_examples:
                examples_image_dir = os.path.join('ego4d_forecasting', 'models', 'prompts', 'examples', 'forecasting', 'images_SoM')
                os.makedirs(examples_image_dir, exist_ok=True)
                examples_image_raw_dir = os.path.join('ego4d_forecasting', 'models', 'prompts', 'examples', 'forecasting', 'images_raw')
                os.makedirs(examples_image_raw_dir, exist_ok=True)
                examples_actions_dir = os.path.join('ego4d_forecasting', 'models', 'prompts', 'examples', 'forecasting', 'actions')
                os.makedirs(examples_actions_dir, exist_ok=True)
                examples_reasoning_dir = os.path.join('ego4d_forecasting', 'models', 'prompts', 'examples', 'forecasting', 'reasoning')
                os.makedirs(examples_reasoning_dir, exist_ok=True)
                for image_idx, image in enumerate(images_samples_ltp):
                    video_image = image
                    os.makedirs(os.path.join(examples_image_dir, f'{clip_id}'), exist_ok=True)
                    video_image.save(os.path.join(examples_image_dir, f'{clip_id}', f'{image_idx}.png'))
                images_ = images[0][0,0].unbind(1)
                for image_idx, image in enumerate(images_):
                    video_image = image.unsqueeze(0) * image_std + image_mean
                    video_image = Image.fromarray((video_image.squeeze(0).permute(1,2,0).cpu().numpy() * 255).astype(np.uint8))
                    os.makedirs(os.path.join(examples_image_raw_dir, f'{clip_id}'), exist_ok=True)
                    video_image.save(os.path.join(examples_image_raw_dir, f'{clip_id}', f'{image_idx}.png'))
                text_gt_forecast = '' 
                for l_idx in range(forecast_labels.shape[1]):
                    if l_idx>0:
                        text_gt_forecast += f'\n'
                    text_gt_forecast += f'{l_idx+1}. {task.idx2verbs[int(forecast_labels[0,l_idx,0].cpu().numpy())]} {task.idx2nouns[int(forecast_labels[0,l_idx,1].cpu().numpy())]}'
                with open(os.path.join(examples_actions_dir, f'{clip_id}.txt'), 'w') as f:
                    f.write(text_gt_forecast)

                examples_image_dir = os.path.join('ego4d_forecasting', 'models', 'prompts', 'examples', 'recognition', 'images_SoM')
                os.makedirs(examples_image_dir, exist_ok=True)
                examples_image_raw_dir = os.path.join('ego4d_forecasting', 'models', 'prompts', 'examples', 'recognition', 'images_raw')
                os.makedirs(examples_image_raw_dir, exist_ok=True)
                examples_actions_dir = os.path.join('ego4d_forecasting', 'models', 'prompts', 'examples', 'recognition', 'actions')
                os.makedirs(examples_actions_dir, exist_ok=True)
                examples_reasoning_dir = os.path.join('ego4d_forecasting', 'models', 'prompts', 'examples', 'recognition', 'reasoning')
                os.makedirs(examples_reasoning_dir, exist_ok=True)
                for image_idx, image in enumerate(images_samples_ltp):
                    video_image = image
                    os.makedirs(os.path.join(examples_image_dir, f'{clip_id}'), exist_ok=True)
                    video_image.save(os.path.join(examples_image_dir, f'{clip_id}', f'{image_idx}.png'))
                images_ = images[0][0,0].unbind(1)
                for image_idx, image in enumerate(images_):
                    video_image = image.unsqueeze(0) * image_std + image_mean
                    video_image = Image.fromarray((video_image.squeeze(0).permute(1,2,0).cpu().numpy() * 255).astype(np.uint8))
                    os.makedirs(os.path.join(examples_image_raw_dir, f'{clip_id}'), exist_ok=True)
                    video_image.save(os.path.join(examples_image_raw_dir, f'{clip_id}', f'{image_idx}.png'))
                text_gt_recognition = '' 
                for l_idx in range(video_labels.shape[1]):
                    if l_idx>0:
                        text_gt_recognition += f'\n'
                    text_gt_recognition += f'{l_idx+1}. {task.idx2verbs[int(video_labels[0,l_idx,0].cpu().numpy())]} {task.idx2nouns[int(video_labels[0,l_idx,1].cpu().numpy())]}'
                with open(os.path.join(examples_actions_dir, f'{clip_id}.txt'), 'w') as f:
                    f.write(text_gt_recognition)
                st()
                if step>=3:
                    break

            if cfg.MAX_EPISODES is not None:
                if step+1>=cfg.MAX_EPISODES:
                    break
                

        


if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args)
    # if args.on_cluster:
    #     copy_and_run_with_config(
    #         main,
    #         cfg,
    #         args.working_directory,
    #         job_name=args.job_name,
    #         time="72:00:00",
    #         partition="devlab,learnlab,learnfair",
    #         gpus_per_node=cfg.NUM_GPUS,
    #         ntasks_per_node=cfg.NUM_GPUS,
    #         cpus_per_task=10,
    #         mem="470GB",
    #         nodes=cfg.NUM_SHARDS,
    #         constraint="volta32gb",
    #     )
    # else:  # local
    main(cfg)

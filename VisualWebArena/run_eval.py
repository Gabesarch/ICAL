"""Script to run end-to-end evaluation on the benchmark.

Modified from https://github.com/web-arena-x/webarena/blob/main/run.py.
"""
import argparse
import glob
import json
import logging
import os
import random
import time
from pathlib import Path
from typing import List

import openai
import requests
import torch
from beartype import beartype
from PIL import Image
from io import BytesIO
from browser_env.actions import action2str
import pickle

from agent import (
    PromptAgent,
    construct_agent,
)
from agent.prompts import *
from agent.prompts.save_memory_example import save_example
from browser_env import (
    Action,
    ActionTypes,
    ScriptBrowserEnv,
    StateInfo,
    Trajectory,
    create_stop_action,
)
from browser_env.actions import is_equivalent
from browser_env.helper_functions import (
    RenderHelper,
    get_action_description,
)
from evaluation_harness import evaluator_router, image_utils
import wandb

import ipdb
st = ipdb.set_trace

LOG_FOLDER = "log_files"
Path(LOG_FOLDER).mkdir(parents=True, exist_ok=True)
LOG_FILE_NAME = f"{LOG_FOLDER}/log_{time.strftime('%Y%m%d%H%M%S', time.localtime())}_{random.randint(0, 10000)}.log"

logger = logging.getLogger("logger")
logger.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
logger.addHandler(console_handler)

file_handler = logging.FileHandler(LOG_FILE_NAME)
file_handler.setLevel(logging.DEBUG)
logger.addHandler(file_handler)

# Set the log format
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)


def config() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run end-to-end evaluation on the benchmark"
    )
    parser.add_argument(
        "--render", action="store_true", help="Render the browser"
    )
    parser.add_argument(
        "--debug", action="store_true", help="Debug mode"
    )

    parser.add_argument(
        "--slow_mo",
        type=int,
        default=0,
        help="Slow down the browser by the specified amount",
    )
    parser.add_argument(
        "--action_set_tag", default="id_accessibility_tree", help="Action type"
    )
    parser.add_argument(
        "--observation_type",
        choices=[
            "accessibility_tree",
            "accessibility_tree_with_captioner",
            "html",
            "image",
            "image_som",
        ],
        default="accessibility_tree",
        help="Observation type",
    )
    parser.add_argument(
        "--current_viewport_only",
        action="store_true",
        help="Only use the current viewport for the observation",
    )
    parser.add_argument("--viewport_width", type=int, default=1280)
    parser.add_argument("--viewport_height", type=int, default=2048)
    parser.add_argument("--save_trace_enabled", action="store_true")
    parser.add_argument("--sleep_after_execution", type=float, default=0.0)

    parser.add_argument("--max_steps", type=int, default=30)

    # agent config
    parser.add_argument("--agent_type", type=str, default="prompt")
    parser.add_argument(
        "--instruction_path",
        type=str,
        default="agents/prompts/state_action_agent.json",
    )
    parser.add_argument(
        "--feedback_instruction_path",
        type=str,
        default="agent/prompts/jsons/p_multimodal_humanfeedback.json",
    )
    parser.add_argument(
        "--parsing_failure_th",
        help="When consecutive parsing failures exceed this threshold, the agent will terminate early.",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--repeating_action_failure_th",
        help="When consecutive repeated actions exceed this threshold, the agent will terminate early.",
        type=int,
        default=5,
    )

    parser.add_argument("--episode_in_try_except", action="store_true", default=False, help="Continue to next episode if assertion error occurs? ")
    parser.add_argument("--skip_if_finished", action="store_true", default=False, help="Continue to next episode if assertion error occurs? ")
    parser.add_argument("--skip_if_exists", action="store_true", default=False, help="Continue to next episode if assertion error occurs? ")

    parser.add_argument("--test_config_base_dir", type=str)

    parser.add_argument(
        "--eval_captioning_model_device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to run eval captioning model on. By default, runs it on CPU.",
    )
    parser.add_argument(
        "--eval_captioning_model",
        type=str,
        default="Salesforce/blip2-flan-t5-xl",
        choices=["Salesforce/blip2-flan-t5-xl"],
        help="Captioning backbone for VQA-type evals.",
    )
    parser.add_argument(
        "--captioning_model",
        type=str,
        default="Salesforce/blip2-flan-t5-xl",
        choices=["Salesforce/blip2-flan-t5-xl", "llava-hf/llava-1.5-7b-hf"],
        help="Captioning backbone for accessibility tree alt text.",
    )

    
    parser.add_argument("--instruction_jsons", type=str, nargs='+', default=[], help="jsons to use for example retrieval")
    parser.add_argument("--feedback_jsons", type=str, nargs='+', default=[], help="jsons to use for human feedback retrieval")
    parser.add_argument(
        "--save_examples_memory", action="store_true", help="save out example data?"
    )
    parser.add_argument(
        "--continually_add_saved_examples", action="store_true", help="save out example data?"
    )

    # lm config
    parser.add_argument("--provider", type=str, default="openai")
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo-0613")
    parser.add_argument("--mode", type=str, default="chat")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=0.1)
    parser.add_argument("--context_length", type=int, default=0)
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--stop_token", type=str, default=None)
    parser.add_argument(
        "--max_retry",
        type=int,
        help="max retry times to perform generations when parsing fails",
        default=1,
    )
    parser.add_argument(
        "--max_obs_length",
        type=int,
        help="when not zero, will truncate the observation to this length before feeding to the model",
        default=0,
    )

    parser.add_argument("--experiment_name", type=str, default="run0")
    parser.add_argument("--wandb_directory", type=str, default='.', help="Path to wandb metadata")

    parser.add_argument("--shopping", type=int, nargs='+', default=None, help="shopping episodes")
    parser.add_argument("--reddit", type=int, nargs='+', default=None, help="reddit episodes")
    parser.add_argument("--classifieds", type=int, nargs='+', default=None, help="classified episodes")

    parser.add_argument("--pick_random_subset", type=int, default=None, help="pick out this number of random episodes across all domains")
    parser.add_argument("--seed", type=int, default=32)
    parser.add_argument("--topk", type=int, default=3)

    parser.add_argument("--w_obs", type=float, default=0.025)
    parser.add_argument("--w_act", type=float, default=0.05)
    parser.add_argument("--w_task", type=float, default=0.9)
    parser.add_argument("--w_vis", type=float, default=0.025)

    parser.add_argument("--using_full_actions_examples", action="store_true", default=False, help="Giving full actions in prompt with in-context examples?")

    parser.add_argument("--shuffle", action="store_true", default=False, help="shuffle examples?")
    parser.add_argument("--no_add_abstractions_system", action="store_true", default=False, help="add abstractions system prompt?")
    parser.add_argument("--ablate_image_context", action="store_true", default=False, help="ablate images in examples?")

    

    # example config
    parser.add_argument("--test_start_idx", type=int, default=0)
    parser.add_argument("--test_end_idx", type=int, default=910)



    # logging related
    parser.add_argument("--result_dir", type=str, default="")
    parser.add_argument("--evalMode", type=str, default="regular")
    args = parser.parse_args()

    # check the whether the action space is compatible with the observation space
    if (
        args.action_set_tag == "id_accessibility_tree"
        and args.observation_type
        not in [
            "accessibility_tree",
            "accessibility_tree_with_captioner",
            "image_som",
        ]
    ):
        raise ValueError(
            f"Action type {args.action_set_tag} is incompatible with the observation type {args.observation_type}"
        )

    return args


@beartype
def early_stop(
    trajectory: Trajectory, max_steps: int, thresholds: dict[str, int]
) -> tuple[bool, str]:
    """Check whether need to stop early"""

    # reach the max step
    num_steps = (len(trajectory) - 1) / 2
    if num_steps >= max_steps:
        return True, f"Reach max steps {max_steps}"

    last_k_actions: list[Action]
    action_seq: list[Action]

    # Case: parsing failure for k times
    k = thresholds["parsing_failure"]
    last_k_actions = trajectory[1::2][-k:]  # type: ignore[assignment]
    if len(last_k_actions) >= k:
        if all(
            [
                action["action_type"] == ActionTypes.NONE
                for action in last_k_actions
            ]
        ):
            return True, f"Failed to parse actions for {k} times"

    # Case: same action for k times
    k = thresholds["repeating_action"]
    last_k_actions = trajectory[1::2][-k:]  # type: ignore[assignment]
    action_seq = trajectory[1::2]  # type: ignore[assignment]

    if len(action_seq) == 0:
        return False, ""

    last_action: Action = action_seq[-1]

    if last_action["action_type"] != ActionTypes.TYPE:
        if len(last_k_actions) >= k:
            if all(
                [
                    is_equivalent(action, last_action)
                    for action in last_k_actions
                ]
            ):
                return True, f"Same action for {k} times"

    else:
        # check the action sequence
        if (
            sum([is_equivalent(action, last_action) for action in action_seq])
            >= k
        ):
            return True, f"Same typing action for {k} times"

    return False, ""


@beartype
def test(
    args: argparse.Namespace,
    config_file_list: list[str]
) -> None:
    scores = []

    trajSOM = {}
    trajImages = {}
    trajActions = {}
    trajInputImages = {}
    trajSuccess = {}
    trajHumanLoopFeedback = {} # dict for human feedfback
    metrics_dict = {}

    if args.save_examples_memory:
        args.result_dir = f'experiments/{args.experiment_name}'
        os.makedirs(args.result_dir, exist_ok=True)
        # load if exists
        if os.path.exists(f'experiments/{args.experiment_name}/metrics.json'):
            with open(f'experiments/{args.experiment_name}/metrics.json') as json_file:
                metrics_dict = json.load(json_file)
        if os.path.exists(f'experiments/{args.experiment_name}/states.pkl'):
            with open(f'experiments/{args.experiment_name}/states.pkl', 'rb') as f:
                trajSOM = pickle.load(f)
        if os.path.exists(f'experiments/{args.experiment_name}/actions.pkl'):
            with open(f'experiments/{args.experiment_name}/actions.pkl', 'rb') as f:
                trajActions = pickle.load(f)
        # if os.path.exists(f'experiments/{args.experiment_name}/images.pkl'):
        #     with open(f'experiments/{args.experiment_name}/images.pkl', 'rb') as f:
        #         trajImages = pickle.load(f)
        # if os.path.exists(f'experiments/{args.experiment_name}/input_images.pkl'):
        #     with open(f'experiments/{args.experiment_name}/input_images.pkl', 'rb') as f:
        #         trajInputImages = pickle.load(f)
        if os.path.exists(f'experiments/{args.experiment_name}/human_feedback.pkl'):
            with open(f'experiments/{args.experiment_name}/human_feedback.pkl', 'rb') as f:
                trajHumanLoopFeedback = pickle.load(f)
        if os.path.exists(f'experiments/{args.experiment_name}/success.pkl'):
            with open(f'experiments/{args.experiment_name}/success.pkl', 'rb') as f:
                trajSuccess = pickle.load(f)

    max_steps = args.max_steps

    memory_save_root = f"data/memory_human_in_the_loop/{args.experiment_name}"
    example_json_file = f"data/memory_human_in_the_loop/{args.experiment_name}/planning_examples.json"

    if args.evalMode=="human_in_the_loop":
        feedback_memory_save_root = f"learned_examples/memory_human_in_the_loop_feedback/{args.experiment_name}"
        feedback_example_json_file = f"learned_examples/memory_human_in_the_loop_feedback/{args.experiment_name}/planning_examples.json"
        args.feedback_jsons.append(args.feedback_instruction_path)
        if args.continually_add_saved_examples:
            args.feedback_jsons.append(feedback_example_json_file)

    args.instruction_jsons.append(args.instruction_path)
    args.instruction_jsons = list(set(args.instruction_jsons))

    if args.continually_add_saved_examples and args.save_examples_memory:
        args.instruction_jsons.append(example_json_file)

    early_stop_thresholds = {
        "parsing_failure": args.parsing_failure_th,
        "repeating_action": args.repeating_action_failure_th,
    }

    if args.observation_type in [
        "accessibility_tree_with_captioner",
        "image_som",
    ]:
        device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        caption_image_fn = image_utils.get_captioning_fn(
            device, dtype, args.captioning_model
        )
    else:
        caption_image_fn = None

    # Load a (possibly different) captioning model for running VQA evals.
    if (
        caption_image_fn
        and args.eval_captioning_model == args.captioning_model
    ):
        eval_caption_image_fn = caption_image_fn
    else:
        eval_caption_image_fn = image_utils.get_captioning_fn(
            args.eval_captioning_model_device,
            torch.float16
            if (
                torch.cuda.is_available()
                and args.eval_captioning_model_device == "cuda"
            )
            else torch.float32,
            args.eval_captioning_model,
        )

    agent = construct_agent(
        args,
        captioning_fn=caption_image_fn
        if args.observation_type == "accessibility_tree_with_captioner"
        else None,
    )  # NOTE: captioning_fn here is used for captioning input images.

    env = ScriptBrowserEnv(
        headless=not args.render,
        slow_mo=args.slow_mo,
        observation_type=args.observation_type,
        current_viewport_only=args.current_viewport_only,
        viewport_size={
            "width": args.viewport_width,
            "height": args.viewport_height,
        },
        save_trace_enabled=args.save_trace_enabled,
        sleep_after_execution=args.sleep_after_execution,
        # NOTE: captioning_fn here is used for LLM + captioning baselines.
        # This can be different from the captioning model used for evals.
        captioning_fn=caption_image_fn,
    )

    seen = set()
    for json_file in args.instruction_jsons:
        if os.path.exists(os.path.join(os.path.split(json_file)[0])):
            with open(os.path.join(os.path.split(json_file)[0], 'seen.json')) as json_file:
                seen_ = json.load(json_file)
            seen.update(seen_["seen"])

    data_dict = {"config":[], "success":[], "num_human_feedbacks":[], "seen":[]}
    
    file_num = 0
    for config_file_idx in range(len(config_file_list)):

        config_file = config_file_list[config_file_idx]

        print(f"Iteration {config_file_idx}/{len(config_file_list)}")
        
        if args.skip_if_exists:
            if config_file in metrics_dict.keys():
                print(f"{config_file} already in metrics... skipping...")
                continue
        
        actionList = []
        imageList = []
        somList = []
        input_images = []
        humanFeedbackList = []
        try:
        # if (1):
            render_helper = RenderHelper(
                config_file, args.result_dir, args.action_set_tag
            )

            # Load task.
            with open(config_file) as f:
                _c = json.load(f)
                intent = _c["intent"]
                task_id = _c["task_id"]
                image_paths = _c.get("image", None)
                site = _c["sites"][0]
            episode_id = f'{site}_{task_id}'
                
            images = []
            # Load input images for the task, if any.
            if image_paths is not None:
                if isinstance(image_paths, str):
                    image_paths = [image_paths]
                inputCount = 0
                for image_path in image_paths:
                    # Load image either from the web or from a local path.
                    if image_path.startswith("http"):
                        import requests
                        from PIL import Image
                        from io import BytesIO
                        try:
                            response = requests.get(image_path)
                            response.raise_for_status()  # ensure the link works
                            input_image = Image.open(BytesIO(response.content))
                            os.makedirs(f'output/{config_file}', exist_ok=True)
                            input_image.save(f'output/{config_file}/input_image{inputCount}.png')
                            inputCount += 1
                            images.append(input_image)
                        except requests.exceptions.RequestException as e:
                            logger.error(f"Request failed: {e}")
                        except PIL.UnidentifiedImageError as e:
                            logger.error(f"Cannot identify image file from the URL: {image_path}")
                        except Exception as e:
                            logger.error(f"Unhandled error: {e}")
                    else:
                        from PIL import Image
                        input_image = Image.open(image_path)
                        os.makedirs(f'output/{config_file}', exist_ok=True)
                        input_image.save(f'output/{config_file}/input_image{inputCount}.png')
                        inputCount += 1
                        images.append(input_image)


            logger.info(f"[Config file]: {config_file}")
            logger.info(f"[Intent]: {intent}")

            agent.reset(config_file)
            trajectory: Trajectory = []
            obs, info = env.reset(options={"config_file": config_file})
            state_info: StateInfo = {"observation": obs, "info": info}
            trajectory.append(state_info)

            meta_data = {"action_history": ["None"]}
            time_step_dict = {"config":[], "time_step":[], "intent":[], "action":[]}
            count = 0
            num_human_feedbacks = 0
            while True:
                imageList.append(obs["image"])
                somList.append(obs["text"])

                early_stop_flag, stop_info = early_stop(
                    trajectory, max_steps, early_stop_thresholds
                )

                if args.evalMode == "regular":
                    if early_stop_flag:
                        action = create_stop_action(f"Early stop: {stop_info}")
                    else:
                        # try:
                        action = agent.next_action(
                            trajectory,
                            intent,
                            images=images,
                            meta_data=meta_data,
                            config_file=config_file,
                        )
                        # except ValueError as e:
                        #     # get the error message
                        #     action = create_stop_action(f"ERROR: {str(e)}")

                elif args.evalMode == "human_in_the_loop":
                    humanFeedbackList.append(None)
                    import matplotlib.pyplot as plt
                    image = np.float32(obs['image'][:,:,:3])
                    # Create a new figure
                    image = image.astype(np.uint8) 
                    plt.figure(1, (24,24)); plt.clf()
                    plt.rcParams['figure.dpi']=500
                    plt.rcParams['savefig.dpi']=500
                    plt.imshow(image)
                    directory = f'output/{config_file}'
                    if not os.path.exists(directory):
                        os.makedirs(directory)
                        print(f"Directory '{directory}' created successfully.")
                    print(f"IMAGE {count} PLOTTING")
                    plt.savefig(f'output/{config_file}/image{count}.png')
                    print("IMAGE SAVED")
                    humanSatisfaction = False
                    if early_stop_flag:
                        action = create_stop_action(f"Early stop: {stop_info}")
                    else:
                        #ipdb.set_trace()
                        action = agent.next_action(
                            trajectory,
                            intent,
                            images=images,
                            meta_data=meta_data,
                            )        
                        # action_str = get_action_description(
                        #     action,
                        #     state_info["info"]["observation_metadata"],
                        #     action_set_tag=args.action_set_tag,
                        #     prompt_constructor=agent.prompt_constructor
                        #     if isinstance(agent, PromptAgent)
                        #     else None,
                        #     )
                        action_str = action2str(action, args.action_set_tag)
                        action_str = action_str.split(' where')[0].replace('[A] ', '')
                        print(f"Agent ACTION: {action_str}")
                        print("Dear User, was my action choice suboptimal?")
                        print("Yes or No?")
                        humanReaction = input("Let me know:")
                        humanReaction = humanReaction.lower()
                        if humanReaction != "no":
                            newAction = action
                            while humanSatisfaction == False:
                                num_human_feedbacks += 1
                                print("I promise to fix my action this time...hopefully?")
                                print("tell me what I did wrong and what I should have done instead!")
                                print("Give me constructive feedback in natural language!")
                                humanFeedback = input("Feedback:")
                                # humanFeedback is the string u want to add to your prompt
                                ### GABE TO DO: Incorporate the human feedback string to the prompt
                                ### want some way to pass the string to the prompt constructor
                                # mutiple feedbacks?
                                newAction = agent.next_action_humanFeedback(
                                    trajectory,
                                    intent,
                                    images=images,
                                    meta_data=meta_data,
                                    humanFeedback=humanFeedback,
                                    prev_action=newAction,
                                    ) 
                                newAction["human_feedback"] = humanFeedback
                                # TIME_STEP, INTENT, STATE, ACTION, FEEDBACK, NEW ACTION
                                feedbackTuple = (count, intent, obs["text"], action, humanFeedback, newAction)
                                if humanFeedbackList[-1] is None:
                                    humanFeedbackList[-1] = [feedbackTuple]
                                else:
                                    humanFeedbackList[-1].append(feedbackTuple)
                                newAction_str = get_action_description(
                                    newAction,
                                    state_info["info"]["observation_metadata"],
                                    action_set_tag=args.action_set_tag,
                                    prompt_constructor=agent.prompt_constructor
                                    if isinstance(agent, PromptAgent)
                                    else None,
                                    )
                                print(f"Agent NEW ACTION: {newAction_str}")                            
                                print("Dear User, was my action choice still suboptimal?")
                                print("Yes or No?")
                                humanReaction = input("Let me know:")
                                humanReaction = humanReaction.lower()
                                if humanReaction == 'no':
                                    print("I trust you! I'm using this action.")
                                    humanSatisfaction = True
                                    action = newAction
                                else:
                                    action = newAction
                                #ipdb.set_trace()
                            # ipdb.set_trace()

                trajectory.append(action)
                actionList.append(action)

                action_str = get_action_description(
                    action,
                    state_info["info"]["observation_metadata"],
                    action_set_tag=args.action_set_tag,
                    prompt_constructor=agent.prompt_constructor
                    if isinstance(agent, PromptAgent)
                    else None,
                )
                render_helper.render(
                    action, state_info, meta_data, args.render_screenshot
                )

                def get_element_line(
                    state,
                    element_id,
                    # prob = 0.2,
                ):
                    state_lines = state.split('\n')
                    element_id_text = '[' + str(element_id) + ']'
                    state_lines_reduced = []
                    element_id_line = ''
                    for s_line in state_lines:
                        if element_id_text in s_line:
                            state_lines_reduced.append(s_line)
                            element_id_line = s_line
                            break
                    return element_id_line
                element_id_line = get_element_line(obs['text'], action['element_id'])
                
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
                action_str_nums = action2str(action, args.action_set_tag)
                action_str_nums = action_str_nums.split(' where')[0].replace('[A] ', '')
                action_str_history = get_action_history_format(action_str_nums, action['element_id'], element_id_line)
                meta_data["action_history"].append(action_str_history)

                print(action_str_history)

                if action["action_type"] == ActionTypes.STOP:
                    break

                visualize_results = False
                if visualize_results:
                    print(action)
                    tag = 'gpt4v'
                    import matplotlib.pyplot as plt
                    rgb = rgb.astype(np.uint8) 
                    plt.figure(1, (24,24)); plt.clf()
                    plt.rcParams['figure.dpi']=500
                    plt.rcParams['savefig.dpi']=500
                    plt.imshow(rgb)
                    plt.title(f'{action_str}')
                    os.makedirs(f'output/{tag}', exist_ok=True)
                    plt.savefig(f'output/{tag}/image{count}.png')
                count += 1

                # rgb = np.float32(obs['image'][:,:,:3])
                from PIL import Image
                from PIL import ImageFont
                from PIL import ImageDraw 
                im = Image.fromarray(obs['image'][:,:,:3])
                draw = ImageDraw.Draw(im)
                # font = ImageFont.load_default(size=60)
                font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeMono.ttf", 48)
                draw.text((40, 40), f"{action_str}", fill=(0, 0, 0), font=font)
                font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeMono.ttf", 24)
                draw.text((40, 80), f"{intent}", fill=(0, 0, 0), font=font)
                if args.save_examples_memory:
                    image_dir = f'experiments/{args.experiment_name}/{config_file}/images'
                    os.makedirs(image_dir, exist_ok=True)
                    im.save(os.path.join(image_dir, f'{count}.png'))

                time_step_dict["config"].append(config_file)
                time_step_dict["time_step"].append(count)
                time_step_dict["intent"].append(intent)
                time_step_dict["action"].append(action_str_nums)
                tbl = wandb.Table(columns=list(time_step_dict.keys()))
                for idx in range(len(time_step_dict["config"])):
                    tbl.add_data(*[
                        time_step_dict["config"][idx], 
                        time_step_dict["time_step"][idx],
                        time_step_dict["intent"][idx],
                        time_step_dict["action"][idx],
                        ])
                wandb.log({f"time_step_action/{config_file}": tbl})

                obs, _, terminated, _, info = env.step(action)
                state_info = {"observation": obs, "info": info}
                trajectory.append(state_info)

                if terminated:
                    # add a action place holder
                    trajectory.append(create_stop_action(""))
                    break

            trajActions[config_file] = actionList
            trajSOM[config_file] = somList
            # trajImages[config_file] = imageList
            trajHumanLoopFeedback[config_file] = humanFeedbackList

            # NOTE: eval_caption_image_fn is used for running eval_vqa functions.

            evaluator = evaluator_router(
                config_file, captioning_fn=eval_caption_image_fn
            )
            score = evaluator(
                trajectory=trajectory,
                config_file=config_file,
                page=env.page,
                client=env.get_page_client(env.page),
            )
            scores.append(score)

            seen_episode = episode_id in seen

            data_dict["config"].append(config_file)
            data_dict["success"].append(score)
            data_dict["num_human_feedbacks"].append(num_human_feedbacks)
            data_dict["seen"].append(seen_episode)
            tbl = wandb.Table(columns=list(data_dict.keys()))
            for idx in range(len(data_dict["config"])):
                tbl.add_data(*[
                    data_dict["config"][idx], 
                    data_dict["success"][idx],
                    data_dict["num_human_feedbacks"][idx],
                    data_dict["seen"][idx],
                    ])
            wandb.log({f"metrics_summary/metrics": tbl})

            metrics_dict[config_file] = {"config":config_file, "success":score, "seen":seen_episode}

            file_num += 1

            if score == 1:
                logger.info(f"[Result] (PASS) {config_file}")
            else:
                logger.info(f"[Result] (FAIL) {config_file}")

            trajSuccess[config_file] = score

            if args.save_trace_enabled:
                env.save_trace(
                    Path(args.result_dir) / "traces" / f"{task_id}.zip"
                )

            if args.save_examples_memory:
                print("Saving jsons...")
                os.makedirs(f'experiments/{args.experiment_name}', exist_ok=True)

                with open(f'experiments/{args.experiment_name}/states.pkl', 'wb') as f:
                    pickle.dump(trajSOM, f)

                # # Write dictionary with lists of strings to disk
                # with open(f'experiments/{args.experiment_name}/images.pkl', 'wb') as f:
                #     pickle.dump(trajImages, f)

                # Write dictionary with lists of dictionaries to disk
                with open(f'experiments/{args.experiment_name}/actions.pkl', 'wb') as f:
                    pickle.dump(trajActions, f)
                
                # with open(f'experiments/{args.experiment_name}/input_images.pkl', 'wb') as f:
                #     pickle.dump(trajInputImages, f)
                
                with open(f'experiments/{args.experiment_name}/human_feedback.pkl', 'wb') as f:
                    pickle.dump(trajHumanLoopFeedback, f)

                with open(f'experiments/{args.experiment_name}/success.pkl', 'wb') as f:
                    pickle.dump(trajSuccess, f)

                with open(f'experiments/{args.experiment_name}/metrics.json', "w") as outfile: 
                    json.dump(metrics_dict, outfile, indent=4, sort_keys=True)

                if args.evalMode=="human_in_the_loop":
                    os.makedirs(memory_save_root, exist_ok=True)
                    os.makedirs(feedback_memory_save_root, exist_ok=True)
                    save_example(
                        actionList,
                        somList,
                        imageList,
                        humanFeedbackList,
                        score,
                        config_file,
                        example_json_file,
                        memory_save_root+'/images',
                        feedback_example_json_file,
                        feedback_memory_save_root+'/images',
                        action_set_tag=args.action_set_tag,
                    )

                    if args.continually_add_saved_examples:
                        agent.prompt_constructor.refresh_examples()
        
        except openai.OpenAIError as e:
            if args.debug:
                raise
            logger.info(f"[OpenAI Error] {repr(e)}")
            if args.save_examples_memory:
                seen_episode = episode_id in seen
                metrics_dict[config_file] = {"config":config_file, "success":f"[OpenAI Error] {repr(e)}", "seen":seen_episode}
        
        except Exception as e:
            if args.debug:
                raise
            logger.info(f"[Unhandled Error] {repr(e)}]")
            import traceback

            # write to error file
            with open(Path(args.result_dir) / "error.txt", "a") as f:
                f.write(f"[Config file]: {config_file}\n")
                f.write(f"[Unhandled Error] {repr(e)}\n")
                f.write(traceback.format_exc())  # write stack trace to file

            if args.save_examples_memory:
                seen_episode = episode_id in seen
                metrics_dict[config_file] = {"config":config_file, "success":f"[Unhandled Error] {repr(e)}]", "seen":seen_episode}
        
        render_helper.close()

    env.close()
    logger.info(f"Average score: {sum(scores) / len(scores)}")


def prepare(args: argparse.Namespace) -> None:
    # convert prompt python files to json
    from agent.prompts import to_json

    to_json.run()

    # prepare result dir
    result_dir = args.result_dir
    if not result_dir:
        result_dir = (
            f"cache/results_{time.strftime('%Y%m%d%H%M%S', time.localtime())}"
        )
    if not Path(result_dir).exists():
        Path(result_dir).mkdir(parents=True, exist_ok=True)
        args.result_dir = result_dir
        logger.info(f"Create result dir: {result_dir}")

    if not (Path(result_dir) / "traces").exists():
        (Path(result_dir) / "traces").mkdir(parents=True)

    # log the log file
    with open(os.path.join(result_dir, "log_files.txt"), "a+") as f:
        f.write(f"{LOG_FILE_NAME}\n")


def get_unfinished(config_files: list[str], result_dir: str) -> list[str]:
    result_files = glob.glob(f"{result_dir}/*.html")
    task_ids = [
        os.path.basename(f).split(".")[0].split("_")[1] for f in result_files
    ]
    unfinished_configs = []
    for config_file in config_files:
        task_id = os.path.basename(config_file).split(".")[0]
        if task_id not in task_ids:
            unfinished_configs.append(config_file)
    return unfinished_configs


@beartype
def dump_config(args: argparse.Namespace) -> None:
    config_file = Path(args.result_dir) / "config.json"
    if not config_file.exists():
        with open(config_file, "w") as f:
            json.dump(vars(args), f, indent=4)
            logger.info(f"Dump config to {config_file}")


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args = config()
    args.sleep_after_execution = 2.5
    prepare(args)

    test_config_base_dir = args.test_config_base_dir

    if args.shopping is not None or args.reddit is not None or args.classifieds is not None:
        test_file_list = []
        if args.shopping is not None:
            config_folder = 'config_files/test_shopping'
            for i in args.shopping:
                test_file_list.append(os.path.join(config_folder, f"{i}.json"))
        if args.reddit is not None:
            config_folder = 'config_files/test_reddit'
            for i in args.reddit:
                test_file_list.append(os.path.join(config_folder, f"{i}.json"))
        if args.classifieds is not None:
            config_folder = 'config_files/test_classifieds'
            for i in args.classifieds:
                test_file_list.append(os.path.join(config_folder, f"{i}.json"))
        if args.shuffle:
            np.random.seed(args.seed)
            random_idxs = np.random.choice(list(range(len(test_file_list))), size=len(test_file_list), replace=False)
            test_file_list = [test_file_list[idx] for idx in list(random_idxs)]
    elif args.pick_random_subset is not None:
        test_file_list_ = []
        config_folder = 'config_files/test_shopping'
        test_file_list_ += glob.glob(config_folder+'/*')
        config_folder = 'config_files/test_reddit'
        test_file_list_ += glob.glob(config_folder+'/*')
        config_folder = 'config_files/test_classifieds'
        test_file_list_ += glob.glob(config_folder+'/*')
        np.random.seed(args.seed)
        random_idxs = np.random.choice(list(range(len(test_file_list_))), size=args.pick_random_subset, replace=False)
        random_idxs = np.flip(random_idxs)
        # random_idxs = random_idxs[100:]
        test_file_list = []
        for idx in list(random_idxs):
            test_file_list.append(test_file_list_[idx])
    else:
        test_file_list = []
        st_idx = args.test_start_idx
        ed_idx = args.test_end_idx
        for i in range(st_idx, ed_idx):
            test_file_list.append(os.path.join(test_config_base_dir, f"{i}.json"))
    if args.skip_if_finished:
        test_file_list = get_unfinished(test_file_list, args.result_dir)
    print(f"Total {len(test_file_list)} tasks left")
    args.render = False
    args.render_screenshot = True
    args.save_trace_enabled = True

    args.current_viewport_only = True
    dump_config(args)

    if args.debug:
        wandb.init(mode="disabled")
    else:
        wandb.init(project="visual-web-arena", name=args.experiment_name, config=args, dir=args.wandb_directory)

    test(args, test_file_list)

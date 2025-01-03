"""
Script to run end-to-end evaluation on VisualWebArena benchmark.
"""

import argparse
import glob
import json
import logging
import os
import pickle
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple, Union

import openai
import requests
import torch
import wandb
from beartype import beartype
from PIL import Image
from io import BytesIO

# Browser environment imports
from browser_env.actions import action2str, is_equivalent
from browser_env import (
    Action,
    ActionTypes,
    ScriptBrowserEnv,
    StateInfo,
    Trajectory,
    create_stop_action,
)
from browser_env.helper_functions import (
    RenderHelper,
    get_action_description,
)

# Agent imports
from agent import PromptAgent, construct_agent
from agent.prompts import *
from agent.prompts.save_memory_example import save_example

# Evaluation harness imports
from evaluation_harness import evaluator_router, image_utils

# Local utility imports
from arguments import config
from human_in_the_loop import handle_human_in_the_loop

# Logging setup
LOG_FOLDER = "log_files"
Path(LOG_FOLDER).mkdir(parents=True, exist_ok=True)
LOG_FILE_NAME = (
    f"{LOG_FOLDER}/log_{time.strftime('%Y%m%d%H%M%S', time.localtime())}_"
    f"{random.randint(0, 10000)}.log"
)

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


@beartype
def early_stop(
    trajectory: Trajectory,
    max_steps: int,
    thresholds: Dict[str, int],
) -> Tuple[bool, str]:
    """
    Determine if the agent should stop early, either due to reaching the maximum
    number of steps or by hitting repeated or failed parsing actions.

    Args:
        trajectory (Trajectory): The full trajectory of states and actions so far.
        max_steps (int): Maximum number of steps before forced early-stop.
        thresholds (Dict[str, int]): Thresholds for repeated actions and parsing failures.

    Returns:
        (bool, str): A tuple indicating whether to stop and the reason string if so.
    """
    num_steps = (len(trajectory) - 1) / 2
    if num_steps >= max_steps:
        return True, f"Reach max steps {max_steps}"

    # Check for consecutive parsing failures
    k = thresholds["parsing_failure"]
    last_k_actions = trajectory[1::2][-k:]  # type: ignore[assignment]
    if len(last_k_actions) >= k:
        if all(action["action_type"] == ActionTypes.NONE for action in last_k_actions):
            return True, f"Failed to parse actions for {k} times"

    # Check for repeated actions
    k = thresholds["repeating_action"]
    last_k_actions = trajectory[1::2][-k:]  # type: ignore[assignment]
    action_seq = trajectory[1::2]  # type: ignore[assignment]

    if not action_seq:
        return False, ""

    last_action: Action = action_seq[-1]

    if last_action["action_type"] != ActionTypes.TYPE:
        if len(last_k_actions) >= k:
            if all(is_equivalent(action, last_action) for action in last_k_actions):
                return True, f"Same action for {k} times"
    else:
        # Special handling for typing actions
        count_equivalent = sum(is_equivalent(action, last_action) for action in action_seq)
        if count_equivalent >= k:
            return True, f"Same typing action for {k} times"

    return False, ""


@beartype
def test(
    args: argparse.Namespace,
    config_file_list: List[str],
) -> None:
    """
    Main testing routine for running tasks end-to-end in a ScriptBrowserEnv
    with a constructed agent.

    Args:
        args (argparse.Namespace): Namespace containing configuration arguments.
        config_file_list (List[str]): List of config files (tasks) to evaluate.
    """
    scores: List[Union[int, float]] = []

    # Trajectories and metadata
    trajSOM = {}
    trajImages = {}
    trajActions = {}
    trajInputImages = {}
    trajSuccess = {}
    trajHumanLoopFeedback = {}
    metrics_dict: Dict[str, Union[int, Dict[str, Union[str, int]]]] = {}

    # If saving example memory, load existing data
    if args.save_examples_memory:
        args.result_dir = f"experiments/{args.experiment_name}"
        os.makedirs(args.result_dir, exist_ok=True)

        metrics_json_path = f"experiments/{args.experiment_name}/metrics.json"
        if os.path.exists(metrics_json_path):
            with open(metrics_json_path) as json_file:
                metrics_dict = json.load(json_file)

        states_path = f"experiments/{args.experiment_name}/states.pkl"
        if os.path.exists(states_path):
            with open(states_path, "rb") as f:
                trajSOM = pickle.load(f)

        actions_path = f"experiments/{args.experiment_name}/actions.pkl"
        if os.path.exists(actions_path):
            with open(actions_path, "rb") as f:
                trajActions = pickle.load(f)

        human_feedback_path = f"experiments/{args.experiment_name}/human_feedback.pkl"
        if os.path.exists(human_feedback_path):
            with open(human_feedback_path, "rb") as f:
                trajHumanLoopFeedback = pickle.load(f)

        success_path = f"experiments/{args.experiment_name}/success.pkl"
        if os.path.exists(success_path):
            with open(success_path, "rb") as f:
                trajSuccess = pickle.load(f)

    max_steps = args.max_steps

    # File paths for saving memory
    memory_save_root = f"data/memory_human_in_the_loop/{args.experiment_name}"
    example_json_file = (
        f"data/memory_human_in_the_loop/{args.experiment_name}/planning_examples.json"
    )

    # Additional human-in-the-loop setup
    if args.evalMode == "human_in_the_loop":
        feedback_memory_save_root = (
            f"learned_examples/memory_human_in_the_loop_feedback/{args.experiment_name}"
        )
        feedback_example_json_file = (
            f"learned_examples/memory_human_in_the_loop_feedback/"
            f"{args.experiment_name}/planning_examples.json"
        )
        args.feedback_jsons.append(args.feedback_instruction_path)
        if args.continually_add_saved_examples:
            args.feedback_jsons.append(feedback_example_json_file)

    # De-duplicate instruction files
    args.instruction_jsons.append(args.instruction_path)
    args.instruction_jsons = list(set(args.instruction_jsons))

    if args.continually_add_saved_examples and args.save_examples_memory:
        args.instruction_jsons.append(example_json_file)

    # Early-stop thresholds
    early_stop_thresholds = {
        "parsing_failure": args.parsing_failure_th,
        "repeating_action": args.repeating_action_failure_th,
    }

    # Prepare captioning if needed
    if args.observation_type in ["accessibility_tree_with_captioner", "image_som"]:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        caption_image_fn = image_utils.get_captioning_fn(
            device, dtype, args.captioning_model
        )
    else:
        caption_image_fn = None

    # Load an additional model for evaluation if necessary
    if caption_image_fn and args.eval_captioning_model == args.captioning_model:
        eval_caption_image_fn = caption_image_fn
    else:
        device_for_eval = args.eval_captioning_model_device
        dtype_for_eval = (
            torch.float16
            if (torch.cuda.is_available() and device_for_eval == "cuda")
            else torch.float32
        )
        eval_caption_image_fn = image_utils.get_captioning_fn(
            device_for_eval,
            dtype_for_eval,
            args.eval_captioning_model,
        )

    # Construct the agent with (possibly) a captioning function
    agent = construct_agent(
        args,
        captioning_fn=caption_image_fn
        if args.observation_type == "accessibility_tree_with_captioner"
        else None,
    )

    # Initialize browser environment
    env = ScriptBrowserEnv(
        headless=not args.render,
        slow_mo=args.slow_mo,
        observation_type=args.observation_type,
        current_viewport_only=args.current_viewport_only,
        viewport_size={"width": args.viewport_width, "height": args.viewport_height},
        save_trace_enabled=args.save_trace_enabled,
        sleep_after_execution=args.sleep_after_execution,
        captioning_fn=caption_image_fn,  # for LLM + captioning baselines
    )

    # Load seen tasks
    seen = set()
    for json_file in args.instruction_jsons:
        folder = os.path.split(json_file)[0]
        if os.path.exists(os.path.join(folder)):
            with open(os.path.join(folder, "seen.json")) as seen_file:
                seen_data = json.load(seen_file)
            seen.update(seen_data["seen"])

    data_dict = {"config": [], "success": [], "num_human_feedbacks": [], "seen": []}

    # Main loop over config files
    for idx, config_file in enumerate(config_file_list):
        logger.info(f"Iteration {idx}/{len(config_file_list)}: {config_file}")

        if args.skip_if_exists and config_file in metrics_dict:
            logger.info(f"{config_file} already in metrics... Skipping.")
            continue

        action_list = []
        image_list = []
        som_list = []
        human_feedback_list = []

        try:
            # Prepare a helper for rendering
            render_helper = RenderHelper(
                config_file,
                args.result_dir,
                args.action_set_tag,
            )

            # Load task data
            with open(config_file) as f:
                config_data = json.load(f)
                intent = config_data["intent"]
                task_id = config_data["task_id"]
                image_paths = config_data.get("image", None)
                site = config_data["sites"][0]

            episode_id = f"{site}_{task_id}"
            images = []

            # If the task has images, download or load them
            if image_paths is not None:
                if isinstance(image_paths, str):
                    image_paths = [image_paths]

                input_count = 0
                for img_path in image_paths:
                    if img_path.startswith("http"):
                        try:
                            response = requests.get(img_path)
                            response.raise_for_status()
                            input_img = Image.open(BytesIO(response.content))
                            os.makedirs(f"output/{config_file}", exist_ok=True)
                            out_path = f"output/{config_file}/input_image{input_count}.png"
                            input_img.save(out_path)
                            images.append(input_img)
                            input_count += 1
                        except requests.exceptions.RequestException as e:
                            logger.error(f"Request failed: {e}")
                        except Image.UnidentifiedImageError:
                            logger.error(f"Cannot identify image file from URL: {img_path}")
                        except Exception as e:
                            logger.error(f"Unhandled error: {e}")
                    else:
                        input_img = Image.open(img_path)
                        os.makedirs(f"output/{config_file}", exist_ok=True)
                        out_path = f"output/{config_file}/input_image{input_count}.png"
                        input_img.save(out_path)
                        images.append(input_img)
                        input_count += 1

            logger.info(f"[Config file] {config_file}")
            logger.info(f"[Intent] {intent}")

            agent.reset(config_file)
            trajectory: Trajectory = []

            # Environment reset
            obs, info = env.reset(options={"config_file": config_file})
            state_info: StateInfo = {"observation": obs, "info": info}
            trajectory.append(state_info)

            meta_data = {"action_history": ["None"]}

            count = 0
            num_human_feedbacks = 0

            # Step through environment until STOP
            while True:
                image_list.append(obs["image"])
                som_list.append(obs["text"])

                # Early-stop check
                should_stop, stop_reason = early_stop(trajectory, max_steps, early_stop_thresholds)

                if args.evalMode == "regular":
                    if should_stop:
                        action = create_stop_action(f"Early stop: {stop_reason}")
                    else:
                        action = agent.next_action(
                            trajectory,
                            intent,
                            images=images,
                            meta_data=meta_data,
                            config_file=config_file,
                        )
                if args.evalMode == "human_in_the_loop":
                    # For interactive feedback mode
                    human_feedback_list.append(None)
                    if should_stop:
                        action = create_stop_action(f"Early stop: {stop_reason}")
                    else:
                        action = agent.next_action(
                            trajectory,
                            intent,
                            images=images,
                            meta_data=meta_data,
                        )
                    
                    # Initialize feedback list for the current task if necessary
                    if len(human_feedback_list) <= count:
                        human_feedback_list.append(None)

                    # Handle the human-in-the-loop interaction
                    action, feedbacks_received = handle_human_in_the_loop(
                        agent=agent,
                        trajectory=trajectory,
                        intent=intent,
                        images=images,
                        meta_data=meta_data,
                        config_file=config_file,
                        action=action,
                        args=args,
                        count=count,
                        human_feedback_list=human_feedback_list,
                    )
                    num_human_feedbacks += feedbacks_received

                trajectory.append(action)
                action_list.append(action)

                # Format action for display/logging
                action_str = get_action_description(
                    action,
                    state_info["info"]["observation_metadata"],
                    action_set_tag=args.action_set_tag,
                    prompt_constructor=(
                        agent.prompt_constructor if isinstance(agent, PromptAgent) else None
                    ),
                )
                logger.info(f"Action selected: {action_str}")
                render_helper.render(action, state_info, meta_data, args.render_screenshot)

                # Helper to get the line from the textual state for the chosen element
                def get_element_line(state: str, element_id: int) -> str:
                    lines = state.split("\n")
                    element_str = f"[{element_id}]"
                    for line in lines:
                        if element_str in line:
                            return line
                    return ""

                element_id_line = get_element_line(obs["text"], action["element_id"])

                # Create an action-history-friendly version of the string
                def get_action_history_format(
                    raw_action_str: str, element_id: int, element_line: str
                ) -> str:
                    element_str = f"[{element_id}]"
                    replaced = element_line.replace(element_str, "")
                    if replaced and replaced.startswith(" "):
                        replaced = replaced[1:]
                    return raw_action_str.replace(element_str, replaced)

                # Remove extra info from action string
                action_str_nums = action2str(action, args.action_set_tag)
                action_str_nums = action_str_nums.split(" where")[0].replace("[A] ", "")
                action_str_history = get_action_history_format(
                    action_str_nums, action["element_id"], element_id_line
                )
                meta_data["action_history"].append(action_str_history)

                if action["action_type"] == ActionTypes.STOP:
                    break

                # Draw action and intent on the image for debug or memory saving
                from PIL import ImageDraw, ImageFont

                rendered_im = Image.fromarray(obs["image"][:, :, :3])
                draw = ImageDraw.Draw(rendered_im)
                # Example fonts; adjust if needed
                font_large = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeMono.ttf", 48)
                font_small = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeMono.ttf", 24)

                draw.text((40, 40), action_str, fill=(0, 0, 0), font=font_large)
                draw.text((40, 80), intent, fill=(0, 0, 0), font=font_small)

                if args.save_examples_memory:
                    image_dir = f"experiments/{args.experiment_name}/{config_file}/images"
                    os.makedirs(image_dir, exist_ok=True)
                    rendered_im.save(os.path.join(image_dir, f"{count}.png"))

                # Log steps to W&B
                time_step_dict = {
                    "config": config_file,
                    "time_step": count,
                    "intent": intent,
                    "action": action_str_nums,
                }
                table = wandb.Table(columns=list(time_step_dict.keys()))
                table.add_data(
                    time_step_dict["config"],
                    time_step_dict["time_step"],
                    time_step_dict["intent"],
                    time_step_dict["action"],
                )
                wandb.log({f"time_step_action/{config_file}": table})

                # Environment step
                obs, _, terminated, _, info = env.step(action)
                state_info = {"observation": obs, "info": info}
                trajectory.append(state_info)

                if terminated:
                    # Ensure the trajectory ends cleanly
                    trajectory.append(create_stop_action(""))
                    break

                count += 1

            # Store trajectory info
            trajActions[config_file] = action_list
            trajSOM[config_file] = som_list
            trajHumanLoopFeedback[config_file] = human_feedback_list

            # Evaluate this trajectory
            evaluator = evaluator_router(config_file, captioning_fn=eval_caption_image_fn)
            score = evaluator(
                trajectory=trajectory,
                config_file=config_file,
                page=env.page,
                client=env.get_page_client(env.page),
            )
            scores.append(score)

            seen_episode = episode_id in seen

            # Summarize data for W&B
            data_dict["config"].append(config_file)
            data_dict["success"].append(score)
            data_dict["num_human_feedbacks"].append(num_human_feedbacks)
            data_dict["seen"].append(seen_episode)
            summary_table = wandb.Table(columns=list(data_dict.keys()))
            for idx_data in range(len(data_dict["config"])):
                summary_table.add_data(
                    data_dict["config"][idx_data],
                    data_dict["success"][idx_data],
                    data_dict["num_human_feedbacks"][idx_data],
                    data_dict["seen"][idx_data],
                )
            wandb.log({"metrics_summary/metrics": summary_table})

            # Save scores in local dictionary
            metrics_dict[config_file] = {
                "config": config_file,
                "success": score,
                "seen": seen_episode,
            }

            if score == 1:
                logger.info(f"[Result] (PASS) {config_file}")
            else:
                logger.info(f"[Result] (FAIL) {config_file}")

            trajSuccess[config_file] = score

            # Optionally save trace
            if args.save_trace_enabled:
                trace_path = Path(args.result_dir) / "traces" / f"{task_id}.zip"
                env.save_trace(trace_path)

            # Save memory and metrics if required
            if args.save_examples_memory:
                logger.info("Saving memory files...")
                os.makedirs(f"experiments/{args.experiment_name}", exist_ok=True)

                with open(f"experiments/{args.experiment_name}/states.pkl", "wb") as f:
                    pickle.dump(trajSOM, f)

                with open(f"experiments/{args.experiment_name}/actions.pkl", "wb") as f:
                    pickle.dump(trajActions, f)

                with open(f"experiments/{args.experiment_name}/human_feedback.pkl", "wb") as f:
                    pickle.dump(trajHumanLoopFeedback, f)

                with open(f"experiments/{args.experiment_name}/success.pkl", "wb") as f:
                    pickle.dump(trajSuccess, f)

                with open(f"experiments/{args.experiment_name}/metrics.json", "w") as outfile:
                    json.dump(metrics_dict, outfile, indent=4, sort_keys=True)

                # For human-in-the-loop mode, store examples if needed
                if args.evalMode == "human_in_the_loop":
                    os.makedirs(memory_save_root, exist_ok=True)
                    os.makedirs(feedback_memory_save_root, exist_ok=True)
                    save_example(
                        action_list,
                        som_list,
                        image_list,
                        human_feedback_list,
                        score,
                        config_file,
                        example_json_file,
                        f"{memory_save_root}/images",
                        feedback_example_json_file,
                        f"{feedback_memory_save_root}/images",
                        action_set_tag=args.action_set_tag,
                    )

                    if args.continually_add_saved_examples:
                        agent.prompt_constructor.refresh_examples()

        except openai.OpenAIError as e:
            if args.debug:
                raise
            logger.error(f"[OpenAI Error] {repr(e)}")

            if args.save_examples_memory:
                seen_episode = episode_id in seen
                metrics_dict[config_file] = {
                    "config": config_file,
                    "success": f"[OpenAI Error] {repr(e)}",
                    "seen": seen_episode,
                }

        except Exception as e:
            if args.debug:
                raise
            logger.error(f"[Unhandled Error] {repr(e)}")

            import traceback
            error_file = Path(args.result_dir) / "error.txt"
            with open(error_file, "a") as err_f:
                err_f.write(f"[Config file]: {config_file}\n")
                err_f.write(f"[Unhandled Error] {repr(e)}\n")
                err_f.write(traceback.format_exc())

            if args.save_examples_memory:
                seen_episode = episode_id in seen
                metrics_dict[config_file] = {
                    "config": config_file,
                    "success": f"[Unhandled Error] {repr(e)}",
                    "seen": seen_episode,
                }
        finally:
            render_helper.close()

    env.close()
    if scores:
        logger.info(f"Average score: {sum(scores) / len(scores)}")
    else:
        logger.info("No scores collected (possibly no valid tasks).")


@beartype
def prepare(args: argparse.Namespace) -> None:
    """
    Prepare directories and convert prompt Python files to JSON prior to running tests.

    Args:
        args (argparse.Namespace): Configuration arguments.
    """
    from agent.prompts import to_json
    to_json.run()

    # Prepare result dir
    result_dir = args.result_dir
    if not result_dir:
        ts = time.strftime("%Y%m%d%H%M%S", time.localtime())
        result_dir = f"cache/results_{ts}"
        args.result_dir = result_dir

    Path(result_dir).mkdir(parents=True, exist_ok=True)
    logger.info(f"Create result dir: {result_dir}")

    # Create traces folder
    traces_dir = Path(result_dir) / "traces"
    traces_dir.mkdir(parents=True, exist_ok=True)

    # Log the log file
    with open(os.path.join(result_dir, "log_files.txt"), "a+") as f:
        f.write(f"{LOG_FILE_NAME}\n")


def get_unfinished(config_files: List[str], result_dir: str) -> List[str]:
    """
    Filter out tasks (config files) that already have results in the result directory.

    Args:
        config_files (List[str]): All config files to be processed.
        result_dir (str): Directory where results are stored.

    Returns:
        unfinished_configs (List[str]): Subset of config_files not yet completed.
    """
    result_files = glob.glob(f"{result_dir}/*.html")
    task_ids = [os.path.basename(f).split(".")[0].split("_")[1] for f in result_files]
    unfinished_configs = []
    for config_file in config_files:
        task_id = os.path.basename(config_file).split(".")[0]
        if task_id not in task_ids:
            unfinished_configs.append(config_file)
    return unfinished_configs


@beartype
def dump_config(args: argparse.Namespace) -> None:
    """
    Dump the current args to config.json in the result directory, if not already dumped.

    Args:
        args (argparse.Namespace): Configuration arguments.
    """
    config_file = Path(args.result_dir) / "config.json"
    if not config_file.exists():
        with open(config_file, "w") as f:
            json.dump(vars(args), f, indent=4)
            logger.info(f"Dump config to {config_file}")


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args = config()
    args.sleep_after_execution = 2.5

    # Prepare directories and prompt conversions
    prepare(args)

    test_config_base_dir = args.test_config_base_dir

    # Build test file list according to user-specified arguments
    if any([args.shopping, args.reddit, args.classifieds]):
        test_file_list = []
        if args.shopping:
            config_folder = "config_files/test_shopping"
            for i in args.shopping:
                test_file_list.append(os.path.join(config_folder, f"{i}.json"))
        if args.reddit:
            config_folder = "config_files/test_reddit"
            for i in args.reddit:
                test_file_list.append(os.path.join(config_folder, f"{i}.json"))
        if args.classifieds:
            config_folder = "config_files/test_classifieds"
            for i in args.classifieds:
                test_file_list.append(os.path.join(config_folder, f"{i}.json"))

        if args.shuffle:
            import numpy as np

            np.random.seed(args.seed)
            indices = np.random.permutation(len(test_file_list))
            test_file_list = [test_file_list[idx] for idx in indices]
    elif args.pick_random_subset is not None:
        import numpy as np

        test_file_list_ = []
        for folder_name in ["test_shopping", "test_reddit", "test_classifieds"]:
            folder_path = f"config_files/{folder_name}"
            test_file_list_.extend(glob.glob(folder_path + "/*"))

        np.random.seed(args.seed)
        random_idxs = np.random.choice(
            range(len(test_file_list_)),
            size=args.pick_random_subset,
            replace=False,
        )
        test_file_list = [test_file_list_[idx] for idx in np.flip(random_idxs)]
    else:
        # Default range-based approach
        test_file_list = []
        for i in range(args.test_start_idx, args.test_end_idx):
            test_file_list.append(os.path.join(test_config_base_dir, f"{i}.json"))

    if args.skip_if_finished:
        test_file_list = get_unfinished(test_file_list, args.result_dir)

    logger.info(f"Total {len(test_file_list)} tasks left")

    # Render settings
    args.render = False
    args.render_screenshot = True
    args.save_trace_enabled = True
    args.current_viewport_only = True

    # Save config
    dump_config(args)

    # Initialize wandb
    if args.debug:
        wandb.init(mode="disabled")
    else:
        wandb.init(
            project="visual-web-arena",
            name=args.experiment_name,
            config=args,
            dir=args.wandb_directory,
        )

    # Run tests
    test(args, test_file_list)

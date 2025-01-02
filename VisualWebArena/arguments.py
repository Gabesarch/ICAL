import argparse

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
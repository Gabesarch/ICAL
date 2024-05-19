"""Config for language models."""

from __future__ import annotations

import argparse
import dataclasses
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class LMConfig:
    """A config for a language model.

    Attributes:
        provider: The name of the API provider.
        model: The name of the model.
        model_cls: The Python class corresponding to the model, mostly for
             Hugging Face transformers.
        tokenizer_cls: The Python class corresponding to the tokenizer, mostly
            for Hugging Face transformers.
        mode: The mode of the API calls, e.g., "chat" or "generation".
    """

    provider: str
    model: str
    model_cls: type | None = None
    tokenizer_cls: type | None = None
    mode: str | None = None
    gen_config: dict[str, Any] = dataclasses.field(default_factory=dict)


def construct_llm_config(args: argparse.Namespace) -> LMConfig:
    llm_config = LMConfig(
        provider=args.provider, model=args.model, mode=args.mode
    )
    if args.provider in ["openai", "google"]:
        llm_config.gen_config["temperature"] = args.temperature
        llm_config.gen_config["top_p"] = args.top_p
        llm_config.gen_config["context_length"] = args.context_length
        llm_config.gen_config["max_tokens"] = args.max_tokens
        llm_config.gen_config["stop_token"] = args.stop_token
        llm_config.gen_config["max_obs_length"] = args.max_obs_length
        llm_config.gen_config["max_retry"] = args.max_retry
        llm_config.gen_config["eval_mode"] = args.evalMode
        llm_config.gen_config["instruction_jsons"] = args.instruction_jsons
        llm_config.gen_config["feedback_jsons"] = args.feedback_jsons
        llm_config.gen_config["topk"] = args.topk
        llm_config.gen_config["w_obs"] = args.w_obs
        llm_config.gen_config["w_act"] = args.w_act
        llm_config.gen_config["w_task"] = args.w_task
        llm_config.gen_config["w_vis"] = args.w_vis
        llm_config.gen_config["experiment_name"] = args.experiment_name
        llm_config.gen_config["using_full_actions_examples"] = args.using_full_actions_examples
        llm_config.gen_config["no_add_abstractions_system"] = args.no_add_abstractions_system
        llm_config.gen_config["ablate_image_context"] = args.ablate_image_context
    elif args.provider == "huggingface":
        llm_config.gen_config["temperature"] = args.temperature
        llm_config.gen_config["top_p"] = args.top_p
        llm_config.gen_config["max_new_tokens"] = args.max_tokens
        llm_config.gen_config["stop_sequences"] = (
            [args.stop_token] if args.stop_token else None
        )
        llm_config.gen_config["max_obs_length"] = args.max_obs_length
        llm_config.gen_config["model_endpoint"] = args.model_endpoint
        llm_config.gen_config["max_retry"] = args.max_retry
    else:
        raise NotImplementedError(f"provider {args.provider} not implemented")
    return llm_config

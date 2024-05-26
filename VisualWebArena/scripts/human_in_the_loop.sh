#!/bin/sh
export AZURE_OPENAI_KEY="YOUR_KEY_HERE"
export AZURE_OPENAI_ENDPOINT="YOUR_ENDPOINT_HERE"
export RESULTS_DIR="output"
python run_eval.py \
  --instruction_path learned_examples/human_demos_with_abstractions/planning_examples.json \
  --test_start_idx 0 \
  --test_end_idx 1 \
  --result_dir $RESULTS_DIR \
  --test_config_base_dir=config_files/test_classifieds \
  --model gpt-4-vision-preview \
  --temperature 0.2 \
  --top_p 0.1 \
  --test_start_idx 70 \
  --test_end_idx 75 \
  --action_set_tag som \
  --save_trace_enabled \
  --wandb_directory . \
  --reddit 70 28 121 \
  --evalMode human_in_the_loop \
  --continually_add_saved_examples \
  --save_examples_memory \
  --observation_type image_som \
  --experiment_name hitl_run0

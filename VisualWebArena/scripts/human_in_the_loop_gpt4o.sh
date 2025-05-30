#!/bin/sh
export AZURE_OPENAI_KEY="YOUR_AZURE_OPENAI_KEY"
export AZURE_OPENAI_ENDPOINT="YOUR_AZURE_OPENAI_ENDPOINT"
export RESULTS_DIR="output"
mkdir -p ./.auth
python browser_env/auto_login.py
python run_eval.py \
  --instruction_path learned_examples/human_demos_with_abstractions/planning_examples.json \
  --result_dir $RESULTS_DIR \
  --test_config_base_dir=config_files/test_classifieds \
  --model gpt-4o \
  --temperature 0.2 \
  --top_p 0.9 \
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

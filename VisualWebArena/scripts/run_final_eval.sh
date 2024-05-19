#!/bin/sh
export AZURE_OPENAI_KEY="YOUR_KEY_HERE"
export AZURE_OPENAI_ENDPOINT="YOUR_ENDPOINT_HERE"
export RESULTS_DIR="output"
mkdir -p ./.auth
python browser_env/auto_login.py
python run_eval.py \
  --instruction_path data/human_demos/planning_examples.json \
  --instruction_jsons data/memory_human_in_the_loop/merged_classifieds_shopping_reddit_V2/planning_examples.json \
  --result_dir $RESULTS_DIR \
  --test_config_base_dir=config_files/test_classifieds \
  --model gpt-4-vision-preview \
  --temperature 0.2 \
  --top_p 0.1 \
  --action_set_tag som \
  --save_trace_enabled \
  --wandb_directory . \
  --evalMode regular \
  --observation_type image_som \
  --pick_random_subset 910 \
  --seed 32 \
  --save_examples_memory \
  --skip_if_exists \
  --topk 5 \
  --experiment_name final_run_expertdemos_hiltdemos_NEW_topk5_oneperepisode_COMBINED_00
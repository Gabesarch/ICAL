#!/bin/sh
export AZURE_OPENAI_KEY="YOUR_AZURE_OPENAI_KEY"
export AZURE_OPENAI_ENDPOINT="YOUR_AZURE_OPENAI_ENDPOINT"
export RESULTS_DIR="output"
mkdir -p ./.auth
python browser_env/auto_login.py
python run_eval.py \
  --instruction_path agent/prompts/jsons/p_som_cot_id_actree_3s.json \
  --result_dir $RESULTS_DIR \
  --test_config_base_dir=config_files/test_classifieds \
  --model gpt-4o \
  --temperature 0.2 \
  --top_p 0.9 \
  --action_set_tag som \
  --save_trace_enabled \
  --wandb_directory . \
  --evalMode regular \
  --observation_type image_som \
  --pick_random_subset 910 \
  --seed 32 \
  --save_examples_memory \
  --skip_if_exists \
  --experiment_name run_ICAL_agent

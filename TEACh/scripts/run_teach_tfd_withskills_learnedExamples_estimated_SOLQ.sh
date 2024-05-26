#!/bin/sh
export AZURE_OPENAI_KEY="YOUR_KEY_HERE"
export AZURE_OPENAI_ENDPOINT="YOUR_ENDPOINT_HERE"
export TEACH_ROOT="TEACH_ROOT_HERE"
python main.py \
 --mode teach_eval_tfd \
 --split valid_seen \
 --teach_data_dir $TEACH_ROOT \
 --precompute_map_path ./data/precomputed_maps_estimated \
 --create_movie \
 --remove_map_vis \
 --log_every 100 \
 --use_attribute_detector \
 --run_error_correction_llm \
 --remove_unusable_slice \
 --use_llm_search \
 --use_constraint_check \
 --wandb_directory ./ \
 --group teach_eval_skill_tfd \
 --server_port 0 \
 --check_success_change_state \
 --skill_folder ./learned_examples/fullmemlearning_idm_00 \
 --precompute_map_path ./data/precomputed_maps_LEARNEDMEM \
 --gpt_model gpt-4-1106-Preview \
 --topk_mem_examples 10 \
 --episode_in_try_except \
 --zoedepth_checkpoint ./checkpoints/model-00020000.pth \
 --solq_checkpoint ./checkpoints/model-00023000.pth \
 --instruct_lambda 1.0 \
 --state_lambda 0.0 \
 --visual_lambda 0.0 \
 --skip_if_exists \
 --set_name run_tfd_learnedMem_VALIDSEEN_idm_SOLQperception_00
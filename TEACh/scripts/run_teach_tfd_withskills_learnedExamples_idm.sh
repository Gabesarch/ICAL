#!/bin/sh
export AZURE_OPENAI_KEY="YOUR_KEY_HERE"
export AZURE_OPENAI_ENDPOINT="YOUR_ENDPOINT_HERE"
python main.py \
 --mode teach_eval_tfd \
 --split valid_seen \
 --teach_data_dir $TEACH_ROOT \
 --precompute_map_path ./data/precomputed_maps_GT \
 --create_movie \
 --remove_map_vis \
 --log_every 1 \
 --use_gt_depth \
 --use_gt_seg \
 --use_gt_success_checker \
 --use_GT_error_feedback \
 --use_GT_constraint_checks \
 --use_gt_attributes \
 --use_attribute_detector \
 --run_error_correction_llm \
 --use_gt_metadata \
 --force_actions \
 --remove_unusable_slice \
 --use_llm_search \
 --use_constraint_check \
 --wandb_directory ./ \
 --max_api_fails 50 \
 --group teach_eval_skill_tfd \
 --server_port 0 \
 --check_success_change_state \
 --do_state_abstraction \
 --skill_folder ./learned_examples/fullmemlearning_idm_00 \
 --precompute_map_path ./data/precomputed_maps_LEARNEDMEM \
 --gpt_model gpt-3.5-turbo-1106 \
 --skip_if_exists \
 --episode_in_try_except \
 --topk_mem_examples 10 \
 --set_name run_tfd_learnedMem_VALIDSEEN_idm_00
#!/bin/sh
export AZURE_OPENAI_KEY="YOUR_KEY_HERE"
export AZURE_OPENAI_ENDPOINT="YOUR_ENDPOINT_HERE"
export TEACH_ROOT="TEACH_ROOT_HERE"
python main.py \
 --mode teach_skill_learning \
 --split train \
 --create_movie \
 --remove_map_vis \
 --teach_data_dir $TEACH_ROOT \
 --use_gt_depth \
 --use_gt_seg \
 --use_gt_centroids \
 --use_gt_success_checker \
 --remove_unusable_slice \
 --wandb_directory ./ \
 --server_port 0 \
 --max_api_fails 50 \
 --group HELPER_skill_learning \
 --add_back_objs_progresscheck \
 --max_episodes 100 \
 --shuffle \
 --use_gt_metadata \
 --use_gt_attributes \
 --use_attribute_detector \
 --skip_if_exists \
 --get_expert_program_idm \
 --load_model_path ./checkpoints/model-00000045.pth \
 --demo_folder ./output/expert_programs_idm/task_demos \
 --seed 0 \
 --W 900 \
 --H 900 \
 --query_for_each_object \
 --num_feature_levels 1 \
 --set_name test00
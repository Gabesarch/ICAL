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
 --use_gt_attributes \
 --wandb_directory ./ \
 --server_port 0 \
 --max_api_fails 50 \
 --group HELPER_skill_learning \
 --add_back_objs_progresscheck \
 --use_attribute_detector \
 --online_skill_learning \
 --shuffle \
 --num_online_learning_iterations 50 \
 --num_environments_skills 1 \
 --demo_folder ./output/expert_programs_idm/task_demos \
 --skill_folder2 ./learned_examples/hand_written_memory_examples \
 --use_gt_metadata \
 --force_actions \
 --use_task_to_instance_programs \
 --use_critic_gt \
 --explore_steps 50 \
 --gpt_model gpt-4-1106-Preview \
 --error_on_action_fail \
 --run_modelbased_refinement \
 --num_nodes 1 \
 --num_refinements_skills 5 \
 --save_memory_images \
 --use_llm_state_abstraction \
 --demos_from_idm \
 --skip_if_exists \
 --episode_in_try_except \
 --seed 18 \
 --set_name fullmemlearning_idm_00
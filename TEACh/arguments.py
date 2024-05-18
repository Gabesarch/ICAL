import argparse
import numpy as np
import os
# from map_and_plan.FILM.film_arguments import get_FILM_args, FILM_adjust_args
import torch
parser = argparse.ArgumentParser()

# simulation
parser.add_argument('--run_modelbased_refinement', default=False, action='store_true', help='run LLM model based offline refinement?')
parser.add_argument('--simulate_actions', default=False, action='store_true', help='simulate actions? Recommend to keep this false')
parser.add_argument("--topk_mem_examples", type=int, default=3, help="number of examples")
parser.add_argument("--num_nodes", type=int, default=3, help="number of search nodes for tree of thought")
parser.add_argument('--save_memory_images', default=False, action='store_true', help='save out memory images')
parser.add_argument("--clip_model", type=str, default="openai/clip-vit-base-patch32", help="which clip model?")
parser.add_argument('--load_script_from_tmp', default=False, action='store_true', help='load memory script from tmp (for debugging)')
parser.add_argument("--max_examples", type=int, default=10, help="maximum number of examples to use")
parser.add_argument('--use_raw_demos', default=False, action='store_true', help='Use raw demos for planning')
parser.add_argument('--use_llm_state_abstraction', default=False, action='store_true', help='Use LLM for state abstraction during memory learning?')
parser.add_argument('--demos_from_idm', default=False, action='store_true', help='Are demos coming from inverse dynamics model? If so, use slightly different prompt to tell LLM this.')
# parser.add_argument('--no_online_phase', default=False, action='store_true')
parser.add_argument('--relabel_unsuccessful', default=False, action='store_true', help='Relabel unsuccessful trajectories')


parser.add_argument("--max_memory_episodes", type=int, default=250, help="maximum number of episodes before ending")

parser.add_argument("--use_first_X_examples_learned", type=int, default=None, help="Saturation experiments. use first X number of examples for memory retrieval")


parser.add_argument('--ablate_offline', default=False, action='store_true', help='Ablate offline phase of memory learning?')


parser.add_argument('--do_tree_of_thought', default=False, action='store_true', help='Do tree of thought planning at test time?')
parser.add_argument('--ft_with_retrieval', default=False, action='store_true', help='Force retrieval with finetuned model?')
parser.add_argument('--zero_shot', default=False, action='store_true', help='Force zero shot?')

# retrieval weighting
parser.add_argument("--instruct_lambda", type=float, default=0.8, help="weight given to instruction embedding distance")
parser.add_argument("--state_lambda", type=float, default=0.1, help="weight given to object state embedding distance")
parser.add_argument("--visual_lambda", type=float, default=0.1, help="weight given to visual embedding distance")


# parser.add_argument("--topk_mem_sample_examples", type=int, default=3, help="number of examples")
parser.add_argument('--task_type', default=None, type=str, nargs='+')

# search arguments
parser.add_argument("--max_search_steps", type=int, default=600, help="maximum searching steps")
parser.add_argument('--check_success_change_state', default=False, action='store_true', help='get skills from demos')
parser.add_argument('--do_state_abstraction', default=False, action='store_true', help='do state abstraction')

parser.add_argument("--use_saved_program_output", default=False, action='store_true', help="cache outputs")

parser.add_argument("--skill_folder2", type=str, default=None, help="second folder for examples")
parser.add_argument("--skill_folder3", type=str, default=None, help="third folder for examples")
parser.add_argument('--get_skills_demos', default=False, action='store_true', help='get skills from demos')
parser.add_argument('--get_expert_program', default=False, action='store_true', help='get the python programs from expert demos')
parser.add_argument("--online_skill_learning", action="store_true", default=False, help="perform online skill learning phase")
parser.add_argument("--num_online_learning_iterations", type=int, default=3, help="How many iterations of skill learning through the task demos?")
parser.add_argument("--skill_folder", type=str, default="./ouput/skill_logging", help="checkpoint for attribute detector")
parser.add_argument("--num_environments_skills", type=int, default=5, help="How many environments to test the skill in?")
parser.add_argument("--num_refinements_skills", type=int, default=3, help="How many times to refine a skill in an environment?")
parser.add_argument("--demo_folder", type=str, default="./output/expert_programs/task_demos", help="checkpoint for attribute detector")
# parser.add_argument('--run_modelbased_refinement', default=False, action='store_true', help='run LLM model based offline refinement?')
parser.add_argument("--use_gt_metadata", action="store_true", default=False, help="use GT metadata?")
parser.add_argument("--run_from_code_file", action="store_true", default=False, help="use GT metadata?")
parser.add_argument("--force_actions", action="store_true", default=False, help="use GT metadata?")
parser.add_argument("--error_on_action_fail", action="store_true", default=False, help="Return error on action failure?")
parser.add_argument("--num_allowable_failure", type=int, default=3, help="How many max times to replan?")
parser.add_argument("--use_instance_programs", action="store_true", default=False, help="Use expert where programs are seperated by task instances, not task type")
parser.add_argument("--use_critic_gt", action="store_true", default=False, help="Use critic feedback from GT?")
parser.add_argument("--explore_steps", type=int, default=150, help="How many steps for exploring?")
parser.add_argument("--use_task_to_instance_programs", action="store_true", default=False, help="Go from task to instance")

parser.add_argument("--use_attribute_detector", action="store_true", default=False, help="use attribute clip detector")
parser.add_argument("--attribute_detector_checkpoint", type=str, default="./checkpoints/state_estimator.pth", help="checkpoint for attribute detector")
parser.add_argument("--use_gt_attributes", action="store_true", default=False, help="use attribute clip detector")


### Iverse dynamics get expert program
parser.add_argument('--get_expert_program_idm', default=False, action='store_true', help='get the python programs from expert demos')



###### INVERSE DYNAMICS MODEL ######
parser.add_argument("--save_instance_masks", action="store_true", default=False, help="save out instance segmentations (less memory efficient)")
parser.add_argument("--alfred_load_train_root", type=str, default=None, help="load root for alfred loader TRAIN")
parser.add_argument("--alfred_load_train_root2", type=str, default=None, help="load root for alfred loader TRAIN - second folder to add to train set")
parser.add_argument("--alfred_load_train_root3", type=str, default=None, help="load root for alfred loader TRAIN - third folder to add to train set")
parser.add_argument("--alfred_load_valid_seen_root", type=str, default='', help="load root for alfred loader VALID SEEN")
parser.add_argument("--alfred_load_valid_unseen_root", type=str, default='', help="load root for alfred loader VALID UNSEEN")
parser.add_argument("--run_valid_seen", action="store_true", default=False, help="evaluate valid seen loss during training")
parser.add_argument("--run_valid_unseen", action="store_true", default=False, help="evaluate valid unseen loss during training")
parser.add_argument('--val_batch_mult', default=1, type=int, help="increase batch size of val dataloader by a multiplicative scale factor")
parser.add_argument('--max_validation_iters', default=None, type=int, help="maximum validation iters")
parser.add_argument('--validation_shuffle', action='store_true', help="shuffle validation dataloader")
parser.add_argument("--load_strict_false", action="store_true", default=False, help="do not load strict checkpoint")
parser.add_argument('--action_loss_coef', default=1, type=float, help="action loss coefficient")
parser.add_argument("--action_weights", type=float, nargs='+', default=None, help="action weightings for CE action loss")
parser.add_argument('--use_action_weights', action='store_true', default=True, help="use class weights for action CE loss")
parser.add_argument('--use_label_weights', action='store_true', default=True, help="use label weights for action CE loss")
parser.add_argument("--pretrained_ddetr", type=str, default='', help="pretrained ddetr path")
parser.add_argument('--max_episodes_train', default=None, type=int, help="maximum episodes for dataloader train")
parser.add_argument('--targets_npz_format', action='store_true', default=False, help="targets are saved in npz format?")
parser.add_argument('--save_freq_epoch', default=1, type=int, help="How often every X epochs to save model")
parser.add_argument("--checkpoint_path", type=str, default='', help="pretrained ddetr path")
parser.add_argument('--cls_loss_coef', default=1, type=float, help="object class loss coefficient")
parser.add_argument('--train_on_teach', action='store_true', default=False, help="train on teach dataset?")
parser.add_argument('--query_for_each_object', action='store_true', default=False, help="Have a seperate query for each object?")
parser.add_argument('--reduce_final_layer', action='store_true', default=False, help="Reduce number of output features in backbone for transformer")


# odin model arguments
parser.add_argument("--use_odin", action="store_true", default=False, help="use odin for detection?")
parser.add_argument("--odin_update_frequency", type=int, default=10, help="Frequency of updating object tracking with odin")
parser.add_argument("--max_odin_images", type=int, default=20, help="Maximum images for odin to process")
parser.add_argument("--save_state", action="store_true", default=False, help="Save state information for training")
parser.add_argument("--STATE_DATA_DIR", type=str, default="./data/state_data", help="where to save state info")
parser.add_argument("--odin_checkpoint", type=str, default="./checkpoints/model_0002999.pth", help="where is the odin model checkpoint?")



parser.add_argument("--seed", type=int, default=39, help="Random seed")
parser.add_argument("--mode", type=str, help="mode to run, see main.py")
parser.add_argument("--verbose", action="store_true", default=False, help="print out actions + other logs during task")
parser.add_argument("--set_name", type=str, help="experiment name")
parser.add_argument("--split", type=str, help="eval split")

parser.add_argument("--shuffle", action="store_true", default=False, help="shuffle files")

parser.add_argument("--use_openai", action="store_true", default=False, help="")
parser.add_argument("--gpt_model", type=str, default="gpt-4", help="options: gpt-3.5-turbo, text-davinci-003, gpt-4, gpt-3.5-turbo-instruct, gpt-3.5-turbo-1106")
parser.add_argument("--max_token_length", type=int, default=32768, help="maximum token length allowable")


parser.add_argument('--skip_if_exists', default=False, action='store_true', help='skip if file exists in teach metrics')

parser.add_argument("--dpi", type=int, default=100, help="DPI for plotting")
parser.add_argument("--max_traj_steps", type=int, default=1000, help="maximum trajectory steps")
parser.add_argument('--remove_map_vis', default=False, action='store_true', help='remove map visual from movies')



parser.add_argument("--root", type=str, default="", help="root folder")
parser.add_argument("--tag", type=str, default="", help="root folder tag")
parser.add_argument("--teleport_to_objs", action="store_true", default=False, help="teleport to objects instead of navigating")
parser.add_argument("--render", action="store_true", default=False, help="render video and logs")
parser.add_argument("--use_gt_objecttrack", action="store_true", default=False, help="if navigating, use GT object masks for getting object detections + centroids?")
parser.add_argument("--use_gt_depth", action="store_true", default=False, help="if navigating, use GT depth maps? ")
# parser.add_argument("--use_GT_seg_for_interaction", action="store_true", default=False, help="use GT segmentation for interaction?")
parser.add_argument("--use_gt_seg", action="store_true", default=False, help="use GT segmentation?")
parser.add_argument("--use_gt_success_checker", action="store_true", default=False, help="use GT segmentation?")
parser.add_argument("--use_gt_centroids", action="store_true", default=False, help="use GT centroids?")

# parser.add_argument("--use_GT_success_checker_for_interaction", action="store_true", default=False, help="use GT success check for interaction?")
# parser.add_argument("--use_GT_success_checker_for_navigation", action="store_true", default=False, help="use GT success check for navigation?")
parser.add_argument("--do_masks", action="store_true", default=False, help="use masks?")
parser.add_argument("--use_solq", action="store_true", default=False, help="use SOLQ?")
parser.add_argument("--use_gt_subgoals", action="store_true", default=False, help="use GT subgoals?")
parser.add_argument("--sample_every_other", action="store_true", default=False, help="run every other episode in the split")
parser.add_argument("--episode_in_try_except", action="store_true", default=False, help="Continue to next episode if assertion error occurs? ")
parser.add_argument("--log_every", type=int, default=1, help="log every X episodes")
# parser.add_argument("--split", type=str, default="valid_seen", help="which split to use")
parser.add_argument("--on_aws", action="store_true", default=False, help="on AWS?")
parser.add_argument("--new_parser", action="store_true", default=False, help="on AWS?")
parser.add_argument('--load_explore', action='store_true', default=False, help="load explore for full task from path?")
parser.add_argument("--movie_dir", type=str, default="./output/movies", help="where to output rendered movies")
parser.add_argument("--precompute_map_path", default='./data/precomputed_maps', type=str, help="load trajectory list from file?")
parser.add_argument("--create_movie", action="store_true", default=False, help="create mp4 movie")
parser.add_argument("--visualize_masks", action="store_true", default=False, help="visualize masks in object tracker visuals")

parser.add_argument("--metrics_dir", type=str, default="./output/metrics", help="where to output rendered movies")
parser.add_argument("--llm_output_dir", type=str, default="./output/llm", help="where to output rendered movies")
parser.add_argument("--gpt_embedding_dir", type=str, default="./dataset", help="where to output rendered movies")
parser.add_argument("--run_error_correction_llm", action="store_true", default=False, help="run error correction for LLM")
parser.add_argument("--run_error_correction_basic", action="store_true", default=False, help="run error correction - manual correction")
parser.add_argument("--use_progress_check", action="store_true", default=False, help="run progress check at the end to replan")
parser.add_argument("--remove_unusable_slice", action="store_true", default=False, help="remove the unusable slice from the environment after slicing")
parser.add_argument("--add_back_objs_progresscheck", action="store_true", default=False, help="add back in objects to object tracker for progress check")
parser.add_argument("--teach_data_dir", type=str, default="./dataset", help="data directory where teach data is held")
parser.add_argument("--data_path", type=str, default="./dataset", help="data directory where teach data is held")


parser.add_argument("--dont_use_controller", action="store_true", default=False, help="dont init controller")
parser.add_argument("--use_constraint_check", action="store_true", default=False, help="use constraint check")
parser.add_argument("--num_continual_iter", type=int, default=3, help="how many continual learning iterations?")


parser.add_argument("--mod_api_continual", action="store_true", default=False, help="setting to modify the api instead of example retrieval for continual learning experiments")
parser.add_argument("--ablate_example_retrieval", action="store_true", default=False, help="ablate example retrieval for GPT?")

parser.add_argument("--increased_explore", action="store_true", default=False, help="increase explore for GT depth")

parser.add_argument("--max_episodes", type=int, default=None, help="maximum episodes to evaluate")

###########%%%%%%% agent parameters %%%%%%%###########
parser.add_argument("--start_startx", action="store_true", default=False, help="start x server upon calling main")
parser.add_argument("--server_port", type=int, default=1, help="server port for x server")
parser.add_argument("--do_headless_rendering", action="store_true", default=False, help="render in headless mode with new Ai2thor version")
parser.add_argument("--HORIZON_DT", type=int, default=30, help="pitch movement delta")
parser.add_argument("--DT", type=int, default=90, help="yaw movement delta")
parser.add_argument("--STEP_SIZE", type=int, default=0.25, help="yaw movement delta")
parser.add_argument("--pitch_range", type=list, default=[-30,60], help="pitch allowable range for the agent. positive is 'down'")
parser.add_argument("--fov", type=int, default=90, help="field of view")
parser.add_argument("--W", type=int, default=480, help="image width")
parser.add_argument("--H", type=int, default=480, help="image height")
parser.add_argument("--visibilityDistance", type=float, default=1.5, help="visibility NOTE: this will not change rearrangement visibility")

parser.add_argument('--debug', default=False, action='store_true')

parser.add_argument("--eval_split", type=str, default="test", help="evaluation mode: combined (rearrange), train, test, val")

parser.add_argument('--use_estimated_depth', action='store_true', help="use estimated depth?")
parser.add_argument('--num_search_locs_object', type=int, default=20, help='number of search locations for searching for object')
parser.add_argument("--dist_thresh", type=float, default=0.5, help="navigation distance threshold to point goal")
parser.add_argument('--use_GT_constraint_checks', default=False, action='store_true', help='use GT constraint checks?')
parser.add_argument('--use_GT_error_feedback', default=False, action='store_true', help='use GT error feedback?')


parser.add_argument("--max_api_fails", type=int, default=30, help="maximum allowable api failures")

parser.add_argument('--use_llm_search', default=False, action='store_true', help='use llm search')
parser.add_argument('--use_mask_rcnn_pred', default=False, action='store_true', help='use maskrcnn')

parser.add_argument("--episode_file", type=str, default=None, help="specify an episode file name to run")



# parser.add_argument('--use_gt_detections', action='store_true', help="Use ground truth detections during evaluation")

###########%%%%%%% object tracker %%%%%%%###########
parser.add_argument("--OT_dist_thresh", type=float, default=0.5, help="distance threshold for NMS for object tracker")
parser.add_argument("--OT_dist_thresh_searching", type=float, default=0.5, help="distance threshold for NMS for object tracker")
parser.add_argument("--confidence_threshold", type=float, default=0.4, help="confidence threshold for detections [0, 0.1]")
parser.add_argument("--confidence_threshold_interm", type=float, default=0.4, help="intermediate object score threshold")
parser.add_argument("--confidence_threshold_searching", type=float, default=0.4, help="confidence threshold for detections when searching for a target object class [0, 0.1]")
parser.add_argument("--nms_threshold", type=float, default=0.5, help="NMS threshold for object tracker")
parser.add_argument("--use_GT_centroids", action="store_true", default=False, help="use GT centroids for object tracker")
parser.add_argument('--only_one_obj_per_cat', action='store_true', default=False, help="only one object per cateogry in the object tracker?")
parser.add_argument('--env_frame_width_FILM', type=int, default=300, help='Frame width (default:84)')
parser.add_argument('--env_frame_height_FILM', type=int, default=300, help='Frame height (default:84)')


# Depth network
parser.add_argument("--randomize_object_placements", action="store_true", default=False, help="")
parser.add_argument("--randomize_scene_lighting_and_material", action="store_true", default=False, help="")
parser.add_argument("--randomize_agent_pickup", action="store_true", default=False, help="")
parser.add_argument("--randomize_object_state", action="store_true", default=False, help="")
parser.add_argument("--load_model", action="store_true", default=False, help="")
parser.add_argument("--load_model_path", type=str, default="", help="load checkpoint path")
parser.add_argument("--lr_scheduler_from_scratch", action="store_true", default=False, help="")
parser.add_argument("--optimizer_from_scratch", action="store_true", default=False, help="")
parser.add_argument("--start_one", action="store_true", default=False, help="")
parser.add_argument('--max_iters', type=int, default=100000, help='max train iterations')
parser.add_argument('--log_freq', type=int, default=500, help='log frequency')
parser.add_argument('--val_freq', type=int, default=500, help='validation frequency')
parser.add_argument('--save_freq', type=int, default=2500, help='save checkpoint frequency')
parser.add_argument("--load_train_agent", action="store_true", default=False, help="")
parser.add_argument('--batch_size', default=2, type=int, help="batch size for model training")
parser.add_argument("--S", type=int, default=2, help="Number of views per trajectory")
parser.add_argument("--radius_min", type=float, default=0.0, help="radius min to spawn near target object")
parser.add_argument("--radius_max", type=float, default=7.0, help="radius max to spawn near target object")
parser.add_argument("--views_to_attempt", type=int, default=8, help="max views to attempt for getting trajectory")
parser.add_argument("--movement_mode", type=str, default="random", help="movement mode for action sampling for getting trajectory (forward_first, random); forward_first: always try to move forward")
parser.add_argument("--fail_if_no_objects", type=bool, default=True, help="fail view if no objects in view")
parser.add_argument("--torch_checkpoint_path", type=str, default="", help="torch hub checkpoint path")
parser.add_argument('--lr_scheduler_freq', type=int, default=20000, help='lr frequency for step')
parser.add_argument('--keep_latest', default=5, type=int, help="number of checkpoints to keep at one time")
parser.add_argument("--val_load_dir", type=str, default="./dataset/val", help="val load dir")
parser.add_argument("--run_val", action="store_true", default=False, help="")
parser.add_argument("--load_val_agent", action="store_true", default=False, help="")
parser.add_argument('--n_val', type=int, default=50, help='number of validation iters')

parser.add_argument("--zoedepth_checkpoint", type=str, default="./checkpoints/ZOEDEPTH-model-00015000.pth", help="zoe depth checkpoint to load for teach")
parser.add_argument("--solq_checkpoint", type=str, default="./checkpoints/SOLQ-model-00023000.pth", help="SOLQ checkpoint to load for teach")

# SOLQ hyperparams
parser.add_argument('--lr', default=2e-5, type=float) 
parser.add_argument('--lr_backbone_names', default=["backbone.0"], type=str, nargs='+')
parser.add_argument('--lr_backbone_mult', default=0.1, type=float)
parser.add_argument('--lr_linear_proj_names', default=['reference_points', 'sampling_offsets'], type=str, nargs='+')
parser.add_argument('--lr_linear_proj_mult', default=0.1, type=float)
parser.add_argument('--lr_text_encoder_mult', default=0.05, type=float)
parser.add_argument('--lr_text_encoder_names', default=["text_encoder"], type=str, nargs='+')

parser.add_argument('--weight_decay', default=1e-4, type=float)
parser.add_argument('--epochs', default=50, type=int)
parser.add_argument('--lr_drop', default=25, type=int)
parser.add_argument('--save_period', default=10, type=int)
parser.add_argument('--lr_drop_epochs', default=None, type=int, nargs='+')
parser.add_argument('--clip_max_norm', default=1.0, type=float,
                    help='gradient clipping max norm')
parser.add_argument('--meta_arch', default='solq', type=str)
parser.add_argument('--sgd', action='store_true')
# Variants of Deformable DETR
parser.add_argument('--with_box_refine', default=True, action='store_true')
parser.add_argument('--two_stage', default=True)
# VecInst
parser.add_argument('--with_vector', default=True, action='store_true')
parser.add_argument('--n_keep', default=256, type=int,
                    help="Number of coeffs to be remained")
parser.add_argument('--gt_mask_len', default=128, type=int,
                    help="Size of target mask")
parser.add_argument('--vector_loss_coef', default=3, type=float)
parser.add_argument('--vector_hidden_dim', default=1024, type=int,
                    help="Size of the vector embeddings (dimension of the transformer)")
parser.add_argument('--no_vector_loss_norm', default=False, action='store_true')
parser.add_argument('--activation', default='relu', type=str, help="Activation function to use")
parser.add_argument('--checkpoint', default=False, action='store_true')
parser.add_argument('--vector_start_stage', default=0, type=int)
parser.add_argument('--num_machines', default=1, type=int)
parser.add_argument('--loss_type', default='l1', type=str)
parser.add_argument('--dcn', default=False, action='store_true')
# Model parameters
parser.add_argument('--frozen_weights', type=str, default=None,
                    help="Path to the pretrained model. If set, only the mask head will be trained")
parser.add_argument('--pretrained', default=None, help='resume from checkpoint')
# * Backbone
parser.add_argument('--backbone', default='resnet50', type=str,
                    help="Name of the convolutional backbone to use")
parser.add_argument('--dilation', action='store_true',
                    help="If true, we replace stride with dilation in the last convolutional block (DC5)")
parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned', 'rel'),
                    help="Type of positional embedding to use on top of the image features")
parser.add_argument('--position_embedding_scale', default=2 * np.pi, type=float,
                    help="position / size * scale")
parser.add_argument('--num_feature_levels', default=4, type=int, help='number of feature levels')
# * Transformer
parser.add_argument('--enc_layers', default=6, type=int,
                    help="Number of encoding layers in the transformer")
parser.add_argument('--dec_layers', default=6, type=int,
                    help="Number of decoding layers in the transformer")
parser.add_argument('--dim_feedforward', default=1024, type=int,
                    help="Intermediate size of the feedforward layers in the transformer blocks")
parser.add_argument('--hidden_dim', default=384, type=int,
                    help="Size of the embeddings (dimension of the transformer)")
parser.add_argument('--dropout', default=0.1, type=float,
                    help="Dropout applied in the transformer")
parser.add_argument('--nheads', default=8, type=int,
                    help="Number of attention heads inside the transformer's attentions")
parser.add_argument('--num_queries', default=300, type=int,
                    help="Number of query slots")
parser.add_argument('--dec_n_points', default=4, type=int)
parser.add_argument('--enc_n_points', default=4, type=int)
# * Segmentation
parser.add_argument('--masks', type=bool, default=True,
                    help="Train segmentation head if the flag is provided")
# Loss
parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                    help="Disables auxiliary decoding losses (loss at each layer)")
# * Matcher
parser.add_argument('--set_cost_class', default=2, type=float,
                    help="Class coefficient in the matching cost")
parser.add_argument('--set_cost_bbox', default=5, type=float,
                    help="L1 box coefficient in the matching cost")
parser.add_argument('--set_cost_giou', default=2, type=float,
                    help="giou box coefficient in the matching cost")

parser.add_argument('--dataset_file', default='coco')
parser.add_argument('--coco_path', default='./data/coco', type=str)
parser.add_argument('--coco_panoptic_path', type=str)
parser.add_argument('--remove_difficult', action='store_true')
parser.add_argument('--alg', default='instformer', type=str)
parser.add_argument('--output_dir', default='',
                    help='path where to save, empty for no saving')
parser.add_argument('--device', default='cuda',
                    help='device to use for training / testing')
parser.add_argument('--resume', default='', help='resume from checkpoint')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='start epoch')
parser.add_argument('--eval', action='store_true')
parser.add_argument('--test', action='store_true')
parser.add_argument('--num_workers', default=2, type=int)
parser.add_argument('--cache_mode', default=False, action='store_true', help='whether to cache images on memory')
# * Loss coefficients
parser.add_argument('--mask_loss_coef', default=1, type=float)
parser.add_argument('--dice_loss_coef', default=1, type=float)
parser.add_argument('--bbox_loss_coef', default=5, type=float)
parser.add_argument('--giou_loss_coef', default=2, type=float)
parser.add_argument('--focal_alpha', default=0.25, type=float)

### WANDB
parser.add_argument("--group", type=str, default="default", help="group name")
parser.add_argument("--wandb_directory", type=str, default='.', help="Path to wandb metadata")



# Beauty DETR Args
parser.add_argument(
   "--contrastive_hdim",
   type=int,
   default=64,
   help="Projection head output size before computing normalized temperature-scaled cross entropy loss",
)

args = parser.parse_args()

args.use_estimated_depth = True
if args.use_gt_depth:
   args.use_estimated_depth = False

args.metrics_dir = os.path.join(args.metrics_dir, args.set_name)
args.llm_output_dir = os.path.join(args.llm_output_dir, args.set_name)
args.movie_dir = os.path.join(args.movie_dir, args.set_name)

if args.use_gt_metadata:
   args.use_gt_centroids = True

if args.ablate_offline:
   args.num_nodes = 0

if args.action_weights is None:
   if args.train_on_teach:
      args.action_weights = [
         0.042795, 0.548919, 0.193116, 0.26202, 3.923681, 3.68427, 0.145809, 0.160138, 
         0.274113, 0.289598, 1.451081, 2.587761, 2.96147, 2.147944, 1.761523, 4.809114
      ]
   else:
      args.action_weights = [ 0.10984484,  0.74386463,  0.79303222,  0.93225415,  1.55291897,
                              0.93917156,  1.        ,  1.76089283,  1.74251135, 12.54684096,
                              4.58154336,  6.38470067
                              ]
   if args.train_on_teach:
      args.label_weights = [
         12.0, 0.018868, 0.007859, 0.020033, 1.714286, 4.0, 4.0, 1.714286, 6.0, 12.0, 
         6.0, 0.292683, 3.0, 0.666667, 4.0, 12.0, 12.0, 0.11215, 1.0, 3.0, 0.016416, 
         2.0, 1.333333, 1.714286, 0.705882, 12.0, 6.0, 6.0, 0.428571, 12.0, 1.5, 
         0.315789, 12.0, 0.352941, 6.0, 12.0, 0.923077, 0.081081, 0.164384, 0.461538, 
         12.0, 2.0, 12.0, 0.545455, 0.24, 12.0, 12.0, 0.193548, 12.0, 0.068571, 0.4, 
         0.705882, 0.666667, 0.117647, 0.157895, 6.0, 3.0, 3.0, 12.0, 0.038585, 12.0, 
         4.0, 0.352941, 0.206897, 0.080537, 1.2, 0.218182, 0.033994, 0.061856, 0.210526, 
         0.040541, 0.705882, 0.521739, 0.082192, 0.050209, 0.109091, 0.085714, 0.075, 
         0.193548, 0.060914, 0.20339, 0.157895, 0.25, 0.052174, 0.428571, 0.375, 
         0.016327, 0.039216, 0.028777, 0.045802, 12.0, 1.0, 0.857143, 0.139535, 12.0, 
         0.521739, 0.545455, 12.0, 12.0, 4.0, 1.090909, 12.0, 12.0, 0.571429, 0.75, 
         0.75, 1.333333, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 
         12.0, 12.0, 12.0, 12.0, 12.0, 1.333333, 0.035088, 12.0, 0.094488, 0.051282, 
         0.048583, 0.000231
      ]
   else:
      args.label_weights = [
         40.0, 0.26, 0.69, 40.0, 40.0, 4.76, 40.0, 1.61, 3.1, 12.35, 
         40.0, 3.46, 1.15, 3.56, 40.0, 30.25, 0.4, 1.98, 3.38, 1.93, 0.34, 
         40.0, 1.82, 5.55, 3.76, 1.51, 21.61, 3.44, 6.58, 3.67, 2.62, 2.29, 
         40.0, 2.19, 5.4, 4.2, 3.76, 0.45, 1.33, 40.0, 40.0, 2.41, 40.0, 
         2.57, 5.55, 40.0, 20.17, 4.65, 1.95, 2.79, 40.0, 6.95, 9.76, 2.27, 
         2.94, 40.0, 30.25, 17.79, 40.0, 0.8, 40.0, 1.82, 7.29, 1.14, 1.57, 
         40.0, 1.83, 0.08, 4.12, 4.38, 0.09, 10.08, 5.0, 2.16, 1.11, 1.2, 1.05, 
         1.36, 11.2, 1.04, 10.08, 0.94, 40.0, 40.0, 3.12, 3.67, 1.04, 1.08, 0.78, 
         1.17, 40.0, 8.29, 40.0, 2.48, 40.0, 40.0, 6.44, 40.0, 3.48, 40.0, 
         40.0, 40.0, 40.0, 40.0, 4.58, 4.88, 5.0, 40.0, 40.0, 40.0, 40.0, 
         40.0, 40.0, 40.0, 40.0, 8.64, 21.61, 40.0, 40.0, 40.0, 40.0, 40.0, 
         1.74, 1.98, 2.64, 1.74, 1.81, 40.0, 0.0039
         ]

args.cuda = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
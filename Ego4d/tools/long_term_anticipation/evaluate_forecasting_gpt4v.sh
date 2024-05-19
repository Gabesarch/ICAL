export AZURE_OPENAI_KEY="YOUR_KEY_HERE"
export AZURE_OPENAI_ENDPOINT="YOUR_ENDPOINT_HERE"

function run(){
    
  NAME=$1
  CONFIG=$2
  shift 2;

  python -m scripts.run_lta_gpt4v \
    --job_name $NAME \
    --working_directory ${WORK_DIR} \
    --cfg $CONFIG \
    ${CLUSTER_ARGS} \
    DATA.PATH_TO_DATA_DIR ${EGO4D_ANNOTS} \
    DATA.PATH_PREFIX ${EGO4D_VIDEOS} \
    CHECKPOINT_LOAD_MODEL_HEAD True \
  	TRAIN.ENABLE False \
    DATA.CHECKPOINT_MODULE_FILE_PATH "" \
    FORECASTING.AGGREGATOR "" \
    FORECASTING.DECODER "" \
    CHECKPOINT_FILE_PATH "" \
    $@
}

#-----------------------------------------------------------------------------------------------#

WORK_DIR=$1

EGO4D_DATA=/media/gsarch/HDD14TB/datasets/ego4d
EGO4D_ANNOTS=$EGO4D_DATA/data/long_term_anticipation/annotations/
EGO4D_VIDEOS=$EGO4D_DATA/data/long_term_anticipation/clips/
# CLUSTER_ARGS="--on_cluster NUM_GPUS 8"
CLUSTER_ARGS="NUM_GPUS 1 TRAIN.BATCH_SIZE 1 TEST.BATCH_SIZE 1"



# GPT4V
# run eval_gpt4v\
#     configs/Ego4dLTA/GPT4V.yaml \
#     FORECASTING.NUM_INPUT_CLIPS 1 \
#     TEST.EVAL_VAL True \
#     TEST.ICAL_TRAIN True \
#     DATA.NUM_FRAMES 16 \
#     DATA_LOADER.NUM_WORKERS 0 \
#     EXPERIMENT_NAME "test00" #"test_zeroshot"


run eval_gpt4v\
  configs/Ego4dLTA/GPT4V.yaml \
  FORECASTING.NUM_INPUT_CLIPS 4 \
  TEST.EVAL_VAL True \
  TEST.ICAL_TRAIN True \
  DATA.NUM_FRAMES 12 \
  DATA_LOADER.NUM_WORKERS 0 \
  EXPERIMENT_NAME "test00" #"test_zeroshot"

# # SlowFast-Transformer
# # LOAD=$PWD/pretrained_models/long_term_anticipation/lta_slowfast_trf.ckpt
# LOAD=$EGO4D_DATA/v2/lta_models/lta_slowfast_trf_v2.ckpt
# run eval_slowfast_trf\
#     configs/Ego4dLTA/MULTISLOWFAST_8x8_R101.yaml \
#     FORECASTING.AGGREGATOR TransformerAggregator \
#     FORECASTING.DECODER MultiHeadDecoder \
#     FORECASTING.NUM_INPUT_CLIPS 4 \
#     DATA_LOADER.NUM_WORKERS 0 \
#     CHECKPOINT_FILE_PATH $LOAD

    # TEST.EVAL_VAL \
#-----------------------------------------------------------------------------------------------#
#                                    OTHER OPTIONS                                              #
# #-----------------------------------------------------------------------------------------------#

# # SlowFast-Recognition (repeat action baseline)
# LOAD=$PWD/pretrained_models/long_term_anticipation/ego4d_slowfast8x8.ckpt
# run eval_slowfast_repeat \
#     configs/Ego4dLTA/MULTISLOWFAST_8x8_R101.yaml \
#     FORECASTING.NUM_INPUT_CLIPS 1 \
#     MODEL.MODEL_NAME RecognitionSlowFastRepeatLabels \
#     CHECKPOINT_FILE_PATH $LOAD


# # SlowFast-Concat
# LOAD=$PWD/pretrained_models/long_term_anticipation/lta_slowfast_concat.ckpt
# run eval_slowfast_concat \
#     configs/Ego4dLTA/MULTISLOWFAST_8x8_R101.yaml \
#     FORECASTING.AGGREGATOR ConcatAggregator \
#     FORECASTING.DECODER MultiHeadDecoder \
#     CHECKPOINT_FILE_PATH $LOAD


# # MViT-Concat
# LOAD=$PWD/pretrained_models/long_term_anticipation/lta_mvit_concat.ckpt
# run eval_mvit_concat \
#     configs/Ego4dLTA/MULTIMVIT_16x4.yaml \
#     FORECASTING.AGGREGATOR ConcatAggregator \
#     FORECASTING.DECODER MultiHeadDecoder \
#     CHECKPOINT_FILE_PATH $LOAD

# Debug locally using a smaller batch size / fewer GPUs
# CLUSTER_ARGS="NUM_GPUS 2 TRAIN.BATCH_SIZE 8 TEST.BATCH_SIZE 32"
# CLUSTER_ARGS="NUM_GPUS 1 TRAIN.BATCH_SIZE 8 TEST.BATCH_SIZE 32"

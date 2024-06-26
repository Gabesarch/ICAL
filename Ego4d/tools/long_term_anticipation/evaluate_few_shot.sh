# export AZURE_OPENAI_KEY="YOUR_KEY_HERE"
# export AZURE_OPENAI_ENDPOINT="YOUR_ENDPOINT_HERE"

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

EGO4D_DATA=EGO4D_DATA_PATH_HERE
EGO4D_DATA=/media/gsarch/HDD14TB/datasets/ego4d
EGO4D_ANNOTS=$EGO4D_DATA/data/long_term_anticipation/annotations/
EGO4D_VIDEOS=$EGO4D_DATA/data/long_term_anticipation/clips/
CLUSTER_ARGS="NUM_GPUS 1 TRAIN.BATCH_SIZE 1 TEST.BATCH_SIZE 1"

run eval_gpt4v\
  configs/Ego4dLTA/GPT4V.yaml \
  FORECASTING.NUM_INPUT_CLIPS 4 \
  TEST.EVAL_VAL True \
  TEST.ICAL_TEST True \
  DATA.NUM_FRAMES 12 \
  DATA_LOADER.NUM_WORKERS 0 \
  MAX_EPISODES 200 \
  ONLY_DO_FORECASTING True \
  FORECASTING.NUM_SEQUENCES_TO_PREDICT 5 \
  EXAMPLE_PATH "ego4d_forecasting/models/prompts/learned_examples/handwritten/forecasting/examples.json" \
  EXPERIMENT_NAME "eval_handwritten_00"

  # Add this to avoid opanai errors
  # IN_TRY_EXCEPT True \
  # Add this one to restart the evaluation from where you left off
  # SKIP_IF_EXISTS True \
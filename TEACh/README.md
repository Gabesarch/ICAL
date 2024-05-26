<h1 align="center">
    Code for running ICAL on TEACh
</h1>

### Contents

<div class="toc">
<ul>
<li><a href="#important-files"> Important Files </a></li>
<li><a href="#installation"> Installation </a></li>
<li><a href="#teach-dataset"> Dataset </a></li>
<li><a href="#running-ical-on-teach"> Running ICAL on TEACh </a></li>
</ul>
</div>

## ICAL examples
You can view the ICAL learned examples [here](https://github.com/Gabesarch/ICAL/TEACh/learned_examples/fullmemlearning_idm_00).

## Important Files

ICAL example learning script: `models/teach_skill_learning.py` (with `run_abstraction_phase` and `run_human_in_the_loop_phase` functions for each phase)
All prompts are located in: `prompt/`
Training inverse dynamics model: `models/train_inverse_dynamics`
Network files: `nets/`
TEACh evaluation script: `models/teach_eval_embodied_llm.py`

This repo uses [HELPER](https://github.com/Gabesarch/HELPER) for navigation and manipulation modules.

View the learned ICAL examples here: [HELPER](https://github.com/Gabesarch/HELPER)

## Installation 

**(1)** Start by cloning the repository:
```bash
git clone https://github.com/Gabesarch/embodied-llm.git
```
**(1a)** (optional) If you are using conda, create an environment: 
```bash
conda create -n embodied_llm python=3.8
```

You also will want to set CUDA paths. For example (on our tested machine with CUDA 11.1): 
```bash
export CUDA_HOME="/opt/cuda/11.1.1"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:$LD_LIBRARY_PATH"
```

**(2)** Install [PyTorch](https://pytorch.org/get-started/locally/) with the CUDA version you have. For example, run the following for CUDA 11.1: 
```bash
pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
```
<!-- pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html -->

**(3)** Install additional requirements: 
```bash
pip install -r requirements.txt
```

**(4)** Install [Detectron2](https://detectron2.readthedocs.io/en/latest/tutorials/install.html) (needed for SOLQ detector) with correct PyTorch and CUDA version. 
E.g. for PyTorch 1.8 & CUDA 11.1:
```bash
python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.10/index.html
```
<!-- python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.8/index.html -->

**(5)** Install teach: 
```bash
pip install -e teach
```
<!-- git clone https://github.com/alexa/teach.git -->

**(6)** Build SOLQ deformable attention:
```bash
cd ./SOLQ/models/ops && sh make.sh && cd ../../..
```

**(7)** Clone ZoeDepth repo
```bash
git clone https://github.com/isl-org/ZoeDepth.git
cd ZoeDepth
git checkout edb6daf45458569e24f50250ef1ed08c015f17a7
```

## TEACh Dataset
1. Download the TEACh dataset following the instructions in the [TEACh repo](https://github.com/alexa/teach)
```bash
teach_download 
```
Place the extracted data in the `./dataset`

## Checkpoints
To run our model with estimated depth and segmentation, download the SOLQ and ZoeDepth checkpoints:

1. Download inverse dynamics model checkpoint: [here](https://drive.google.com/file/d/1fcuAvrF93zqNV_lKrCLz9YmwMeGlZfpk/view?usp=drive_link). Place it in the `./checkpoints` folder (or anywhere you want and specify the path with `--load_model_path`). 

2. Download SOLQ checkpoint: [here](https://drive.google.com/file/d/1hTCtTuygPCJnhAkGeVPzWGHiY3PHNE2j/view?usp=sharing). Place it in the `./checkpoints` folder (or anywhere you want and specify the path with `--solq_checkpoint`). 

3. Download ZoeDepth checkpoint: [here](https://drive.google.com/file/d/1gMe8_5PzaNKWLT5OP-9KKEYhbNxRjk9F/view?usp=drive_link). Place it in the `./checkpoints` folder (or anywhere you want and specify the path with `--zoedepth_checkpoint`). (Also make sure you clone the ZoeDepth repo: `git clone https://github.com/isl-org/ZoeDepth.git`)

## Running ICAL on TEACh
To evaluate the agent using the learned ICAL examples, replace your openai keys and teach data path in `scripts/run_teach_tfd_withskills_learnedExamples_idm.sh`. Then run the script:
```bash
sh scripts/run_teach_tfd_withskills_learnedExamples_idm.sh
```

### Running the ICAL learning
To to run the ICAL to learn the examples, first generate the labeled demonstrations:
```bash
sh scripts/run_teach_get_expert_programs_idm.sh
```
If you wish to skip this step, we provide the labeled demos for you in `./output/expert_programs_idm/task_demos`.
Then, run the ICAL abstraction and human-in-the-loop phases:
```bash
sh scripts/run_teach_online_skill_learning_idm.sh
```

### Remote Server Setup
To run the Ai2THOR simulator on a headless machine, you must either stat an X-server or use Ai2THOR's new headless mode. 
To start an X-server with any of the scripts, you can simply append `--startx` to the arguments. You can specify the X-server port use the `--server_port` argument.
Alternatively, you can use [Ai2THOR's new headless rendering](https://ai2thor.allenai.org/ithor/documentation/#headless-setup) by appending `--do_headless_rendering` to the arguments. 

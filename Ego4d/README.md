<h1 align="center">
    Running ICAL in Ego4d Forecasting
</h1>

### Contents
<div class="toc">
<ul>
<li><a href="#installation"> Installation </a></li>
<li><a href="#dataset"> Dataset </a></li>
<li><a href="#running-the-ical-evaluation"> Running the ICAL evaluation </a></li>
<li><a href="#citation"> Citation </a></li>
</ul>
</div>

## ICAL examples
You can view the ICAL learned examples [here](https://github.com/Gabesarch/ICAL/tree/main/Ego4d/ego4d_forecasting/models/prompts/learned_examples/examples_ICAL_abstraction_phase/forecasting).

## Installation
This code requires Python>=3.8. If you are using Anaconda, you can create a clean virtual environment with the required Python version with the following command:

`conda create -n ego4d_forecasting python=3.8`

To proceed with the installation, you should activate the virtual environment with the following command:

`conda activate ego4d_forecasting`

We provide two ways to install the repository: a manual installation and a package-based installation. 

### Manual installation
This installation is recommended if you want to modify the code in place and see the results immediately (without having to re-build). On the downside, you will have to add this repository to the PYTHONPATH environment variable manually.

Run the following commands to install the requirements:

`cat requirements.txt | xargs -n 1 -L 1 pip install`

In order to make the `ego4d` module loadable, you should add the current directory to the Python path:

`export PYTHONPATH=$PWD:$PYTHONPATH`

Please note that the command above is not persistent and hence you should run it every time you open a new shell.

### Package-based installation
This installation is recommended if you want import the code of this repo in a separate project. Following these instructions, you will install an "ego4d_forecasting" package which will be accessible in any python project.

To build and install the package run the command:

`pip install .`

To check if the package is installed, move to another directory and try to import a module from the package. For instance:

```
cd ..
python -c "from ego4d_forecasting.models.head_helper import ResNetRoIHead"
```

### Download Model Checkpoints
To download the requirements and models for [Track-Anything-with-DEVA](https://github.com/hkchengrex/Tracking-Anything-with-DEVA/tree/main) that is used for set of marks image annotations, run the following:
```
cd Tracking-Anything-with-DEVA
pip install -e .
bash scripts/download_models.sh
```

## Dataset
Please follow the Ego4d Action Anticipation to download the videos and annotations.

After downloading, copy the ICAL evaluation splits to the `data/long_term_anticipation/annotations` folder in the downloaded dataset:

```
cp PATH_TO_ICAL_REPO/Ego4d/data/fho_lta_ICAL_* PATH_TO_EGO4D/data/long_term_anticipation/annotations
```

## Running the ICAL evaluation
To run the evaluation using ICAL examples, run:
```
bash tools/long_term_anticipation/evaluate_ICAL_abstract_only.sh output/
```

See [this file](https://github.com/Gabesarch/ICAL/blob/main/Ego4d/scripts/run_lta_gpt4v.py) for the main run file for ICAL.

To run the few-shot evaluation using handwritten examples, run:
```
bash tools/long_term_anticipation/evaluate_few_shot.sh output/
```

## Acknowledgements
This code builds on [Ego4d Action Anticipation](https://github.com/EGO4D/forecasting).

## Citation
```
@inproceedings{sarch2024vlm,
  title={VLM Agents Generate Their Own Memories: Distilling Experience into Embodied Programs of Thought},
  author={Sarch, Gabriel Herbert and Jang, Lawrence and Tarr, Michael J and Cohen, William W and Marino, Kenneth and Fragkiadaki, Katerina},
  booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
  year={2024}
}
```

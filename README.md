# ICAL: Continual Learning of Multimodal Agents by Transforming Trajectories into Actionable Insights

## NeurIPS 2024 Spotlight

## Overview
This repository contains the code and resources for the paper titled: "ICAL: Continual Learning of Multimodal Agents by Transforming Trajectories into Actionable Insights." This repository is organized into three main directories, each representing a different domain evaluated in our work: Ego4D, VisualWebArena, and TEACh.

## Abstract
Large-scale LLMs and VLMs excel at few-shot learning but require high-quality demonstrations. We introduce In-Context Abstraction Learning (ICAL), which iteratively refines suboptimal trajectories into high-quality data with optimized actions and detailed reasoning. Given an inefficient demonstration, a VLM corrects actions and annotates causal relationships, object states, subgoals, and task-relevant visuals, forming “programs of thought.” With human feedback, these programs are improved as the agent executes them in a similar environment. The resulting examples, used as prompts or fine-tuning data, significantly boost decision-making while reducing human feedback needs. ICAL surpasses state-of-the-art in TEACh (dialogue-based instruction following), VisualWebArena (multimodal web agents), and Ego4D (action anticipation). In TEACh, combining fine-tuning and retrieval on ICAL examples outperforms raw human demonstrations and expert examples, achieving a 17.5% increase in goal-condition success. In VisualWebArena, retrieval-augmented GPT-4V with ICAL improves task success rate 1.6× over GPT-4V, while fine-tuning Qwen2-VL achieves a 2.8× improvement. In Ego4D, ICAL outperforms few-shot GPT-4V and remains competitive with supervised models. Overall, ICAL scales 2× better than raw human demonstrations and reduces manual prompt engineering.

## Repository Structure
```
ICAL
├── README.md
├── Ego4D
│   ├── README.md
│   ├── domain_files
│   └── environment
├── VisualWebArena
│   ├── README.md
│   ├── domain_files
│   └── environment
└── TEACh
    ├── README.md
    ├── domain_files
    └── environment
```

### Contents of Each Folder
Each folder (Ego4D, VisualWebArena, TEACh) includes:
- **README.md:** Instructions specific to running ICAL in that domain.
- **domain_files:** Files required to run ICAL in the domain.
- **environment:** Configuration and setup files for the environment.

## Running ICAL in Each Domain

### Ego4D
To learn more about running ICAL in Ego4D Forecasting, please refer to the [Ego4D README](Ego4d/README.md).

### VisualWebArena
For instructions on running ICAL in VisualWebArena, please refer to the [VisualWebArena README](VisualWebArena/README.md).

### TEACh
Detailed steps for running ICAL in the TEACh domain can be found in the [TEACh README](TEACh/README.md).

## Learned Examples
The ICAL learned examples for each domain can be found in the following locations:

- **Ego4D:** [Ego4D ICAL Learned Examples](Ego4d/ego4d_forecasting/models/prompts/learned_examples/examples_ICAL_abstraction_phase/forecasting)
- **VisualWebArena:** [VisualWebArena ICAL Learned Examples](VisualWebArena/learned_examples)
- **TEACh:** [TEACh ICAL Learned Examples](TEACh/learned_examples/fullmemlearning_idm_00)

## Installation
To install and set up the ICAL environment, follow these general steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/Gabesarch/ICAL.git
    cd ICAL
    ```

2. Navigate to the domain directory you are interested in (e.g., `Ego4D`), and follow the installation instructions provided in its README file.

<!-- ## Citation
If you find our work useful, please consider citing our paper:
```
@article{sarch2024ical,
  title={ICAL: Continual Learning of Multimodal Agents by Transforming Trajectories into Actionable Insights},
  author={},
  journal={},
  year={2024}
}
``` -->

## Acknowledgements
This work builds on the existing research and frameworks in each of the respective domains. We would like to thank the contributors to the Ego4D, VisualWebArena, and TEACh projects for their invaluable resources.

## Citation
```
@inproceedings{sarch2024vlm,
  title={VLM Agents Generate Their Own Memories: Distilling Experience into Embodied Programs of Thought},
  author={Sarch, Gabriel Herbert and Jang, Lawrence and Tarr, Michael J and Cohen, William W and Marino, Kenneth and Fragkiadaki, Katerina},
  booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
  year={2024}
}
```

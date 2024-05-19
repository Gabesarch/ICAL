# ICAL: Continual Learning of Multimodal Agents by Transforming Trajectories into Actionable Insights

## Overview
This repository contains the code and resources for the paper titled: "ICAL: Continual Learning of Multimodal Agents by Transforming Trajectories into Actionable Insights." This repository is organized into three main directories, each representing a different domain evaluated in our work: Ego4D, VisualWebArena, and TEACh.

## Abstract
Humans excel at rapidly learning new skills by synthesizing and generalizing information from diverse sources, presenting a model for developing automated systems with similar adaptive learning capabilities. Although large-scale generative language and vision-language models (LLMs and VLMs) excel in few-shot learning and aid decision-making through in-context learning, they require high-quality examples and costly prompt engineering. We propose In-Context Abstraction Learning (ICAL) to enable language and vision model agents to efficiently learn new behaviors. ICAL builds a memory of experience insights from sub-optimal demonstrations and human feedback. Given a noisy demonstration in a new domain, VLMs abstract the trajectory into a general program by fixing inefficient actions and annotating cognitive abstractions: task relationships, object state changes, temporal subgoals, and task construals. These abstractions are refined by executing the trajectory with human feedback. This method quickly learns useful in-context multimodal experiences, significantly improving decision-making in retrieval-augmented LLM and VLM agents. Our ICAL agent surpasses the state-of-the-art in dialogue-based instruction following in TEACh, multimodal web agents in VisualWebArena, and action anticipation in Ego4D. In TEACh, we achieve a 12.6% improvement in goal-condition success. In VisualWebArena, our task success rate improves over the SOTA from 14.3% to 22.7%. In Ego4D action forecasting, we improve over few-shot GPT-4V and remain competitive with supervised models. Comparing weight fine-tuning and retrieval-augmented generation using our learned examples, we find a combination yields the best performance. By leveraging learned abstractions, our approach significantly reduces reliance on expertly-crafted examples and consistently outperforms in-context learning from action plans that lack such insights.

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

## Installation
To install and set up the ICAL environment, follow these general steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/Gabesarch/ICAL.git
    cd ICAL
    ```

2. Navigate to the domain directory you are interested in (e.g., `Ego4D`), and follow the installation instructions provided in its README file.

## Citation
If you find our work useful, please consider citing our paper:
```
@article{sarch2024ical,
  title={ICAL: Continual Learning of Multimodal Agents by Transforming Trajectories into Actionable Insights},
  author={},
  journal={},
  year={2024}
}
```

## Acknowledgements
This work builds on the existing research and frameworks in each of the respective domains. We would like to thank the contributors to the Ego4D, VisualWebArena, and TEACh projects for their invaluable resources and support.

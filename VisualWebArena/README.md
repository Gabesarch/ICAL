<h1 align="center">
    Running ICAL in VisualWebArena
</h1>

### Contents
<div class="toc">
<ul>
<li><a href="#install"></a>Install</li>
<li><a href="#dataset"> Dataset </a></li>
<li><a href="#running-the-ical-evaluation"> Running the ICAL evaluation </a></li>
<li><a href="#citation"> Citation </a></li>
</ul>
</div>

## ICAL examples
You can view the ICAL learned examples [here](https://github.com/Gabesarch/ICAL/VisualWebArena/learned_examples).

## Install
```bash
# Python 3.10+
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
playwright install
pip install -e .
```

You can also run the unit tests to ensure that VisualWebArena is installed correctly:
```
pytest -x
```

## End-to-end Evaluation
1. Setup the standalone environments.
Please check out [this page](environment_docker/README.md) for details.

2. Configurate the urls for each website.
```bash
export CLASSIFIEDS="<your_classifieds_domain>:9980"
export CLASSIFIEDS_RESET_TOKEN="4b61655535e7ed388f0d40a93600254c"  # Default reset token for classifieds site, change if you edited its docker-compose.yml
export SHOPPING="<your_shopping_site_domain>:7770"
export REDDIT="<your_reddit_domain>:9999"
export WIKIPEDIA="<your_wikipedia_domain>:8888"
export HOMEPAGE="<your_homepage_domain>:4399"
```

3. Generate config files for each test example:
```bash
python scripts/generate_test_data.py
```
You will see `*.json` files generated in the [config_files](./config_files) folder. Each file contains the configuration for one test example.

4. Obtain and save the auto-login cookies for all websites:
```
bash prepare.sh
```

5. Set up API keys.

If using OpenAI models, set a valid OpenAI API key (starting with `sk-`) as the environment variable:
```
export OPENAI_API_KEY=your_key
```

## Running the ICAL evaluation
To run the evaluation, replace the paths in `scripts/run_final_eval.sh` with your local paths. Then run the script:
```bash
sh scripts/run_final_eval.sh
```

### ICAL scripts
We provide our scripts for the VLM abstraction and human-in-the-loop in `ICAL_scripts`.

## Citation
If you find our environment or our models useful, please consider citing ICAL as well as VisualWebArena:
```
@article{sarch2024ical,
  title={},
  author={},
  journal={},
  year={2024}
}
@article{koh2024visualwebarena,
  title={VisualWebArena: Evaluating Multimodal Agents on Realistic Visual Web Tasks},
  author={Koh, Jing Yu and Lo, Robert and Jang, Lawrence and Duvvur, Vikram and Lim, Ming Chong and Huang, Po-Yu and Neubig, Graham and Zhou, Shuyan and Salakhutdinov, Ruslan and Fried, Daniel},
  journal={arXiv preprint arXiv:2401.13649},
  year={2024}
}
```

## Acknowledgements

This code builds on [VisualWebArena](https://github.com/web-arena-x/visualwebarena). Be sure to check them out!

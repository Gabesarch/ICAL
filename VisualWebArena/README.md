<h1 align="center">
    Running ICAL in VisualWebArena
</h1>

### Contents
<div class="toc">
<ul>
<li><a href="#install"> Install </a></li>
<li><a href="#setup"> Setup </a></li>
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

## Setup
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

You can also run the unit tests to ensure that VisualWebArena is installed correctly:
```
pytest -x
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

## Running the Human-in-the-Loop Feature
To run the human-in-the-loop ICAL agent and to collect human correctable trajectories, replace the paths in `scripts/human_in_the_loop.sh` with your local paths. Then run the script:
```bash
sh scripts/human_in_the_loop.sh
```

### ICAL scripts
We provide our scripts for the VLM abstraction and human-in-the-loop in `ICAL_scripts`.

## Acknowledgements

This code builds on [VisualWebArena](https://github.com/web-arena-x/visualwebarena). Be sure to check them out!

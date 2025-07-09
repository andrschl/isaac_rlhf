# isaac_rlhf (Preference-Based RL Library for Robotic Manipulation)
[**Installation**](#installation) | [**Training**](#training)

Library for inverse reinforcement learning and RLHF for robotic manipulation.

## Installation
1. Install IsaacLab v2.0 following the instructions in on the [IsaacLab documentation](https://isaac-sim.github.io/IsaacLab/v2.0.2/source/setup/installation/index.html).
2. Clone this repository
3. Activate the virtual environment with the isaaclab installation and run `python -m pip install -e source/isaac_rlhf`

## api key
To use free api, 
1. create an account in [openrouter](https://openrouter.ai/)
2. In `api_keys` folder, create `.env.api_keys` file and add your key like:
``` 
OPENROUTER_API_KEY=your_key
```
import the key by running:
```bash
source api_key_importer.sh
```

try running `eureka/llm_manager.py`. If key is correctly imported, it should query LLM and print a preference response.

`llm_manager` uses deepseek r1 model by default. However, model availability from OpenRouter may change, free models might become priced. Check for model availability.

## Adding a new task
To run `isaac_rlhf` on a new task, you must manually define a success metric for the task.
Add your task to `ENV_ID_TO_RL_TASK` of `config/rlhf_cfg.py`
Write your own success metric inside `eureka/success_metric` folder, and use the same name for python file as above.
Ex: 
for "Isaac-Humanoid-v0" task, we use 'humanoid' in ENV_ID_TO_RL_TASK, and success metric is defined in `humanoid.py`.

Write a function named `compute_success_metric`, this will be attached to env instance of isaaclab.
You can think of 'self' as isaaclab env instance.
Return a dictionary of key and values, including "success_metric".
These will be saved under `Eureka/` in tensorboard.

## How to run LLM preference

set relevant parameters in `scripts/rlhf/train_rlhf_config.yaml` 
`num_rlhf_iterations` is like number of generations, `num_rl_runs` is number of policies per generation. 
In total, it trains `num_rlhf_iterations` * `num_rl_runs` policies.

for preference mode,
"llm" will use preference from llm,
"sm" will just prefer policy with higher success metric,
"gt" will use the synthetic preference derived from ground truth reward weight.

run `train_rlhf_jiwon.py`
(set up vscode launch.json or run from terminal)
`train_rlhf_jiwon.py` will read parameters from `train_rlhf_config.yaml` and override `cli_args`.

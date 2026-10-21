# Assignment 2-1: Naive Agent

## Overview
The self-reflection agent predicts the effect of the chosen action at each step and evaluates its prediction using the observation at the next step, if the prediction was incorrect stores a lesson about it in the memory.
This evaluation allows it to change its plan and update its understanding of the environment.

## Setup

1. Git clone the BALROG Repo:
```bash
git clone https://github.com/balrog-ai/BALROG.git
cd BALROG
```

2. Create a Conda environment:
```bash
conda create -n balrog python=3.10 -y
conda activate balrog
pip install -e .
balrog-post-install
```

3. Add `self_reflection.py` to agents folder, and add `run_self_reflection.sh` to the BALROG directory.

4. Setup api key in SECRETS or local LLM (see BALROG ReadMe), by default our run script is setup to use openrouter,
   so running all models only requires one api key (enter the openrouter key in the openai key var in SECRETS):
```bash
OPENAI_API_KEY=<your_api_key>
```

5. Add the agents to the `__init__.py` AgentFactory: create_agent() method as shown below, or replace the `__init__.py` file in the BALROG repo with the `__init__.py` file in Ass2-3/code.
```python
from .self_reflection import PredictiveReflectionAgent 
.
.
.
elif self.config.agent.type == "self_reflection":
    return PredictiveReflectionAgent(
        client_factory,
        prompt_builder,
        max_memory_entries=self.config.agent.get("max_memory_entries", 15),
    )
```

## Running the experiments
To run our experiments (the naive agent across three diffferent llms on Crafter and BabyAI) run the following: 
```bash
bash run_self_reflection.sh
```

## Results and Analysis
The results and interpretation can be found in our report. Our raw results can be found in in `Ass2-3/results`.

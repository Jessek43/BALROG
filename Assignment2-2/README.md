# Assignment 2-2: Reasoning Agent


## Overview
How does combining three step reasoning with task specific rules influence performance over a naive agent?

The reasoning agent makes the LLM answer three questions before every action:
1. **What do I see?**: describe observations that are relevant.  
2. **What is the goal?**: formulate the goal given the observed environment and task.
3. **Given the current environment and goal, which action leads to reaching the goal the quickest?**: pick one action.

After the reasoning steps the prompt contains one rule per task environment, which were picked based on naive agent failures observed in logs.
- **BabyAI (a simple navigation game)**: "If an object is 1 step in front of you, it is directly in front of you, meaning "go forward" does nothing. You are able to pick up the object."
- **Crafter**: "Standing still does not protect from enemies. If you are chased by zombies and are too low health to fight, walk away instead of waiting."


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

3. Add `reasoning.py` to agents folder, and add `run_experiments.py` to the BALROG directory.

4. Setup api key in SECRETS or local LLM (see BALROG ReadMe):
```bash
OPENAI_API_KEY=<your_api_key>
```

5. Add the agents to the `__init__.py` AgentFactory: create_agent() method as shown below, or replace the `__init__.py` file in the BALROG repo with the `__init__.py` file in assignment2-2/code.
```python
from .reasoning import ReasoningAgent
.
.
.
elif self.config.agent.type == "reasoning":
    return ReasoningAgent(client_factory, prompt_builder)
```


6. Select experiments to run in `run_experiments.py` inside the BALROG repo by adding their names to `RUN`. Available experiments are found in the `EXPERIMENTS` dictionary in `run_experiments.py`.

## Running the experiments
When you have completed the setup and you chose your desired experiments inside `run_experiments.py`

Simply run the following: 
```bash
python run_experiments.py
```
Results of your runs will be saved under `full_results/results.csv` and logs under `full_results/<experiment>/run.log`

## Results and Analysis
The results and interpretation can be found in our report. Our raw results can be found in in `Assignment2-2/results`.
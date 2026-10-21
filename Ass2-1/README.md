# Assignment 2-1: Naive Agent

## Overview
The naive agent outputs only the action and no other text each step.
This agent was already implemented by BALROG.

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

3. Adding the files from the agent dir to the BALROG repo is not necessary as we did not change the naive agent, but you can do it if you want.
   Then add run_naive.sh to the BALROG repo.

4. Setup api key in SECRETS or local LLM (see BALROG ReadMe), by default our run script is setup to use openrouter,
   so running all models only requires one api key (enter the openrouter key in the openai key var in SECRETS):
```bash
OPENAI_API_KEY=<your_api_key>
```

## Running the experiments
To run our experiments (the naive agent across three diffferent llms on Crafter and BabyAI) run the following: 
```bash
bash run_naive.sh
```

## Results and Analysis
The results and interpretation can be found in our report. Our raw results can be found in in `Ass2-1/results`.

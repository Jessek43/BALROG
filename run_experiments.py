import os, subprocess, sys, glob



# TO DO:
# Fix results output
# add experiment grids



# run config
ENVS = "babyai-crafter"
CLIENT_NAME = "xai"
MODEL_ID = "upstage/solar-mini4"
AGENT_TYPE = "reasoning"
RUN_TYPE = "standard"
CLIENT_BASE_URL = "https://openrouter.ai/api/v1"
N_WORKERS = 10

presets = {
    "smoke": {"num_episodes_babyai": 1, "num_episodes_crafter": 1, "max_tokens": 8192},
    "standard": {"num_episodes_babyai": 10, "num_episodes_crafter": 10, "max_tokens": 8192}
}

run_cfg = presets[RUN_TYPE]



print(f"Starting run")
EVAL_OUTPUT_DIR = f"full_results/{RUN_TYPE}/{AGENT_TYPE}"
os.makedirs(EVAL_OUTPUT_DIR, exist_ok=True)
LOG_PATH = f"{EVAL_OUTPUT_DIR}/run.log"


cmd = [sys.executable, "eval.py", f"agent.type={AGENT_TYPE}", "agent.max_image_history=0",
    "agent.max_text_history=16", f"eval.num_workers={N_WORKERS}", f"eval.num_episodes.babyai={run_cfg['num_episodes_babyai']}",
    f"eval.num_episodes.crafter={run_cfg['num_episodes_crafter']}", f"client.generate_kwargs.max_tokens={run_cfg['max_tokens']}", 
    f"envs.names={ENVS}", f"client.client_name={CLIENT_NAME}", f"client.model_id={MODEL_ID}", f"client.base_url={CLIENT_BASE_URL}",
    f"eval.output_dir={EVAL_OUTPUT_DIR}"]

with open(LOG_PATH, "w") as log:
    result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)

if result.returncode != 0:
    print(f"FAILED {RUN_TYPE} run with {AGENT_TYPE} agent and model {MODEL_ID}: {EVAL_OUTPUT_DIR}")









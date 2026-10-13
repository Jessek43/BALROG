import os, subprocess, sys, glob

# TO DO:
# add option to do a full rerun and deleting old result, currently you have to delete results manually
# add better progress logging in terminal
# Run different models and agent lists sequentially through loops
# add argparse for things like --run-type and --experiment (exercise 1,2 and 3)
# add exercise to results output path



# run config
ENVS = "babyai-crafter"
CLIENT_NAME = "vllm"
MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
AGENT_TYPE = "naive"
RUN_TYPE = "standard"
CLIENT_BASE_URL = "http://0.0.0.0:8080/v1"

presets = {
    "smoke": {"num_episodes_babyai": 1, "num_episodes_crafter": 1, "max_tokens": 512, "n_runs":1},
    "standard": {"num_episodes_babyai": 10, "num_episodes_crafter": 10, "max_tokens": 8192, "n_runs":5}
}

run_cfg = presets[RUN_TYPE]

N_RUNS = run_cfg["n_runs"]

for k in range(1, N_RUNS + 1): # run same config k times
    print(f"Starting run {k} out of {N_RUNS}")
    EVAL_OUTPUT_DIR = f"results/{RUN_TYPE}/{AGENT_TYPE}/run_{k}"
    os.makedirs(EVAL_OUTPUT_DIR, exist_ok=True)
    LOG_PATH = f"{EVAL_OUTPUT_DIR}/run.log"

    done = glob.glob(f"{EVAL_OUTPUT_DIR}/*/summary.json")

    if done:
        print(f"Skipping run {k}, already done")
        continue

    cmd = [sys.executable, "eval.py", f"agent.type={AGENT_TYPE}", "agent.max_image_history=0",
       "agent.max_text_history=16", "eval.num_workers=32", f"eval.num_episodes.babyai={run_cfg['num_episodes_babyai']}",
       f"eval.num_episodes.crafter={run_cfg['num_episodes_crafter']}", f"client.generate_kwargs.max_tokens={run_cfg['max_tokens']}", 
       f"envs.names={ENVS}", f"client.client_name={CLIENT_NAME}", f"client.model_id={MODEL_ID}", f"client.base_url={CLIENT_BASE_URL}",
       f"eval.output_dir={EVAL_OUTPUT_DIR}"]

    with open(LOG_PATH, "w") as log:
        result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)

    if result.returncode != 0:
        print(f"FAILED Run {k} of {RUN_TYPE} run with {AGENT_TYPE} agent and model {MODEL_ID}: {EVAL_OUTPUT_DIR}")









import os, subprocess, sys, glob, json, csv



# run config
ENVS = "babyai-crafter"
CLIENT_NAME = "xai"
CLIENT_BASE_URL = "https://openrouter.ai/api/v1"
N_WORKERS = 10



MODELS = {
    "haiku-5.5":"anthropic/claude-haiku-5.5",
    "solar-mini-4":"upstage/solar-mini4"
}


presets = {
    "smoke": {"num_episodes_babyai": 1, "num_episodes_crafter": 1, "max_tokens": 8192},
    "standard": {"num_episodes_babyai": 10, "num_episodes_crafter": 10, "max_tokens": 8192}
}


EXPERIMENTS = {
    "haiku_reason_smoke": {"model": "haiku-5.5", "agent":"reasoning", "run_type":"smoke"},
    "solar_reason_smoke": {"model":"solar-mini-4", "agent":"reasoning", "run_type":"smoke"},
    "haiku_naive": {"model": "haiku-5.5", "agent":"naive", "run_type":"standard"},
    "solar_naive": {"model":"solar-mini-4", "agent":"naive", "run_type":"standard"},
    "haiku_reasoning": {"model":"haiku-5.5", "agent":"reasoning", "run_type":"standard"},
    "solar_reasoning": {"model":"solar-mini-4", "agent":"reasoning", "run_type":"standard"}
}

# which experiments to run right now
RUN = ["haiku_reasoning"]

RESULTS_CSV = "full_results/results.csv"
COLUMNS = ["experiment", "run_type", "model", "agent", "babyai", "babyai_se", "crafter", "crafter_se",
           "average", "average_se", "input_tokens", "output_tokens", "run_dir"]



for exp_name in RUN:
    exp = EXPERIMENTS[exp_name]
    RUN_TYPE, AGENT_TYPE, MODEL_ID = exp["run_type"], exp["agent"], MODELS[exp["model"]]
    run_cfg = presets[RUN_TYPE]
    


    print(f"Starting run {exp_name}: {RUN_TYPE} run with {AGENT_TYPE} agent and model {MODEL_ID} ")
    EVAL_OUTPUT_DIR = f"full_results/{exp_name}"
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
        print(f"FAILED {exp_name} run: see {LOG_PATH}")
        continue

    # take newest summary folder
    summary_path = max(glob.glob(f"{EVAL_OUTPUT_DIR}/*/summary.json"), key=os.path.getmtime)
    with open(summary_path) as f:
        summary = json.load(f)
    envs = summary["environments"]

    row = {
        "experiment": exp_name,
        "run_type": RUN_TYPE,
        "model": exp["model"],
        "agent": AGENT_TYPE,
        "babyai": envs.get("babyai", {}).get("progression_percentage"),
        "babyai_se": envs.get("babyai", {}).get("standard_error"),
        "crafter": envs.get("crafter", {}).get("progression_percentage"),
        "crafter_se": envs.get("crafter", {}).get("standard_error"),
        "average": summary["average_progress"],
        "average_se": summary["standard_error"],
        "input_tokens": summary["total_input_tokens"],
        "output_tokens": summary["total_output_tokens"],
        "run_dir": os.path.dirname(summary_path)
    }

    new_file = not os.path.exists(RESULTS_CSV)
    with open(RESULTS_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if new_file:
            writer.writeheader()
        writer.writerow(row)

    print(f"Finished {exp_name} run: average progression: {summary['average_progress']:.2f}%, see {RESULTS_CSV}")






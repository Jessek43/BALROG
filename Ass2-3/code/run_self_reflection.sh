#!/bin/bash

export OPENAI_BASE_URL="https://openrouter.ai/api/v1"

# Define the models to evaluate
MODELS=(
    "upstage/solar-mini4"
    "mistralai/mistral-nemo"
    "anthropic/claude-haiku-5.5"
)

# Loop through each model and run the evaluation
for MODEL_ID in "${MODELS[@]}"; do
    echo "========================================================"
    echo "Starting evaluation for model: $MODEL_ID"
    echo "========================================================"
    
    python eval.py \
        agent.type=self_reflection \
        envs.names=babyai-crafter \
        eval.num_episodes.babyai=10 \
        eval.num_episodes.crafter=10 \
        eval.num_workers=25 \
        agent.max_text_history=16 \
        client.client_name=openai \
        client.model_id="$MODEL_ID"
        
    echo "Finished evaluation for $MODEL_ID."
    echo ""
done

echo "All evaluations complete!"


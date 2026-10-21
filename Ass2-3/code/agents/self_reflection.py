import re
from balrog.agents.base import BaseAgent
from balrog.prompt_builder.history import Message

class PredictiveReflectionAgent(BaseAgent):
    """A self-reflection agent with memory."""

    def __init__(self, client_factory, prompt_builder, max_memory_entries=15):
        super().__init__(client_factory, prompt_builder)
        self.client = client_factory()
        self.max_memory_entries = max_memory_entries
        self._init_episode_state()

    def _init_episode_state(self):
        self.step_count = 0
        self.plan = None
        self.last_prediction = None
        self.last_action = None
        self.memory = []

    def reset(self):
        super().reset()
        self._init_episode_state()

    def act(self, obs, prev_action=None):
        if prev_action:
            self.prompt_builder.update_action(prev_action)
        self.prompt_builder.update_observation(obs)

        # Build the state context and step-specific instructions
        state_context = self._build_state_context(prev_action)
        if self.step_count == 0 or self.last_prediction is None:
            instructions = self._get_first_step_instructions()
        else:
            instructions = self._get_next_step_instructions()

        messages = self.prompt_builder.get_prompt()
        if messages and messages[-1].role == "user":
            messages[-1].content += "\n\n" + state_context + "\n\n" + instructions

        response = self.client.generate(messages)
        parsed = self._parse_response(response.completion)

        # Store new memory if the agent decided to record one
        if parsed["new_memory"] and parsed["new_memory"].upper() != "NONE":
            if parsed["new_memory"] not in self.memory:
                self.memory.append(parsed["new_memory"])

        # Compress memory if it gets too large
        if len(self.memory) > self.max_memory_entries:
            self._compress_memory()

        # Update active plan
        if parsed["plan"] and parsed["plan"].upper() != "KEEP":
            self.plan = parsed["plan"]

        # Save prediction and action for the next step's evaluation
        self.last_prediction = parsed["prediction"]
        self.last_action = parsed["action"]
        self.step_count += 1

        return response._replace(reasoning=response.completion, completion=parsed["action"])

    def _build_state_context(self, prev_action):
        memory_str = (
            "\n".join(f"{i}. {mem}" for i, mem in enumerate(self.memory, start=1))
            if self.memory
            else "None recorded yet."
        )
        plan_str = self.plan if self.plan else "You have no plan yet."
        
        pred_str = "None (this is the first step)."
        if self.last_prediction and prev_action:
            pred_str = f"Previous Action Executed: {prev_action}\nPredicted Effect: {self.last_prediction}"

        return f"""
=== AGENT INTERNAL STATE ===
{pred_str}

Current Plan:
{plan_str}

Important Memories (Discovered rules, blocked paths, item locations):
{memory_str}
=============================
""".strip()

    def _get_first_step_instructions(self):
        return """
This is the first step.
1. Make a step-by-step plan to achieve your goal based on what you currently see.
2. Select your first action and predict what will change in your view when you execute it.

Format your answer EXACTLY like this (keep all values on a single line, do not write paragraphs):
NEW_MEMORY: <record any important visible items or layout facts, or "NONE">
PLAN: <your step-by-step plan>
PREDICTION: <expected direct effect of your chosen action>
ACTION: <exact command as defined in the rules without extra text>
""".strip()

    def _get_next_step_instructions(self):
        return """
Follow these steps carefully in order:
1. Did your "Previous Action Executed" succeed AND make progress? 
   - If you tried to move forward but the view did not change, you hit a wall (FAILED).
   - If you are just turning in circles without moving closer to your goal, you are stuck (FAILED).
2. If you FAILED, you MUST write a NEW_MEMORY noting the blocked path or loop, and you MUST write a new PLAN to break the loop.
3. Select your next action.

Format your answer EXACTLY like this (keep all values on a single line, do not write paragraphs):
OBSERVATION_CHECK: <"SUCCESS" or "FAILED: [reason]">
NEW_MEMORY: <"NONE" or a short note about a blocked path or being stuck in a loop>
PLAN: <"KEEP" or your updated step-by-step plan>
PREDICTION: <short expected effect of your chosen action>
ACTION: <exact command as defined in the rules without extra text>
""".strip()

    def _parse_response(self, text):
        def extract_line_field(prefix, default=""):
            match = re.search(rf"^{prefix}:\s*([^\n\r]+?)(?:\r?\n|\Z)", text, re.IGNORECASE | re.MULTILINE)
            return match.group(1).strip() if match else default

        new_memory = extract_line_field("NEW_MEMORY", "NONE")
        plan = extract_line_field("PLAN", "KEEP")
        prediction = extract_line_field("PREDICTION", "")
        raw_action = extract_line_field("ACTION", "")

        clean_action = raw_action.strip("`\"'* -")

        return {
            "new_memory": new_memory.strip("`\"' "),
            "plan": plan,
            "prediction": prediction,
            "action": clean_action,
        }

    def _compress_memory(self):
        """Ask the model to select the indices of the most important memories to keep."""
        target_size = max(5, self.max_memory_entries // 2)
        numbered_list = "\n".join(f"{i}. {mem}" for i, mem in enumerate(self.memory, start=1))
        
        prompt = f"""
The Agent Memory has reached its limit ({len(self.memory)} entries).
Keep the most important environmental facts, blocked paths, and game rules. Drop redundant or outdated notes.

Current Memory:
{numbered_list}

Select at most {target_size} items to keep.
Output ONLY the item numbers separated by commas on a single line, like this:
KEEP: 1, 3, 5
""".strip()

        messages = self.prompt_builder.get_prompt()
        if messages and messages[-1].role == "user":
            compress_msg = [Message(role="user", content=prompt)]
            resp = self.client.generate(compress_msg)

            keep_match = re.search(r"KEEP:\s*([0-9,\s]+)", resp.completion, re.IGNORECASE)
            search_text = keep_match.group(1) if keep_match else resp.completion

            picked_indices = []
            for num_str in re.findall(r"\d+", search_text):
                idx = int(num_str) - 1
                if 0 <= idx < len(self.memory) and idx not in picked_indices:
                    picked_indices.append(idx)

            if picked_indices:
                self.memory = [self.memory[i] for i in picked_indices[:target_size]]
            else:
                # Fallback: keep the most recent
                self.memory = self.memory[-target_size:]

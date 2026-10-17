import re

from balrog.agents.base import BaseAgent


class ReasoningAgent(BaseAgent):
    """An agent that generates actions based on three step reasoning."""

    def __init__(self, client_factory, prompt_builder):
        super().__init__(client_factory, prompt_builder)
        self.client = client_factory()

    def act(self, obs, prev_action=None):
        """Generate reasoning and the next action

        Args:
            obs (dict): The current observation in the environment.
            prev_action (str, optional): The previous action taken.

        Returns:
            LLMResponse: The response containing the step reasoning and selected action.
        """
        if prev_action:
            self.prompt_builder.update_action(prev_action)
        self.prompt_builder.update_observation(obs)


    

        reasoning_instructions = """
Ask yourself the following questions: 1. What do I see? Observe the environment and map everything that you think is relevant. 
2. What is the goal? Formulate what the exact goal is given the environment and task. 
3. Given the current environment and goal, which action leads to reaching the goal the most? 
Consider what the next best action is given your formulations of the environment observations and the main goal you formulated. 
Select the single best possible action that will lead you to this goal the quickest. Produce
a single action at every step. 
If you are playing a simple navigation game, keep these rules in mind:
- IF an object is 1 step in front of you, it is directly in front of you, meaning "go forward" does nothing. You are able to pick up the object.
If you are playing Crafter, keep these rules in mind:
- Standing still does not protect from enemies. If you are chased by zombies and are too low health to fight, walk away instead of waiting.
- Never sleep unless enclosed: when energy is low and you have stone and a pickaxe, place stone on every open side (water counts), sleep, then mine a stone to leave once no zombies are near.
- If you are on low-health (under 4) and there is a zombie within three steps, place a stone between you and the zombie if possible and move away from it.
Format your answer EXACTLY in the following way, while keeping all values on a single line without paragraphs:
REASONING: <your formulated answers to the three questions: (1) what do i see? (2) what is the goal? and (3) Given the current environment and goal, which action leads to reaching the goal the most?>
ACTION: <your next action, exactly one action from the list of legal actions> """.strip()

        messages = self.prompt_builder.get_prompt()
        if messages and messages[-1].role == "user":
            messages[-1].content += "\n\n" + reasoning_instructions

        response = self.client.generate(messages)

        # Extract the reasoning and action from the LLM's response
        reasoning, action = self._extract_reasoning_and_action(response.completion)

        # Save the reasoning in the response.reasoning field and the action in response.completion
        response = response._replace(reasoning=reasoning, completion=action)
        return response

    def _extract_reasoning_and_action(self, response_text):
        """Extract the reasoning and action from the LLM's response text.

        Args:
            response_text (str): The response text from the LLM.

        Returns:
            tuple: A tuple containing the extracted reasoning and action.
        """
        reasoning_match = re.search(r"REASONING:\s*(.*?)(?=\nACTION:|\Z)", response_text, re.IGNORECASE | re.DOTALL)
        action_match = re.search(r"ACTION:\s*(.*)", response_text, re.IGNORECASE)

        reasoning = reasoning_match.group(1).strip() if reasoning_match else "No changes to the reasoning."
        action = action_match.group(1).strip() if action_match else None

        return reasoning, action

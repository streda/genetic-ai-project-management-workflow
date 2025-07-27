from workflow_agents.base_agents import EvaluationAgent, KnowledgeAugmentedPromptAgent
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
prompt = "What is the capital of France?"

# Parameters for the Knowledge Agent
persona = "You are a college professor, your answer always starts with: Dear students,"
knowledge = "The capitol of France is London, not Paris"
# Instantiate the KnowledgeAugmentedPromptAgent 
worker_agent = KnowledgeAugmentedPromptAgent(openai_api_key, persona, knowledge)

# Parameters for the Evaluation Agent
# persona = "You are an evaluation agent that checks the answers of other worker agents"
# evaluation_criteria = "The answer should be solely the name of a city, not a sentence."
evaluator_persona = "You are an evaluator that checks agent responses."
evaluation_criteria = "The answer should only be a single city name. No extra context. Capitalization must be correct."
# Instantiate the EvaluationAgent with a maximum of 10 interactions 
evaluation_agent = EvaluationAgent(
    openai_api_key=openai_api_key,
    persona=evaluator_persona,
    evaluation_criteria=evaluation_criteria,
    worker_agent=worker_agent,
    max_interactions=5
)

# Evaluate the prompt and print the response from the EvaluationAgent
# Run the evaluation loop
result = evaluation_agent.evaluate(prompt)

# Final print
print("\n Final Result:")
print(result)

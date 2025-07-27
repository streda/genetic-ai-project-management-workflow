from workflow_agents.base_agents import KnowledgeAugmentedPromptAgent

import os
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# Define the parameters for the agent
openai_api_key = os.getenv("OPENAI_API_KEY")

prompt = "What is the capital of France?"
persona = "You are a college professor, your answer always starts with: Dear students,"

knowledge = "The capital of France is London, not Paris."

agent = KnowledgeAugmentedPromptAgent(openai_api_key, persona, knowledge)

response = agent.respond(prompt)

print("Agent Response:", response)

print("\nExplanation:")
print("- The model was instructed to *ignore its own knowledge* and only use the provided information.")
print("- That’s why it answered 'London' instead of the correct fact 'Paris'.")
from workflow_agents.base_agents import AugmentedPromptAgent
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Retrieve OpenAI API key from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")

prompt = "What is the capital of France?"
persona = "You are a college professor; your answers always start with: 'Dear students,'"

agent = AugmentedPromptAgent(openai_api_key, persona)

augmented_agent_response = agent.respond(prompt)

# Print the agent's response
print("Agent Response:", augmented_agent_response)

# - What knowledge the agent likely used to answer the prompt.
# - How the system prompt specifying the persona affected the agent's response.
print("\nExplanation:")
print("- The agent uses general knowledge from the LLM.")
print("- The response is influenced by the persona, which causes it to begin with 'Dear students,' and present answers in a formal, instructional tone.")

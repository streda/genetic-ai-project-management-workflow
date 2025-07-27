from workflow_agents.base_agents import DirectPromptAgent 
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

prompt = "What is the Capital of France?"

direct_agent = DirectPromptAgent(openai_api_key)
direct_agent_response = direct_agent.respond(prompt)

# Print the response from the agent
print("Agent Response:", direct_agent_response)

print("\nExplanation: This response is generated purely based on general knowledge embedded in the LLM model (GPT-3.5-turbo), without any added context, tools, or memory.")

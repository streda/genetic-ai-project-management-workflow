
# Import the KnowledgeAugmentedPromptAgent and RoutingAgent
from workflow_agents.base_agents import RoutingAgent, KnowledgeAugmentedPromptAgent
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

# Define the Texas Knowledge Augmented Prompt Agent
texas_agent = KnowledgeAugmentedPromptAgent(
    openai_api_key=openai_api_key,
    persona = "You are a college professor",
    knowledge = "You know everything about Texas"
)

# Define the Europe Knowledge Augmented Prompt Agent
europe_agent = KnowledgeAugmentedPromptAgent(
    openai_api_key=openai_api_key,
    persona="You are a professor of European history.",
    knowledge = "You know everything about Europe"
)

# Define the Math Knowledge Augmented Prompt Agent
math_agent = KnowledgeAugmentedPromptAgent(
    openai_api_key=openai_api_key,
    persona = "You are a college math professor",
    knowledge = "You know everything about math, you take prompts with numbers, extract math formulas, and show the answer without explanation"
)

agents = [
    {
        "name": "texas agent",
        "description": "Answer a question about Texas",
        # Call the Texas Agent to respond to prompts
        "func": lambda x: texas_agent.respond(x)
    },
    {
        "name": "europe agent",
        "description": "Answer a question about Europe",
        # Define a function to call the Europe Agent
        "func": lambda x: europe_agent.respond(x)
    },
    {
        "name": "math agent",
        "description": "When a prompt contains numbers, respond with a math formula",
        # Define a function to call the Math Agent
        "func": lambda x: math_agent.respond(x)
    }
]
routing_agent = RoutingAgent(openai_api_key, agents)

# Printing the RoutingAgent responses to the following prompts:
#           - "Tell me about the history of Rome, Texas"
#           - "Tell me about the history of Rome, Italy"
#           - "One story takes 2 days, and there are 20 stories"

print("\n--- Routing Test: Rome, Texas ---")
print(routing_agent.route("Tell me about the history of Rome, Texas"))

print("\n--- Routing Test: Rome, Italy ---")
print(routing_agent.route("Tell me about the history of Rome, Italy"))

print("\n--- Routing Test: Math ---")
print(routing_agent.route("One story takes 2 days, and there are 20 stories"))
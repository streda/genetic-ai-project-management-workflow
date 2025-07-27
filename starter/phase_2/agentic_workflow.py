# agentic_workflow.py
import sys
import os
sys.path.append(os.path.dirname(__file__))

# Import the following agents: ActionPlanningAgent, KnowledgeAugmentedPromptAgent, EvaluationAgent, RoutingAgent from the workflow_agents.base_agents module
from starter.phase_1.workflow_agents.base_agents import (
    ActionPlanningAgent,
    KnowledgeAugmentedPromptAgent,
    EvaluationAgent,
    RoutingAgent
)

import os
from dotenv import load_dotenv

# Load the OpenAI key into a variable called openai_api_key
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Load the product spec document Product-Spec-Email-Router.txt into a variable called product_spec
with open("starter/phase_2/Product-Spec-Email-Router.txt", "r") as f:
    product_spec = f.read()
# Instantiate all the agents

# Action Planning Agent
knowledge_action_planning = (
    "Stories are defined from a product spec by identifying a "
    "persona, an action, and a desired outcome for each story. "
    "Each story represents a specific functionality of the product "
    "described in the specification. \n"
    "Features are defined by grouping related user stories. \n"
    "Tasks are defined for each story and represent the engineering "
    "work required to develop the product. \n"
    "A development Plan for a product contains all these components"
)
# Instantiates an action_planning_agent using the 'knowledge_action_planning'
action_planning_agent = ActionPlanningAgent(openai_api_key, knowledge_action_planning)

# Product Manager - Knowledge Augmented Prompt Agent
persona_product_manager = "You are a Product Manager, you are responsible for defining the user stories for a product."
knowledge_product_manager = (
    "Stories are defined by writing sentences with a persona, an action, and a desired outcome. "
    "The sentences always start with: As a "
    "Write several stories for the product spec below, where the personas are the different users of the product. "
)
# Instantiates a product_manager_knowledge_agent using 'persona_product_manager' and the completed 'knowledge_product_manager'
knowledge_product_manager += product_spec
product_manager_knowledge_agent = KnowledgeAugmentedPromptAgent(
    openai_api_key,
    persona_product_manager,
    knowledge_product_manager
)


# Product Manager - Evaluation Agent
# Defines the persona and evaluation criteria for a Product Manager evaluation agent and instantiate it as product_manager_evaluation_agent. This agent will evaluate the product_manager_knowledge_agent.

# The evaluation_criteria specifies the expected structure for user stories (e.g., "As a [type of user], I want [an action or feature] so that [benefit/value].").
product_manager_evaluation_agent = EvaluationAgent(
    openai_api_key,
    persona="You are an evaluation agent that checks user stories for quality.",
    evaluation_criteria="Each user story must follow the format: As a [type of user], I want [an action or feature] so that [benefit/value].",
    worker_agent=product_manager_knowledge_agent,
    max_interactions=3
)

# Program Manager - Knowledge Augmented Prompt Agent
persona_program_manager = "You are a Program Manager, you are responsible for defining the features for a product."
knowledge_program_manager = "Features of a product are defined by organizing similar user stories into cohesive groups."
# Instantiates a program_manager_knowledge_agent using 'persona_program_manager' and 'knowledge_program_manager'

program_manager_knowledge_agent = KnowledgeAugmentedPromptAgent(
    openai_api_key,
    persona_program_manager,
    knowledge_program_manager + "\n" + product_spec
)

# Program Manager - Evaluation Agent
persona_program_manager_eval = "You are an evaluation agent that checks the answers of other worker agents."

# Instantiate a program_manager_evaluation_agent using 'persona_program_manager_eval' and the evaluation criteria below.
#                      "The answer should be product features that follow the following structure: " \
#                      "Feature Name: A clear, concise title that identifies the capability\n" \
#                      "Description: A brief explanation of what the feature does and its purpose\n" \
#                      "Key Functionality: The specific capabilities or actions the feature provides\n" \
#                      "User Benefit: How this feature creates value for the user"

program_manager_evaluation_agent = EvaluationAgent(
    openai_api_key,
    persona_program_manager_eval,
    evaluation_criteria=(
        "The answer should be product features that follow the structure:\n"
        "Feature Name: A clear title\n"
        "Description: What the feature does\n"
        "Key Functionality: Specific actions/capabilities\n"
        "User Benefit: How it helps users"
    ),
    worker_agent=program_manager_knowledge_agent,
    max_interactions=3
)

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

# Development Engineer - Knowledge Augmented Prompt Agent
persona_dev_engineer = "You are a Development Engineer, you are responsible for defining the development tasks for a product."
knowledge_dev_engineer = "Development tasks are defined by identifying what needs to be built to implement each user story."

# Instantiates a development_engineer_knowledge_agent using 'persona_dev_engineer' and 'knowledge_dev_engineer'
development_engineer_knowledge_agent = KnowledgeAugmentedPromptAgent(
    openai_api_key,
    persona_dev_engineer,
    knowledge_dev_engineer + "\n" + product_spec
)

# Development Engineer - Evaluation Agent
persona_dev_engineer_eval = "You are an evaluation agent that checks the answers of other worker agents."

# Instantiate a development_engineer_evaluation_agent using 'persona_dev_engineer_eval' and the evaluation criteria below.
#                      "The answer should be tasks following this exact structure: " \
#                      "Task ID: A unique identifier for tracking purposes\n" \
#                      "Task Title: Brief description of the specific development work\n" \
#                      "Related User Story: Reference to the parent user story\n" \
#                      "Description: Detailed explanation of the technical work required\n" \
#                      "Acceptance Criteria: Specific requirements that must be met for completion\n" \
#                      "Estimated Effort: Time or complexity estimation\n" \
#                      "Dependencies: Any tasks that must be completed first"

development_engineer_evaluation_agent = EvaluationAgent(
    openai_api_key,
    persona_dev_engineer_eval,
    evaluation_criteria=(
        "Each task must follow:\n"
        "Task ID, Task Title, Related User Story, Description, Acceptance Criteria, Estimated Effort, Dependencies"
    ),
    worker_agent=development_engineer_knowledge_agent,
    max_interactions=3
)


# Routing Agent
# Instantiate a routing_agent. Need to define a list of agent dictionaries (routes) for Product Manager, Program Manager, and Development Engineer. Each dictionary should contain 'name', 'description', and 'func' (linking to a support function). Assign this list to the routing_agent's 'agents' attribute.


# Job function persona support functions
# Define the support functions for the routes of the routing agent (e.g., product_manager_support_function, program_manager_support_function, development_engineer_support_function).
# Each support function should:
#   1. Take the input query (e.g., a step from the action plan).
#   2. Get a response from the respective Knowledge Augmented Prompt Agent.
#   3. Have the response evaluated by the corresponding Evaluation Agent.
#   4. Return the final validated response.

def product_manager_support_function(prompt):
    return product_manager_evaluation_agent.evaluate(prompt)["final_response"]

def program_manager_support_function(prompt):
    return program_manager_evaluation_agent.evaluate(prompt)["final_response"]

def development_engineer_support_function(prompt):
    return development_engineer_evaluation_agent.evaluate(prompt)["final_response"]

routing_agent = RoutingAgent(openai_api_key, agents=[
    {
        "name": "Product Manager",
        "description": "Define user stories from the product spec.",
        "func": product_manager_support_function
    },
    {
        "name": "Program Manager",
        "description": "Organize user stories into feature groups.",
        "func": program_manager_support_function
    },
    {
        "name": "Development Engineer",
        "description": "Create engineering tasks from user stories.",
        "func": development_engineer_support_function
    }
])

# Run the workflow

print("\n*** Workflow execution started ***\n")
# Workflow Prompt
# ****
# workflow_prompt = "What would the development tasks for this product be?"
workflow_prompt = "Create a complete development plan for the Email Router product, including user stories, features, and engineering tasks."
# ****
print(f"Task to complete in this workflow, workflow prompt = {workflow_prompt}")

# Implement the workflow.
#   1. Use the 'action_planning_agent' to extract steps from the 'workflow_prompt'.
#   2. Initialize an empty list to store 'completed_steps'.
#   3. Loop through the extracted workflow steps:
#      a. For each step, use the 'routing_agent' to route the step to the appropriate support function.
#      b. Append the result to 'completed_steps'.
#      c. Print information about the step being executed and its result.
#   4. After the loop, print the final output of the workflow (the last completed step).

print("\nDefining workflow steps from the workflow prompt")
steps = action_planning_agent.extract_steps_from_prompt(workflow_prompt)
completed_steps = []

for idx, step in enumerate(steps):
    print(f"\n[Step {idx+1}] {step}")
    result = routing_agent.route(step)
    print(f"→ Result:\n{result}\n")
    completed_steps.append(result)

# print("\n*** Final Output ***")
# print(completed_steps[-1])


# print("\n*** Final Output: All Completed Steps ***")
# for idx, step_output in enumerate(completed_steps, start=1):
    # print(f"\n[Output {idx}]\n{step_output}")


final_output_path = "starter/phase_2/Final-Output-Email-Router.md"

labels = ["USER STORIES", "PRODUCT FEATURES", "ENGINEERING TASKS"]

with open(final_output_path, "w") as f:
    f.write("# Email Router – Final Development Plan\n")
    for idx, step_output in enumerate(completed_steps, start=1):
        section_title = labels[idx - 1]
        f.write(f"\n## {section_title}\n")
        f.write(f"{step_output.strip()}\n")
print(f"\n Final structured output saved to: {final_output_path}")
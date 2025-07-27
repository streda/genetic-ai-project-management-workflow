from workflow_agents.base_agents import RAGKnowledgePromptAgent

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Define the parameters for the agent
openai_api_key = os.getenv("OPENAI_API_KEY")

persona = "You are a college professor, your answer always starts with: Dear students,"

# Defining filename manually so both functions can refer to the same file
from datetime import datetime
import uuid

unique_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
chunk_filename = f"chunks-{unique_id}.csv"
embedding_filename = f"embeddings-{unique_id}.csv"

RAG_knowledge_prompt_agent = RAGKnowledgePromptAgent(openai_api_key, persona, chunk_size=800)
RAG_knowledge_prompt_agent.unique_filename = unique_id

knowledge_text = """
Clara, a marine biologist, created a podcast called Crosscurrents. The show explored the intersection of science, culture, and ethics. She interviewed researchers and activists on topics ranging from neuroplasticity to decentralized identity.
"""

import csv

chunks = RAG_knowledge_prompt_agent.chunk_text(knowledge_text)
with open(f"chunks-{RAG_knowledge_prompt_agent.unique_filename}", 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=["text", "chunk_size"])
    writer.writeheader()
    for chunk in chunks:
        writer.writerow({k: chunk[k] for k in ["text", "chunk_size"]})


embeddings_df = RAG_knowledge_prompt_agent.calculate_embeddings()

prompt = "What is the podcast that Clara hosts about?"

prompt = "What is the podcast that Clara hosts about?"
print(f"\nPrompt: {prompt}")
response = RAG_knowledge_prompt_agent.find_prompt_in_knowledge(prompt)
print(f"\nAgent Response:\n{response}")
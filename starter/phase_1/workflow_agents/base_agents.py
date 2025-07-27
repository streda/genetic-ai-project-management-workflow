# Import the OpenAI class from the openai library
from openai import OpenAI
import numpy as np
import pandas as pd
import re
import csv
import uuid
from datetime import datetime


# DirectPromptAgent class definition
class DirectPromptAgent:
    
    def __init__(self, openai_api_key):
        # Initializing the agent
        # Defining an attribute named openai_api_key to store the OpenAI API key 
        self.client = OpenAI(
            api_key=openai_api_key                    
        )
        
    def respond(self, prompt):
        # Generate a response using the OpenAI API
        response = self.client.chat.completions.create(
            model= "gpt-3.5-turbo", # Specify the model to use (gpt-3.5-turbo)
            messages=[
                # Providing the user's prompt 
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        # Returns only the textual content of the response (not the full JSON response).
        return response.choices[0].message.content

# AugmentedPromptAgent class definition
class AugmentedPromptAgent:
    def __init__(self, openai_api_key, persona):
        """Initialize the agent with given attributes."""
        # Create an attribute for the agent's persona
        self.persona = persona

        self.client = OpenAI(
            api_key=openai_api_key
        )

    def respond(self, input_text):
        """Generate a response using OpenAI API."""

        # Declare a variable 'response' that calls OpenAI's API for a chat completion.
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                # Add a system prompt instructing the agent to assume the defined persona and explicitly forget previous context.
                {
                    "role": "system",
                    "content": f"You are a helpful assistant. Assume the following persona and always respond accordingly: {self.persona}. Forget all previous context."
                },
                {"role": "user", "content": input_text}
            ],
            temperature=0
        )

        # Return only the textual content of the response, not the full JSON payload.
        return response.choices[0].message.content
    
# KnowledgeAugmentedPromptAgent class definition
class KnowledgeAugmentedPromptAgent:
    def __init__(self, openai_api_key, persona, knowledge):
        """Initialize the agent with provided attributes."""
        self.persona = persona
        # Create an attribute to store the agent's knowledge.
        self.knowledge = knowledge
        self.client = OpenAI(
            api_key=openai_api_key
        )

    def respond(self, input_text):
        """Generate a response using the OpenAI API."""

        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": (
                        f"You are {self.persona}, a knowledge-based assistant. Forget all previous context.\n"
                        f"Use only the following knowledge to answer, do not use your own knowledge:\n{self.knowledge}\n"
                        f"Answer the prompt based on this knowledge, not your own."
                    )
                },
                
                # Add the user's input prompt as a user message.
                 {
                    "role": "user",
                    "content": input_text
                }
            ],
            temperature=0
        )
        return response.choices[0].message.content
    
class RAGKnowledgePromptAgent:
    """
    An agent that uses Retrieval-Augmented Generation (RAG) to find knowledge from a large corpus
    and leverages embeddings to respond to prompts based solely on retrieved information.
    """

    def __init__(self, openai_api_key, persona, chunk_size=2000, chunk_overlap=100):
        """
        Initializes the RAGKnowledgePromptAgent with API credentials and configuration settings.

        Parameters:
        openai_api_key (str): API key for accessing OpenAI.
        persona (str): Persona description for the agent.
        chunk_size (int): The size of text chunks for embedding. Defaults to 2000.
        chunk_overlap (int): Overlap between consecutive chunks. Defaults to 100.
        """
        self.persona = persona
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.openai_api_key = openai_api_key
        self.unique_filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.csv"

    def get_embedding(self, text):
        """
        Fetches the embedding vector for given text using OpenAI's embedding API.

        Parameters:
        text (str): Text to embed.

        Returns:
        list: The embedding vector.
        """
        client = OpenAI(api_key=self.openai_api_key)
        response = client.embeddings.create(
            model="text-embedding-3-large",
            input=text,
            encoding_format="float"
        )
        return response.data[0].embedding

    def calculate_similarity(self, vector_one, vector_two):
        """
        Calculates cosine similarity between two vectors.

        Parameters:
        vector_one (list): First embedding vector.
        vector_two (list): Second embedding vector.

        Returns:
        float: Cosine similarity between vectors.
        """
        vec1, vec2 = np.array(vector_one), np.array(vector_two)
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def chunk_text(self, text):
        """
        Splits text into manageable chunks, attempting natural breaks.

        Parameters:
        text (str): Text to split into chunks.

        Returns:
        list: List of dictionaries containing chunk metadata.
        """
        separator = "\n"
        text = re.sub(r'\s+', ' ', text).strip()

        if len(text) <= self.chunk_size:
            return [{"chunk_id": 0, "text": text, "chunk_size": len(text)}]

        chunks, start, chunk_id = [], 0, 0

        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            if separator in text[start:end]:
                end = start + text[start:end].rindex(separator) + len(separator)

            chunks.append({
                "chunk_id": chunk_id,
                "text": text[start:end],
                "chunk_size": end - start,
                "start_char": start,
                "end_char": end
            })

            start = end - self.chunk_overlap
            chunk_id += 1

        with open(f"chunks-{self.unique_filename}", 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=["text", "chunk_size"])
            writer.writeheader()
            for chunk in chunks:
                writer.writerow({k: chunk[k] for k in ["text", "chunk_size"]})

        return chunks

    def calculate_embeddings(self):
        """
        Calculates embeddings for each chunk and stores them in a CSV file.

        Returns:
        DataFrame: DataFrame containing text chunks and their embeddings.
        """
        df = pd.read_csv(f"chunks-{self.unique_filename}", encoding='utf-8')
        df['embeddings'] = df['text'].apply(self.get_embedding)
        df.to_csv(f"embeddings-{self.unique_filename}", encoding='utf-8', index=False)
        return df

    def find_prompt_in_knowledge(self, prompt):
        """
        Finds and responds to a prompt based on similarity with embedded knowledge.

        Parameters:
        prompt (str): User input prompt.

        Returns:
        str: Response derived from the most similar chunk in knowledge.
        """
        prompt_embedding = self.get_embedding(prompt)
        df = pd.read_csv(f"embeddings-{self.unique_filename}", encoding='utf-8')
        df['embeddings'] = df['embeddings'].apply(lambda x: np.array(eval(x)))
        df['similarity'] = df['embeddings'].apply(lambda emb: self.calculate_similarity(prompt_embedding, emb))

        best_chunk = df.loc[df['similarity'].idxmax(), 'text']

        client = OpenAI(api_key=self.openai_api_key)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": f"You are {self.persona}, a knowledge-based assistant. Forget previous context."},
                {"role": "user", "content": f"Answer based only on this information: {best_chunk}. Prompt: {prompt}"}
            ],
            temperature=0
        )

        return response.choices[0].message.content

class EvaluationAgent:
    
    def __init__(self, openai_api_key, persona, evaluation_criteria, worker_agent, max_interactions):
        # Initialize the EvaluationAgent with given attributes.
        self.persona = persona
        self.evaluation_criteria = evaluation_criteria
        self.worker_agent = worker_agent
        self.max_interactions = max_interactions
        self.client = OpenAI(
            api_key=openai_api_key
        )
    
    def evaluate(self, initial_prompt):
        # This method manages interactions between agents to achieve a solution.
        prompt_to_evaluate = initial_prompt

        # Set loop to iterate up to the maximum number of interactions:
        for i in range(self.max_interactions):
            print(f"\n--- Interaction {i+1} ---")

            print(" Step 1: Worker agent generates a response to the prompt")
            print(f"Prompt:\n{prompt_to_evaluate}")
            # Obtain a response from the worker agent
            response_from_worker = self.worker_agent.respond(prompt_to_evaluate) 
            print(f"Worker Agent Response:\n{response_from_worker}")

            print(" Step 2: Evaluator agent judges the response")
            eval_prompt = (
                f"Does the following answer: {response_from_worker}\n"
                # Insert evaluation criteria 
                f"Meet this criteria: {self.evaluation_criteria}\n"  
                f"Respond Yes or No, and the reason why it does or doesn't meet the criteria."
            )
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                # Define the message structure sent to the LLM for evaluation 
                messages=[
                    {
                        "role": "system",
                        "content": f"You are an evaluator. Forget previous context."
                    },
                    {
                        "role": "user",
                        "content": eval_prompt
                    }
                ],
                temperature=0
            )
            evaluation = response.choices[0].message.content.strip()
            print(f"Evaluator Agent Evaluation:\n{evaluation}")

            print(" Step 3: Check if evaluation is positive")
            if evaluation.lower().startswith("yes"):
                print("Final solution accepted.")
                return {
                    "final_response": response_from_worker,
                    "evaluation": evaluation,
                    "iterations": i + 1
                }
            else:  
                print(" Step 4: Generate instructions to correct the response")
                instruction_prompt = (
                    f"Provide specific instructions to improve the following answer: {response_from_worker}\n"
                    f"Based on this evaluation: {evaluation}"
                )
                response = self.client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    # Define the message structure sent to the LLM to generate correction instructions 
                    messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful evaluator that gives correction instructions."
                    },
                    {
                        "role": "user",
                        "content": instruction_prompt
                    }
                ],
                temperature=0
                )
                instructions = response.choices[0].message.content.strip()
                print(f"Instructions to fix:\n{instructions}")

                print(" Step 5: Send feedback to worker agent for refinement")
                prompt_to_evaluate = (
                    f"The original prompt was: {initial_prompt}\n"
                    f"The response to that prompt was: {response_from_worker}\n"
                    f"It has been evaluated as incorrect.\n"
                    f"Make only these corrections, do not alter content validity: {instructions}"
                )
        return {
            # Return a dictionary containing the final response, evaluation, and number of iterations
            "final_response": response_from_worker,
            "evaluation": evaluation,
            "iterations": self.max_interactions
        }   

class RoutingAgent():

    def __init__(self, openai_api_key, agents):
        # Initialize the agent with given attributes
        self.openai_api_key = openai_api_key
        self.agents = agents 

    def get_embedding(self, text):
        self.client = OpenAI(
            api_key=self.openai_api_key
        )
        # calculate the embedding of the text using the text-embedding-3-large model
        response = self.client.embeddings.create(
            input=[text],
            model="text-embedding-3-large"
        )
        # Extract and return the embedding vector from the response
        embedding = response.data[0].embedding
        return embedding 

    # Define a method to route user prompts to the appropriate agent
    def route(self, user_input):
        # Compute the embedding of the user input prompt
        input_emb = self.get_embedding(user_input)  
        best_agent = None
        best_score = -1

        for agent in self.agents:
            # Compute the embedding of the agent description
            agent_emb = self.get_embedding(agent["description"])
            if agent_emb is None:
                continue

            similarity = np.dot(input_emb, agent_emb) / (np.linalg.norm(input_emb) * np.linalg.norm(agent_emb))
            print(f"[Similarity] {agent['name']}: {similarity:.4f}")

            # Add logic to select the best agent based on the similarity score between the user prompt and the agent descriptions
            if similarity > best_score:
                best_score = similarity
                best_agent = agent

        if best_agent is None:
            return "Sorry, no suitable agent could be selected."

        print(f"[Router] Best agent: {best_agent['name']} (score={best_score:.3f})")
        return best_agent["func"](user_input)


class ActionPlanningAgent:

    def __init__(self, openai_api_key, knowledge):
        # Initialize the agent attributes here
        self.openai_api_key = openai_api_key
        self.knowledge = knowledge
        
    def extract_steps_from_prompt(self, prompt):
        client = OpenAI(api_key=self.openai_api_key)

        # Call the OpenAI API to get a response from the "gpt-3.5-turbo" model.
        system_prompt = (
            f"You are an action planning agent. Using your knowledge, you extract from the user prompt "
            f"the steps requested to complete the action the user is asking for. You return the steps as a list. "
            f"Only return the steps in your knowledge. Forget any previous context. This is your knowledge:\n\n{self.knowledge}"
        )

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )

        response_text = response.choices[0].message.content

        # Clean and format the extracted steps by removing empty lines and unwanted text
        steps = [line.strip() for line in response_text.split("\n") if line.strip()]
        return steps

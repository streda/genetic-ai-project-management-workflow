# TODO: 1 - import the OpenAI class from the openai library
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
        # TODO: 1 - Create an attribute for the agent's persona
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
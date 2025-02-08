''' meta-llamaMeta-Llama-3.1-8B '''

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
    BitsAndBytesConfig
)

#############################################
# GENERAL CONFIGURATION
#############################################

# Set the path to your base model directory.
# Make sure this folder contains all the necessary model files (e.g., config.json, tokenizer files, etc.)
BASE_MODEL_PATH = os.path.join(os.getcwd(), "E:\ia\AWS\meta-llamaMeta-Llama-3.1-8B")

# File to store conversation history
CHAT_HISTORY_FILE = os.path.join(os.getcwd(), "chat_history.txt")

# Initial system prompt defining the bot's role
SYSTEM_PROMPT = (
     "You are a helpful assistant named Zetalbot. "
    "Answer all queries concisely and directly, with more details possible, and if it needed, instruct the user to email support at support@support.com. Remember, your name is Zetalbot.\n\n"
)

# Generation parameters
MAX_NEW_TOKENS = 500
TEMPERATURE = 0.7
TOP_P = 0.95

#############################################
# 4-BIT QUANTIZATION CONFIGURATION
#############################################

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                     # Enable 4-bit quantization
    bnb_4bit_compute_dtype=torch.float16,  # Use FP16 for computation
    bnb_4bit_use_double_quant=True         # Improve memory management
)

#############################################
# FUNCTIONS FOR CONVERSATION HISTORY MANAGEMENT
#############################################

def load_history(filename, default_prompt):
    """
    If the file 'filename' exists, return its contents; otherwise, return the default_prompt.
    """
    if os.path.exists(filename):
        with open(filename, 'r', encoding='utf-8') as f:
            return f.read()
    else:
        return default_prompt

def save_history(filename, history):
    """
    Save the conversation history to the file 'filename'.
    """
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(history)

#############################################
# MAIN CHAT SCRIPT
#############################################

def main():
    # Load conversation history from file; if it doesn't exist, use the SYSTEM_PROMPT.
    conversation_history = load_history(CHAT_HISTORY_FILE, SYSTEM_PROMPT)
    print("Loaded conversation history:\n")
    print(conversation_history)
    
    # Load the tokenizer from the base model path.
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token  # Set the pad token
    
    # Load the base model with 4-bit quantization.
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        quantization_config=bnb_config,
        device_map="auto"
    )
    
    # Create a text-generation pipeline.
    chat_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        do_sample=True
    )
    
    print("\n🤖 Chatbot started! Type your message (type 'exit' to quit).\n")
    
    # Continuous chat loop
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ["exit", "quit"]:
            print("Exiting chatbot. Goodbye!")
            break
        
        # Update conversation history by appending the user's message and the prompt for the assistant.
        conversation_history += f"User: {user_input}\nAssistant: "
        
        # Generate the assistant's response using the complete conversation history as prompt.
        output = chat_pipeline(conversation_history)[0]["generated_text"]
        
        # Extract the assistant's reply by removing the prompt portion.
        assistant_response = output[len(conversation_history):].strip()
        if "User:" in assistant_response:
            assistant_response = assistant_response.split("User:")[0].strip()
        
        print(f"\n🤖 Bot: {assistant_response}\n")
        
        # Append the assistant's reply to the conversation history.
        conversation_history += assistant_response + "\n"
        # Save the updated conversation history to file.
        save_history(CHAT_HISTORY_FILE, conversation_history)

if __name__ == "__main__":
    main()


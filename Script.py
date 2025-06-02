#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import json
import shutil
import torch
import threading

from datasets import Dataset
from sklearn.model_selection import train_test_split

# PEFT / LoRA
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel

# Transformers / BitsAndBytes
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    TextIteratorStreamer,
    logging as hf_logging
)

# SentenceTransformer e ChromaDB
import chromadb
from sentence_transformers import SentenceTransformer

#############################################
# GENERAL CONFIGURATION
#############################################

MODEL_MODE = "base"  # "base" for the base model, "trained" for the fine-tuned model with LoRA
BASE_MODEL_PATH = "meta-llama/Llama-3.1-8B-Instruct"
TRAINED_LORA_PATH = "output_lora/mio-lora-checkpoint"
DATASET_FOLDER = "dataset_json"
RAG_FOLDER = "rag_documents"
OUTPUT_LORA_DIR = "output_lora/mio-lora-checkpoint"

# We use a JSON file to store the structured history
HISTORY_BACKUP_FILE = "history_backup.json"

SYSTEM_PROMPT = (
    "You are a helpful assistant named Zetalbot, created by Vioze. "
    "Answer all questions directly and concisely. \n\n"
)

MAX_NEW_TOKENS = 256      # Reserved space for the answer
TEMPERATURE = 0.5
TOP_P = 0.7
MAX_CONTEXT_TOKENS = 8192  # Maximum number of tokens in context

#############################################
# INITIAL CLEANING (CACHE & CHROMADB)
#############################################

cache_dir = os.path.expanduser("~/.cache/torch/sentence_transformers")
if os.path.exists(cache_dir):
    shutil.rmtree(cache_dir)
    print("🧹 Cache of sentence-transformers deleted.")

if os.path.exists("chromadb"):
    shutil.rmtree("chromadb")
    print("🗑️ ChromaDB deleted. Will be recreated automatically.")

#############################################
# PYTORCH AND GPU OPTIMIZATIONS
#############################################

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

#############################################
# QUANTIZATION CONFIGURATION
#############################################

def get_bnb_config():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16 if device == "cuda" else torch.float32,
        bnb_4bit_use_double_quant=True
    )

bnb_config = get_bnb_config()

#############################################
# IMPROVED HISTORY HANDLING (JSON FORMAT)
#############################################

def save_history(filename, messages):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(messages, f)

def load_history(filename):
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    except:
        # If the file does not exist, return the initial history with the system prompt
        return [{"role": "system", "content": SYSTEM_PROMPT.strip()}]

def truncate_messages(messages, tokenizer, max_tokens=MAX_CONTEXT_TOKENS - MAX_NEW_TOKENS):
    total_tokens = 0
    truncated = []
    # Iterate starting from the last (most recent) message
    for msg in reversed(messages):
        msg_tokens = len(tokenizer.encode(msg["content"]))
        # We always keep the system message, truncating only if necessary
        if total_tokens + msg_tokens > max_tokens and msg["role"] != "system":
            continue
        truncated.insert(0, msg)
        total_tokens += msg_tokens
    return truncated

#############################################
# LOADING AND DIVIDING THE DATASET FOR FINE-TUNING
#############################################

def load_json_dataset(folder_path):
    all_records = []
    json_files = glob.glob(os.path.join(folder_path, "*.json"))
    jsonl_files = glob.glob(os.path.join(folder_path, "*.jsonl"))
    all_files = json_files + jsonl_files

    if not all_files:
        print(f"No JSON or JSONL files found in {folder_path}.")
        return None, None

    for filepath in all_files:
        with open(filepath, 'r', encoding='utf-8') as f:
            if filepath.endswith(".jsonl"):
                for line in f:
                    line = line.strip()
                    if line:
                        all_records.append(json.loads(line))
            else:
                content = json.load(f)
                if isinstance(content, list):
                    all_records.extend(content)
                else:
                    all_records.append(content)

    if not all_records:
        print("No records found in JSON/JSONL files.")
        return None, None

    train_data, eval_data = train_test_split(all_records, test_size=0.2, random_state=42)
    return Dataset.from_list(train_data), Dataset.from_list(eval_data)

#############################################
# DATA COLLATOR
#############################################

def data_collator(batch, tokenizer):
    texts = []
    for item in batch:
        instruction = item.get("instruction", "")
        user_input = item.get("input", "")
        response = item.get("response", "")
        prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{user_input}\n\n### Response:\n{response}"
        texts.append(prompt)
    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=1024,
        return_tensors="pt"
    )
    enc["labels"] = enc["input_ids"].clone()
    return enc

#############################################
# FINE-TUNING WITH LoRA
#############################################

def lora_fine_tuning(base_model_path, dataset_folder, output_dir, tokenizer):
    train_dataset, eval_dataset = load_json_dataset(dataset_folder)
    if train_dataset is None or len(train_dataset) == 0:
        print("Dataset empty or not found. Training aborted.")
        return

    print(f"Found {len(train_dataset)} training records and {len(eval_dataset)} validation records.")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        quantization_config=bnb_config,
        device_map="auto"
    )
    model = prepare_model_for_kbit_training(model)

    # Optimized LoRA configuration for Llama 3.1
    lora_config = LoraConfig(
        r=16,
        lora_alpha=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=3,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=5,
        save_steps=50,
        save_total_limit=1,
        evaluation_strategy="epoch",
        eval_accumulation_steps=2,
        disable_tqdm=False,
        remove_unused_columns=False,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=lambda batch: data_collator(batch, tokenizer)
    )

    trainer.train()
    trainer.save_model(output_dir)
    print(f"✅ Fine-tuning completed. LoRA checkpoint saved in: {output_dir}")
    print("Log history during training:")
    for log in trainer.state.log_history:
        print(log)

#############################################
# RAG IMPLEMENTATION WITH CHROMADB
#############################################

def init_chroma_client(path="chromadb"):
    return chromadb.PersistentClient(path=path)

def load_rag_documents(folder_path):
    docs = []
    json_files = glob.glob(os.path.join(folder_path, "*.json"))
    jsonl_files = glob.glob(os.path.join(folder_path, "*.jsonl"))
    all_files = json_files + jsonl_files
    for filepath in all_files:
        with open(filepath, 'r', encoding='utf-8') as f:
            if filepath.endswith(".jsonl"):
                for line in f:
                    line = line.strip()
                    if line:
                        docs.append(json.loads(line))
            else:
                content = json.load(f)
                if isinstance(content, list):
                    docs.extend(content)
                else:
                    docs.append(content)
    return docs

def rag_train_documents(chroma_client, collection_name, docs, embedding_model):
    collection = chroma_client.get_or_create_collection(name=collection_name)
    for doc in docs:
        embedding = embedding_model.encode(doc["content"]).tolist()
        collection.add(
            ids=[doc["id"]],
            documents=[doc["content"]],
            metadatas=[doc.get("metadata", {})],
            embeddings=[embedding]
        )
    print("✅ RAG training completed: Documents indexed in ChromaDB.")

def rag_query(query, chroma_client, collection_name, embedding_model, n_results=1):
    collection = chroma_client.get_or_create_collection(name=collection_name)
    query_vector = embedding_model.encode([query]).tolist()
    results = collection.query(query_embeddings=query_vector, n_results=n_results)
    if results["documents"] and results["documents"][0]:
        return "\n".join(results["documents"][0])
    else:
        return ""

#############################################
# GENERAZIONE DELLA RISPOSTA CON FORMATTAMENTO CORRETTO
#############################################

def generate_answer(model, tokenizer, prompt):
    """
    Generate the response by applying tokenization with truncation and using special tokens to extract the response correctly.
    """
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_CONTEXT_TOKENS - MAX_NEW_TOKENS
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
        )
    
    output_text = tokenizer.decode(
        outputs[0],
        skip_special_tokens=False
    ).split("<|start_header_id|>assistant<|end_header_id|>")[-1]
    assistant_response = output_text.split("<|eot_id|>")[0].strip()
    return assistant_response

#############################################
# MAIN SCRIPT
#############################################

def main():
    # Carica il tokenizer e aggiungi i token speciali necessari
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, use_fast=False)
    tokenizer.pad_token = "<|reserved_special_token_0|>"
    tokenizer.add_special_tokens({"pad_token": "<|reserved_special_token_0|>"})

    print("🟢 Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        quantization_config=bnb_config,
        device_map="auto"
    )
    base_model = prepare_model_for_kbit_training(base_model)

    try:
        base_model = torch.compile(base_model)
        print("✅ Model compiled with torch.compile for further optimization.")
    except Exception as e:
        print("⚠️ torch.compile not available or generated an error. Continue without compile.")

    base_model.eval()

    if MODEL_MODE == "trained":
        print("🔄 Loading LoRA checkpoint...")
        model = PeftModel.from_pretrained(base_model, TRAINED_LORA_PATH)
        if isinstance(model, PeftModel):
            print("✅ Fine-tuned model (LoRA) loaded correctly.")
        else:
            print("⚠️ The model loaded is not a PeftModel. Check your checkpoint.")
    else:
        model = base_model

    chroma_client = init_chroma_client()
    embedding_model = SentenceTransformer("sentence-transformers/msmarco-distilbert-base-v3", device="cuda")
    print(f"🔍 Embedding model loaded on GPU. Dimension: {embedding_model.get_sentence_embedding_dimension()}")

    print("\n🤖 Chatbot is running! Type a message (type 'exit' to quit).")
    print("✏️ Type 'start training' for fine-tuning or 'start rag' to index RAG documents.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ["exit", "quit"]:
            print("👋 Exiting chat.")
            break

        if user_input.lower() in ["start training", "train"]:
            print("⚙️ Starting fine-tuning...")
            lora_fine_tuning(BASE_MODEL_PATH, DATASET_FOLDER, OUTPUT_LORA_DIR, tokenizer)
            print("✅ Fine-tuning completed. Set MODEL_MODE='trained' to use the new checkpoint.")
            continue

        if user_input.lower() in ["start rag", "rag"]:
            print("⚙️ Starting RAG document indexing...")
            docs = load_rag_documents(RAG_FOLDER)
            if docs:
                rag_train_documents(chroma_client, "company_docs", docs, embedding_model)
                print("✅ RAG indexing completed.")
            else:
                print("⚠️ No documents found in the RAG folder.")
            continue

        # Load and truncate history
        messages = load_history(HISTORY_BACKUP_FILE)
        messages = truncate_messages(messages, tokenizer)

        # Add new user message
        messages.append({"role": "user", "content": user_input})

        # Run the RAG query to retrieve the relevant context
        rag_context = rag_query(user_input, chroma_client, "company_docs", embedding_model)

        # Generates the formatted prompt, including the Rag context if available
        if rag_context:
            prompt = f"Context:\n{rag_context}\n\n" + tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )


        # Generate the response
        assistant_response = generate_answer(model, tokenizer, prompt)

        print(f"\n🤖 Bot: {assistant_response}\n")

        # Add the assistant's response to history and save
        messages.append({"role": "assistant", "content": assistant_response})
        save_history(HISTORY_BACKUP_FILE, messages)

if __name__ == "__main__":
    main()

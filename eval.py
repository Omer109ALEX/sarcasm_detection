import my_process_data as mpd
import create_messages as cm
import my_rag as mrag
import pandas as pd
import os
import csv
from tqdm import tqdm
import math
import warnings
from tabulate import tabulate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate, FewShotPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_community.vectorstores import FAISS
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
import asyncio
import nest_asyncio
from dotenv import load_dotenv
import time

# Load variables from .env file
load_dotenv()

# Fetch the API key from environment variables
HF_TOKEN = os.getenv("HF_TOKEN")
gilat_key = os.getenv("GROQ_API_KEY_GILAT")
my_key = os.getenv("GROQ_API_KEY_MY")
rotem_key = os.getenv("GROQ_API_KEY_ROTEM")

# Initialize embeddings
embedding_name = "sentence-transformers/all-roberta-large-v1" # Description: Based on RoBERTa-large, this model has been fine-tuned for various semantic similarity tasks. 
embedding = HuggingFaceEmbeddings(model_name=embedding_name) 

models = ["llama-3.1-70b-versatile", "llama-3.1-8b-instant", "llama3-70b-8192"]
keys = [my_key, rotem_key, gilat_key]
score_threshold = 0.0
k=3


def create_llm(model_name, key):
    # Initialize LLM
    llm = ChatGroq(model=model_name, temperature=0.6 ,groq_api_key=key, model_kwargs={
        "top_p" : 0.7,
        "seed" : 109,
        "response_format" : {"type": "json_object"},
        })
    
    return llm

async def eval_zero_shot(file_path, label_name, key):
    any_empty = mpd.check_empty_cells(file_path, label_name)
    model_index = 0

    while any_empty:
        model_name = models[model_index] 
        llm = create_llm(model_name, key)
        print(f'Starting with model {model_name} for {file_path}\n\n ++++++++++++++++++++++++++++++++++\n\n')
        try:
            await asyncio.to_thread(mrag.ask_llm_from_csv_zero_shot, file_path, llm, label_name, wanted_speed=5)
            any_empty = mpd.check_empty_cells(file_path, label_name)  # Check for remaining empty rows
            if not any_empty:
                print("All rows have been processed.")
                break  # Exit the function when all rows are processed
        except Exception as e:
            model_index = model_index + 1
            print(f"An unexpected error occurred: {e}")

async def to_run(files, labels, keys):
    try:
        tasks = [eval_zero_shot(file_path, label_name, key) for file_path, label_name, key in zip(files, labels, keys)]
        await asyncio.gather(*tasks)
    except Exception as e:
        print(f"An error occurred during processing: {e}")

# Define your file paths and labels
files = ['./data/all/data_all.csv', './data/all/data_all_70.csv', './data/all/data_all_100.csv']
labels = ["zero_shot", "zero_shot", "zero_shot"]

# Run the asyncio tasks
asyncio.run(to_run(files, labels, keys))

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
from langchain_nomic.embeddings import NomicEmbeddings
from langchain_community.embeddings import OllamaEmbeddings, HuggingFaceEmbeddings, HuggingFaceInferenceAPIEmbeddings
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



def eval_zero_shot(file_path, label_name, key):
    empty100 = mpd.get_empty_first_rows(file_path, 100, label_name)
    model_index = 0

    while empty100:
        model_name = models[model_index] 
        llm = create_llm(model_name, key)
        try:
            mrag.ask_llm_from_csv_zero_shot(file_path, llm, label_name)
            empty100 = mpd.get_empty_first_rows(file_path, 100, label_name)  # Check for remaining empty rows
            if not empty100:
                print("All rows have been processed.")
                return  # Exit the function when all rows are processed
            
        except Exception as e:
            if hasattr(e, 'response') and e.response.status_code == 503:
                print("Server error (503) encountered. Retrying after 5 seconds...")
                time.sleep(5)
            elif hasattr(e, 'response') and e.response.status_code == 429:
                if "TPM" or "RPM" in str(e):
                    print("Minute limit reached. Retrying with the same model after 60 seconds...")
                    time.sleep(60)
                    continue  # Retry the same model after waiting
                else:
                    print("Daily limit reached. Trying the next model...")
                    model_index = (model_index + 1) % len(models)
                    continue  # move to the next model
            else:
                print(f"An unexpected error occurred: {e}")
                return  # Exit the function on unexpected errors
                


#to run
print("start")
file_path = f'./data/all/data_all.csv'
label_name = f"zero_shot"
eval_zero_shot(file_path, label_name, my_key)


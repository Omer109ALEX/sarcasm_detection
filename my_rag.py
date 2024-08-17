import my_process_data as mpd
import create_messages as cm
import os
import csv
import pandas as pd
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
import asyncio
from tqdm.asyncio import tqdm as tqdm_async
import httpx
import time
from langchain_community.callbacks import UpstashRatelimitError, UpstashRatelimitHandler
from upstash_ratelimit import FixedWindow, Ratelimit
from upstash_redis import Redis


# Suppress specific warnings
warnings.filterwarnings("ignore", message="Relevance scores must be between 0 and 1")
warnings.filterwarnings("ignore", message="No relevant docs were retrieved using the relevance score threshold")
warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)


wait_time = 61

os.environ["UPSTASH_REDIS_REST_URL"] = "https://gusc1-infinite-lab-30700.upstash.io"
os.environ["UPSTASH_REDIS_REST_TOKEN"] = "AXfsASQgMWYxNjRhNTQtNDIzMS00YWFiLWE2ZjYtZTE0ODc1OTE5YTBkODlkMTFhZDI5YzdhNDMyOWFmYzc2YmRlMGFjNjQyZTc="


def save_faiss_no_split(embedding, embedding_name, input_path):
    
    #for each dataset
    for dataset in tqdm(mpd.datasets, desc='Processing'):
        file_path = f'{input_path}/{dataset}/data.csv'
        data = load_doc_from_csv(file_path=file_path)
        vectorstore = FAISS.from_documents(documents=data, embedding=embedding)
        to_save_path = f'./faiss/data/{embedding_name}/{dataset}/'
        vectorstore.save_local(to_save_path)
        


def save_vectorstore_faiss(embedding, embedding_name, input_path, filtered=False):
    if filtered:
        #for each dataset
        for dataset in tqdm(mpd.datasets, desc='Processing'):
            file_path = f'{input_path}/{dataset}/train_data_filtered.csv'
            data = load_doc_from_csv(file_path=file_path)
            vectorstore = FAISS.from_documents(documents=data, embedding=embedding)
            to_save_path = f'./faiss/{embedding_name}/{dataset}/filtered/'
            vectorstore.save_local(to_save_path)
        
        # for train_data_all
        file_path = f'{input_path}/all/train_data_filtered_all.csv'
        data = load_doc_from_csv(file_path=file_path)
        vectorstore = FAISS.from_documents(documents=data, embedding=embedding)
        to_save_path = f'./faiss/{embedding_name}/filtered/'
        vectorstore.save_local(to_save_path)
        
    else:
        #for each dataset
        for dataset in tqdm(mpd.datasets, desc='Processing'):
            file_path = f'{input_path}/{dataset}/train_data.csv'
            data = load_doc_from_csv(file_path=file_path)
            vectorstore = FAISS.from_documents(documents=data, embedding=embedding)
            to_save_path = f'./faiss/{embedding_name}/{dataset}'
            vectorstore.save_local(to_save_path)
        
        # for train_data_all
        file_path = f'{input_path}/all/train_data_all.csv'
        data = load_doc_from_csv(file_path=file_path)
        vectorstore = FAISS.from_documents(documents=data, embedding=embedding)
        to_save_path = f'./faiss/{embedding_name}/all'
        vectorstore.save_local(to_save_path)


def load_doc_from_csv(file_path):
    loader = CSVLoader(file_path=file_path, encoding='latin1')
    data = loader.load()
    for doc in data:
        # clean label, text, dataset data
        split_doc = doc.page_content.split('label: ')
        split_doc = split_doc[1].split('\ntext: ')
        label = split_doc[0]
        split_doc = split_doc[1].split('\ndataset: ')
        text = split_doc[0]
        dataset = split_doc[1]

        # store text as page_content and label and dataset as metadata
        doc.page_content = text
        doc.metadata["label"] = label
        doc.metadata["dataset"] = dataset

    return data


def get_retriever_similarity_score_threshold(index_path, embedding, score_threshold, k=5):
    vectorstore = FAISS.load_local(index_path, embeddings=embedding, allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever(search_type="similarity_score_threshold",
                                        search_kwargs={
                                            "score_threshold": score_threshold,
                                            "k": k,
                                        })
    return retriever


def ask_llm_with_rag(retriever, llm, sentence, print_prompt=False, with_dataset=False):

    # Define your desired data structure.
    class Sarcasm(BaseModel):
        prediction: str = Field(description="1 if the input sentence is sarcastic or 0 if not")
        explain: str = Field(description="explain your determine, explain if you used the information from the labeld similar sentences only if exists")

        
    # Set up a parser + inject instructions into the prompt template.
    parser = JsonOutputParser(pydantic_object=Sarcasm)

    rag_prompt  = PromptTemplate(
        template = cm.rag_template,           
        input_variables=["sentence", "context"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    
    def format_docs(docs, with_dataset=False):
        formatted_docs = []
        for doc in docs:
            text = doc.page_content
            label = doc.metadata["label"]
            dataset = doc.metadata.get("dataset", None)

            # Create a dictionary for each document
            if with_dataset and dataset is not None:
                formatted_doc = {
                    "sentence": text,
                    "prediction": label,
                    "dataset": dataset
                }
            else:
                formatted_doc = {
                    "sentence": text,
                    "prediction": label
                }

            # Convert dictionary to string and append to the list
            formatted_docs.append(str(formatted_doc))
        
        # Join the string representations with two empty lines in between
        return "\n".join(formatted_docs)


   
    rag_chain = (
    {"context": retriever | format_docs, "sentence": RunnablePassthrough()}
    | rag_prompt
    | llm
    | parser
    )

    response = rag_chain.invoke(sentence)
    docs = retriever.invoke(sentence)
    
    if print_prompt:
        context_prompt = format_docs(docs)
        formatted_prompt = rag_prompt.format(context=context_prompt, sentence=sentence)
        print(formatted_prompt)
        
    return response, docs


def ask_llm_with_rag_result_only(retriever, llm, sentence, print_prompt=False, with_dataset=False, handler=None):

    # Define your desired data structure.
    class Sarcasm(BaseModel):
        prediction: str = Field(description="1 if the input sentence is sarcastic or 0 if not")
        #explain: str = Field(description="explain your determine, explain if you used the information from the labeld similar sentences only if exists")

        
    # Set up a parser + inject instructions into the prompt template.
    parser = JsonOutputParser(pydantic_object=Sarcasm)

    rag_prompt  = PromptTemplate(
        template = cm.rag_template,           
        input_variables=["sentence", "context"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    
    def format_docs(docs, with_dataset=False):
        formatted_docs = []
        for doc in docs:
            text = doc.page_content
            label = doc.metadata["label"]
            dataset = doc.metadata.get("dataset", None)

            # Create a dictionary for each document
            if with_dataset and dataset is not None:
                formatted_doc = {
                    "sentence": text,
                    "prediction": label,
                    "dataset": dataset
                }
            else:
                formatted_doc = {
                    "sentence": text,
                    "prediction": label
                }

            # Convert dictionary to string and append to the list
            formatted_docs.append(str(formatted_doc))
        
        # Join the string representations with two empty lines in between
        return "\n".join(formatted_docs)

   
    rag_chain = (
    {"context": retriever | format_docs, "sentence": RunnablePassthrough()}
    | rag_prompt
    | llm
    | parser
    )
    
    if handler:
        response = rag_chain.invoke(sentence, config={"callbacks": [handler]})

    else:
        response = rag_chain.invoke(sentence)
    
    
    docs = retriever.invoke(sentence)
        

    
    if print_prompt:
        context_prompt = format_docs(docs)
        formatted_prompt = rag_prompt.format(context=context_prompt, sentence=sentence)
        print(formatted_prompt)
        
    return response


def ask_llm_from_csv_similarity_score_threshold(file_path, embedding, llm, label_name, model_name, score_threshold, all=False, filtered=False) :
    context_label = f'context_{label_name}'
    
    # Read the existing CSV file
    df = pd.read_csv(file_path)
    
    # Check if the new column header already exists
    if label_name not in df.columns:
        # Add the new column with the header and initialize with empty strings
        df[label_name] = ''
        df[context_label] = ''
        # Save the updated DataFrame back to the CSV file
        df.to_csv(file_path, index=False)
        print(f"New column '{label_name}' added successfully.")
    
    # Read the column contents from the CSV file    
    text_list = mpd.get_column_content(file_path, 'text')
    dataset_list = mpd.get_column_content(file_path, 'dataset')
    label_list = mpd.get_column_content(file_path, label_name)
    context_list = mpd.get_column_content(file_path, context_label)

    # Set initial index to start processing
    need_to_calculate_index = len(text_list)
    
    # Find the first index with a NaN value in the label list
    for index, value in enumerate(label_list):
        if math.isnan(value):
            need_to_calculate_index = index
            break
    
    # Process each row starting from the first NaN value found
    try:
        with tqdm(range(need_to_calculate_index, len(text_list)), desc='Processing') as progress_bar:
            for row in progress_bar:
                try:
                    if all:
                        dataset = "all"
                    else:
                        dataset = dataset_list[row]
                        
                    if filtered:
                        index_path = f"./faiss/{model_name}/{dataset}/filtered/"
                    else:
                        index_path = f"./faiss/{model_name}/{dataset}"                        
                    retriever = get_retriever_similarity_score_threshold(index_path, embedding, score_threshold)
                    resp, docs = ask_llm_with_rag(retriever, llm, str(text_list[row]))
                    label_list[row] = resp['prediction']
                    context = [doc.metadata for doc in docs]
                    context_list[row] = context

                    if row%100==0:
                        mpd.add_column_to_csv(file_path, label_name, label_list)
                        mpd.add_column_to_csv(file_path, context_label, context_list)
                        
                except Exception as e:
                    print(f"An error occurred: {e}")
        
        # Save the updated label list back to the CSV file
        mpd.add_column_to_csv(file_path, label_name, label_list)
        mpd.add_column_to_csv(file_path, context_label, context_list)
    except Exception as e:
        print(f"An error occurred during processing: {e}")


def ask_llm_from_csv_similarity_score_threshold_result_only(file_path, embedding, llm, label_name, model_name, score_threshold, all=False, filtered=False) :
    
    # Read the existing CSV file
    df = pd.read_csv(file_path)
    
    # Check if the new column header already exists
    if label_name not in df.columns:
        # Add the new column with the header and initialize with empty strings
        df[label_name] = ''
        # Save the updated DataFrame back to the CSV file
        df.to_csv(file_path, index=False)
        print(f"New column '{label_name}' added successfully.")
    
    # Read the column contents from the CSV file    
    text_list = mpd.get_column_content(file_path, 'text')
    dataset_list = mpd.get_column_content(file_path, 'dataset')
    label_list = mpd.get_column_content(file_path, label_name)

    # Set initial index to start processing
    need_to_calculate_index = len(text_list)
    
    # Find the first index with a NaN value in the label list
    for index, value in enumerate(label_list):
        if math.isnan(value):
            need_to_calculate_index = index
            break
    
    # Process each row starting from the first NaN value found
    try:
        with tqdm(range(need_to_calculate_index, len(text_list)), desc='Processing') as progress_bar:
            for row in progress_bar:
                try:
                    if all:
                        dataset = "all"
                    else:
                        dataset = dataset_list[row]
                        
                    if filtered:
                        index_path = f"./faiss/{model_name}/{dataset}/filtered/"
                    else:
                        index_path = f"./faiss/{model_name}/{dataset}"                        
                    retriever = get_retriever_similarity_score_threshold(index_path, embedding, score_threshold)
                    resp = ask_llm_with_rag_result_only(retriever, llm, str(text_list[row]))
                    label_list[row] = resp['prediction']


                    if row%100==0:
                        mpd.add_column_to_csv(file_path, label_name, label_list)
                        
                except Exception as e:
                    print(f"An error occurred: {e}")
        
        # Save the updated label list back to the CSV file
        mpd.add_column_to_csv(file_path, label_name, label_list)
    except Exception as e:
        print(f"An error occurred during processing: {e}")


def ask_llm_from_csv_zero_shot(file_path, llm, label_name) :
    
    # Read the existing CSV file
    df = pd.read_csv(file_path)
    
    # Check if the new column header already exists
    if label_name not in df.columns:
        # Add the new column with the header and initialize with empty strings
        df[label_name] = ''
        # Save the updated DataFrame back to the CSV file
        df.to_csv(file_path, index=False)
        print(f"New column '{label_name}' added successfully.")
    
    # Read the column contents from the CSV file    
    text_list = mpd.get_column_content(file_path, 'text')
    dataset_list = mpd.get_column_content(file_path, 'dataset')
    label_list = mpd.get_column_content(file_path, label_name)
        
    # Define your desired data structure.
    class Sarcasm(BaseModel):
        prediction: str = Field(description="1 if the input sentence is sarcastic or 0 if not")
        #explain: str = Field(description="explain your determine, explain if you used the information from the labeld similar sentences only if exists")

        
    # Set up a parser + inject instructions into the prompt template.
    parser = JsonOutputParser(pydantic_object=Sarcasm)

    prompt = PromptTemplate(
        template=cm.zero_shot_template,
        input_variables=["sentence"],
        partial_variables={"format_instructions": cm.result_only_format},
    )
    
    # Process each row starting from the first NaN value found
    try:
        with tqdm(range(len(text_list)), desc='Processing') as progress_bar:
            for row in progress_bar:
                try:
                    if pd.isna(label_list[row]):
                        sentence = str(text_list[row])
                        chain = (prompt| llm | parser)
                        resp = chain.invoke(sentence)
                        label_list[row] = resp['prediction']
                        
                        if row%100==0:
                            mpd.add_column_to_csv(file_path, label_name, label_list)
                        
                except Exception as e:
                    print(f"An unexpected error occurred while processing row {row}: {e}")
        
        # Save the updated label list back to the CSV file
        mpd.add_column_to_csv(file_path, label_name, label_list)
    except Exception as e:
        print(f"An error occurred during processing: {e}")


"""
ID	Requests per Minute	Requests per Day	Tokens per Minute	Tokens per Day
gemma-7b-it	                            30 	14,400	15,000	(No limit)
gemma2-9b-it	                        30	14,400	15,000	(No limit)
llama-3.1-70b-versatile	               100	14,400	131,072	1,000,000       # 25 min day batch=100 
llama-3.1-8b-instant                 	30	14,400	131,072	1,000,000       # 8 hours (2,500 req per day)
llama-guard-3-8b	                    30	14,400	15,000	(No limit)
llama3-70b-8192	                        30	14,400	6,000	(No limit)
llama3-8b-8192	                        30	14,400	30,000	(No limit)
llama3-groq-70b-8192-tool-use-preview	30	14,400	15,000	(No limit)
llama3-groq-8b-8192-tool-use-preview	30	14,400	15,000	(No limit)
mixtral-8x7b-32768                  	30	14,400	5,000	(No limit)
"""

async def process_row(llm, sentence, row, label_list, prompt, parser):
    try:
        chain = (prompt | llm | parser)
        resp = await chain.ainvoke(sentence)
        label_list[row] = resp['prediction']
    except Exception as e:
        if e.response.status_code == 503:
            retry_after = int(5)
            await asyncio.sleep(retry_after)  # Asynchronous sleep
        else:
            print(f"An unexpected error occurred while processing row {row}: {e}")

        
async def ask_llm_from_csv_zero_shot_async(file_path, llm, label_name, batch_size=30):
    # Read the existing CSV file
    df = pd.read_csv(file_path)
    
    # Check if the new column header already exists
    if label_name not in df.columns:
        # Add the new column with the header and initialize with empty strings
        df[label_name] = ''
        # Save the updated DataFrame back to the CSV file
        df.to_csv(file_path, index=False)
        print(f"New column '{label_name}' added successfully.")
    
    # Read the column contents from the CSV file    
    text_list = mpd.get_column_content(file_path, 'text')
    label_list = mpd.get_column_content(file_path, label_name)

    # Define your desired data structure.
    class Sarcasm(BaseModel):
        prediction: str = Field(description="1 if sarcastic or 0 if not")

    # Set up a parser + inject instructions into the prompt template.
    parser = JsonOutputParser(pydantic_object=Sarcasm)
    
    prompt = PromptTemplate(
        template=cm.zero_shot_template,
        input_variables=["sentence"],
        partial_variables={"format_instructions": cm.result_only_format},
    )

    
    tasks = []
    start_time = time.time()

    # Process each row with a NaN value in the label list
    try:
        async for row in tqdm_async(range(len(text_list)), desc='Processing'):
            if pd.isna(label_list[row]):
                sentence = str(text_list[row])
                tasks.append(process_row(llm, sentence, row, label_list, prompt, parser))
                
                if len(tasks) >= batch_size:  # Control concurrency with batch size
                    await asyncio.gather(*tasks)
                    tasks.clear()
                    mpd.add_column_to_csv(file_path, label_name, label_list)
                    
                    elapsed_time = time.time() - start_time
                    if elapsed_time < wait_time:
                        await asyncio.sleep(wait_time - elapsed_time)  # Ensure wait_time seconds have passed
                    start_time = time.time()  # Reset start time after sleep
        
        # Process remaining tasks
        if tasks:
            await asyncio.gather(*tasks)
        
        # Save the updated label list back to the CSV file
        mpd.add_column_to_csv(file_path, label_name, label_list)
    except Exception as e:
        print(f"An error occurred during processing: {e}")


async def process_similarity_row(llm, retriever, sentence, row, label_list):
    retry_attempts = 3  # Define the number of retry attempts
    for attempt in range(retry_attempts):        
        try:
            resp = await asyncio.to_thread(ask_llm_with_rag_result_only, retriever, llm, sentence)
            label_list[row] = resp['prediction']
            break  # Exit loop if successful
        except Exception as e:
            if e.response.status_code == 429:
                retry_after = int(e.response.headers.get("Retry-After", 5))
                print(f"An error occurred while processing row {row}: {e}")
                print(f"Rate limit hit. Retrying after {retry_after} seconds.")
                await asyncio.sleep(retry_after + 1)  # Asynchronous sleep
            else:
                print(f"An error occurred while processing row {row}: {e}")
                break  # Exit if it's a different kind of error

  
async def process_similarity_row_with_tokens(llm, retriever, sentence, row, label_list, handler):
    try:
        resp = await asyncio.to_thread(ask_llm_with_rag_result_only(retriever, llm, sentence, handler=handler))
        label_list[row] = resp['prediction']
    except UpstashRatelimitError as e:
        await asyncio.sleep(60)  # Wait before retrying
        await process_similarity_row_with_tokens(llm, retriever, sentence, row, label_list, handler)
    except Exception as e:
        print(f"An error occurred while processing row {row}: {e}")       
            

async def ask_llm_from_csv_similarity_score_threshold_result_only_async(file_path, embedding, llm, label_name, model_name, score_threshold, batch_size=30, all=False, filtered=False):
    # Read the existing CSV file
    df = pd.read_csv(file_path)
    
    # Check if the new column header already exists
    if label_name not in df.columns:
        # Add the new column with the header and initialize with empty strings
        df[label_name] = ''
        # Save the updated DataFrame back to the CSV file
        df.to_csv(file_path, index=False)
        print(f"New column '{label_name}' added successfully.")
    
    # Read the column contents from the CSV file    
    text_list = mpd.get_column_content(file_path, 'text')
    dataset_list = mpd.get_column_content(file_path, 'dataset')
    label_list = mpd.get_column_content(file_path, label_name)

    tasks = []
    start_time = time.time()
    
    # Initialize Redis and Rate Limiters
    redis = Redis.from_env()
    request_ratelimit = Ratelimit(
        redis=redis,
        limiter=FixedWindow(max_requests=30, window=60)  # 30 requests per 60 seconds
    )

    # Define rate limit for 6000 tokens per minute (input + output tokens)
    token_ratelimit = Ratelimit(
        redis=redis,
        limiter=FixedWindow(max_requests=6000, window=60)  # 6000 tokens per 60 seconds
    )

    # Create the handler with both request and token limits
    handler = UpstashRatelimitHandler(
        identifier="user_id",  # Customize based on your application's user identifier logic
        request_ratelimit=request_ratelimit,
        token_ratelimit=token_ratelimit,
        include_output_tokens=True  # Include response tokens in the token count
    )

    # Process each row with a NaN value in the label list
    try:
        async for row in tqdm_async(range(len(text_list)), desc='Processing'):
            if pd.isna(label_list[row]):
                sentence = str(text_list[row])
                
                if all:
                    dataset = "all"
                else:
                    dataset = dataset_list[row]
                    
                if filtered:
                    index_path = f"./faiss/{model_name}/{dataset}/filtered/"
                else:
                    index_path = f"./faiss/{model_name}/{dataset}"
                    
                retriever = get_retriever_similarity_score_threshold(index_path, embedding, score_threshold)
                start_time = time.time()
                tasks.append(process_similarity_row_with_tokens(llm, retriever, sentence, row, label_list, handler))
                
                if len(tasks) >= batch_size:  # Control concurrency with batch size
                    await asyncio.gather(*tasks)
                    tasks.clear()
                    mpd.add_column_to_csv(file_path, label_name, label_list)
                    
                    elapsed_time = time.time() - start_time
                    if elapsed_time < wait_time:
                        await asyncio.sleep(wait_time - elapsed_time)  # Ensure wait_time seconds have passed
        
        # Process remaining tasks
        if tasks:
            await asyncio.gather(*tasks)
        
        # Save the updated label list back to the CSV file
        mpd.add_column_to_csv(file_path, label_name, label_list)
    except Exception as e:
        print(f"An error occurred during processing: {e}")
        
   
        
        
        
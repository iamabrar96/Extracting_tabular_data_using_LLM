# **************************************Importing neccessary libraries ***************************************************
import os
import re
import ast
import csv
import json
import pickle
import tiktoken
import pytesseract
import pandas as pd
from PIL import Image
from tqdm import tqdm
from PyPDF2 import PdfReader
from PyPDF2 import PdfReader
from transformers import pipeline
from langchain.llms import OpenAI
from pdf2image import convert_from_path
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import CharacterTextSplitter
from flask import Flask, request, render_template, redirect
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import (
    VectorDBQA,
    RetrievalQA,
    ConversationalRetrievalChain,
)

# ****************************************************************************************************************#


def preprocess_texts(raw_text):
    """
    @param raw_text: the concatinated text to be processed
    @return texts: the splitted and tokenized text
    """
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1024,
        chunk_overlap=200,
        length_function=len,
    )
    texts = text_splitter.split_text(raw_text)
    return texts


def read_pdf_text(path, preprocess_langchain=False):
    """
    @param path: the pdf object path
    @param preprocess_langchain: preprocessing flag from langchain
    @return texts: all the text from the pdf concatinated
    """
    reader = PdfReader(path)
    raw_text = ""

    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            raw_text += text

    if preprocess_langchain:
        texts = preprocess_texts(raw_text)
    else:
        texts = raw_text
    return texts


def process_single_pdf(file_path, preprocess_langchain=False):
    """
    @param file_path: path to the PDF file
    @param preprocess_langchain: if the preprocess for langchain to optimize token in chunks should be done
    @return: text from the PDF
    """
    return read_pdf_text(file_path, preprocess_langchain)

#********************************Total cost evaluation of each pdf file and storing them in the csv file ***********************************************

def count_tokens(model_name, text_list):
    # Mapping custom model names to recognized model names
    model_mapping = {
        "gpt-3.5_turbo-16k_with_fine_tuning": "gpt-3.5-turbo-16k",
        "gpt-3.5_turbo-16k_without_fine_tuning": "gpt-3.5-turbo-16k",
        "gpt-4o_without_fine_tuning": "gpt-4o" 
    }
    recognized_model_name = model_mapping.get(model_name, model_name)
    
    encoder = tiktoken.encoding_for_model(recognized_model_name)
    total_tokens = sum(len(encoder.encode(text)) for text in text_list)
    return total_tokens

def calculate_cost(input_tokens, output_tokens, prompt_tokens, cost_per_million_tokens):
    input_cost = (input_tokens / 1_000_000) * cost_per_million_tokens
    output_cost = (output_tokens / 1_000_000) * cost_per_million_tokens
    prompt_cost = (prompt_tokens / 1_000_000) * cost_per_million_tokens
    return input_cost + output_cost + prompt_cost

# Configuration for different models
model_config = {
    "gpt-3.5_turbo-16k_with_fine_tuning": {
        "prompt_text": ["Your task is to identify the following items from the context: ..."],
        "cost_per_million_tokens": 0.5
    },
    "gpt-3.5_turbo-16k_without_fine_tuning": {
        "prompt_text": ["Your task is to identify the tables from the context and extract the contents from it. Format your response as a json object."],
        "cost_per_million_tokens": 0.5
    },
    "gpt-4o_without_fine_tuning": {
        "prompt_text": ["Your task is to identify the tables from the context and extract the contents from it. Please also extract the Main Title/Heading of the entire context as well as the DOI. Format your response as a json object with Main Title and DOI as keys:"],
        "cost_per_million_tokens": 5
    }
}

# Select the model configuration
specific_llm_model_name = "gpt-3.5_turbo-16k_without_fine_tuning"  


# Define CSV file name
csv_file_name = "Token_cost_evaluation.csv"

# Write the header row outside the loop
header = ["PDF File Name", "Chunk Size", "LLM Model Name", "Input Token Count", "Prompt Token Count", "Output Token Count", "Total Cost"]
with open(csv_file_name, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(header)

# ********************* open api key **********************
os.environ["OPENAI_API_KEY"] = "************"

# ************** embedding model and large langauge model name *****************************
embedding_model_name = "text-embedding-ada-002"
llm_model_name = "gpt-3.5-turbo-16k"

# initialize the embeddings using openAI ada text embedding library and the llm model using gpt-3.5-turbo-16k
embeddings = OpenAIEmbeddings(model=embedding_model_name)
llm = OpenAI(temperature=0, model_name=llm_model_name)

# Directory containing the PDF files
pdf_directory = "/home/abrar/LLM_document_search/Research_Papers/"

# Get a list of all PDF files in the directory
pdf_paths = [
    os.path.join(pdf_directory, filename)
    for filename in os.listdir(pdf_directory)
    if filename.endswith(".pdf")
]
print(len(pdf_paths))

# Directory containing tables data of each paper in txt format
Txt_directory = f"/home/abrar/LLM_document_search/Tables_data_{llm_model_name}/"

# Initialize tqdm to measure progress
progress_bar = tqdm(total=len(pdf_paths), desc="Processing PDFs")

# Loop through each PDF file extract the tabular data, calculate the total cost based on the token count and model used
for pdf_path in pdf_paths:
    # Extract the last part of the PDF path without the extension
    pdf_file_name = os.path.splitext(os.path.basename(pdf_path))[0]

    # Generate the txt file name based on the last part of the PDF path
    Txt_file_name = f"{pdf_file_name}_{llm_model_name}.txt"
    save_path_table = f"/home/abrar/LLM_document_search/Tables_data_{llm_model_name}"
    output_file_path = f"{save_path_table}/{Txt_file_name}"

    # Check if the corresponding txt file exists
    if os.path.exists(output_file_path):
        # txt file already exists, print confirmation message
        print(f"Txt file containing the extracted tabular data for pdf {pdf_file_name} already exists. Skipping processing.")
        progress_bar.update(1)
        continue  # Move to the next PDF file
    else:
        # txt file does not exist, continue with processing
        print(f"Currently Processing {pdf_file_name}")
        texts = process_single_pdf(pdf_path, preprocess_langchain=True)

        if len(texts) <= 1:

            def extract_text_with_ocr(pdf_path):
                """
                @param pdf_path: path to the PDF file
                @return: text extracted using OCR
                """
                # For each image page, use OCR to extract text
                text = ""
                images = convert_pdf_to_images(pdf_path)
                for image in images:
                    text += pytesseract.image_to_string(image)

                return text

            # Implement a function to convert PDF to images using pdf2image or other tools
            def convert_pdf_to_images(pdf_path):
                images = convert_from_path(pdf_path)
                return images

            texts = extract_text_with_ocr(pdf_path)

            def preprocess_texts(texts):
                """
                @param raw_text: the concatinated text to be processed
                @return texts: the splitted and tokenized text
                """
                text_splitter = CharacterTextSplitter(
                    separator="\n",
                    chunk_size=1024,
                    chunk_overlap=200,
                    length_function=len,
                )
                texts = text_splitter.split_text(texts)
                return texts

            texts = preprocess_texts(texts)
            print(
                f"total number of text chunk static after converting the image to text {len(texts)}"
            )


        # ************************** Removing reference section by emphasizing the case sensitivity ********

        full_text = " ".join(texts)

        # Keywords to search for
        reference_keywords = ["REFERENCES", "Reference"]

        # Find the start index of the reference section
        start_index = -1
        for keyword in reference_keywords:
            if keyword[0] == "R":  # Check if the keyword starts with capital letter 'R'
                start_index = full_text.find(keyword)
                if start_index != -1:
                    break

        # Set the end index to the length of the full text
        end_index = len(full_text)

        # Extract the text between the start and end indices
        extracted_text = full_text[start_index:end_index]

        updated_full_text = full_text.replace(extracted_text, "")

        # max_token_limit is the maximum allowed by the GPT-3.5-turbo-16k model
        max_token_limit = 16385

        # max_token_limit is the maximum allowed by the gpt-4o model
        # max_token_limit = 128000

        # Calculate average_token_length based on your text
        average_token_length = len(updated_full_text.split()) / len(updated_full_text)

        max_tokens_per_chunk = int(
            max_token_limit * 0.90
        )  # Use 90% of the limit to leave room for response tokens
        # Calculate adjusted max_chunk_size
        max_chunk_size = min(
            max_tokens_per_chunk, int(max_tokens_per_chunk / average_token_length)
        )
        # Use a fraction of the max_chunk_size for max_chunk_overlap (adjust as needed)
        max_chunk_overlap = int(max_chunk_size * 0.37)  # 37% overlap, for example

        def preprocess_texts_dynamic(raw_text, max_chunk_size, max_chunk_overlap):
            """
            @param raw_text: the concatenated text to be processed
            @param chunk_size: size of each text chunk
            @param chunk_overlap: overlap between adjacent chunks
            @return texts: the splitted and tokenized text
            """
            texts = [
                raw_text[i : i + max_chunk_size]
                for i in range(0, len(raw_text), max_chunk_size - max_chunk_overlap)
            ]
            return texts

        cropped_texts_2 = preprocess_texts_dynamic(
            updated_full_text, max_chunk_size, max_chunk_overlap
        )

        print(f"Total number of text chunks dynamic is  {len(cropped_texts_2)}")

        # ***************************** Title Extraction ******************************************************
        title = texts[0][0:175]
        title = [title.replace("\n", "")]
        # *************************** DOI Extraction ***********************************************************

        def extract_doi(texts):
            dois = []
            for text_chunk in texts:
                # Define a regular expression pattern for matching DOIs
                doi_pattern = r"\b10\.\d{4,}/[-._;()/:a-zA-Z0-9]+\b"

                # Search for DOIs in the text chunk
                dois_in_chunk = re.findall(doi_pattern, text_chunk)

                # Add the found DOIs to the overall list
                dois.extend(dois_in_chunk)

            if dois:
                return dois
            else:
                return "DOI not found"

        dois = extract_doi(texts)[0]

        # ****************************************************************************************************************************
        # **************************************************************************************************************************
        # ***************************************************************************************************************************

        def format_sample(sample):
            formatted_text = ""

            # Split the sample into lines
            lines = sample.split("\n")

            # Iterate through lines in the sample
            for line in lines:
                # Remove unnecessary characters
                cleaned_line = line.replace("ﬂ", "fl").replace("¢", "")

                # Add a newline if the line is not empty
                if cleaned_line.strip():
                    formatted_text += cleaned_line + "\n"

            return formatted_text

        # List to store results for each modified chunk
        results_list = []# Loop through all chunks in cropped_texts_2
        for i, item in enumerate(cropped_texts_2):
            # Dynamically create variable names (Modified_0, Modified_1, etc.)
            modified_variable_name = "Chunck_{}".format(i)
            globals()[modified_variable_name] = item

            # Clean the text
            cleaned_text = format_sample(globals()[modified_variable_name])

            # Convert cleaned text to a list for FAISS
            formatted_sample = [cleaned_text]

            # Create a FAISS document store
            docsearch = FAISS.from_texts(formatted_sample, embeddings)
            retriever = docsearch.as_retriever(
                search_type="similarity", search_kwargs={"k": 2}
            )

    #************************prompt template without fine tuned for gpt-3.5-turbo-16k ***********************
        
            prompt_template = """Your task is to identify the tables from the context and extract the contents from it . Format your response as a json object:


                        Context: {context} User: {question} System: """

    #************************prompt template without fine tuned for gpt-4o ***********************

            # prompt_template = """Your task is to identify the tables from the context and extract the contents from it . Please also extract the Main Title/Heading of the entire context as well as the DOI. Format your response as a json object with Main Title and DOI as keys:


            #         Context: {context} User: {question} System: """

    # ************************ fine tuned prompt template for Gpt-3.5-turbo-16k **********************************************
        
            # prompt_template = """Your task is to identify the following items from the context:

            #         Please follow the below instructions step by step to successfully query the task.

            #         1) Look for the keyword "\nTable i\n" . where i can be a number (ranging from 1 to 10) or Roman numerals (I to X). only the keyword "\nTable i\n" has to be considered as valid tables.
            #         2) Following this (example: "\nTable \n"), there is the title of the table.
            #         3) Next look for the Title of the table which is usually adjacent to the key word "\nTable \n".
            #         4) After the Title, you will find the header of the table.
            #         5) The table contains values filled inside it.
            #         6) Ensure to capture the entire table structure, including the title, header, and values.
            #         7) Format your response as a JSON object with Table [i] (where i can be any number) as keys.
            #         8) If the information isn't present, use "unable to detect from the given context" as the value.
            #         9) After detecting and formatting the tables in step 7, check if there are any JSON objects where all tables are marked as "unable to detect from the given context."
            #         10) If you find such JSON objects in step 10, please remove those Tables completely from the final output.
            #         11) Format your final response as a JSON object with only the valid tables included, using Table [i] (where i can be any number) as keys.
            #         12) If all tables in a JSON object are marked as "unable to detect from the given context," treat the entire JSON object as if it is not present in the final output.

            #             Note: Tables marked as "unable to detect from the given context" needs to be excluded from the final output.

            #             Context: {context} User: {question} System: """

            qa_prompt = PromptTemplate(
                input_variables=["context", "question"], template=prompt_template
            )

            chain_type_kwargs = {"prompt": qa_prompt}

            # Create a RetrievalQA model
            qa = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever,
                chain_type_kwargs=chain_type_kwargs,
                return_source_documents=False,
            )

            query = "Please Extract only the Valid Tables from the given context"

            # Query the model
            result = qa({"query": query})

            # Store the result in the list
            results_list.append((modified_variable_name, result["result"]))


        def is_empty_dict(d):
            return isinstance(d, dict) and all(not bool(value) for value in d.values())

        def remove_tables_with_text(data):
            # If data is a string, convert it to a dictionary
            if isinstance(data, str):
                try:
                    data = json.loads(data)
                except json.JSONDecodeError:
                    # If the string cannot be decoded, return the original data
                    return data

            # Initialize tables_to_delete list
            tables_to_delete = []

            try:
                # Try accessing 'Tables' key, if KeyError occurs, execute the except block
                if isinstance(data.get('Tables'), list):
                    for table in data['Tables']:
                        if isinstance(table, dict):
                            for table_name, table_info in table.items():
                                if table_info == "unable to detect from the given context":
                                    tables_to_delete.append(table_name)
            except KeyError:
                # If KeyError occurs, execute this block
                filtered_dict = {key: value for key, value in data.items() if value != 'unable to detect from the given context'}
                return filtered_dict
            else:
                # If no KeyError, execute this block
                for table_name in tables_to_delete:
                    data['Tables'] = [table for table in data['Tables'] if not (isinstance(table, dict) and table_name in table)]

                # Check if the dictionary is empty or contains only {}
                if is_empty_dict(data):
                    return None  # Returning None for empty dictionary to indicate removal
                else:
                    return data

        # Initialize a list to store non-empty dictionaries
        prompt_table_data_results = []

        # Iterate over results_list
        for i in range(len(results_list)):
            x = results_list[i][1]
            x = [x]
            results_dict_2 = x[0]
            results_dict_2 = remove_tables_with_text(results_dict_2)

            # Only append results_dict_2 to non_empty_results if it is not None and not an empty dictionary
            if results_dict_2 is not None and results_dict_2 != {}:
                prompt_table_data_results.append(results_dict_2)
        
        # Initialize a list to store non-empty dictionaries
        prompt_table_data_results = []

        # Iterate over results_list
        for i in range(len(results_list)):
            x = results_list[i][1]
            x = [x]
            results_dict_2 = x[0]
            results_dict_2 = remove_tables_with_text(results_dict_2)

            # Only append results_dict_2 to non_empty_results if it is not None and not an empty dictionary
            if results_dict_2 is not None and results_dict_2 != {}:
                prompt_table_data_results.append(results_dict_2)

        # ******************  Saving the tabular data obtained from LLM model in .txt format **************************** 

        folder_name = "gpt-3.5-turbo-16k"
        # folder_name = "gpt-3.5k-turbo-16k_with_finetuning"
        # folder_name = "gpt-4o"

        # Define the directory and file paths
        file_path =  f'/home/abrar/LLM_document_search/Tables_data_{folder_name}/{pdf_file_name}_{folder_name}.txt'

        # Writing to a txt file
        with open(file_path, mode="w") as file:
            # Write each item in prompt_table_data_results list as a line in the txt file
            for item in prompt_table_data_results:
                file.write("%s\n" % item)


        # **************** Saving the tabular data obtained from LLM model in .json format **************************** 


        # Define the directory and file paths
        file_path_json = f'/home/abrar/LLM_document_search/Tables_data_{folder_name}/{pdf_file_name}_{folder_name}.json'

        with open(file_path_json, 'w') as output_file:
            json.dump(prompt_table_data_results, output_file, indent=4)



            input_token_count = count_tokens(specific_llm_model_name, cropped_texts_2)
            prompt_token_count = count_tokens(specific_llm_model_name, model_config[specific_llm_model_name]["prompt_text"])
            output_token_count = count_tokens(specific_llm_model_name, [json.dumps(data) for data in prompt_table_data_results])

            total_cost = calculate_cost(input_token_count, output_token_count, prompt_token_count, model_config[specific_llm_model_name]["cost_per_million_tokens"])

            # Print the result
            print(f"Total cost of extracting the tables from the given pdf {pdf_file_name} with chunk size {len(cropped_texts_2)} using LLM_model : {specific_llm_model_name} with No.of input_tokens: {input_token_count}, No.of prompt_tokens: {prompt_token_count}, and No.of output_tokens: {output_token_count} is ${total_cost}")

            # Append the results to the CSV file
            csv_data = [
                [pdf_file_name, len(cropped_texts_2), specific_llm_model_name, input_token_count, prompt_token_count, output_token_count, total_cost]
            ]

            with open(csv_file_name, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerows(csv_data)


        # Update tqdm progress bar
        progress_bar.update(1)

# Close the tqdm progress bar
progress_bar.close()



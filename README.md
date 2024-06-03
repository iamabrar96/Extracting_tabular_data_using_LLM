# Extracting_tabular_data_using_LLM

![Extracting_tabular_data_using_LLM](/home/abrar/LLM_document_search/flow_chart.png)

This repository is designed to facilitate the extraction and analysis of tabular data and question answering from research articles using large language models (LLM) like GPT-3.5-turbo-16k and GPT-4o.

## Tabular Data Extraction

The `Tabular_data_extraction.py` script focuses on extracting tabular data from research articles utilizing LLM models like gpt-3.5-turbo-16k,gpt-4o. The tabular extracted with both fine tuning prompt as well as without fine tuning prompt in order to analyze the output. The script also calculates the overall token cost for processing each PDF file. The extracted tabular data is stored in both `.txt` and `.json` formats for further analysis.

## Accuracy Testing

The `accuracy_test.py` file is dedicated to testing the accuracy of tabular data extraction. It accomplishes this by comparing the contents of tabular data extracted by the LLM model with the original data from the PDF files. The results are plotted for visual inspection, providing insights into the performance of the extraction process.

## Question Answering

In `question_answering.py`, a collection of research papers is gathered, cleaned, and processed using the RAG (Retrieval-Augmented Generation) approach. The papers are formatted into templates suitable for question answering using GPT-3.5 models. The responses generated are document-based and tailored to the specific domain of the papers, enabling quick extraction of relevant information.

This repository aims to provide a toolkit for extracting, analyzing, and retrieving information from research articles.
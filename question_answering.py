import os
from langchain.llms import OpenAI
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.embeddings import OpenAIEmbeddings



# ********************* open api key **********************

os.environ["OPENAI_API_KEY"] = "*********************"

# ************** embedding model and large langauge model name *****************************
embedding_model_name = "text-embedding-ada-002"
llm_model_name = "gpt-4o"

# initialize the embeddings using openAI ada text embedding library and the llm model using gpt-3.5-turbo-16k
embeddings = OpenAIEmbeddings(model=embedding_model_name)
llm = OpenAI(temperature=0, model_name=llm_model_name)


# Define the prompt for QA
QA_PROMPT_DOCUMENT_CHAT = """You are a helpful AI assistant. Use the following pieces of context to answer the question at the end.
Make sure the answer is between 2-3 sentences.
If the question is not related to the context, just say Sorry for this question my AI has no answer.
If you don't know the answer, just say Sorry for this question my AI has no answer. DO NOT try to make up an answer.
For every answer you provide, include the title of the source document or its DOI if available.

{context}

User: {question}
System: """
qa_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=QA_PROMPT_DOCUMENT_CHAT,
)

# Load FAISS index
faiss_docstore_path = ""
embeddings = OpenAIEmbeddings()
docsearch = FAISS.load_local(faiss_docstore_path, embeddings)

# Create a retriever
retriever = docsearch.as_retriever()

# Create a conversation buffer memory
memory = ConversationBufferMemory(
    memory_key="chat_history", return_messages=True
)

# Create a Conversational Retrieval Chain
qa = ConversationalRetrievalChain.from_llm(
    llm=llm,  # you need to define and initialize `llm` somewhere
    retriever=retriever,
    memory=memory,
    combine_docs_chain_kwargs={"prompt": qa_prompt},
)

# Interactive loop for Q&A
chat_history = []
while True:
    query = input("Enter your question: ")
    result = qa({"question": query})
    print(f'Answer: {result["answer"]}')
    chat_history.append((query, result["answer"]))

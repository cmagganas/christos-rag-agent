import chainlit as cl
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings import CacheBackedEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.storage import LocalFileStore
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
import chainlit as cl

import build_langchain_vector_store

build_langchain_vector_store

from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
import openai
import os

openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_base = 'https://api.openai.com/v1' # default

embedding_model_name = "text-embedding-ada-002"
embedding_model = OpenAIEmbeddings(model=embedding_model_name)
read_vector_store = Chroma(
        persist_directory="langchain-chroma-pulze-docs", embedding_function=embedding_model
    )
query_results = read_vector_store.similarity_search("How do I use Pulze?")
print(query_results[0].page_content)
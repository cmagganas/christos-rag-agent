import chainlit as cl
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings import CacheBackedEmbeddings, OpenAIEmbeddings
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma #, FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.storage import LocalFileStore
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
import chainlit as cl

from build_langchain_vector_store import chunk_docs, load_gitbook_docs, tiktoken_len
from tiktoken import Encoding, encoding_for_model

import openai
# import os

# openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_base = 'https://api.openai.com/v1' # default

docs_url = "https://docs.pulze.ai/"
embedding_model_name = "text-embedding-ada-002"
langchain_documents = load_gitbook_docs(docs_url)
chunked_langchain_documents = chunk_docs(
    langchain_documents,
    tokenizer=encoding_for_model(embedding_model_name),
    chunk_size=200,
)

embedding_model = OpenAIEmbeddings(model=embedding_model_name)
vector_store = Chroma.from_documents(
    chunked_langchain_documents, embedding=embedding_model, persist_directory="langchain-chroma-pulze-docs"
)
read_vector_store = Chroma(
    persist_directory="langchain-chroma-pulze-docs", embedding_function=embedding_model
)
print(read_vector_store.similarity_search("How do I use Pulze?"))
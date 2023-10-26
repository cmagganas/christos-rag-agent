import chainlit as cl
from langchain.agents.agent_toolkits import create_conversational_retrieval_agent, create_retriever_tool
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

@cl.on_chat_start
async def init():
    
    msg = cl.Message(content="Building Index...")
    await msg.send()
    
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
    
    msg.content = "Index built!"
    await msg.send()
    
    # set up search pulze docs retriever tool
    tool = create_retriever_tool(
        read_vector_store.as_retriever(), 
        "search_pulze_docs",
        "Searches and returns documents regarding Pulze."
    )
    tools = [tool]

    #set llm and agent
    llm = ChatOpenAI(temperature = 0)
    agent_executor = create_conversational_retrieval_agent(llm, tools, verbose=True)

    cl.user_session.set("agent_executor", agent_executor)

@cl.on_message
async def main(message):
    chain: Chain = cl.user_session.get("agent_executor")
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=False, answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    cb.answer_reached = True
    answer = chain.run({"input": message})

    await cl.Message(content=answer["output"]).send()
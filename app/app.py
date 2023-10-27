import chainlit as cl
from langchain.agents.agent_toolkits import create_conversational_retrieval_agent, create_retriever_tool
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings import CacheBackedEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

from build_langchain_vector_store import chunk_docs, load_gitbook_docs, tiktoken_len
from tiktoken import encoding_for_model

import openai
# import os

# openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_base = 'https://api.openai.com/v1' # default

@cl.on_chat_start
async def initialize_chat():
    
    index_building_message = cl.Message(content="Building Index...")
    await index_building_message.send()
    
    documents_url = "https://docs.pulze.ai/"
    embedding_model_identifier = "text-embedding-ada-002"
    langchain_documents = load_gitbook_docs(documents_url)
    chunked_langchain_documents = chunk_docs(
        langchain_documents,
        tokenizer=encoding_for_model(embedding_model_identifier),
        chunk_size=200,
    )
    
    embedding_model = OpenAIEmbeddings(model=embedding_model_identifier)
    Chroma.from_documents(
        chunked_langchain_documents, embedding=embedding_model, persist_directory="langchain-chroma-pulze-docs"
    )
    read_vector_store = Chroma(
        persist_directory="langchain-chroma-pulze-docs", embedding_function=embedding_model
    )
    
    index_building_message.content = "Index built!"
    await index_building_message.send()
    
    # set up search pulze docs retriever tool
    pulze_docs_retriever_tool = create_retriever_tool(
        read_vector_store.as_retriever(), 
        "search_pulze_docs",
        "Searches and returns documents regarding Pulze."
    )
    retrieval_tools = [pulze_docs_retriever_tool]

    #set llm and agent
    chat_model = ChatOpenAI(temperature = 0)
    agent_executor = create_conversational_retrieval_agent(chat_model, retrieval_tools, verbose=True)

    cl.user_session.set("agent_executor", agent_executor)

@cl.on_message
async def process_message(message):
    agent_executor = cl.user_session.get("agent_executor")
    callback_handler = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=False, answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    callback_handler.answer_reached = True
    response = agent_executor({"input": message})

    await cl.Message(content=response["output"]).send()
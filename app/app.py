import chainlit as cl
import openai
from build_langchain_vector_store import chunk_docs, load_gitbook_docs
from langchain.agents.agent_toolkits import (
    create_conversational_retrieval_agent,
    create_retriever_tool,
)
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from tiktoken import encoding_for_model

# import os

# openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_base = "https://api.openai.com/v1"  # default


@cl.on_chat_start
async def start_chat():
    index_building_notification = cl.Message(content="Building Index...")
    await index_building_message.send()

    docs_url = "https://docs.pulze.ai/"
    embedding_model_id = "text-embedding-ada-002"
    loaded_documents = load_gitbook_docs(docs_url)
    chunked_documents = chunk_docs(
        loaded_documents,
        tokenizer=encoding_for_model(embedding_model_id),
        chunk_size=200,
    )

    embedding_model = OpenAIEmbeddings(model=embedding_model_id)
    Chroma.from_documents(
        chunked_documents,
        embedding=embedding_model,
        persist_directory="langchain-chroma-pulze-docs",
    )
    vector_store = Chroma(
        persist_directory="langchain-chroma-pulze-docs",
        embedding_function=embedding_model,
    )

    index_building_notification.content = "Index built!"
    await index_building_notification.send()

    # set up search pulze docs retriever tool
    docs_retriever_tool = create_retriever_tool(
        vector_store.as_retriever(),
        "search_pulze_docs",
        "Searches and returns documents regarding Pulze.",
    )
    retriever_tools = [docs_retriever_tool]

    # set llm and agent
    openai_chat_model = ChatOpenAI(temperature=0)
    retrieval_agent = create_conversational_retrieval_agent(
        openai_chat_model, retriever_tools, verbose=True
    )

    cl.user_session.set("retrieval_agent", retrieval_agent)


@cl.on_message
async def handle_message(message):
    retrieval_agent = cl.user_session.get("retrieval_agent")
    answer_handler = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=False, answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    answer_handler.answer_reached = True
    agent_response = retrieval_agent({"input": message})

    await cl.Message(content=agent_response["output"]).send()

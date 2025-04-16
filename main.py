import streamlit as st
from langchain.agents import AgentType, initialize_agent
from langchain_core.tools import StructuredTool
from langchain_community.chat_models import ChatOpenAI
from dotenv import load_dotenv
from datetime import datetime, timezone
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.retrieval_qa.base import RetrievalQA

import os

load_dotenv()

def get_current_time() -> str:
    sTime = datetime.now(timezone.utc).isoformat()
    return sTime

def rag_search(query: str)-> str:
    with open("data.txt") as f:
        raw_text = f.read()

    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = splitter.split_text(raw_text)

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts, embedding=embeddings)

    qa = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())
    return qa.run(query)
        

api_key = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="AI Assistant", page_icon="🤖")

user_question = st.text_input("Enter your query:")

if user_question:
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    tools = [
        StructuredTool.from_function(
            func=get_current_time,
            name="get_current_time",
            description="Returns the current time in UTC timezone"
        ),
         StructuredTool.from_function(
            func=rag_search,
            name="rag_search",
            description="Employee birthdays"
        )
    ]

    with st.spinner("Thinking..."):
        agent = initialize_agent(
            tools,
            llm,
            agent=AgentType.OPENAI_FUNCTIONS,
            verbose=True)
        response = agent.run(user_question)
        st.success("Answer:")
        st.write(response)
import streamlit as st
from langchain.agents import Tool, initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from dotenv import load_dotenv
from datetime import datetime

import os

load_dotenv()


def get_current_time(_):
    return datetime.now().strftime("%H:%M, %d %B %Y")

api_key = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="AI Assistant", page_icon="🤖")

user_question = st.text_input("Enter your query:")

if user_question:
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    tools = [
        Tool(
            name="get_current_time",
            func=get_current_time,
            description="Returns current time. Use when a user asks how many time"
        )
    ]

    with st.spinner("Thinking..."):
        agent = initialize_agent(
            tools,
            llm,
            agent="chat-zero-shot-react-description",
            verbose=True)
        response = agent.run(user_question)
        st.success("Answer:")
        st.write(response)
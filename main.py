import streamlit as st
from langchain.agents import AgentType, initialize_agent
from langchain_core.tools import StructuredTool
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from dotenv import load_dotenv
from datetime import datetime

import os

load_dotenv()

def get_current_time() -> str:
    return datetime.now().isoformat()

api_key = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="AI Assistant", page_icon="🤖")

user_question = st.text_input("Enter your query:")

if user_question:
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    tools = [
        StructuredTool.from_function(
            func=get_current_time,
            name="get_current_time",
            description="Returns the current time"
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
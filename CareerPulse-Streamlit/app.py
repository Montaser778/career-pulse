import streamlit as st
import os
import re
import docx
import pdfplumber
import requests
from typing import List, TypedDict, Annotated
import operator
from groq import Groq
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage

# --- Page Setup & CSS ---
st.set_page_config(page_title="CareerPulse AI", layout="centered")

st.markdown("""
    <style>
    /* إزالة الفراغ العلوي الافتراضي */
    .block-container { padding-top: 2rem; }
    
    /* خلفية متحركة */
    .stApp {
        background: linear-gradient(-45deg, #ee7752, #e73c7e, #23a6d5, #23d5ab);
        background-size: 400% 400%;
        animation: gradient 15s ease infinite;
    }
    @keyframes gradient {
        0% {background-position: 0% 50%;}
        50% {background-position: 100% 50%;}
        100% {background-position: 0% 50%;}
    }
    
    .main-box { background-color: rgba(255,255,255,0.95); padding: 30px; border-radius: 20px; box-shadow: 0 10px 30px rgba(0,0,0,0.1); }
    .title-style { margin-bottom: 0; padding-bottom: 0; }
    .subtitle-style { color: #555; margin-top: 0; margin-bottom: 20px; font-weight: 300; }
    .stButton>button { width: 100%; border-radius: 10px; background-color: #2c3e50; color: white; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

# --- Logic Setup ---
api_key = st.secrets.get("GROQ_API_KEY")
llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=api_key)

@tool
def job_posting_tool(job_link: str) -> str:
    """Extracts content from a job URL."""
    try:
        r = requests.get(job_link, headers={'User-Agent': 'Mozilla/5.0'})
        return r.text[:8000]
    except: return "Error accessing job link."

@tool
def extract_cv_text(file_path: str) -> str:
    """Extracts text from a CV file."""
    try:
        if file_path.endswith(".docx"):
            return '\n'.join([p.text for p in docx.Document(file_path).paragraphs])
        elif file_path.endswith(".pdf"):
            with pdfplumber.open(file_path) as pdf:
                return '\n'.join([p.extract_text() for p in pdf.pages])
    except: return "Error reading CV."
    return "Unsupported format."

tools = [job_posting_tool, extract_cv_text]
llm_with_tools = llm.bind_tools(tools)

# --- Graph Workflow ---
class AgentState(TypedDict):
    messages: Annotated[List, operator.add]

def agent_node(state: AgentState):
    msg = llm_with_tools.invoke(state["messages"])
    return {"messages": [msg]}

def tool_node(state: AgentState):
    last_msg = state["messages"][-1]
    results = []
    for tool_call in last_msg.tool_calls:
        res = extract_cv_text.invoke(tool_call["args"]["file_path"]) if tool_call["name"] == "extract_cv_text" else job_posting_tool.invoke(tool_call["args"]["job_link"])
        results.append(res)
    return {"messages": [AIMessage(content=str(results))]}

workflow = StateGraph(AgentState)
workflow.add_node("agent", agent_node)
workflow.add_node("tools", tool_node)
workflow.set_entry_point("agent")
workflow.add_conditional_edges("agent", lambda s: "tools" if s["messages"][-1].tool_calls else END)
workflow.add_edge("tools", "agent")
app = workflow.compile()

# --- UI Content ---
st.markdown("<div class='main-box'>", unsafe_allow_html=True)
st.markdown("<h1 class='title-style' style='text-align:center;'>🚀 CareerPulse AI Pro</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle-style' style='text-align:center;'>Developed by Eng. Montaser</p>", unsafe_allow_html=True)

cv_file = st.file_uploader("Upload your CV", type=['pdf', 'docx'])
job_link = st.text_input("Job Posting URL")
question = st.text_area("Your Question", placeholder="Evaluate my CV against this job...", height=150)

if st.button("Analyze"):
    if cv_file and job_link and question:
        with open("temp_cv.pdf", "wb") as f: f.write(cv_file.getbuffer())
        with st.spinner("Agent is analyzing..."):
            initial_state = {"messages": [HumanMessage(content=f"CV at temp_cv.pdf. Job at {job_link}. Task: {question}")]}
            final_state = app.invoke(initial_state)
            st.write("### Analysis Result:")
            st.write(final_state["messages"][-1].content)
    else:
        st.warning("Please fill all fields.")
st.markdown("</div>", unsafe_allow_html=True)

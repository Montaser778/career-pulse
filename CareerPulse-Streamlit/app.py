import streamlit as st
import os
import docx
import pdfplumber
import requests
from typing import List, Annotated
import operator
from groq import Groq
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END, MessagesState
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage

# --- Page Setup & Professional Background ---
st.set_page_config(page_title="CareerPulse AI Pro", layout="centered")

st.markdown("""
    <style>
    /* خلفية احترافية ثابتة */
    .stApp {
        background: linear-gradient(135deg, #1a2a6c, #b21f1f, #fdbb2d);
        background-attachment: fixed;
    }
    .main-container {
        background-color: rgba(255, 255, 255, 0.95);
        padding: 40px;
        border-radius: 25px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        margin-top: -20px;
    }
    .header-text { color: #2c3e50; font-weight: 800; text-align: center; margin-bottom: 0px; }
    .sub-text { color: #7f8c8d; text-align: center; margin-bottom: 25px; }
    </style>
""", unsafe_allow_html=True)

# --- Logic Setup ---
# تأكد من أن الـ API Key صحيح ومفعل في الـ Secrets
api_key = st.secrets.get("GROQ_API_KEY")
llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=api_key)

# (نفس دوال الأدوات والـ Workflow السابقة)
@tool
def job_posting_tool(job_link: str) -> str:
    """Extracts content from a job URL."""
    try:
        r = requests.get(job_link, headers={'User-Agent': 'Mozilla/5.0'})
        return r.text[:5000]
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

def agent(state: MessagesState):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

workflow = StateGraph(MessagesState)
workflow.add_node("agent", agent)
workflow.add_node("tools", ToolNode(tools))
workflow.set_entry_point("agent")
workflow.add_conditional_edges("agent", lambda s: "tools" if s["messages"][-1].tool_calls else END)
workflow.add_edge("tools", "agent")
app = workflow.compile()

# --- UI Content ---
st.markdown("<div class='main-container'>", unsafe_allow_html=True)
st.markdown("<h1 class='header-text'>Career Pulse AI Pro</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-text'>Developed by Eng. Montaser</p>", unsafe_allow_html=True)

cv_file = st.file_uploader("Upload your CV", type=['pdf', 'docx'])
job_link = st.text_input("Job Posting URL")
question = st.text_area("Your Question", placeholder="Evaluate my CV against this job...", height=150)

if st.button("Analyze"):
    if cv_file and job_link and question:
        save_path = "temp_cv.pdf"
        with open(save_path, "wb") as f: f.write(cv_file.getbuffer())
        
        with st.spinner("Agent is working..."):
            try:
                initial_state = {"messages": [HumanMessage(content=f"CV at {save_path}. Job at {job_link}. Task: {question}")]}
                final_state = app.invoke(initial_state)
                st.success("Analysis Result:")
                st.write(final_state["messages"][-1].content)
            except Exception as e:
                st.error("Error occurred! (Check if your API Key is active or limits reached).")
    else:
        st.warning("Please fill all fields.")
st.markdown("</div>", unsafe_allow_html=True)

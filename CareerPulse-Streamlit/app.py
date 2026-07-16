import streamlit as st
import os
import re
import docx
import pdfplumber
import requests
from typing import List, TypedDict
from groq import Groq
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END

# --- Page Setup ---
st.set_page_config(page_title="CareerPulse AI", layout="centered")

# Developed by Eng. Montaser Header
st.markdown("<p style='text-align: center; color: #7f8c8d; font-family: sans-serif;'>Developed by Eng. Montaser</p>", unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center; color: #2c3e50;'>🚀 CareerPulse AI</h1>", unsafe_allow_html=True)

# --- Groq & Config ---
api_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=api_key)

# --- Tools (Exact logic from your Notebook) ---
@tool
def job_posting_tool(job_link: str) -> str:
    """Extracts structured information from a job posting at the provided URL."""
    try:
        r = requests.get(job_link, headers={'User-Agent': 'Mozilla/5.0'})
        return r.text[:5000]
    except: return "Job posting unavailable."

@tool
def extract_cv_text(file_path: str) -> str:
    """Extracts text content from CV (PDF/DOCX)."""
    ext = os.path.splitext(file_path)[-1].lower()
    try:
        if ".docx" in ext:
            doc = docx.Document(file_path)
            return '\n'.join([para.text for para in doc.paragraphs])
        elif ".pdf" in ext:
            with pdfplumber.open(file_path) as pdf:
                return '\n'.join([page.extract_text() for page in pdf.pages])
    except Exception as e: return f"Error: {e}"
    return "Unsupported format."

# --- ReWOO Graph Logic ---
class ReWOO(TypedDict):
    task: str
    plan_string: str
    steps: List
    results: dict
    result: str

def planner_node(state: ReWOO):
    # Ensure this matches your notebook prompt exactly
    planner_prompt = "Plan: Explain steps and assign tool (CV or JobPost) #E1 = TOOL[input]\nTask: {task}"
    raw_plan = llm.invoke(planner_prompt.format(task=state["task"])).content
    steps = re.findall(r"Plan:\s*(.+?)\s*(#E\d+)\s*=\s*(\w+)\[(.+?)\]", raw_plan, flags=re.S)
    return {"plan_string": raw_plan, "steps": steps}

def executor_node(state: ReWOO):
    idx = len(state.get("results", {}))
    if idx >= len(state["steps"]): return {}
    _, _, tool_name, tool_input = state["steps"][idx]
    inp = tool_input.strip("'\"")
    
    if tool_name == "CV": out = extract_cv_text.invoke(inp)
    elif tool_name == "JobPost": out = job_posting_tool.invoke(inp)
    else: out = llm.invoke(inp).content
    
    results = dict(state.get("results", {}))
    results[state["steps"][idx][1]] = str(out)
    return {"results": results}

def solver_node(state: ReWOO):
    lines = [f"Plan: {s[0]}\nEvidence {s[1]}: {state['results'].get(s[1])}" for s in state["steps"]]
    prompt = f"Solve task: {state['task']}\nEvidence:\n{'\n'.join(lines)}"
    return {"result": llm.invoke(prompt).content}

# --- Graph Compilation ---
builder = StateGraph(ReWOO)
builder.add_node("plan", planner_node)
builder.add_node("tool", executor_node)
builder.add_node("solve", solver_node)
builder.set_entry_point("plan")
builder.add_edge("plan", "tool")
builder.add_conditional_edges("tool", lambda s: "solve" if len(s.get("results", {})) >= len(s["steps"]) else "tool")
builder.add_edge("solve", END)
rewoo_graph = builder.compile()

# --- UI ---
cv_file = st.file_uploader("Upload your CV", type=['pdf', 'docx'])
job_link = st.text_input("Job Posting URL")
question = st.text_area("Your Question", placeholder="e.g., Evaluate my CV based on this job...")

if st.button("Analyze"):
    if cv_file and job_link and question:
        with open("temp_cv.pdf", "wb") as f: f.write(cv_file.getbuffer())
        with st.spinner("Agent is working..."):
            state = {"task": f"{question} | Job: {job_link} | CV: temp_cv.pdf", "steps": [], "results": {}, "result": ""}
            for event in rewoo_graph.stream(state):
                if "solve" in event:
                    st.success("Analysis Result:")
                    st.write(event["solve"]["result"])
    else:
        st.warning("Please fill in all fields.")

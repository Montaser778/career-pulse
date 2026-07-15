import streamlit as st
import os
import docx
import pdfplumber
import requests
from bs4 import BeautifulSoup
import re
from typing import List
from typing_extensions import TypedDict
from groq import Groq
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END

# --- إعدادات الصفحة ---
st.set_page_config(page_title="CareerPulse-AI", page_icon="🚀")
st.title("🚀 CareerPulse-AI")

# --- إعدادات Groq ---
api_key = st.secrets.get("GROQ_API_KEY")
client = Groq(api_key=api_key)
llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=api_key)

# --- الأدوات ---
@tool
def job_posting_tool(job_link: str) -> str:
    """Extracts job details from URL."""
    try:
        r = requests.get(job_link, headers={'User-Agent': 'Mozilla/5.0'})
        soup = BeautifulSoup(r.content, 'html.parser')
        return " ".join([p.text for p in soup.find_all(['p', 'li'])])[:5000]
    except: return "Error fetching link."

@tool
def extract_cv_text(file_path: str) -> str:
    """Extracts text from PDF/DOCX."""
    ext = os.path.splitext(file_path)[-1].lower()
    if ".docx" in ext: return '\n'.join([p.text for p in docx.Document(file_path).paragraphs])
    elif ".pdf" in ext:
        with pdfplumber.open(file_path) as pdf: return '\n'.join([p.extract_text() for p in pdf.pages])
    return "Unsupported format."

# --- منطق الـ ReWOO ---
class ReWOO(TypedDict):
    task: str; steps: List; results: dict; result: str

def planner_node(state: ReWOO):
    plan = llm.invoke(f"Task: {state['task']}\nPlan: Explain step by step and assign tool #E1 = TOOL[input]").content
    return {"steps": re.findall(r"Plan:\s*(.+?)\s*(#E\d+)\s*=\s*(\w+)\[(.+?)\]", plan, flags=re.S)}

def executor_node(state: ReWOO):
    idx = len(state.get("results", {}))
    _, _, tool, inp = state["steps"][idx]
    out = extract_cv_text.invoke(inp) if tool == "CV" else job_posting_tool.invoke(inp)
    results = dict(state.get("results", {}))
    results[state["steps"][idx][1]] = str(out)
    return {"results": results}

def solver_node(state: ReWOO):
    return {"result": llm.invoke(f"Analyze: {state['task']} using {state['results']}").content}

# --- بناء الـ Graph ---
builder = StateGraph(ReWOO)
builder.add_node("plan", planner_node); builder.add_node("tool", executor_node); builder.add_node("solve", solver_node)
builder.set_entry_point("plan"); builder.add_edge("plan", "tool"); builder.add_edge("solve", END)
builder.add_conditional_edges("tool", lambda s: "solve" if len(s.get("results", {})) >= len(s["steps"]) else "tool")
rewoo_graph = builder.compile()

# --- واجهة المستخدم ---
with st.sidebar:
    cv_file = st.file_uploader("📄 ارفع السيرة الذاتية", type=['pdf', 'docx'])
    job_link = st.text_input("🔗 رابط الوظيفة")

msg = st.text_input("❓ ماذا تريد أن تعرف؟")

if st.button("تحليل ذكي 🔍"):
    if cv_file and job_link and msg:
        with st.spinner("جاري العمل..."):
            # حفظ الملف مؤقتاً للمعالجة
            with open("temp_cv.pdf", "wb") as f: f.write(cv_file.getbuffer())
            
            state = {"task": f"{msg} | Job: {job_link} | CV: temp_cv.pdf", "steps": [], "results": {}, "result": ""}
            for event in rewoo_graph.stream(state):
                if "solve" in event:
                    st.success("التحليل:")
                    st.write(event["solve"]["result"])
    else:
        st.error("يرجى إكمال البيانات")
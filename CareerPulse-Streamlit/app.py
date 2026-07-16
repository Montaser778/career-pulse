import streamlit as st
import os
import docx
import pdfplumber
import requests
import re
from typing import List, TypedDict
from groq import Groq
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END

# --- إعداد الصفحة وتنسيق CSS خرافي ---
st.set_page_config(page_title="CareerPulse AI", layout="centered")

st.markdown("""
    <style>
    .stApp { background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); }
    .main-title { color: #2c3e50; text-align: center; font-weight: 800; }
    .stButton>button { width: 100%; border-radius: 20px; background-color: #3498db; color: white; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

st.markdown("<h1 class='main-title'>🚀 CareerPulse AI</h1>", unsafe_allow_html=True)

# --- إعدادات Groq ---
api_key = st.secrets.get("GROQ_API_KEY")
llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=api_key)

# --- الأدوات والمنطق (Backend) ---
class ReWOO(TypedDict):
    task: str; steps: List; results: dict; result: str

@tool
def job_posting_tool(job_link: str) -> str:
    """Extracts job details from URL."""
    try:
        r = requests.get(job_link, headers={'User-Agent': 'Mozilla/5.0'})
        soup = BeautifulSoup(r.content, 'html.parser')
        return " ".join([p.text for p in soup.find_all(['p', 'li'])])[:5000]
    except: return "Job posting unavailable or contains no job details."

@tool
def extract_cv_text(file_path: str) -> str:
    """Extracts text from PDF/DOCX file path."""
    try:
        if file_path.endswith(".docx"):
            doc = docx.Document(file_path)
            return '\n'.join([p.text for p in doc.paragraphs])
        elif file_path.endswith(".pdf"):
            with pdfplumber.open(file_path) as pdf:
                return '\n'.join([p.extract_text() for p in pdf.pages])
    except Exception as e: return f"Error reading file: {e}"
    return "Unsupported format."

# --- Graph Logic ---
def planner_node(state: ReWOO):
    prompt = f"Task: {state['task']}\nPlan: Explain steps and assign tool (CV or JobPost) #E1 = TOOL[input]"
    raw = llm.invoke(prompt).content
    steps = re.findall(r"Plan:\s*(.+?)\s*(#E\d+)\s*=\s*(\w+)\[(.+?)\]", raw, flags=re.S)
    return {"steps": steps}

def executor_node(state: ReWOO):
    idx = len(state.get("results", {}))
    steps = state.get("steps", [])
    if idx >= len(steps): return {}
    _, _, tool, inp = steps[idx]
    inp = inp.strip("'\"")
    out = extract_cv_text.invoke(inp) if tool == "CV" else job_posting_tool.invoke(inp)
    results = dict(state.get("results", {}))
    results[steps[idx][1]] = str(out)
    return {"results": results}

def solver_node(state: ReWOO):
    ans = llm.invoke(f"Solve {state['task']} using {state['results']}").content
    return {"result": ans}

# --- بناء وتجميع الـ Graph ---
builder = StateGraph(ReWOO)
builder.add_node("plan", planner_node); builder.add_node("tool", executor_node); builder.add_node("solve", solver_node)
builder.set_entry_point("plan"); builder.add_edge("plan", "tool"); builder.add_edge("solve", END)
builder.add_conditional_edges("tool", lambda s: "solve" if len(s.get("results", {})) >= len(s["steps"]) else "tool")
rewoo_graph = builder.compile()

# --- واجهة المستخدم (Streamlit) ---
with st.container():
    cv_file = st.file_uploader("📄 ارفع السيرة الذاتية (PDF/DOCX)", type=['pdf', 'docx'])
    job_link = st.text_input("🔗 رابط الوظيفة")
    question = st.text_area("❓ سؤالك أو تقييمك للوظيفة؟")

    if st.button("تحليل ذكي 🔍"):
        if cv_file and job_link and question:
            with open("temp_cv.pdf", "wb") as f: f.write(cv_file.getbuffer())
            with st.spinner("🤖 الـ Agent يعمل الآن..."):
                state = {"task": f"{question} | Job: {job_link} | CV: temp_cv.pdf", "steps": [], "results": {}, "result": ""}
                for event in rewoo_graph.stream(state):
                    if "solve" in event:
                        st.markdown("### 🎯 النتيجة النهائية:")
                        st.info(event["solve"]["result"])
        else:
            st.warning("⚠️ يرجى تعبئة كافة الحقول!")

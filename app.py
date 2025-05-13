import numpy as np
import streamlit as st
import pandas as pd
import tempfile
import ast
import re
import nltk
import language_tool_python
import base64
from collections import Counter
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from pdfminer.high_level import extract_text
from difflib import get_close_matches
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from rag_utils import (
    initialize_embedding_model,
    get_embeddings,
    create_faiss_index,
    retrieve_similar_texts,
    generate_response
)

nltk.download('stopwords')

# === Streamlit Page Setup
st.set_page_config(page_title="Resume Skill Matcher", layout="centered")

# === Background and Styling
def set_background_and_styles():
    st.markdown("""
        <style>
        .stApp {
            font-family: 'Segoe UI', sans-serif;
            background-image: 
                linear-gradient(to bottom right, rgba(255,255,255,0.85), rgba(240,240,240,0.85)),
                url("https://img.freepik.com/free-photo/top-view-desk-concept-with-copy-space_23-2148236824.jpg?semt=ais_hybrid&w=740");
            background-size: cover;
            background-attachment: fixed;
            background-position: center;
        }

        .glass-box {
            background: rgba(255, 255, 255, 0.9);
            backdrop-filter: blur(12px);
            padding: 20px;
            border-radius: 16px;
            margin-bottom: 20px;
        }

        .score-circle {
            width: 160px;
            height: 160px;
            border-radius: 50%;
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 32px;
            font-weight: bold;
            border: 2px solid #ccc;
            color: #222;
            margin: 20px auto;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }

        .pill {
            display: inline-block;
            background: rgba(0, 0, 0, 0.07);
            padding: 6px 12px;
            margin: 4px;
            border-radius: 20px;
            font-size: 14px;
            color: #000;
        }

        .info-box {
            background: rgba(255, 255, 255, 0.9);
            backdrop-filter: blur(6px);
            padding: 15px;
            border-left: 5px solid #f59e0b;
            border-radius: 10px;
            margin-top: 25px;
            font-size: 15px;
            color: #222;
        }

        h1, h2, h3, p {
            color: #222 !important;
        }
        </style>
    """, unsafe_allow_html=True)


set_background_and_styles()

# === Constants
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english')).union(ENGLISH_STOP_WORDS)

ROLE_KEYWORDS = {
    "Data Scientist": ["data", "ml", "ai", "analytics", "statistics", "python", "nlp"],
    "Software Engineer": ["software", "developer", "backend", "frontend", "java", "spring", "sql", "api"],
    "Project Manager": ["project", "manager", "scrum", "agile", "planning", "jira", "kanban"],
    "Android Developer": ["android", "mobile", "kotlin", "java", "firebase", "xml"]
}

# === Utility Functions
def extract_text_from_pdf_file(file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file.read())
        return extract_text(tmp.name)

def clean_and_tokenize(text):
    text = re.sub(r'[^A-Za-z0-9\s\-]', ' ', text)
    text = re.sub(r'-', ' ', text)
    tokens = text.split()
    return [t for t in tokens if len(t) > 0 and t.lower() not in stop_words]

def generate_phrases(tokens):
    unigrams = tokens
    bigrams = [' '.join(tokens[i:i+2]) for i in range(len(tokens)-1)]
    trigrams = [' '.join(tokens[i:i+3]) for i in range(len(tokens)-2)]
    return set(unigrams + bigrams + trigrams)

def extract_skills_from_resume(text):
    tokens = clean_and_tokenize(text)
    phrases = generate_phrases(tokens)
    normalized = set()
    for phrase in phrases:
        lower = phrase.lower()
        stem = stemmer.stem(lower)
        normalized.update([phrase.strip(), lower.strip(), stem.strip()])
    return set(normalized)

def normalize_skills(skills):
    normalized = set()
    for skill in skills:
        skill = skill.lower().strip().replace('-', ' ')
        skill = re.sub(r'[^a-z0-9\s]', '', skill)
        normalized.add(skill)
        normalized.add(stemmer.stem(skill))
    return set(normalized)

def fuzzy_match(skills, reference):
    matched = set()
    for skill in skills:
        close = get_close_matches(skill, reference, cutoff=0.8)
        matched.update(close)
    return matched

@st.cache_data
def load_and_enrich_skills():
    df = pd.read_csv("resume_data.csv")
    df.columns = df.columns.str.strip().str.lower()
    def map_to_title(pos_list_str):
        try:
            positions = ast.literal_eval(pos_list_str)
            for pos in positions:
                pos_lower = pos.lower()
                for role, keywords in ROLE_KEYWORDS.items():
                    if any(k in pos_lower for k in keywords):
                        return role
        except:
            return None
        return None

    df["job title"] = df["positions"].apply(map_to_title)
    df = df[df["job title"].notnull()]

    skill_map = {}
    for role in df["job title"].unique():
        role_df = df[df["job title"] == role]
        all_skills = []
        for skill_list in role_df["skills"].dropna():
            try:
                parsed = ast.literal_eval(skill_list)
                all_skills.extend(normalize_skills(parsed))
            except:
                continue
        enriched = all_skills + ROLE_KEYWORDS.get(role, [])
        top_skills = [s for s, _ in Counter(enriched).most_common(40)]
        skill_map[role] = set(top_skills)
    return skill_map

role_skill_map = load_and_enrich_skills()

def compute_skill_match(resume_skills, required_skills, resume_text=None):
    resume_skills = normalize_skills(resume_skills)
    required_skills = normalize_skills(required_skills)
    fuzzy_matches = fuzzy_match(resume_skills, required_skills)
    matched = resume_skills & required_skills | fuzzy_matches
    missing = required_skills - matched

    def dedup(skills):
        seen = {}
        for s in sorted(skills, key=lambda x: -len(x)):
            norm = re.sub(r'[^a-z0-9]', '', s.lower().replace(' ', ''))
            stem = stemmer.stem(norm)
            if not any(stem == existing or stem in existing or existing in stem for existing in seen):
                seen[stem] = s
        return set(seen.values())

    matched = dedup(matched)
    missing = dedup(missing)

    skill_score = len(matched) / max(len(required_skills), 1)

    tool = language_tool_python.LanguageTool('en-US')
    matches = tool.check(resume_text or "")
    grammar_penalty = len(matches)
    grammar_score = max(0.0, 1.0 - grammar_penalty / 25)

    final_score = round((0.9 * skill_score + 0.1 * grammar_score) * 100, 2)
    return final_score, matched, missing, matches

# === App UI
st.markdown("<div class='glass-box'>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align:center;'>🔍 Resume Skill Matcher</h2>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Upload your resume PDF and get personalized skill feedback for different tech roles.</p>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    role = st.selectbox("🎯 Select Target Role", list(role_skill_map.keys()))
with col2:
    file = st.file_uploader("📄 Upload Your Resume (PDF)", type=["pdf"])
if file:
    with st.spinner("🧠 Analyzing your resume..."):
        text = extract_text_from_pdf_file(file)
        resume_skills = extract_skills_from_resume(text)
        required_skills = role_skill_map[role]

    st.markdown("<div class='glass-box'>", unsafe_allow_html=True)
    st.markdown("### 📊 Match Summary")
    st.markdown(f"<div class='score-circle'>{final_score}%</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if final_score >= 80:
        st.success("✅ Great job! Your resume is a strong fit for this role.")
    elif final_score >= 50:
        st.warning("⚠️ Decent match. Consider improving your skill alignment.")
    else:
        st.error("❌ Low match. Try gaining experience in the missing skill areas.")

    with st.expander("🧠 View Skill Breakdown"):
        st.markdown("✅ **Matched Skills:**")
        st.markdown("".join([f"<span class='pill'>{s}</span>" for s in sorted(matched)]), unsafe_allow_html=True)
        st.markdown("❌ **Missing Skills:**")
        st.markdown("".join([f"<span class='pill'>{s}</span>" for s in sorted(missing)]), unsafe_allow_html=True)

    
    st.markdown("<div class='info-box'>💡 Tip: Use Coursera, Udemy, or LinkedIn Learning to build missing skills.</div>", unsafe_allow_html=True)
    st.markdown("### 🤖 Ask About Your Resume")
    query = st.text_input("💬 Ask a question (e.g., 'What are my strengths?' or 'What can I improve?')")

    if query:
        with st.spinner("🔍 Analyzing with Gemini..."):
            embedding_model = initialize_embedding_model()
            docs = [text]
            embeddings = get_embeddings(embedding_model, docs)
            faiss_index = create_faiss_index(np.array(embeddings, dtype=np.float32))
            retrieved = retrieve_similar_texts(embedding_model, faiss_index, query, docs)
            response = generate_response(retrieved, query)

        st.markdown("### 💡 Gemini Insight")
        st.markdown(f"<div class='info-box'>{response.strip()}</div>", unsafe_allow_html=True)
else:
    st.info("📤 Upload a resume to start the analysis.")

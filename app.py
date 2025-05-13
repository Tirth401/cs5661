# app.py
import numpy as np
import streamlit as st
import pandas as pd
import tempfile
import ast
import re
import nltk
import language_tool_python
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

    # Grammar penalty using language_tool_python
    tool = language_tool_python.LanguageTool('en-US')
    matches = tool.check(resume_text or "")
    grammar_penalty = len(matches)
    grammar_score = max(0.0, 1.0 - grammar_penalty / 25)

    final_score = round((0.9 * skill_score + 0.1 * grammar_score) * 100, 2)
    return final_score, matched, missing, matches

# === App UI
st.markdown("## 🔍 Resume Skill Matcher")
st.markdown("<p style='text-align:center;'>Upload your resume PDF and compare your skills to top industry roles.</p>", unsafe_allow_html=True)

role = st.selectbox("🎯 Select Job Title", list(role_skill_map.keys()))
file = st.file_uploader("📄 Upload Resume (PDF)", type=["pdf"])

if file:
    with st.spinner("🧠 Analyzing your resume..."):
        text = extract_text_from_pdf_file(file)
        resume_skills = extract_skills_from_resume(text)
        required_skills = role_skill_map[role]
        final_score, matched, missing, grammar_matches = compute_skill_match(resume_skills, required_skills, resume_text=text)

    st.markdown("### 📊 Match Results")
    st.markdown(f"<div class='score-circle'>{final_score}%</div>", unsafe_allow_html=True)

    if final_score >= 80:
        st.success("✅ Excellent match!")
    elif final_score >= 50:
        st.warning("⚠️ Fair match — improve key skills.")
    else:
        st.error("❌ Weak match — build more relevant skills.")

    with st.expander("🧠 Show Skill Details"):
        st.markdown("✅ **Matched Skills**")
        st.markdown("<div style='display: flex; flex-wrap: wrap; gap: 8px; padding: 8px;'>" + "".join([f"<span class='pill'>{s}</span>" for s in sorted(set(matched))]) + "</div>", unsafe_allow_html=True)

        st.markdown("❌ **Missing Skills**")
        st.markdown("<div style='display: flex; flex-wrap: wrap; gap: 8px; padding: 8px;'>" + "".join([f"<span class='pill'>{s}</span>" for s in sorted(set(missing))]) + "</div>", unsafe_allow_html=True)

    # Grammar Issues
    with st.expander("📝 Grammar Feedback"):
        if grammar_matches:
            st.markdown(f"**Found {len(grammar_matches)} issue(s). Top issues:**")
            for m in grammar_matches[:5]:
                st.markdown(f"""
                <div class='info-box'>
                    <b>Context:</b> {m.context}<br>
                    <b>Message:</b> {m.message}<br>
                    <b>Suggestions:</b> {', '.join(m.replacements) if m.replacements else 'None'}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.success("✅ No major grammar issues detected.")

    st.markdown("<div class='info-box'>💡 Tip: Use Coursera, Udemy, or LinkedIn Learning to build missing skills based on the list above.</div>", unsafe_allow_html=True)

    # === RAG Integration UI
    st.markdown("### 🤖 Ask About Your Resume")
    query = st.text_input("💬 Enter a question (e.g., What are my strengths?)")

    if query:
        with st.spinner("🔍 Querying resume with Gemini..."):
            embedding_model = initialize_embedding_model()
            docs = [text]
            embeddings = get_embeddings(embedding_model, docs)
            faiss_index = create_faiss_index(np.array(embeddings, dtype=np.float32))
            retrieved = retrieve_similar_texts(embedding_model, faiss_index, query, docs)
            response = generate_response(retrieved, query)

        st.markdown("### 💡 Gemini Insight")
        st.markdown(f"<div class='info-box'>{response}</div>", unsafe_allow_html=True)

else:
    st.info("📤 Upload a resume file above to begin.")

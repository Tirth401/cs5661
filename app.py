import streamlit as st
import pandas as pd
import numpy as np
import tempfile
import ast
from collections import Counter
import spacy
from pdfminer.high_level import extract_text

# === Page config
st.set_page_config(page_title="Resume Skill Matcher", layout="centered")

# === Load NLP model
nlp = spacy.load("en_core_web_sm")

# === Load and categorize dataset
@st.cache_data
def load_and_prepare_dataset():
    df = pd.read_csv("resume_data.csv")

    # Map raw positions to 4 standard job titles
    role_keywords = {
        "Data Scientist": ["data", "ml", "ai", "analyst"],
        "Software Engineer": ["software", "developer", "backend", "frontend"],
        "Project Manager": ["project", "manager", "scrum", "agile"],
        "Android Developer": ["android", "mobile", "kotlin", "java"]
    }

    def map_to_title(pos):
        pos_lower = str(pos).lower()
        for title, keywords in role_keywords.items():
            if any(k in pos_lower for k in keywords):
                return title
        return None

    df["Job Title"] = df["positions"].apply(map_to_title)
    return df[df["Job Title"].notnull()]

df = load_and_prepare_dataset()

# === Utility functions
def extract_text_from_pdf_file(file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file.read())
        return extract_text(tmp.name)

def extract_skills_from_resume(text):
    doc = nlp(text)
    return {token.text.lower().strip() for token in doc if token.pos_ in {"NOUN", "PROPN"} and len(token.text) > 2}

def get_required_skills(df, role):
    skills = []
    role_df = df[df["Job Title"] == role]
    for skill_list in role_df["skills"].dropna():
        parsed = ast.literal_eval(skill_list) if isinstance(skill_list, str) else []
        skills.extend([s.strip().lower() for s in parsed if isinstance(s, str)])
    return set(skills)

def get_top_skills_for_role(df, target_role, top_n=30):
    role_df = df[df["Job Title"] == target_role]
    all_skills = []

    for skill_list in role_df["skills"].dropna():
        try:
            parsed = ast.literal_eval(skill_list)
            all_skills.extend(s.strip().lower() for s in parsed if isinstance(s, str))
        except:
            continue

    # Count and return top N most common skills
    counter = Counter(all_skills)
    top_skills = {skill for skill, _ in counter.most_common(top_n)}
    return top_skills


def compute_skill_match(resume_skills, required_skills):
    matched = resume_skills & required_skills
    missing = required_skills - resume_skills
    score = round((len(matched) / max(len(required_skills), 1)) * 100, 2)
    return score, matched, missing

# === Streamlit UI
st.title("🔍 Resume Skill Matcher")
st.markdown("Upload your resume and see how well your skills align with your target job role.")

roles = ["Data Scientist", "Software Engineer", "Project Manager", "Android Developer"]
selected_role = st.selectbox("🎯 Select Job Title", roles)
uploaded_file = st.file_uploader("📄 Upload Resume (PDF)", type=["pdf"])

if uploaded_file:
    with st.spinner("Processing resume..."):
        resume_text = extract_text_from_pdf_file(uploaded_file)
        resume_skills = extract_skills_from_resume(resume_text)
        required_skills = get_top_skills_for_role(df, selected_role, top_n=30)
        score, matched, missing = compute_skill_match(resume_skills, required_skills)

        # === Results
        st.subheader("📊 Skill Match Results")
        st.metric("Match Score", f"{score}%")
        st.markdown(f"✅ **Matched Skills:** {', '.join(matched) if matched else 'None'}")
        st.markdown(f"❌ **Missing Skills:** {', '.join(missing) if missing else 'None'}")

        if score >= 80:
            st.success("Excellent match! You're well-qualified for this role.")
        elif score >= 50:
            st.warning("Fair match. You may want to strengthen a few areas.")
        else:
            st.error("Weak match. Consider building more relevant skills for this role.")

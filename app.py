import streamlit as st
import pandas as pd
import tempfile
import ast
from collections import Counter
from pdfminer.high_level import extract_text

# === Streamlit Page Setup
st.set_page_config(page_title="Resume Skill Matcher", layout="centered")

# === Abbreviation Mapping
ABBREVIATION_MAP = {
    "natural language processing": "nlp",
    "artificial intelligence": "ai",
    "machine learning": "ml",
    "deep learning": "dl"
}

# === Normalize Skills
def normalize_skills(skills):
    normalized = set()
    for skill in skills:
        skill = skill.lower().strip()
        for full, abbr in ABBREVIATION_MAP.items():
            if full in skill:
                normalized.add(abbr)
        normalized.add(skill)
    return normalized

# === Inject full/abbreviation forms into text for inclusive matching
def normalize_text(text):
    text = text.lower()
    for full, abbr in ABBREVIATION_MAP.items():
        text += " " + abbr + " " + full
    return text

# === Load and Prepare Dataset
@st.cache_data
def load_and_prepare_dataset():
    df = pd.read_csv("resume_data.csv")

    role_keywords = {
        "Data Scientist": ["data", "ml", "ai", "analyst"],
        "Software Engineer": ["software", "developer", "backend", "frontend"],
        "Project Manager": ["project", "manager", "scrum", "agile"],
        "Android Developer": ["android", "mobile", "kotlin", "java"]
    }

    def map_to_title(pos_list_str):
        try:
            pos_list = ast.literal_eval(pos_list_str)
            for pos in pos_list:
                pos_lower = pos.lower()
                for title, keywords in role_keywords.items():
                    if any(k in pos_lower for k in keywords):
                        return title
        except:
            return None
        return None

    df["Job Title"] = df["positions"].apply(map_to_title)
    return df[df["Job Title"].notnull()]

df = load_and_prepare_dataset()

# === Extract Text from PDF
def extract_text_from_pdf_file(file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file.read())
        return extract_text(tmp.name)

def get_all_skills_for_role(df, target_role):
    role_df = df[df["Job Title"] == target_role]
    all_skills = set()
    for skill_list in role_df["skills"].dropna():
        try:
            parsed = ast.literal_eval(skill_list)
            all_skills.update(normalize_skills(parsed))
        except:
            continue
    return all_skills

# === Compute Skill Match Based on Full Text
def compute_skill_match_from_text(resume_text, required_skills):
    normalized_text = normalize_text(resume_text)
    matched = {s for s in required_skills if s in normalized_text}
    missing = set(required_skills) - matched
    score = round((len(matched) / max(len(required_skills), 1)) * 100, 2)
    return score, matched, missing

# === UI Styling
st.markdown("""
    <style>
    html, body, .stApp {
        background: linear-gradient(135deg, #1e3c72, #2a5298, #a4508b, #5f0a87);
        background-size: 400% 400%;
        animation: gradientShift 20s ease infinite;
        color: white;
    }

    @keyframes gradientShift {
        0% {background-position: 0% 50%;}
        50% {background-position: 100% 50%;}
        100% {background-position: 0% 50%;}
    }

    h1, h2, h3 {
        text-align: center;
        color: #fff;
    }

    .score-circle {
        width: 160px;
        height: 160px;
        border-radius: 50%;
        background: rgba(255,255,255,0.1);
        backdrop-filter: blur(10px);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 32px;
        font-weight: bold;
        border: 3px solid rgba(255,255,255,0.4);
        margin: 20px auto;
        box-shadow: 0 0 25px rgba(255, 255, 255, 0.2);
    }

    .pill {
        display: inline-block;
        background: rgba(255,255,255,0.15);
        padding: 6px 12px;
        margin: 5px;
        border-radius: 30px;
        font-size: 14px;
    }

    .info-box {
        background: rgba(255,255,255,0.08);
        padding: 15px;
        border-left: 4px solid #f59e0b;
        border-radius: 10px;
        margin-top: 25px;
        font-size: 15px;
    }

    .stTextInput>div>div>input, .stSelectbox>div>div>div>input {
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# === App UI
st.markdown("## 🔍 Resume Skill Matcher")
st.markdown("<p style='text-align:center;'>Upload your resume PDF and compare your skills to top industry roles.</p>", unsafe_allow_html=True)

role = st.selectbox("🎯 Select Job Title", ["Data Scientist", "Software Engineer", "Project Manager", "Android Developer"])
file = st.file_uploader("📄 Upload Resume (PDF)", type=["pdf"])

if file:
    with st.spinner("🧠 Analyzing your resume..."):
        text = extract_text_from_pdf_file(file)
        required_skills = get_all_skills_for_role(df, role)
        score, matched, missing = compute_skill_match_from_text(text, required_skills)

    st.markdown("### 📊 Match Results")
    st.markdown(f"<div class='score-circle'>{score}%</div>", unsafe_allow_html=True)

    if score >= 80:
        st.success("✅ Excellent match!")
    elif score >= 50:
        st.warning("⚠️ Fair match — improve key skills.")
    else:
        st.error("❌ Weak match — build more relevant skills.")

    with st.expander("🧠 Show Skill Details"):
        st.markdown("✅ **Matched Skills**")
        st.markdown("".join([f"<span class='pill'>{s}</span>" for s in sorted(matched)]), unsafe_allow_html=True)

        st.markdown("❌ **Missing Skills**")
        st.markdown("".join([f"<span class='pill'>{s}</span>" for s in sorted(missing)]), unsafe_allow_html=True)

    st.markdown("<div class='info-box'>💡 Tip: Use Coursera, Udemy, or LinkedIn Learning to build missing skills based on the list above.</div>", unsafe_allow_html=True)
else:
    st.info("📤 Upload a resume file above to begin.")

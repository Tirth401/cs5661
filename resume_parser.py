import re
import ast
import pandas as pd
import spacy
from pdfminer.high_level import extract_text

# === Load NLP model
nlp = spacy.load("en_core_web_sm")

# === Extract full text from PDF
def extract_text_from_pdf(file_path):
    return extract_text(file_path)

# === Extract skills from full resume text
def extract_skills_from_text(text):
    doc = nlp(text)
    return {token.text.lower().strip() for token in doc if token.pos_ in {"NOUN", "PROPN"} and len(token.text) > 2}


# === Categorize positions into 4 standard job roles
def map_job_titles(df):
    role_keywords = {
        "Data Scientist": ["data", "ml", "ai", "analyst"],
        "Software Engineer": ["software", "developer", "backend", "frontend"],
        "Project Manager": ["project", "manager", "scrum", "agile"],
        "Android Developer": ["android", "mobile", "kotlin", "java"]
    }

    def map_title(pos):
        pos = str(pos).lower()
        for role, keywords in role_keywords.items():
            if any(k in pos for k in keywords):
                return role
        return None

    df["Job Title"] = df["positions"].apply(map_title)
    return df[df["Job Title"].notnull()]  # Drop rows that don't match any role

# === Aggregate skills for selected job title
def get_skills_for_role(df, target_role):
    role_df = df[df["Job Title"] == target_role]
    skill_set = set()

    for skill_list in role_df["skills"].dropna():
        try:
            parsed = ast.literal_eval(skill_list)
            skill_set.update(s.strip().lower() for s in parsed if isinstance(s, str))
        except:
            continue

    return skill_set 


# === Skill match scoring
def match_skills(resume_skills, role_skills):
    matched = resume_skills & role_skills
    missing = role_skills - resume_skills
    score = round((len(matched) / max(len(role_skills), 1)) * 100, 2)
    return {
        "score": score,
        "matched_skills": sorted(matched),
        "missing_skills": sorted(missing)
    }

import re
import ast
import pandas as pd
import spacy
from pdfminer.high_level import extract_text

nlp = spacy.load("en_core_web_sm")

ABBREVIATION_MAP = {
    "natural language processing": "nlp",
    "artificial intelligence": "ai",
    "machine learning": "ml",
    "deep learning": "dl",
}

def normalize_skills(skills):
    normalized = set()
    for skill in skills:
        skill = skill.lower().strip()
        for full, abbr in ABBREVIATION_MAP.items():
            if full in skill:
                normalized.add(abbr)
        normalized.add(skill)
    return normalized

def normalize_text(text):
    text = text.lower()
    for full, abbr in ABBREVIATION_MAP.items():
        text += f" {abbr} {full}"
    return text

def extract_text_from_pdf(file_path):
    return extract_text(file_path)

def map_job_titles(df):
    role_keywords = {
        "Data Scientist": ["data scientist"],
        "Software Engineer": ["software engineer"],
        "Project Manager": ["project manager"],
        "Android Developer": ["android developer"]
    }

    def map_title(pos_list):
        pos_list = ast.literal_eval(pos_list)
        for pos in pos_list:
            pos_lower = pos.lower()
            for role, patterns in role_keywords.items():
                if any(p in pos_lower for p in patterns):
                    return role
        return None

    df["Job Title"] = df["positions"].apply(map_title)
    return df[df["Job Title"].notnull()]

def get_all_skills_for_role(df, target_role):
    role_df = df[df["Job Title"] == target_role]
    skill_set = set()
    for skill_list in role_df["skills"].dropna():
        try:
            parsed = ast.literal_eval(skill_list)
            skill_set.update(normalize_skills(parsed))
        except:
            continue
    return skill_set

def match_skills_in_text(resume_text, role_skills):
    normalized_text = normalize_text(resume_text)
    matched = {s for s in role_skills if s in normalized_text}
    missing = role_skills - matched
    score = round((len(matched) / max(len(role_skills), 1)) * 100, 2)
    return {
        "score": score,
        "matched_skills": sorted(matched),
        "missing_skills": sorted(missing)
    }

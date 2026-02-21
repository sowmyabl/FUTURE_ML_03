import os
from typing import List

import PyPDF2

# A simple skills vocabulary; extend as needed.
SKILLS_LIST = {
    "python", "java", "c++", "c#", "javascript", "react", "node.js",
    "tensorflow", "pytorch", "sql", "nosql", "mongodb", "docker",
    "kubernetes", "aws", "azure", "gcp", "linux", "git", "html",
    "css", "rest", "graphql", "flask", "django", "scikit-learn",
    "nlp", "computer vision", "pandas", "numpy", "matplotlib",
    "data analysis", "machine learning", "deep learning", "spark",
    "hadoop", "microservices", "api", "redis", "postgresql",
    "mysql", "excel", "power bi", "tableau", "sass", "bootstrap",
}


def read_resume(filepath: str) -> str:
    """Return the text contents of a resume file.

    Supports .txt and .pdf. If the file extension is unrecognized, raises
    a ValueError.
    """

    _, ext = os.path.splitext(filepath)
    ext = ext.lower()
    if ext == ".txt":
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    elif ext == ".pdf":
        text = []
        with open(filepath, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text.append(page.extract_text() or "")
        return "\n".join(text)
    else:
        raise ValueError(f"Unsupported file type: {ext}")


def extract_skills(text: str) -> List[str]:
    """Return a list of skills found in the provided text.

    The matching is case-insensitive and simple word-based; for a real project
    you might use fuzzy matching or a more comprehensive ontology.
    """
    text_lower = text.lower()
    found = [skill for skill in SKILLS_LIST if skill in text_lower]
    return sorted(found)


def missing_skills(resume_skills: List[str], required_skills: List[str]) -> List[str]:
    """Return skills that are in required_skills but not present in resume_skills."""
    return sorted(set(required_skills) - set(resume_skills))

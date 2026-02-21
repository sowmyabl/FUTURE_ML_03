from typing import List, Tuple

import matplotlib.pyplot as plt
import pandas as pd

from model import ResumeModel
from utils import extract_skills, missing_skills


class Ranker:
    def __init__(self, ngram_range: Tuple[int, int] = (1, 2)) -> None:
        self.model = ResumeModel(ngram_range=ngram_range)

    def score_and_rank(
        self,
        resume_texts: List[str],
        job_desc: str,
        candidate_names: List[str],
        raw_texts: List[str] | None = None,
        raw_job: str | None = None,
    ) -> pd.DataFrame:
        """Return a DataFrame containing score, matched and missing skills.

        The returned DataFrame is sorted so the highest scoring candidate is first.

        ``resume_texts`` and ``job_desc`` are expected to be preprocessed versions
        of the inputs (lowercased, lemmatized, etc.) which are used only for
        scoring.  If ``raw_texts`` or ``raw_job`` are provided they will be used
        for skill extraction so that punctuation-based skills like "c++" or
        "c#" are not lost by the cleaning step.
        """
        if len(resume_texts) != len(candidate_names):
            raise ValueError("Each resume must have a corresponding candidate name")

        # scoring always operates on the cleaned/preprocessed texts
        processed_resumes = resume_texts.copy()
        score_list = self.model.batch_score(processed_resumes, job_desc)

        # determine which version to use for skill extraction
        skill_texts = raw_texts if raw_texts is not None else resume_texts
        job_skill_source = raw_job if raw_job is not None else job_desc

        required = extract_skills(job_skill_source)
        rows = []
        for name, text, score in zip(candidate_names, skill_texts, score_list):
            skills = extract_skills(text)
            miss = missing_skills(skills, required)
            rows.append(
                {
                    "name": name,
                    "score": score,
                    "matched_skills": ", ".join(skills),
                    "missing_skills": ", ".join(miss),
                    "num_matched": len(skills),
                    "num_missing": len(miss),
                }
            )

        df = pd.DataFrame(rows)
        df = df.sort_values("score", ascending=False).reset_index(drop=True)
        return df

    # ---------------- plotting helpers ----------------

    @staticmethod
    def plot_scores(df: pd.DataFrame) -> plt.Figure:
        """Bar chart showing candidate scores.

        Uses dark background with yellow bars to match Streamlit theme, adds grid
        lines for readability, and ensures labels remain legible on dark.
        """
        fig, ax = plt.subplots(figsize=(8, 5))
        fig.patch.set_facecolor('#001f3f')
        ax.set_facecolor('#001f3f')
        bars = ax.bar(df["name"], df["score"], color="#ffdc00")
        ax.set_xlabel("Candidate", color="white")
        ax.set_ylabel("Score (%)", color="white")
        ax.set_title("Resume similarity scores", color="white")
        ax.set_xticklabels(df["name"], rotation=45, ha="right", color="white")
        ax.tick_params(colors="white")
        ax.grid(True, linestyle="--", alpha=0.3)
        fig.tight_layout()
        return fig

    @staticmethod
    def plot_skill_match_pie(matched: int, missing: int) -> plt.Figure:
        """Pie chart for one candidate's skill match versus missing.

        Applies a dark face with contrasting labels and uses theme colors.
        """
        fig, ax = plt.subplots(figsize=(4, 4))
        fig.patch.set_facecolor('#001f3f')
        ax.set_facecolor('#001f3f')
        wedges, texts, autotexts = ax.pie(
            [matched, missing],
            labels=["Matched", "Missing"],
            colors=["#2ca02c", "#d62728"],
            autopct="%1.1f%%",
            textprops={"color": "white"},
        )
        for txt in texts + autotexts:
            txt.set_color('white')
        ax.set_title("Skill match percentage", color="white")
        return fig

    @staticmethod
    def plot_missing_skills(missing_skills_list: List[str]) -> plt.Figure:
        """Horizontal bar chart showing which skills are missing.

        Uses a darker background and adds grid lines; bars use an accent orange.
        """
        fig, ax = plt.subplots(figsize=(6, max(2, len(missing_skills_list) * 0.5)))
        fig.patch.set_facecolor('#001f3f')
        ax.set_facecolor('#001f3f')
        counts = [1] * len(missing_skills_list)
        ax.barh(missing_skills_list, counts, color="#ff7f0e")
        ax.set_xlabel("Missing skill (count)", color="white")
        ax.set_title("Skills not found in resume", color="white")
        ax.tick_params(colors="white")
        ax.grid(axis='x', linestyle="--", alpha=0.3)
        fig.tight_layout()
        return fig

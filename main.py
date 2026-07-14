"""Streamlit application for LearnBridge; AI attribution is in README.md."""
from pathlib import Path
import subprocess
import sys

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from learnbridge.app_utils import explanation_mode
from learnbridge.config import FEATURE_FILE, FORMATS, TOPICS
from learnbridge.data_loader import load_resources
from learnbridge.inference import recommend
from learnbridge.metrics import (
    categorical_diversity,
    constraint_satisfaction,
    knowledge_gap_coverage,
    topic_diversity,
)
from learnbridge.schemas import LearnerProfile


@st.cache_resource
def load_system():
    """Load the fitted model and data, generating deterministic artifacts if absent."""
    if not FEATURE_FILE.exists():
        subprocess.run(
            [sys.executable, str(ROOT / "scripts/make_dataset.py")],
            cwd=ROOT,
            check=True,
        )
        subprocess.run(
            [sys.executable, str(ROOT / "scripts/build_features.py")],
            cwd=ROOT,
            check=True,
        )
    return load_resources(), joblib.load(FEATURE_FILE)


def render_recommendation_cards(recommendations: pd.DataFrame) -> None:
    """Render accessible recommendation summaries from dataset metadata."""
    for _, resource in recommendations.iterrows():
        cost = (
            "Free"
            if resource.cost_type == "Free"
            else f"${resource.price_usd:.0f}"
        )
        st.markdown(
            f"""<div class='card'>
            <h3>#{resource['rank']} · {resource.title}</h3>
            <p><b>Match:</b> {resource.match_score:.0%} &nbsp;
            <b>Responsible score:</b> {resource.responsible_score:.0%}</p>
            <p class='small'>{resource.difficulty} · {resource['format']} ·
            {resource.duration_minutes} min · {resource.provider} · {cost}</p>
            <p><b>Topics:</b> {str(resource.topics).replace('|', ', ')}</p>
            <p>{resource.reason}</p>
            <p><b>Knowledge gap addressed:</b>
            {resource.knowledge_gap_addressed}</p>
            <p><b>Responsible note:</b> {resource.responsible_note}</p>
            <a href='{resource.resource_url}'>Open synthetic resource link ↗</a>
            </div>""",
            unsafe_allow_html=True,
        )


def main() -> None:
    """Run the complete Streamlit user interface."""
    st.set_page_config(page_title="LearnBridge", page_icon="🌉", layout="wide")
    st.markdown(
        "<style>.card{padding:1rem;border:1px solid #d9e2ec;border-radius:12px;"
        "margin:.7rem 0;background:#fff}.small{color:#52606d}</style>",
        unsafe_allow_html=True,
    )
    st.title("🌉 LearnBridge")
    st.subheader("Responsible Learning Resource Recommender")
    st.info(
        "Educational prototype: recommendations use synthetic data and do not "
        "replace teachers, academic advisors, or professional training guidance."
    )

    try:
        resources, bundle = load_system()
    except Exception as error:
        st.error(
            f"Could not initialize LearnBridge: {error}. Run `make all` from the "
            "project directory."
        )
        st.stop()

    with st.sidebar:
        st.header("Learner profile")
        goal = st.text_input(
            "Learning goal", "Build practical machine learning systems"
        )
        level = st.selectbox(
            "Current level", ["Beginner", "Intermediate", "Advanced"]
        )
        weak_topics = st.multiselect(
            "Weak or missing topics",
            TOPICS,
            default=["Statistics", "Responsible AI"],
        )
        preferred_formats = st.multiselect(
            "Preferred formats",
            FORMATS,
            default=["Video", "Interactive exercise"],
        )
        weekly_time = st.slider(
            "Weekly study time (minutes)", 30, 600, 180, 30
        )
        free_only = st.checkbox("Free resources only", True)
        captions_required = st.checkbox("Captions required for video", False)
        exploration = st.slider(
            "Exploration",
            0.0,
            1.0,
            0.35,
            0.05,
            help="Higher values favor novelty and diversity over pure relevance.",
        )
        recommendation_count = st.slider("Number of recommendations", 3, 10, 5)
        model_label = st.radio(
            "Recommendation model",
            ["Responsible hybrid", "TF-IDF content", "Popularity baseline"],
        )

    profile = LearnerProfile(
        goal,
        level,
        weak_topics,
        preferred_formats,
        weekly_time,
        free_only,
        captions_required,
        exploration,
    )
    model_name = {
        "Responsible hybrid": "responsible",
        "TF-IDF content": "content",
        "Popularity baseline": "popularity",
    }[model_label]
    recommendations, rejected = recommend(
        resources, profile, bundle, model_name, recommendation_count
    )

    if recommendations.empty:
        st.warning(
            "No candidates satisfy all hard constraints. Try increasing time, "
            "allowing paid resources, or relaxing caption requirements."
        )
    else:
        metric_columns = st.columns(6)
        metrics = [
            ("Avg relevance", recommendations.match_score.mean()),
            ("Topic diversity", topic_diversity(recommendations)),
            (
                "Format diversity",
                categorical_diversity(recommendations["format"].tolist()),
            ),
            (
                "Provider diversity",
                categorical_diversity(recommendations.provider.tolist()),
            ),
            (
                "Constraint satisfaction",
                constraint_satisfaction(recommendations, profile),
            ),
            (
                "Gap coverage",
                knowledge_gap_coverage(recommendations, weak_topics),
            ),
        ]
        for column, (label, value) in zip(metric_columns, metrics):
            column.metric(label, f"{value:.0%}")
        render_recommendation_cards(recommendations)

    st.divider()
    st.subheader("Compare all three systems")
    if st.checkbox("Show same-profile comparison"):
        comparison = []
        for comparison_model in ["popularity", "content", "responsible"]:
            frame, _ = recommend(
                resources, profile, bundle, comparison_model, 5
            )
            comparison.extend(
                {
                    "Model": comparison_model.title(),
                    "Rank": row["rank"],
                    "Resource": row["title"],
                    "Provider": row["provider"],
                    "Score": round(row["responsible_score"], 3),
                }
                for row in frame.to_dict("records")
            )
        st.dataframe(
            pd.DataFrame(comparison), use_container_width=True, hide_index=True
        )

    with st.expander("Transparency: method, constraints, and limitations"):
        st.markdown(
            f"""**How it works.** A fitted TF-IDF model retrieves at least 20
            candidates from the onboarding profile. The responsible system combines
            relevance, gap coverage, quality, accessibility, and diversity, then uses
            MMR and caps each provider at two top-five positions.

            **Constraints applied.** Free-only: `{free_only}`; video captions:
            `{captions_required}`; level: `{level}`; weekly time preference:
            `{weekly_time}` minutes. {len(rejected)} candidates were filtered with
            auditable reasons.

            **Explanation mode.**
            `{explanation_mode(ROOT / 'models/learnbridge-lora')}`. The deployed
            default never loads an LLM.

            **Limitations.** Resources and judgments are synthetic, relevance is
            rule-generated, URLs are illustrative, and no real learner outcome or
            provider quality claim is established."""
        )
        if not rejected.empty:
            st.dataframe(
                rejected[["title", "filter_reason"]].head(20),
                hide_index=True,
                use_container_width=True,
            )


if __name__ == "__main__":
    main()

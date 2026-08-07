"""
Streamlit dashboard for Job Market Analysis.

Run from the repo root:
    streamlit run dashboard.py
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys

import pandas as pd
import streamlit as st

from config import DATA_DIR, GRAPH_DIR, OUTPUT_DIR, REPO_ROOT

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CSV_FILES = {
    "LinkedIn historical": DATA_DIR / "linkedin_historical.csv",
    "LinkedIn (no skills, cleaned)": DATA_DIR / "linkedin_no_skills_cleaned.csv",
    "Indeed webscrape": DATA_DIR / "indeed_webscrape.csv",
}

# Relative paths from repo root → analysis scripts the sidebar can run
SCRIPTS = {
    "Clean LinkedIn dates": "analysis/clean_linkedin.py",
    "Clean skills + TF-IDF plots": "analysis/clean_skills_plot.py",
    "Most common job titles": "analysis/most_common_job_titles.py",
    "Skill demand evolution": "analysis/skill_demand_evolution.py",
    "Top hiring companies": "analysis/top_hiring_companies.py",
    "Top hiring locations": "analysis/top_hiring_locations.py",
    "Hiring trends over time": "analysis/hiring_trends_overtime.py",
    "Forecast job postings": "analysis/forecast_job_postings.py",
}


def run_script(rel_path: str) -> tuple[bool, str]:
    """Run an analysis script as a subprocess from the repo root."""
    path = REPO_ROOT / rel_path
    if not path.exists():
        return False, f"Script not found: {rel_path}"

    try:
        result = subprocess.run(
            [sys.executable, str(path)],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
            timeout=600,
        )
        if result.returncode == 0:
            msg = f"Ran {path.name} successfully"
            if result.stdout.strip():
                msg += f"\n{result.stdout.strip()}"
            return True, msg
        err = result.stderr.strip() or result.stdout.strip() or "Unknown error"
        return False, f"Error running {path.name}:\n{err}"
    except subprocess.TimeoutExpired:
        return False, f"Timed out: {path.name}"
    except Exception as exc:
        return False, f"Failed to run {path.name}: {exc}"


def list_graph_files():
    if not GRAPH_DIR.exists():
        return []
    exts = {".png", ".jpg", ".jpeg", ".svg"}
    files = [f for f in GRAPH_DIR.glob("*") if f.suffix.lower() in exts]
    return sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)


def list_output_files():
    if not OUTPUT_DIR.exists():
        return []
    files = [f for f in OUTPUT_DIR.glob("*") if f.is_file() and f.name != ".gitkeep"]
    return sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Job Market Analysis",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Job Market Analysis")
st.caption(
    "Explore cleaned datasets, run analysis scripts, and review graphs / outputs "
    "generated into graph_dir/ and output_dir/."
)
st.divider()

# Sidebar
st.sidebar.header("Run scripts")
selected = st.sidebar.multiselect(
    "Analysis scripts",
    options=list(SCRIPTS.keys()),
)

if st.sidebar.button("Run selected", type="primary"):
    if not selected:
        st.sidebar.warning("Select at least one script.")
    else:
        for label in selected:
            ok, msg = run_script(SCRIPTS[label])
            if ok:
                st.sidebar.success(msg)
            else:
                st.sidebar.error(msg)

st.sidebar.divider()
st.sidebar.header("Data files")
for name, path in CSV_FILES.items():
    if path.exists():
        size_kb = max(1, path.stat().st_size // 1024)
        st.sidebar.write(f"{name} — {size_kb} KB")
    else:
        st.sidebar.write(f"{name} — missing")

tab_data, tab_graphs, tab_out = st.tabs(["Data explorer", "Graphs", "Outputs"])

with tab_data:
    st.subheader("CSV preview")
    for label, path in CSV_FILES.items():
        st.markdown(f"**{label}**")
        if path.exists():
            try:
                df = pd.read_csv(path, nrows=200)
                st.dataframe(df.head(10), use_container_width=True)
                with st.expander("Describe"):
                    st.write(df.describe(include="all").transpose())
            except Exception as exc:
                st.warning(f"Could not read {path.name}: {exc}")
        else:
            st.info(
                f"`{path.name}` not found in `dat_dir/`. "
                "See dat_dir/README.md for required files."
            )

with tab_graphs:
    st.subheader("Generated graphs")
    graphs = list_graph_files()
    if not graphs:
        st.info("No images in graph_dir/. Run analysis scripts to generate plots.")
    else:
        for img in graphs:
            st.image(str(img), caption=img.name, use_container_width=True)

with tab_out:
    st.subheader("Analysis outputs")
    outputs = list_output_files()
    if not outputs:
        st.info("No files in output_dir/. Run analysis / forecast scripts to generate results.")
    else:
        for out in outputs:
            st.markdown(f"**{out.name}**")
            if out.suffix.lower() == ".csv":
                try:
                    st.dataframe(pd.read_csv(out).head(20), use_container_width=True)
                except Exception as exc:
                    st.warning(f"Could not read {out.name}: {exc}")
            else:
                st.write(f"Size: {max(1, out.stat().st_size // 1024)} KB")

st.divider()
st.caption(f"Repo root: {REPO_ROOT}")

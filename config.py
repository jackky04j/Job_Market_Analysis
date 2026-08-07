"""
Central paths for the Job Market Analysis project.

All scripts should import from here instead of hardcoding local folders:

    from config import DATA_DIR, GRAPH_DIR, OUTPUT_DIR

    df = pd.read_csv(DATA_DIR / "indeed_webscrape.csv")
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

DATA_DIR = REPO_ROOT / "dat_dir"
GRAPH_DIR = REPO_ROOT / "graph_dir"
OUTPUT_DIR = REPO_ROOT / "output_dir"

DATA_DIR.mkdir(exist_ok=True)
GRAPH_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Backwards-compatible aliases
dat_dir = DATA_DIR
graph_dir = GRAPH_DIR
output_dir = OUTPUT_DIR

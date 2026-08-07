# This code displays the most common job titles (top 10).

# Uses: ALL 3 DATASETS (INDEED, LINKEDIN SKILLS, LINKEDIN DATES)

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import matplotlib
matplotlib.use("Agg")
import pandas as pd
import matplotlib.pyplot as plt
from config import DATA_DIR, GRAPH_DIR

# Step 1: Load all three datasets
indeed_data = pd.read_csv(DATA_DIR / "indeed_webscrape.csv")
linkedin_historical_data = pd.read_csv(DATA_DIR / "linkedin_historical.csv")
linkedin_no_skills_data = pd.read_csv(DATA_DIR / "linkedin_no_skills.csv")

# Step 2: Extract job titles from each dataset
indeed_job_titles = indeed_data['job_title']  # Extract from Indeed
linkedin_historical_titles = linkedin_historical_data['job_title']  # Extract from LinkedIn Historical
linkedin_no_skills_titles = linkedin_no_skills_data['title']  # Extract from LinkedIn No Skills

# Step 3: Combine all job titles into one series
all_job_titles = pd.concat([indeed_job_titles, linkedin_historical_titles, linkedin_no_skills_titles])

# Step 4: Count the occurrences of each job title
top_job_titles = all_job_titles.value_counts().head(10)

# Step 5: Plot the top 10 most common job titles
plt.figure(figsize=(12, 6))  # Increase the figure size to accommodate longer labels
top_job_titles.plot(kind='bar', color='purple')
plt.title("Top 10 Job Titles from All Datasets")
plt.xlabel("Job Title")
plt.ylabel("Number of Job Postings")
plt.xticks(rotation=45, ha='right')  # Rotate labels and align them to the right
plt.tight_layout()  # Adjust the layout to ensure everything fits
plt.savefig(GRAPH_DIR / "Top10JobTitles.png", dpi=300, bbox_inches="tight")
plt.close()

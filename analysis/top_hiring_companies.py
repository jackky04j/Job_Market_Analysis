# TOP HIRING COMPANIES
# USES ALL 3 DATASETS (INDEED, LINKEDIN SKILLS, LINKEDIN DATES)

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

# Step 2: Extract company names from each dataset
indeed_companies = indeed_data['company']  # Extract from Indeed
linkedin_historical_companies = linkedin_historical_data['company']  # Extract from LinkedIn Historical
linkedin_no_skills_companies = linkedin_no_skills_data['company']  # Extract from LinkedIn No Skills

# Step 3: Combine all companies into one series
all_companies = pd.concat([indeed_companies, linkedin_historical_companies, linkedin_no_skills_companies])

# Step 4: Count the occurrences of each company
top_companies = all_companies.value_counts().head(10)

# Step 5: Plot the top 10 hiring companies
plt.figure(figsize=(12, 6))  # Increase the figure size to accommodate longer labels
top_companies.plot(kind='bar', color='orange')
plt.title("Top 10 Hiring Companies from All Datasets")
plt.xlabel("Company")
plt.ylabel("Number of Job Postings")
plt.xticks(rotation=45, ha='right')  # Rotate labels and align them to the right
plt.tight_layout()  # Adjust the layout to ensure everything fits
plt.savefig(GRAPH_DIR / "Top10HiringCompanies.png", dpi=300, bbox_inches="tight")
plt.close()

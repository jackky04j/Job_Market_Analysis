import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import matplotlib
matplotlib.use("Agg")
import pandas as pd
import matplotlib.pyplot as plt
from config import DATA_DIR, GRAPH_DIR, OUTPUT_DIR

# Step 1: Load cleaned datasets
indeed_data = pd.read_csv(DATA_DIR / "indeed_webscrape_cleaned.csv")
linkedin_data = pd.read_csv(DATA_DIR / "linkedin_historical_cleaned.csv")

# Step 2: Define a list of terms or patterns to exclude (e.g., company names, locations, etc.)
exclude_terms = ['inc.', 'labs', 'corp', 'company', 'university', 'research', 'technologies', 'manager', 'director']

# Step 3: Clean the skills data
def extract_skills(skill_column):
    """Splits comma-separated skill strings into a list of individual skills and filters out unwanted terms."""
    all_skills = []
    for skills in skill_column.dropna():
        # Split skills and clean them
        skill_list = [skill.strip().lower() for skill in skills.split(',')]
        # Filter out unwanted terms (e.g., company names, generic terms)
        filtered_skills = [skill for skill in skill_list if not any(exclude_term in skill for exclude_term in exclude_terms)]
        all_skills.extend(filtered_skills)
    return pd.Series(all_skills).value_counts()

# Compute skill frequencies for Indeed and LinkedIn data
indeed_skill_counts = extract_skills(indeed_data['skills'])
linkedin_skill_counts = extract_skills(linkedin_data['job_skills'])

# Step 4: Merge skill frequencies into a single DataFrame
skill_trends = pd.DataFrame({
    'LinkedIn (Historical)': linkedin_skill_counts,
    'Indeed (Recent)': indeed_skill_counts
}).fillna(0)  # Fill missing values with 0 (skills that didn't appear in one dataset)

# Step 5: Calculate percentage change in skill demand
skill_trends['% Change'] = ((skill_trends['Indeed (Recent)'] - skill_trends['LinkedIn (Historical)']) /
                            skill_trends['LinkedIn (Historical)'].replace(0, 1)) * 100  # Avoid division by zero

# Step 6: Identify rising skills based on the change percentage
rising_skills = skill_trends.sort_values('% Change', ascending=False).head(10)

# Remove '(required)' from skill labels
rising_skills.index = rising_skills.index.str.replace(r'\s*\(required\)', '', regex=True).str.strip()
rising_skills.index = rising_skills.index.str.replace(r'\s*matching qualification', '', regex=True).str.strip()


# Step 7: Plot skill trends (only rising skills)
plt.figure(figsize=(14, 8))  # Increased height for better spacing
plt.yticks(fontsize=8)  # Reduce font size of y-axis labels
plt.barh(rising_skills.index, rising_skills['% Change'], color='green')
plt.xlabel('% Change in Demand')
plt.title('Top 10 Emerging Skills')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(GRAPH_DIR / "Top10EmergingSkills.png", dpi=300, bbox_inches="tight")
plt.close()

# Save results
skill_trends.to_csv(OUTPUT_DIR / "skill_demand_trends.csv")
rising_skills.to_csv(OUTPUT_DIR / "top_rising_skills.csv")

print("Skill demand evolution analysis completed! Results saved as CSV files.")

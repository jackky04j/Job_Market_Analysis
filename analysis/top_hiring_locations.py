# TOP HIRING LOCATIONS
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

# Step 2: Extract location names from each dataset
indeed_locations = indeed_data['location']  # Extract from Indeed
linkedin_historical_locations = linkedin_historical_data['job_location']  # Extract from LinkedIn Historical
linkedin_no_skills_locations = linkedin_no_skills_data['location']  # Extract from LinkedIn No Skills

# Step 3: Combine all locations into one series
all_locations = pd.concat([indeed_locations, linkedin_historical_locations, linkedin_no_skills_locations])

# Step 4: List of country names to exclude
country_names = ['united states', 'canada', 'united kingdom', 'australia', 'germany', 'france', 'india', 'china']

# Step 5: Filter out locations that match any country name
filtered_locations = all_locations[~all_locations.str.lower().isin(country_names)]

# Step 6: Count the occurrences of each location
top_locations = filtered_locations.value_counts().head(10)

# Step 7: Plot the top 10 hiring locations
plt.figure(figsize=(12, 6))  # Increase the figure size to accommodate longer labels
top_locations.plot(kind='bar', color='green')
plt.title("Top 10 Hiring Locations from All Datasets")
plt.xlabel("Location")
plt.ylabel("Number of Job Postings")
plt.xticks(rotation=45, ha='right')  # Rotate labels and align them to the right
plt.tight_layout()  # Adjust the layout to ensure everything fits
plt.savefig(GRAPH_DIR / "Top10HiringLocations.png", dpi=300, bbox_inches="tight")
plt.close()

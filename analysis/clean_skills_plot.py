import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import matplotlib
matplotlib.use("Agg")
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from config import DATA_DIR, GRAPH_DIR

# Step 1: Load the CSV files into pandas DataFrames
indeed_data = pd.read_csv(DATA_DIR / "indeed_webscrape.csv")
linkedin_data = pd.read_csv(DATA_DIR / "linkedin_historical.csv")

# Step 2: Clean and preprocess Indeed data
# Remove any rows where 'skills' column is NaN or blank
indeed_data['skills'] = indeed_data['skills'].fillna('')  # Fill NaN values with empty string
indeed_data = indeed_data[indeed_data['skills'].str.strip() != '']  # Remove rows with blank skills

# Group Indeed data by 'job_title-href' and aggregate other columns
indeed_data_cleaned = indeed_data.groupby(['job_title-href'], as_index=False).agg({
    'job_title': 'first',  # Keep the first occurrence of job_title for each 'job_title-href'
    'location': 'first',   # Keep the first occurrence of location for each 'job_title-href'
    'company': 'first',    # Keep the first occurrence of company for each 'job_title-href'
    # Combine unique skills into a comma-separated list
    'skills': lambda x: ', '.join(x.dropna().unique())  
})

# Save cleaned Indeed data to CSV
indeed_data_cleaned.to_csv(DATA_DIR / "indeed_webscrape_cleaned.csv", index=False)

# Step 3: Clean and preprocess LinkedIn data
linkedin_data = linkedin_data.drop_duplicates()  # Remove duplicate rows
linkedin_data['job_summary'] = linkedin_data['job_summary'].fillna("No summary available")  # Fill missing job summaries
linkedin_data = linkedin_data.dropna(subset=['job_title', 'company', 'job_skills'])  # Drop rows with missing critical columns

# Analysis Process

# Initialize TF-IDF Vectorizer
# TF-IDF is a technique used in NLP to convert textual data 
# into numerical form for machine learning models

# The TfidfVectorizer is initialized with stop_words='english' 
# and max_features=10 to focus on the most important words
# It transforms the skills data into a numerical representation where 
# more important or unique skills get higher weights.
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=10)

# Step 4: Apply TF-IDF on Indeed data
indeed_tfidf_matrix = tfidf_vectorizer.fit_transform(indeed_data_cleaned['skills'])

# Step 5: Apply TF-IDF on LinkedIn data
linkedin_tfidf_matrix = tfidf_vectorizer.fit_transform(linkedin_data['job_skills'])

# Step 6: Get the top terms (skills)
# calculates the sum of TF-IDF scores for each skill across all job postings
# and extracts the top 10 skills with the highest scores.
top_skills_indeed_idx = indeed_tfidf_matrix.sum(axis=0).A1.argsort()[-10:][::-1]  # Get indices of top 10 skills
top_skills_linkedin_idx = linkedin_tfidf_matrix.sum(axis=0).A1.argsort()[-10:][::-1]  # Get indices of top 10 skills

top_skills_indeed = [tfidf_vectorizer.get_feature_names_out()[i] for i in top_skills_indeed_idx]
top_skills_linkedin = [tfidf_vectorizer.get_feature_names_out()[i] for i in top_skills_linkedin_idx]

top_skills_indeed_values = indeed_tfidf_matrix.sum(axis=0).A1[top_skills_indeed_idx]
top_skills_linkedin_values = linkedin_tfidf_matrix.sum(axis=0).A1[top_skills_linkedin_idx]

# Step 7: Plot the results

# Plot for Indeed data
plt.figure(figsize=(10, 6))
plt.barh(top_skills_indeed, top_skills_indeed_values, color='skyblue')
plt.title('Top Skills in Indeed Job Postings')
plt.xlabel('Sum of TF-IDF')
plt.ylabel('Skills')
plt.gca().invert_yaxis()  # To have the highest value at the top
plt.tight_layout()
plt.savefig(GRAPH_DIR / "TopSkillsIndeedJobPostings.png", dpi=300, bbox_inches="tight")
plt.close()

# Plot for LinkedIn data
plt.figure(figsize=(10, 6))
plt.barh(top_skills_linkedin, top_skills_linkedin_values, color='lightgreen')
plt.title('Top Skills in LinkedIn Job Postings')
plt.xlabel('Sum of TF-IDF')
plt.ylabel('Skills')
plt.gca().invert_yaxis()  # To have the highest value at the top
plt.tight_layout()
plt.savefig(GRAPH_DIR / "TopSkillsLinkedInJobPostings.png", dpi=300, bbox_inches="tight")
plt.close()

# Save cleaned LinkedIn data to CSV
linkedin_data.to_csv(DATA_DIR / "linkedin_historical_cleaned.csv", index=False)

print("Data cleaning and analysis completed! Results saved as CSV files.")

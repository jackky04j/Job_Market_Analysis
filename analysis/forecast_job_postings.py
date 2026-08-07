# This code provides a prediction of job postings for the
# next few months provided the data.

# ML

# Uses: linkedin_no_skills_cleaned (since we need the dates)
# Produces:

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import matplotlib
matplotlib.use("Agg")
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from config import DATA_DIR, GRAPH_DIR, OUTPUT_DIR

# Load your LinkedIn data
linkedin_recent = pd.read_csv(DATA_DIR / "linkedin_no_skills_cleaned.csv")

# Convert datePosted to datetime
linkedin_recent['datePosted'] = pd.to_datetime(linkedin_recent['datePosted'])

# Aggregate job postings per month
linkedin_recent['year_month'] = linkedin_recent['datePosted'].dt.to_period('M')
job_trends = linkedin_recent.groupby('year_month').size().reset_index(name='job_postings')

# Convert to datetime
job_trends['ds'] = job_trends['year_month'].astype(str)
job_trends['ds'] = pd.to_datetime(job_trends['ds'])
job_trends = job_trends[['ds', 'job_postings']]

# Plot historical job postings
plt.figure(figsize=(10, 5))
plt.plot(job_trends['ds'], job_trends['job_postings'], marker='o', label='Actual Job Postings')
plt.title("Historical Job Postings")
plt.xlabel("Date")
plt.ylabel("Number of Job Postings")
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig(GRAPH_DIR / "HistoricalJobPostings.png", dpi=300, bbox_inches="tight")
plt.close()

# Build the ARIMA model
# p = trend lag, d = differencing, q = error term lag
model = ARIMA(job_trends['job_postings'], order=(2, 1, 2))  # This combo is very stable
model_fit = model.fit()

# Forecast the next 6 months
forecast = model_fit.forecast(steps=6)

# Create a future dataframe
future_dates = pd.date_range(start=job_trends['ds'].max(), periods=7, freq='M')[1:]
forecast_df = pd.DataFrame({'ds': future_dates, 'job_postings': forecast})

# Plot the forecast
plt.figure(figsize=(10, 5))
plt.plot(job_trends['ds'], job_trends['job_postings'], marker='o', label='Actual Job Postings')
plt.plot(forecast_df['ds'], forecast_df['job_postings'], marker='x', color='red', label='Predicted Job Postings')
plt.title("Predicted Job Postings (Next 6 Months)")
plt.xlabel("Date")
plt.ylabel("Number of Job Postings")
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig(GRAPH_DIR / "PredictedJobPostings6Months.png", dpi=300, bbox_inches="tight")
plt.close()

# Save predictions to CSV
forecast_df.to_csv(OUTPUT_DIR / "arima_job_postings.csv", index=False)

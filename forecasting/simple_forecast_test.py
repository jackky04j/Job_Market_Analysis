"""
Simple Forecast Testing Script for Small Datasets
This script is designed to work with limited data and provides basic accuracy metrics.
"""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
from config import DATA_DIR, GRAPH_DIR, OUTPUT_DIR
warnings.filterwarnings('ignore')

def simple_forecast_test(data, test_size=1):
    """
    Simple forecast testing for small datasets
    """
    print("="*60)
    print("SIMPLE FORECAST ACCURACY TEST")
    print("="*60)
    
    n = len(data)
    print(f"Dataset size: {n} data points")
    
    if n < 3:
        print("ERROR: Dataset too small for forecasting (need at least 3 data points)")
        return None
    
    # Adjust test size for small datasets
    if n < 6:
        test_size = 1
        print("WARNING: Small dataset: Using test_size = 1")
    
    results = []
    
    # Test different models
    models_to_test = []
    
    # Linear Regression (most robust for small datasets)
    if n >= 3:
        models_to_test.append(('linear', 'Linear Regression'))
    
    # ARIMA (if enough data)
    if n >= 4:
        models_to_test.append(('arima', 'ARIMA(1,1,1)'))
    
    for model_type, model_name in models_to_test:
        print(f"\nTesting {model_name}...")
        
        try:
            # Split data
            train_data = data[:-test_size]
            test_data = data[-test_size:]
            
            if model_type == 'linear':
                # Linear regression forecast
                X = np.arange(len(train_data)).reshape(-1, 1)
                y = train_data
                
                model = LinearRegression()
                model.fit(X, y)
                
                # Forecast
                future_X = np.arange(len(train_data), len(train_data) + test_size).reshape(-1, 1)
                forecast = model.predict(future_X)
                
            elif model_type == 'arima':
                # ARIMA forecast
                model = ARIMA(train_data, order=(1, 1, 1))
                model_fit = model.fit()
                forecast = model_fit.forecast(steps=test_size)
            
            # Calculate metrics
            mae = mean_absolute_error(test_data, forecast)
            rmse = np.sqrt(mean_squared_error(test_data, forecast))
            mape = np.mean(np.abs((test_data - forecast) / test_data)) * 100
            
            # Calculate accuracy ratio
            accuracy_ratio = np.mean(forecast / test_data)
            
            # Calculate bias
            bias = np.mean(forecast - test_data)
            
            results.append({
                'model': model_name,
                'mae': mae,
                'rmse': rmse,
                'mape': mape,
                'accuracy_ratio': accuracy_ratio,
                'bias': bias,
                'actual': test_data.tolist(),
                'predicted': forecast.tolist()
            })
            
            print(f"  SUCCESS - MAE: {mae:.2f}")
            print(f"  SUCCESS - RMSE: {rmse:.2f}")
            print(f"  SUCCESS - MAPE: {mape:.2f}%")
            print(f"  SUCCESS - Accuracy Ratio: {accuracy_ratio:.3f}")
            print(f"  SUCCESS - Bias: {bias:.2f}")
            
        except Exception as e:
            print(f"  ERROR: {str(e)}")
            continue
    
    return results

def plot_simple_results(results, data):
    """
    Plot simple forecast results
    """
    if not results:
        print("No results to plot")
        return
    
    fig, axes = plt.subplots(1, len(results), figsize=(6*len(results), 4))
    if len(results) == 1:
        axes = [axes]
    
    for i, result in enumerate(results):
        actual = result['actual']
        predicted = result['predicted']
        
        # Create time axis
        time_axis = np.arange(len(actual))
        
        axes[i].plot(time_axis, actual, 'o-', label='Actual', linewidth=2, markersize=8)
        axes[i].plot(time_axis, predicted, 's-', label='Predicted', linewidth=2, markersize=8)
        axes[i].set_title(f'{result["model"]}\nMAE: {result["mae"]:.1f}, MAPE: {result["mape"]:.1f}%')
        axes[i].set_xlabel('Time Steps')
        axes[i].set_ylabel('Job Postings')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(GRAPH_DIR / "simple_forecast_results.png", dpi=300, bbox_inches="tight")
    plt.close()

def generate_simple_report(results):
    """
    Generate simple performance report
    """
    if not results:
        print("No results to report")
        return
    
    print("\n" + "="*60)
    print("PERFORMANCE SUMMARY")
    print("="*60)
    
    # Find best model
    best_model = min(results, key=lambda x: x['mae'])
    
    print(f"\nBEST MODEL: {best_model['model']}")
    print(f"   MAE: {best_model['mae']:.2f}")
    print(f"   RMSE: {best_model['rmse']:.2f}")
    print(f"   MAPE: {best_model['mape']:.2f}%")
    print(f"   Accuracy Ratio: {best_model['accuracy_ratio']:.3f}")
    print(f"   Bias: {best_model['bias']:.2f}")
    
    # Performance interpretation
    mape = best_model['mape']
    print(f"\nPERFORMANCE INTERPRETATION:")
    if mape < 10:
        print("   EXCELLENT: Excellent accuracy (MAPE < 10%)")
    elif mape < 20:
        print("   GOOD: Good accuracy (MAPE < 20%)")
    elif mape < 30:
        print("   MODERATE: Moderate accuracy (MAPE < 30%)")
    else:
        print("   POOR: Poor accuracy (MAPE > 30%)")
    
    # Accuracy ratio interpretation
    ratio = best_model['accuracy_ratio']
    print(f"\nACCURACY RATIO: {ratio:.3f}")
    if 0.9 <= ratio <= 1.1:
        print("   EXCELLENT: Very accurate predictions")
    elif 0.8 <= ratio <= 1.2:
        print("   GOOD: Good predictions")
    elif 0.7 <= ratio <= 1.3:
        print("   MODERATE: Moderate predictions")
    else:
        print("   POOR: Poor predictions")
    
    # Bias interpretation
    bias = best_model['bias']
    print(f"\nBIAS: {bias:.2f}")
    if abs(bias) < 10:
        print("   LOW: Low bias - predictions are well-centered")
    elif abs(bias) < 50:
        print("   MODERATE: Moderate bias - some systematic error")
    else:
        print("   HIGH: High bias - significant systematic error")
    
    return best_model

def main():
    """
    Main function for simple forecast testing
    """
    print("Loading LinkedIn job postings data...")
    
    try:
        df = pd.read_csv(DATA_DIR / "linkedin_no_skills_cleaned.csv")
        print(f"Loaded {len(df)} job postings")
    except FileNotFoundError:
        print(f"ERROR: {DATA_DIR / 'linkedin_no_skills_cleaned.csv'} not found!")
        return
    
    # Prepare data
    df['datePosted'] = pd.to_datetime(df['datePosted'])
    df['year_month'] = df['datePosted'].dt.to_period('M')
    job_trends = df.groupby('year_month').size().reset_index(name='job_postings')
    
    print(f"Time series data points: {len(job_trends)}")
    print(f"Date range: {job_trends['year_month'].min()} to {job_trends['year_month'].max()}")
    
    # Check data size
    if len(job_trends) < 6:
        print("\nWARNING: Small dataset detected!")
        print("   - For reliable forecasting, you need at least 12+ months of data")
        print("   - Current results may not be representative of model performance")
        print("   - Consider collecting more historical data for better accuracy")
        print("   - This test will provide basic accuracy metrics for available data")
    
    # Run simple forecast test
    results = simple_forecast_test(job_trends['job_postings'].values)
    
    if not results:
        print("ERROR: No valid results obtained!")
        return
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_simple_results(results, job_trends['job_postings'].values)
    
    # Generate report
    best_model = generate_simple_report(results)
    
    # Save results
    if results:
        results_df = pd.DataFrame(results)
        results_df.to_csv(OUTPUT_DIR / "simple_forecast_results.csv", index=False)
        print(f"\nResults saved to '{OUTPUT_DIR / 'simple_forecast_results.csv'}'")
        print(f"Visualization saved to '{GRAPH_DIR / 'simple_forecast_results.png'}'")
    
    print("\n" + "="*60)
    print("SIMPLE FORECAST TESTING COMPLETED")
    print("="*60)

if __name__ == "__main__":
    main()

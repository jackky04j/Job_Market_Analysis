"""
Comprehensive Testing Script for Job Postings Forecast Model
This script evaluates the accuracy and precision of the ARIMA forecasting model
using time series cross-validation and multiple performance metrics.
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
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
from config import DATA_DIR, GRAPH_DIR, OUTPUT_DIR
warnings.filterwarnings('ignore')

def calculate_mape(actual, predicted):
    """Calculate Mean Absolute Percentage Error"""
    return np.mean(np.abs((actual - predicted) / actual)) * 100

def calculate_smape(actual, predicted):
    """Calculate Symmetric Mean Absolute Percentage Error"""
    return np.mean(2 * np.abs(actual - predicted) / (np.abs(actual) + np.abs(predicted))) * 100

def time_series_cv(data, order=(2, 1, 2), n_splits=5, test_size=3):
    """
    Perform time series cross-validation
    """
    results = []
    n = len(data)
    
    # Create different train/test splits
    for i in range(n_splits):
        # Calculate split point
        split_point = n - test_size - (n_splits - i - 1) * 2
        
        if split_point < 12:  # Need at least 12 months for training
            continue
            
        train_data = data[:split_point]
        test_data = data[split_point:split_point + test_size]
        
        try:
            # Fit ARIMA model
            model = ARIMA(train_data, order=order)
            model_fit = model.fit()
            
            # Make predictions
            forecast = model_fit.forecast(steps=test_size)
            
            # Calculate metrics
            mae = mean_absolute_error(test_data, forecast)
            rmse = np.sqrt(mean_squared_error(test_data, forecast))
            mape = calculate_mape(test_data, forecast)
            smape = calculate_smape(test_data, forecast)
            
            results.append({
                'split': i + 1,
                'train_size': len(train_data),
                'test_size': len(test_data),
                'mae': mae,
                'rmse': rmse,
                'mape': mape,
                'smape': smape,
                'actual': test_data.tolist(),
                'predicted': forecast.tolist()
            })
            
        except Exception as e:
            print(f"Error in split {i+1}: {str(e)}")
            continue
    
    return results

def test_different_arima_orders(data):
    """
    Test different ARIMA parameter combinations
    """
    orders = [
        (1, 1, 1), (1, 1, 2), (1, 2, 1), (1, 2, 2),
        (2, 1, 1), (2, 1, 2), (2, 2, 1), (2, 2, 2),
        (3, 1, 1), (3, 1, 2), (3, 2, 1), (3, 2, 2)
    ]
    
    best_order = None
    best_score = float('inf')
    order_results = []
    
    for order in orders:
        try:
            # Use a simple train/test split for parameter selection
            split_point = int(len(data) * 0.8)
            train_data = data[:split_point]
            test_data = data[split_point:]
            
            model = ARIMA(train_data, order=order)
            model_fit = model.fit()
            forecast = model_fit.forecast(steps=len(test_data))
            
            mae = mean_absolute_error(test_data, forecast)
            rmse = np.sqrt(mean_squared_error(test_data, forecast))
            mape = calculate_mape(test_data, forecast)
            
            order_results.append({
                'order': order,
                'mae': mae,
                'rmse': rmse,
                'mape': mape
            })
            
            if mae < best_score:
                best_score = mae
                best_order = order
                
        except Exception as e:
            print(f"Error with order {order}: {str(e)}")
            continue
    
    return best_order, order_results

def plot_forecast_accuracy(results, title="Forecast Accuracy Analysis"):
    """
    Create visualization of forecast accuracy
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # MAE across splits
    axes[0, 0].bar([r['split'] for r in results], [r['mae'] for r in results])
    axes[0, 0].set_title('Mean Absolute Error by Split')
    axes[0, 0].set_xlabel('Split')
    axes[0, 0].set_ylabel('MAE')
    
    # RMSE across splits
    axes[0, 1].bar([r['split'] for r in results], [r['rmse'] for r in results])
    axes[0, 1].set_title('Root Mean Square Error by Split')
    axes[0, 1].set_xlabel('Split')
    axes[0, 1].set_ylabel('RMSE')
    
    # MAPE across splits
    axes[1, 0].bar([r['split'] for r in results], [r['mape'] for r in results])
    axes[1, 0].set_title('Mean Absolute Percentage Error by Split')
    axes[1, 0].set_xlabel('Split')
    axes[1, 0].set_ylabel('MAPE (%)')
    
    # SMAPE across splits
    axes[1, 1].bar([r['split'] for r in results], [r['smape'] for r in results])
    axes[1, 1].set_title('Symmetric Mean Absolute Percentage Error by Split')
    axes[1, 1].set_xlabel('Split')
    axes[1, 1].set_ylabel('SMAPE (%)')
    
    plt.tight_layout()
    plt.savefig(GRAPH_DIR / "forecast_accuracy_analysis.png", dpi=300, bbox_inches="tight")
    plt.close()

def plot_prediction_vs_actual(results):
    """
    Plot actual vs predicted values for each split
    """
    n_splits = len(results)
    fig, axes = plt.subplots(1, n_splits, figsize=(5*n_splits, 4))
    
    if n_splits == 1:
        axes = [axes]
    
    for i, result in enumerate(results):
        actual = result['actual']
        predicted = result['predicted']
        
        axes[i].plot(actual, 'o-', label='Actual', linewidth=2, markersize=6)
        axes[i].plot(predicted, 's-', label='Predicted', linewidth=2, markersize=6)
        axes[i].set_title(f'Split {result["split"]} (MAE: {result["mae"]:.2f})')
        axes[i].set_xlabel('Time Steps')
        axes[i].set_ylabel('Job Postings')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(GRAPH_DIR / "prediction_vs_actual.png", dpi=300, bbox_inches="tight")
    plt.close()

def generate_performance_report(results, order_results):
    """
    Generate a comprehensive performance report
    """
    print("="*60)
    print("FORECAST MODEL PERFORMANCE REPORT")
    print("="*60)
    
    # Overall performance metrics
    avg_mae = np.mean([r['mae'] for r in results])
    avg_rmse = np.mean([r['rmse'] for r in results])
    avg_mape = np.mean([r['mape'] for r in results])
    avg_smape = np.mean([r['smape'] for r in results])
    
    print(f"\nOVERALL PERFORMANCE METRICS:")
    print(f"Average MAE: {avg_mae:.2f}")
    print(f"Average RMSE: {avg_rmse:.2f}")
    print(f"Average MAPE: {avg_mape:.2f}%")
    print(f"Average SMAPE: {avg_smape:.2f}%")
    
    # Performance interpretation
    print(f"\nPERFORMANCE INTERPRETATION:")
    if avg_mape < 10:
        print("✓ Excellent forecasting accuracy (MAPE < 10%)")
    elif avg_mape < 20:
        print("✓ Good forecasting accuracy (MAPE < 20%)")
    elif avg_mape < 30:
        print("⚠ Moderate forecasting accuracy (MAPE < 30%)")
    else:
        print("✗ Poor forecasting accuracy (MAPE > 30%)")
    
    # Model stability
    mae_std = np.std([r['mae'] for r in results])
    print(f"\nMODEL STABILITY:")
    print(f"MAE Standard Deviation: {mae_std:.2f}")
    if mae_std < avg_mae * 0.2:
        print("✓ Model shows good stability across different time periods")
    else:
        print("⚠ Model shows high variability across different time periods")
    
    # Best ARIMA order
    if order_results:
        best_order = min(order_results, key=lambda x: x['mae'])
        print(f"\nBEST ARIMA PARAMETERS:")
        print(f"Order: {best_order['order']}")
        print(f"MAE: {best_order['mae']:.2f}")
        print(f"RMSE: {best_order['rmse']:.2f}")
        print(f"MAPE: {best_order['mape']:.2f}%")
    
    # Detailed split results
    print(f"\nDETAILED SPLIT RESULTS:")
    for result in results:
        print(f"Split {result['split']}: MAE={result['mae']:.2f}, RMSE={result['rmse']:.2f}, MAPE={result['mape']:.2f}%")
    
    return {
        'avg_mae': avg_mae,
        'avg_rmse': avg_rmse,
        'avg_mape': avg_mape,
        'avg_smape': avg_smape,
        'mae_std': mae_std,
        'best_order': best_order if order_results else None
    }

def main():
    """
    Main function to run the forecast accuracy testing
    """
    print("Loading LinkedIn job postings data...")
    
    # Load the data
    try:
        df = pd.read_csv(DATA_DIR / "linkedin_no_skills_cleaned.csv")
        print(f"Loaded {len(df)} job postings")
    except FileNotFoundError:
        print(f"Error: {DATA_DIR / 'linkedin_no_skills_cleaned.csv'} not found!")
        print("Please make sure the file exists in the data directory.")
        return
    
    # Convert datePosted to datetime
    df['datePosted'] = pd.to_datetime(df['datePosted'])
    
    # Aggregate job postings per month
    df['year_month'] = df['datePosted'].dt.to_period('M')
    job_trends = df.groupby('year_month').size().reset_index(name='job_postings')
    
    print(f"Time series data points: {len(job_trends)}")
    print(f"Date range: {job_trends['year_month'].min()} to {job_trends['year_month'].max()}")
    
    # Extract the time series
    ts_data = job_trends['job_postings'].values
    
    print("\nTesting different ARIMA parameter combinations...")
    best_order, order_results = test_different_arima_orders(ts_data)
    
    print(f"\nBest ARIMA order found: {best_order}")
    
    print("\nPerforming time series cross-validation...")
    cv_results = time_series_cv(ts_data, order=best_order, n_splits=5, test_size=3)
    
    if not cv_results:
        print("Error: No valid cross-validation results obtained!")
        return
    
    print(f"Completed {len(cv_results)} cross-validation splits")
    
    # Generate visualizations
    print("\nGenerating accuracy visualizations...")
    plot_forecast_accuracy(cv_results)
    plot_prediction_vs_actual(cv_results)
    
    # Generate performance report
    print("\nGenerating performance report...")
    performance_metrics = generate_performance_report(cv_results, order_results)
    
    # Save results to CSV
    results_df = pd.DataFrame(cv_results)
    results_df.to_csv(OUTPUT_DIR / "forecast_accuracy_results.csv", index=False)
    print(f"\nResults saved to '{OUTPUT_DIR / 'forecast_accuracy_results.csv'}'")
    
    # Save performance summary
    summary_df = pd.DataFrame([performance_metrics])
    summary_df.to_csv(OUTPUT_DIR / "forecast_performance_summary.csv", index=False)
    print(f"Performance summary saved to '{OUTPUT_DIR / 'forecast_performance_summary.csv'}'")
    
    print("\n" + "="*60)
    print("FORECAST ACCURACY TESTING COMPLETED")
    print("="*60)

if __name__ == "__main__":
    main()

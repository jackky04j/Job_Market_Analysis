"""
Advanced Forecasting Model Testing Script
This script provides more sophisticated testing including:
- Multiple model comparison (ARIMA, Exponential Smoothing, Linear Regression)
- Advanced accuracy metrics (Theil's U, Directional Accuracy)
- Statistical significance testing
- Confidence intervals
- Model diagnostics
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
try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
except ImportError:
    try:
        from statsmodels.tsa.exponential_smoothing import ExponentialSmoothing
    except ImportError:
        print("Warning: ExponentialSmoothing not available. Some tests will be skipped.")
        ExponentialSmoothing = None

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy import stats
import warnings
from config import DATA_DIR, GRAPH_DIR, OUTPUT_DIR
warnings.filterwarnings('ignore')

class AdvancedForecastTester:
    def __init__(self, data):
        self.data = data
        self.results = {}
        
    def calculate_theils_u(self, actual, predicted):
        """Calculate Theil's U statistic (forecast accuracy measure)"""
        numerator = np.sqrt(np.mean((actual - predicted) ** 2))
        denominator = np.sqrt(np.mean(actual ** 2)) + np.sqrt(np.mean(predicted ** 2))
        return numerator / denominator if denominator != 0 else np.inf
    
    def calculate_directional_accuracy(self, actual, predicted):
        """Calculate directional accuracy (percentage of correct direction changes)"""
        if len(actual) < 2 or len(predicted) < 2:
            return 0
        
        actual_direction = np.diff(actual) > 0
        predicted_direction = np.diff(predicted) > 0
        return np.mean(actual_direction == predicted_direction) * 100
    
    def calculate_bias(self, actual, predicted):
        """Calculate forecast bias"""
        return np.mean(predicted - actual)
    
    def calculate_accuracy_ratio(self, actual, predicted):
        """Calculate accuracy ratio (predicted/actual)"""
        return np.mean(predicted / actual)
    
    def arima_forecast(self, train_data, test_size, order=(2, 1, 2)):
        """ARIMA forecasting"""
        try:
            model = ARIMA(train_data, order=order)
            model_fit = model.fit()
            forecast = model_fit.forecast(steps=test_size)
            return forecast, model_fit
        except Exception as e:
            print(f"ARIMA error: {e}")
            return None, None
    
    def exponential_smoothing_forecast(self, train_data, test_size):
        """Exponential Smoothing forecasting"""
        if ExponentialSmoothing is None:
            print("ExponentialSmoothing not available")
            return None, None
            
        try:
            model = ExponentialSmoothing(train_data, trend='add', seasonal=None)
            model_fit = model.fit()
            forecast = model_fit.forecast(steps=test_size)
            return forecast, model_fit
        except Exception as e:
            print(f"Exponential Smoothing error: {e}")
            return None, None
    
    def linear_regression_forecast(self, train_data, test_size):
        """Linear Regression forecasting (trend-based)"""
        try:
            X = np.arange(len(train_data)).reshape(-1, 1)
            y = train_data
            
            model = LinearRegression()
            model.fit(X, y)
            
            # Forecast future periods
            future_X = np.arange(len(train_data), len(train_data) + test_size).reshape(-1, 1)
            forecast = model.predict(future_X)
            return forecast, model
        except Exception as e:
            print(f"Linear Regression error: {e}")
            return None, None
    
    def walk_forward_validation(self, data, model_type='arima', test_size=1, step_size=1):
        """Walk-forward validation for time series"""
        results = []
        n = len(data)
        
        # Adjust test_size for small datasets
        if n < 6:
            test_size = 1
            min_train_size = max(3, n - test_size)
        else:
            min_train_size = max(6, n - test_size)
        
        for i in range(0, n - test_size, step_size):
            train_data = data[:i + test_size]
            test_data = data[i + test_size:i + test_size + test_size]
            
            if len(train_data) < min_train_size:  # Need minimum data for training
                continue
            
            # Get forecast based on model type
            if model_type == 'arima':
                forecast, model = self.arima_forecast(train_data, len(test_data))
            elif model_type == 'exponential':
                forecast, model = self.exponential_smoothing_forecast(train_data, len(test_data))
            elif model_type == 'linear':
                forecast, model = self.linear_regression_forecast(train_data, len(test_data))
            else:
                continue
            
            if forecast is None:
                continue
            
            # Calculate all metrics
            mae = mean_absolute_error(test_data, forecast)
            rmse = np.sqrt(mean_squared_error(test_data, forecast))
            mape = np.mean(np.abs((test_data - forecast) / test_data)) * 100
            theils_u = self.calculate_theils_u(test_data, forecast)
            directional_acc = self.calculate_directional_accuracy(test_data, forecast)
            bias = self.calculate_bias(test_data, forecast)
            accuracy_ratio = self.calculate_accuracy_ratio(test_data, forecast)
            
            results.append({
                'model': model_type,
                'train_size': len(train_data),
                'test_size': len(test_data),
                'mae': mae,
                'rmse': rmse,
                'mape': mape,
                'theils_u': theils_u,
                'directional_accuracy': directional_acc,
                'bias': bias,
                'accuracy_ratio': accuracy_ratio,
                'actual': test_data.tolist(),
                'predicted': forecast.tolist()
            })
        
        return results
    
    def compare_models(self, data, test_size=None):
        """Compare different forecasting models"""
        n = len(data)
        
        # Adjust test_size for small datasets
        if test_size is None:
            if n < 6:
                test_size = 1
            elif n < 12:
                test_size = 2
            else:
                test_size = 3
        
        print(f"Dataset size: {n} data points")
        print(f"Using test_size: {test_size}")
        
        if n < 4:
            print("Warning: Dataset too small for reliable forecasting. Results may be unreliable.")
        
        models = ['linear']  # Start with linear as it's most robust for small datasets
        if n >= 4:
            models.append('arima')
        if ExponentialSmoothing is not None and n >= 6:
            models.append('exponential')
        else:
            print("Note: Exponential Smoothing not available or dataset too small, skipping exponential model")
        
        all_results = []
        
        for model in models:
            print(f"Testing {model} model...")
            results = self.walk_forward_validation(data, model, test_size)
            all_results.extend(results)
        
        return all_results
    
    def statistical_significance_test(self, results1, results2, metric='mae'):
        """Test statistical significance between two model results"""
        values1 = [r[metric] for r in results1]
        values2 = [r[metric] for r in results2]
        
        # Paired t-test
        t_stat, p_value = stats.ttest_rel(values1, values2)
        
        return {
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'mean_diff': np.mean(values1) - np.mean(values2)
        }
    
    def confidence_intervals(self, results, confidence=0.95):
        """Calculate confidence intervals for predictions"""
        all_predictions = []
        for result in results:
            all_predictions.extend(result['predicted'])
        
        mean_pred = np.mean(all_predictions)
        std_pred = np.std(all_predictions)
        
        # Calculate confidence interval
        alpha = 1 - confidence
        z_score = stats.norm.ppf(1 - alpha/2)
        margin_error = z_score * std_pred / np.sqrt(len(all_predictions))
        
        return {
            'mean': mean_pred,
            'std': std_pred,
            'lower_bound': mean_pred - margin_error,
            'upper_bound': mean_pred + margin_error,
            'confidence_level': confidence
        }
    
    def model_diagnostics(self, data, model_type='arima'):
        """Perform model diagnostics"""
        if model_type == 'arima':
            try:
                model = ARIMA(data, order=(2, 1, 2))
                model_fit = model.fit()
                
                # AIC and BIC
                aic = model_fit.aic
                bic = model_fit.bic
                
                # Residuals analysis
                residuals = model_fit.resid
                residual_std = np.std(residuals)
                
                # Ljung-Box test for residual autocorrelation
                from statsmodels.stats.diagnostic import acorr_ljungbox
                lb_stat, lb_pvalue = acorr_ljungbox(residuals, lags=10, return_df=False)
                
                return {
                    'aic': aic,
                    'bic': bic,
                    'residual_std': residual_std,
                    'ljung_box_stat': lb_stat,
                    'ljung_box_pvalue': lb_pvalue,
                    'residuals_normal': stats.normaltest(residuals)[1] > 0.05
                }
            except Exception as e:
                return {'error': str(e)}
        
        return {}
    
    def plot_model_comparison(self, results):
        """Plot comparison of different models"""
        models = list(set([r['model'] for r in results]))
        metrics = ['mae', 'rmse', 'mape', 'theils_u', 'directional_accuracy']
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            if i >= len(axes):
                break
                
            model_means = {}
            model_stds = {}
            
            for model in models:
                model_results = [r for r in results if r['model'] == model]
                values = [r[metric] for r in model_results]
                model_means[model] = np.mean(values)
                model_stds[model] = np.std(values)
            
            x_pos = np.arange(len(models))
            means = [model_means[model] for model in models]
            stds = [model_stds[model] for model in models]
            
            axes[i].bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7)
            axes[i].set_title(f'{metric.upper()} Comparison')
            axes[i].set_xlabel('Model')
            axes[i].set_ylabel(metric.upper())
            axes[i].set_xticks(x_pos)
            axes[i].set_xticklabels(models, rotation=45)
            axes[i].grid(True, alpha=0.3)
        
        # Remove empty subplot
        if len(metrics) < len(axes):
            fig.delaxes(axes[-1])
        
        plt.tight_layout()
        plt.savefig(GRAPH_DIR / "model_comparison.png", dpi=300, bbox_inches="tight")
        plt.close()
    
    def generate_advanced_report(self, results):
        """Generate comprehensive advanced report"""
        print("="*80)
        print("ADVANCED FORECAST MODEL ANALYSIS REPORT")
        print("="*80)
        
        # Group results by model
        model_groups = {}
        for result in results:
            model = result['model']
            if model not in model_groups:
                model_groups[model] = []
            model_groups[model].append(result)
        
        # Model comparison
        print("\nMODEL PERFORMANCE COMPARISON:")
        print("-" * 50)
        
        comparison_data = []
        for model, model_results in model_groups.items():
            metrics = {
                'model': model,
                'mae': np.mean([r['mae'] for r in model_results]),
                'rmse': np.mean([r['rmse'] for r in model_results]),
                'mape': np.mean([r['mape'] for r in model_results]),
                'theils_u': np.mean([r['theils_u'] for r in model_results]),
                'directional_accuracy': np.mean([r['directional_accuracy'] for r in model_results]),
                'bias': np.mean([r['bias'] for r in model_results]),
                'accuracy_ratio': np.mean([r['accuracy_ratio'] for r in model_results])
            }
            comparison_data.append(metrics)
            
            print(f"\n{model.upper()} MODEL:")
            print(f"  MAE: {metrics['mae']:.2f}")
            print(f"  RMSE: {metrics['rmse']:.2f}")
            print(f"  MAPE: {metrics['mape']:.2f}%")
            print(f"  Theil's U: {metrics['theils_u']:.3f}")
            print(f"  Directional Accuracy: {metrics['directional_accuracy']:.1f}%")
            print(f"  Bias: {metrics['bias']:.2f}")
            print(f"  Accuracy Ratio: {metrics['accuracy_ratio']:.3f}")
        
        # Best model selection
        best_model = min(comparison_data, key=lambda x: x['mae'])
        print(f"\nBEST MODEL: {best_model['model'].upper()}")
        print(f"Best MAE: {best_model['mae']:.2f}")
        
        # Statistical significance tests
        if len(model_groups) >= 2:
            print(f"\nSTATISTICAL SIGNIFICANCE TESTS:")
            print("-" * 40)
            
            models = list(model_groups.keys())
            for i in range(len(models)):
                for j in range(i+1, len(models)):
                    model1_results = model_groups[models[i]]
                    model2_results = model_groups[models[j]]
                    
                    sig_test = self.statistical_significance_test(
                        model1_results, model2_results, 'mae'
                    )
                    
                    print(f"\n{models[i].upper()} vs {models[j].upper()}:")
                    print(f"  t-statistic: {sig_test['t_statistic']:.3f}")
                    print(f"  p-value: {sig_test['p_value']:.3f}")
                    print(f"  Significant: {'Yes' if sig_test['significant'] else 'No'}")
                    print(f"  Mean difference: {sig_test['mean_diff']:.2f}")
        
        # Confidence intervals
        ci = self.confidence_intervals(results)
        print(f"\nPREDICTION CONFIDENCE INTERVALS:")
        print(f"  Mean prediction: {ci['mean']:.2f}")
        print(f"  Standard deviation: {ci['std']:.2f}")
        print(f"  {ci['confidence_level']*100}% CI: [{ci['lower_bound']:.2f}, {ci['upper_bound']:.2f}]")
        
        # Model diagnostics
        print(f"\nMODEL DIAGNOSTICS:")
        diagnostics = self.model_diagnostics(self.data, 'arima')
        if 'error' not in diagnostics:
            print(f"  AIC: {diagnostics['aic']:.2f}")
            print(f"  BIC: {diagnostics['bic']:.2f}")
            print(f"  Residual std: {diagnostics['residual_std']:.2f}")
            print(f"  Ljung-Box p-value: {diagnostics['ljung_box_pvalue']:.3f}")
            print(f"  Residuals normal: {'Yes' if diagnostics['residuals_normal'] else 'No'}")
        else:
            print(f"  Diagnostics error: {diagnostics['error']}")
        
        return comparison_data

def main():
    """Main function for advanced testing"""
    print("Loading LinkedIn job postings data...")
    
    try:
        df = pd.read_csv(DATA_DIR / "linkedin_no_skills_cleaned.csv")
        print(f"Loaded {len(df)} job postings")
    except FileNotFoundError:
        print(f"Error: {DATA_DIR / 'linkedin_no_skills_cleaned.csv'} not found!")
        return
    
    # Prepare data
    df['datePosted'] = pd.to_datetime(df['datePosted'])
    df['year_month'] = df['datePosted'].dt.to_period('M')
    job_trends = df.groupby('year_month').size().reset_index(name='job_postings')
    
    print(f"Time series data points: {len(job_trends)}")
    print(f"Date range: {job_trends['year_month'].min()} to {job_trends['year_month'].max()}")
    
    # Check data size and provide recommendations
    if len(job_trends) < 6:
        print("\nWARNING: Small dataset detected!")
        print("   - For reliable forecasting, you need at least 12+ months of data")
        print("   - Current results may not be representative of model performance")
        print("   - Consider collecting more historical data for better accuracy")
    
    # Initialize tester
    tester = AdvancedForecastTester(job_trends['job_postings'].values)
    
    print("\nComparing different forecasting models...")
    results = tester.compare_models(job_trends['job_postings'].values)
    
    if not results:
        print("Error: No valid results obtained!")
        return
    
    print(f"Completed testing with {len(results)} forecast periods")
    
    # Generate visualizations
    print("\nGenerating model comparison plots...")
    tester.plot_model_comparison(results)
    
    # Generate advanced report
    print("\nGenerating advanced analysis report...")
    comparison_data = tester.generate_advanced_report(results)
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_DIR / "advanced_forecast_results.csv", index=False)
    print(f"\nResults saved to '{OUTPUT_DIR / 'advanced_forecast_results.csv'}'")
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df.to_csv(OUTPUT_DIR / "model_comparison_summary.csv", index=False)
    print(f"Model comparison saved to '{OUTPUT_DIR / 'model_comparison_summary.csv'}'")
    
    print("\n" + "="*80)
    print("ADVANCED FORECAST TESTING COMPLETED")
    print("="*80)

if __name__ == "__main__":
    main()

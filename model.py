import pickle
import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


# Paths
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')

LR_MODEL_PATH       = os.path.join(MODELS_DIR, 'lr_model.pkl')
XGB_MODEL_PATH      = os.path.join(MODELS_DIR, 'xgb_model.pkl')
FEATURE_NAMES_PATH  = os.path.join(MODELS_DIR, 'feature_names.pkl')
ENCODERS_PATH       = os.path.join(MODELS_DIR, 'encoders.pkl')
METRICS_PATH        = os.path.join(MODELS_DIR, 'metrics.pkl')


# Load Artifacts
def load_artifacts():
    """Load all pipeline artifacts from pkl files."""

    print("Loading pipeline artifacts...")

    with open(LR_MODEL_PATH, 'rb') as f:
        lr_model = pickle.load(f)

    with open(XGB_MODEL_PATH, 'rb') as f:
        xgb_model = pickle.load(f)

    with open(FEATURE_NAMES_PATH, 'rb') as f:
        feature_names = pickle.load(f)

    with open(ENCODERS_PATH, 'rb') as f:
        encoders = pickle.load(f)

    with open(METRICS_PATH, 'rb') as f:
        metrics = pickle.load(f)

    print("All artifacts loaded\n")

    return lr_model, xgb_model, feature_names, encoders, metrics


# Preprocessing
def preprocess_input(data: dict, feature_names: list, encoders: dict) -> np.ndarray:

    df = pd.DataFrame([data])

    df['tech_role'] = df['job_title'].str.contains(
        'AI|ML|Data|Cloud|DevOps|Backend', case=False
    ).astype(int)

    df = pd.get_dummies(df, columns=['job_title', 'industry', 'location'], drop_first=True)

    df['education_level_encoded'] = encoders['education_level'].get(data['education_level'], -1)
    df['company_size_encoded'] = encoders['company_size'].get(data['company_size'], -1)
    df['remote_work_encoded'] = int(encoders['remote_work'].transform([data['remote_work']])[0])

    df = df.select_dtypes(include=[np.number])

    df = df.reindex(columns=feature_names, fill_value=0)

    return df.to_numpy()


# Prediction
def predict_salary(input_data: dict, model_choice: str = 'both') -> dict:
    """
    Predict employee salary from raw input data.

    Args:
        input_data:   Dictionary of employee features
        model_choice: 'lr', 'xgb', or 'both'

    Returns:
        Dictionary with predicted salary/salaries
    """

    lr_model, xgb_model, feature_names, encoders, metrics = load_artifacts()

    X = preprocess_input(input_data, feature_names, encoders)

    results = {}

    if model_choice in ('lr', 'both'):
        lr_pred = lr_model.predict(X)[0]
        results['linear_regression'] = {
            'predicted_salary': round(lr_pred, 2),
            'model_r2':  round(metrics['linear_regression']['R2'], 4),
            'model_mae': round(metrics['linear_regression']['MAE'], 2),
        }

    if model_choice in ('xgb', 'both'):
        xgb_pred = xgb_model.predict(X)[0]
        results['xgboost'] = {
            'predicted_salary': round(float(xgb_pred), 2),
            'model_r2':  round(metrics['xgboost']['R2'], 4),
            'model_mae': round(metrics['xgboost']['MAE'], 2),
        }

    if model_choice == 'both':
        avg = (results['linear_regression']['predicted_salary'] +
               results['xgboost']['predicted_salary']) / 2
        results['ensemble_average'] = round(avg, 2)

    return results


# Display Metrics
def display_metrics():
    """Print a summary of both model evaluation metrics."""

    with open(METRICS_PATH, 'rb') as f:
        metrics = pickle.load(f)

    print(f"{'Model':<25} {'R²':>6} {'MAE':>8} {'RMSE':>10}")

    for model_name, m in metrics.items():
        name = model_name.replace('_', ' ').title()
        print(f"{name:<25} {m['R2']:>6.4f} {m['MAE']:>8.2f} {m['RMSE']:>10.2f}")


# Main
if __name__ == '__main__':

    # Example employee input
    sample_input = {
        'experience_years': 8,
        'education_level':  'Master',
        'skills_count':     10,
        'industry':         'Finance',
        'company_size':     'Large',
        'location':         'Australia',
        'remote_work':      'No',
        'certifications':   2,
        'job_title':        'Data Analyst',
    }

    print("Employee Salary Prediction")
    print("\nInput:")
    for key, value in sample_input.items():
        print(f"  {key:<20} {value}")

    print("\nModel Metrics:")
    display_metrics()

    print("\nPredictions:")
    results = predict_salary(sample_input, model_choice='both')

    for model_name, result in results.items():
        if model_name == 'ensemble_average':
            print(f"\n  Ensemble Average Salary:  ${result:,.2f}")
        else:
            name = model_name.replace('_', ' ').title()
            print(f"\n  {name}")
            print(f"    Predicted Salary: ${result['predicted_salary']:,.2f}")
            print(f"    Model R²:         {result['model_r2']}")
            print(f"    Model MAE:        ${result['model_mae']:,.2f}")
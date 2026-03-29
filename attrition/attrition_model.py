"""
Employee Attrition Prediction Module

This module implements classification models to predict employee attrition.
Uses RandomForestClassifier and DecisionTreeClassifier for comparison.
Loads employee data from CSV file.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from pathlib import Path


def create_attrition_csv():
    """
    Create and save sample employee dataset to CSV file.
    
    Returns:
        str: Path to the created CSV file
    """
    np.random.seed(42)
    
    # Create sample data
    n_samples = 100
    data = {
        'employee_id': range(1, n_samples + 1),
        'salary': np.random.uniform(30000, 150000, n_samples),
        'work_hours': np.random.uniform(30, 60, n_samples),
        'experience': np.random.uniform(0, 30, n_samples),
        'attrition': np.random.randint(0, 2, n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Save to CSV
    csv_path = Path(__file__).parent.parent / 'data' / 'employee_attrition.csv'
    csv_path.parent.mkdir(exist_ok=True)
    df.to_csv(csv_path, index=False)
    
    print(f"✓ CSV file created: {csv_path}")
    return csv_path


def load_attrition_dataset(csv_path):
    """
    Load employee dataset from CSV file.
    
    Args:
        csv_path: Path to the CSV file
        
    Returns:
        pd.DataFrame: Employee data loaded from CSV
    """
    df = pd.read_csv(csv_path)
    return df


def train_attrition_models(X_train, X_test, y_train, y_test, verbose=True):
    """
    Train both RandomForest and DecisionTree models for attrition prediction.
    
    Args:
        X_train: Training features
        X_test: Testing features
        y_train: Training target
        y_test: Testing target
        
    Returns:
        dict: Dictionary containing both trained models and their accuracies
    """
    models = {}
    
    # Train RandomForestClassifier
    if verbose:
        print("\n" + "="*60)
        print("TRAINING RANDOM FOREST CLASSIFIER")
        print("="*60)
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_accuracy = accuracy_score(y_test, rf_pred)
    if verbose:
        print(f"Random Forest Accuracy: {rf_accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, rf_pred))
    models['random_forest'] = {
        'model': rf_model,
        'accuracy': rf_accuracy,
        'predictions': rf_pred
    }
    
    # Train DecisionTreeClassifier
    if verbose:
        print("\n" + "="*60)
        print("TRAINING DECISION TREE CLASSIFIER")
        print("="*60)
    dt_model = DecisionTreeClassifier(random_state=42)
    dt_model.fit(X_train, y_train)
    dt_pred = dt_model.predict(X_test)
    dt_accuracy = accuracy_score(y_test, dt_pred)
    if verbose:
        print(f"Decision Tree Accuracy: {dt_accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, dt_pred))
    models['decision_tree'] = {
        'model': dt_model,
        'accuracy': dt_accuracy,
        'predictions': dt_pred
    }
    
    return models


def predict_employee_attrition(models, new_employee_features, verbose=True):
    """
    Make attrition predictions for a new employee using trained models.
    
    Args:
        models: Dictionary of trained models
        new_employee_features: Features of the new employee as array/list
        
    Returns:
        dict: Predictions from both models
    """
    predictions = {}
    
    if verbose:
        print("\n" + "="*60)
        print("ATTRITION PREDICTION FOR NEW EMPLOYEE")
        print("="*60)
        print(f"\nEmployee Features:")
        print(f"  Salary: ${new_employee_features[0]:.2f}")
        print(f"  Work Hours: {new_employee_features[1]:.1f}")
        print(f"  Experience: {new_employee_features[2]:.1f} years")
    
    rf_pred = models['random_forest']['model'].predict([new_employee_features])[0]
    dt_pred = models['decision_tree']['model'].predict([new_employee_features])[0]
    
    predictions['random_forest'] = 'LEAVE' if rf_pred == 1 else 'STAY'
    predictions['decision_tree'] = 'LEAVE' if dt_pred == 1 else 'STAY'
    
    if verbose:
        print(f"\nPredictions:")
        print(f"  Random Forest: {predictions['random_forest']}")
        print(f"  Decision Tree: {predictions['decision_tree']}")
    
    return predictions


def run_attrition_model(verbose=True):
    """
    Main function to run the complete attrition prediction pipeline.
    """
    if verbose:
        print("\n" + "#"*60)
        print("# EMPLOYEE ATTRITION PREDICTION MODULE")
        print("#"*60)
    
    # Create CSV file if it doesn't exist
    data_dir = Path(__file__).parent.parent / 'data'
    csv_path = data_dir / 'employee_attrition.csv'
    
    if not csv_path.exists():
        create_attrition_csv()
    
    # Load dataset from CSV
    if verbose:
        print(f"\n📂 Loading data from: {csv_path}")
    df = load_attrition_dataset(csv_path)
    if verbose:
        print(f"✓ Dataset loaded with {len(df)} employees")
        print(f"✓ Attrition rate: {df['attrition'].mean()*100:.1f}%")
    
        # Display first few rows
        print(f"\nFirst 5 records:")
        print(df[['employee_id', 'salary', 'work_hours', 'experience', 'attrition']].head().to_string(index=False))
    
    # Prepare features and target
    X = df[['salary', 'work_hours', 'experience']]
    y = df['attrition']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train models
    models = train_attrition_models(X_train, X_test, y_train, y_test, verbose=verbose)
    
    # Predict for new employee
    # New employee: $90,000 salary, 45 work hours/week, 5 years experience
    new_employee = [90000, 45, 5]
    predictions = predict_employee_attrition(models, new_employee, verbose=verbose)
    
    return predictions

"""
Employee ML Project - Main Orchestrator

This is the main entry point for the machine learning project.
It executes both the employee attrition prediction and fraud detection modules.
"""

import sys
from pathlib import Path

# Add the project root to the path for imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from attrition.attrition_model import run_attrition_model
from fraud.fraud_detection import run_fraud_detection


def main():
    """
    Main function to execute all ML models in the project.
    """
    print("\n" + "█"*60)
    print("█" + " "*58 + "█")
    print("█" + "  EMPLOYEE ML PROJECT - MAIN ORCHESTRATOR".center(58) + "█")
    print("█" + " "*58 + "█")
    print("█"*60)
    
    try:
        # Run Employee Attrition Prediction
        print("\n\n[1/2] Running Employee Attrition Prediction...")
        attrition_results = run_attrition_model()
        
        # Run Fraud Anomaly Detection
        print("\n\n[2/2] Running Fraud Anomaly Detection...")
        fraud_df, fraud_outliers = run_fraud_detection()
        
        # Final Summary
        print("\n\n" + "█"*60)
        print("█" + " "*58 + "█")
        print("█" + "  PROJECT EXECUTION COMPLETED SUCCESSFULLY".center(58) + "█")
        print("█" + " "*58 + "█")
        print("█"*60)
        
        print("\n📊 PROJECT SUMMARY:")
        print(f"  ✓ Employee Attrition Predictions: Generated")
        print(f"  ✓ Fraud Anomalies Detected: {len(fraud_outliers)} transactions")
        print("\n")
        
    except Exception as e:
        print(f"\n❌ ERROR: An error occurred during execution")
        print(f"   Error message: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

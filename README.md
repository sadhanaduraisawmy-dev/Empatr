# Employee Attrition Predictor

A machine learning web application for employee attrition prediction using scikit-learn.

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Installation & Setup](#installation--setup)
- [Web Deployment](#web-deployment)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Features](#features)
- [Output Examples](#output-examples)
- [Requirements](#requirements)

---

## 🎯 Project Overview

This application implements employee attrition prediction using machine learning:

### Key Features
- ✅ Multiple classification algorithms (Random Forest, Decision Tree)
- ✅ Interactive web interface for real-time predictions
- ✅ Dynamic input forms for employee data
- ✅ Feature normalization with StandardScaler
- ✅ Sample datasets with realistic data
- ✅ Modular, well-documented code

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Step 1: Navigate to Project Directory

```powershell
cd c:\anomaly detection\employee_ml_project
```

### Step 2: Create Virtual Environment

```powershell
python -m venv venv
```

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Step 1: Navigate to Project Directory

```powershell
cd c:\anomaly detection\employee_ml_project
```

### Step 2: Create Virtual Environment

```powershell
python -m venv venv
```

### Step 3: Activate Virtual Environment

**Windows PowerShell:**
```powershell
.\venv\Scripts\Activate.ps1
```

**Windows Command Prompt:**
```cmd
.\venv\Scripts\activate.bat
```

**Linux/Mac:**
```bash
source venv/bin/activate
```

You should see `(venv)` appear in your terminal prompt.

### Step 4: Install Dependencies

```powershell
pip install -r requirements.txt
```

This installs:
- `pandas>=3.0.1` - Data manipulation and analysis
- `numpy>=2.4.4` - Numerical computing
- `scikit-learn>=1.8.0` - Machine learning algorithms
- `flask>=3.0.0` - Web framework

---

## 🌐 Web Deployment

The project includes a Flask web application that displays ML results in a beautiful web interface.

### Local Development

```powershell
# Run the web application locally
python app.py
```

Visit `http://localhost:5000` to see the results.

### Deploy to Render

1. **Connect your GitHub repository** to Render
2. **Create a new Web Service** with the following settings:
   - **Runtime**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `python app.py`
3. **Environment Variables** (if needed):
   - `PYTHON_VERSION`: `3.11`

The web app will be available at your Render URL, showing:
- Employee attrition predictions with interactive input
- Model performance metrics
- Real-time predictions based on user input

---

---

## 📁 Project Structure

```
employee_ml_project/
│
├── requirements.txt                 # Project dependencies
├── main.py                         # Main orchestrator script
├── README.md                       # This file
│
├── data/                           # Data directory (auto-created)
│   ├── employee_attrition.csv     # Employee dataset (100 records)
│   └── transactions_fraud.csv     # Transaction dataset (100 records)
│
├── attrition/
│   └── attrition_model.py         # Employee attrition prediction module
│
├── fraud/
│   └── fraud_detection.py         # Fraud anomaly detection module
│
└── venv/                          # Virtual environment (auto-created)
    ├── Scripts/
    ├── Lib/
    └── pyvenv.cfg
```

---

## ⚡ Quick Start

### Run the Complete Project

Once the virtual environment is activated:

```powershell
python main.py
```

This will:
1. Generate a sample employee dataset (100 employees)
2. Train Random Forest and Decision Tree classifiers
3. Predict attrition for a new employee
4. Generate transaction data (100 transactions)
5. Apply DBSCAN clustering for anomaly detection
6. Identify and report fraudulent transactions

### Expected Output

```
████████████████████████████████████████████████████████████
█          EMPLOYEE ML PROJECT - MAIN ORCHESTRATOR         █
████████████████████████████████████████████████████████████

[1/2] Running Employee Attrition Prediction...
✓ Dataset created with 100 employees
✓ Random Forest Accuracy: 0.4000
✓ Decision Tree Accuracy: 0.4000
✓ New employee prediction: LEAVE

[2/2] Running Fraud Anomaly Detection...
✓ Dataset created with 100 transactions
✓ Anomalies detected: 2 transactions
⚠️ FRAUD ALERT - Transaction ID: 99, Amount: $25028.94
⚠️ FRAUD ALERT - Transaction ID: 96, Amount: $16257.98

████████████████████████████████████████████████████████████
📊 PROJECT SUMMARY:
✓ Employee Attrition Predictions: Generated
✓ Fraud Anomalies Detected: 2 transactions
```

---

## 🔧 Modules

### 1. Employee Attrition Prediction (`attrition/attrition_model.py`)

**Purpose**: Predict which employees are likely to leave the company

**Data Source**: CSV file (`data/employee_attrition.csv`)
- Auto-generated on first run
- Contains 100 employee records
- Persists between runs for reproducibility

**Algorithms**:
- RandomForestClassifier (ensemble method)
- DecisionTreeClassifier (tree-based method)

**Features Used**:
- `salary` - Employee salary ($30K-$150K)
- `work_hours` - Weekly work hours (30-60 hrs)
- `experience` - Years of experience (0-30 yrs)

**Target Variable**:
- `attrition` - 1 = Employee leaves, 0 = Employee stays

**Key Functions**:
```python
create_attrition_csv()                 # Generate and save employee CSV
load_attrition_dataset()               # Load data from CSV file
train_attrition_models()               # Train both classifiers
predict_employee_attrition()           # Make predictions
run_attrition_model()                  # Execute complete pipeline
```

**CSV Format**:
```
employee_id,salary,work_hours,experience,attrition
1,74944.81,30.94,19.26,0
2,144085.72,49.09,2.52,1
3,117839.27,39.43,4.85,1
...
```

**Example Prediction**:
```
Employee: $90,000 salary, 45 hrs/week, 5 years experience
Random Forest: LEAVE
Decision Tree: LEAVE
```

---

### 2. Fraud Anomaly Detection (`fraud/fraud_detection.py`)

**Purpose**: Identify suspicious transactions using clustering

**Data Source**: CSV file (`data/transactions_fraud.csv`)
- Auto-generated on first run
- Contains 100 transaction records
- Persists between runs for reproducibility

**Algorithm**:
- DBSCAN (Density-Based Spatial Clustering)

**Features Used**:
- `transaction_amount` - Transaction amount ($50-$50K)
- `transaction_time` - Hour of transaction (0-23h)
- `account_age_days` - Account age in days (1-3650 days)

**Preprocessing**:
- StandardScaler normalization for feature scaling

**Key Functions**:
```python
create_fraud_csv()                     # Generate and save fraud CSV
load_fraud_dataset()                   # Load data from CSV file
apply_standardscaler()                 # Normalize features
train_dbscan_model()                   # Train clustering model
run_fraud_detection()                  # Execute complete pipeline
```

**CSV Format**:
```
transaction_id,transaction_amount,transaction_time,account_age_days
7,337.51,16.82,2677.14
25,2307.55,15.17,2628.24
6,822.17,6.53,1339.69
...
```

**Anomaly Detection**:
- Clusters labeled 0+ are normal transactions
- Clusters labeled -1 are outliers (fraud)

**Example Detection**:
```
Transaction ID 99: $25,028.94 at 1:48 AM, 31-day-old account → FRAUD
Transaction ID 96: $16,257.98 at 12:09 AM, 26-day-old account → FRAUD
```

---

## 📊 Output Examples

### Attrition Model Output
```
EMPLOYEE ATTRITION PREDICTION MODULE
Dataset created with 100 employees
Attrition rate: 54.0%

TRAINING RANDOM FOREST CLASSIFIER
Random Forest Accuracy: 0.4000
Classification Report:
              precision    recall  f1-score   support
           0       0.40      0.40      0.40        10
           1       0.40      0.40      0.40        10

ATTRITION PREDICTION FOR NEW EMPLOYEE
Employee Features:
  Salary: $90000.00
  Work Hours: 45.0
  Experience: 5.0 years

Predictions:
  Random Forest: LEAVE
  Decision Tree: LEAVE
```

### Fraud Detection Output
```
FRAUD ANOMALY DETECTION MODULE
Dataset created with 100 transactions

Dataset Statistics:
  Transaction Amount - Min: $77.33, Max: $35679.16, Mean: $3719.36
  Transaction Time - Min: 0.2h, Max: 22.8h
  Account Age - Min: 5.2 days, Max: 3614.7 days

FRAUD DETECTION RESULTS
Normal Transactions: 98
Anomalies detected: 2

Fraudulent Transactions (Outliers - Cluster = -1):
 transaction_id  transaction_amount  transaction_time  account_age_days
             99        25028.943824          1.800953         31.521633
             96        16257.979583          0.152501         26.589920

⚠️ FRAUD ALERT - Transaction ID: 99
   Amount: $25028.94
   Time: 1.8:00 hours
   Account Age: 31.5 days
```

---

## 📦 Requirements

### Dependencies
- **pandas 2.0.3** - Data manipulation, DataFrame operations
- **numpy 1.24.3** - Numerical arrays and computations
- **scikit-learn 1.3.0** - ML algorithms (classifiers, clustering, preprocessing)

### System Requirements
- RAM: 2GB minimum (1GB for venv + dependencies)
- Disk Space: ~500MB
- Python: 3.8+

---

## 🔍 Code Quality

- ✅ Modular design with separate modules
- ✅ Comprehensive docstrings and comments
- ✅ Type hints where applicable
- ✅ Error handling in main orchestrator
- ✅ Clear function separation of concerns
- ✅ Reproducible results (random seed = 42)
- ✅ Data persisted in CSV files for reproducibility
- ✅ Auto-creates data directory and CSV files on first run

---

## 🛠️ Troubleshooting

### Issue: "command not found: python"
**Solution**: Ensure Python is installed and added to PATH

### Issue: "No module named 'sklearn'"
**Solution**: Activate venv and reinstall requirements:
```powershell
.\venv\Scripts\pip install -r requirements.txt
```

### Issue: "Permission denied" on Activate.ps1
**Solution**: Run PowerShell as Administrator or change execution policy:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteS​igned -Scope CurrentUser
```

### Issue: ModuleNotFoundError in imports
**Solution**: Ensure you're using the correct Python from venv:
```powershell
.\venv\Scripts\python main.py
```

---

## 📝 File Descriptions

| File | Purpose |
|------|---------|
| `main.py` | Main orchestrator - imports and runs both modules |
| `attrition/attrition_model.py` | Classification models for attrition prediction |
| `fraud/fraud_detection.py` | DBSCAN clustering for anomaly detection |
| `data/employee_attrition.csv` | Employee dataset (auto-generated on first run) |
| `data/transactions_fraud.csv` | Transaction dataset (auto-generated on first run) |
| `requirements.txt` | Project dependencies and versions |
| `README.md` | Project documentation |

---

## 🚀 Next Steps

1. **Replace Data**: Edit CSV files in `data/` folder with your own data
   - Keep same column names and format
   - Modify `create_*_csv()` functions for custom data generation
2. **Tune Models**: Adjust hyperparameters in `train_*_models()` functions
3. **Add Features**: Add new columns to CSV files and update feature selection
4. **Export Results**: Modify code to save predictions to output CSV files
5. **Evaluate**: Add cross-validation and additional metrics
6. **Automate Data Refresh**: Use `create_*_csv()` functions in scheduled tasks

---






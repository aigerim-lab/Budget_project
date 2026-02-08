# Budget Risk Alert & Budget Allocation (My Money Mentor)

Machine learning–based personal budgeting project that (1) detects budget risk alerts from transaction behavior and (2) recommends budget allocation across categories based on historical spending.

The project trains a neural network (TensorFlow/Keras) on large-scale transaction datasets (PaySim / Credit Card Transactions) to classify spending periods as **RISK** or **OK**. In parallel, it computes a data-driven **budget allocation plan** from personal expense category history.

---

## Features
- **Budget Risk Alerts (Neural Network)**
  - Learns spending behavior from transactions aggregated into daily/weekly windows
  - Predicts `RISK` vs `OK` and returns a confidence score
- **Budget Allocation (Planning)**
  - Recommends category-level monthly budget split based on past category proportions
  - Outputs a simple plan (Food/Transport/etc.)
- **App Demo (Streamlit)**
  - Upload CSV + enter monthly budget → get alert + allocation table

---

## Project Structure

my_money_mentor_ml/
├── data/
│   ├── paysim/                       # PaySim CSV (heavy dataset)
│   ├── creditcard/                   # Credit Card Transactions CSV
│   ├── personal_expenses/            # Personal categorized expenses CSV
│   └── processed/                    # Aggregated features (generated)
├── models/
│   ├── risk_model.keras              # Trained TensorFlow model
│   ├── scaler.pkl                    # Feature scaler
│   ├── feature_schema.json           # Feature list/order
│   └── figures/
│       ├── learning_curve.png
│       └── confusion_matrix.png
├── src/
│   ├── data_loader.py                # Loads raw CSV datasets
│   ├── preprocessing.py              # Cleaning + aggregation (daily/weekly windows)
│   ├── features.py                   # Feature engineering for risk detection
│   ├── model.py                      # Keras NN architecture
│   ├── train.py                      # Training pipeline + saving artifacts
│   ├── evaluate.py                   # Metrics + confusion matrix + plots
│   ├── allocation.py                 # Budget allocation logic (category proportions)
│   └── utils.py                      # Common helpers (paths, logging)
├── app/
│   ├── streamlit_app.py              # Frontend demo app
│   └── ui_helpers.py                 # UI helpers (tables, charts)
├── main.py                           # CLI entry point
├── requirements.txt
└── README.md

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

LendIQ — AI-Powered Credit Risk & Portfolio Optimization Platform

Live Demo

LendIQ Streamlit App:
https://peterson-muriuki-lendiq.streamlit.app

Overview

LendIQ is an end-to-end AI-driven platform for credit scoring, risk-based pricing, portfolio optimization, and financial risk analytics.
It integrates machine learning, convex optimization, and robust risk metrics to support data-driven lending decisions.

This platform provides:

Machine-learning credit scoring

Real-time risk-based pricing

Portfolio optimization using Modern Portfolio Theory (MPT) and CVXPY

Risk analytics (VaR, CVaR, stress testing)

Explainable AI using SHAP

Interactive Streamlit dashboard

Features
1. AI Credit Scoring

Predicts default probability (PD)

Uses XGBoost with advanced feature engineering

Achieves 82% AUC-ROC

2. Risk-Based Pricing

Converts PD → recommended interest rate

Built into the dashboard for real-time pricing

3. Portfolio Optimization

Mean–variance optimization

Diversification constraints

Efficient frontier analysis using CVXPY

4. Risk Management Tools

Value at Risk (VaR)

Conditional VaR (CVaR)

Stress-test scenarios

Concentration analysis

5. Explainable AI

SHAP values for global & local interpretability

Feature contribution breakdowns

6. Interactive Dashboard

Real-time scoring and pricing

Portfolio analytics

Risk visualizations

Key Results
Metric	Value
AUC-ROC	82%
Default Risk Reduction	30%
Approval Rate Increase	15%
Scoring Speed	< 500 ms
Tech Stack

Python 3.9+

XGBoost, LightGBM

Scikit-Learn, NumPy, Pandas

CVXPY (optimization)

Streamlit (dashboard UI)

Plotly, Matplotlib, Seaborn

Faker (synthetic data generation)

Project Structure
LendIQ/
├── data/
│   ├── generate_data.py          # Synthetic dataset generator
│
├── models/
│   ├── train_credit_model.py     # ML model training pipeline
│   ├── portfolio_optimizer.py    # CVXPY optimization engine
│
├── dashboard/
│   ├── app.py                    # Streamlit dashboard
│
├── figures/                      # Exported plots
│
└── requirements.txt              # Dependencies

Local Installation
1. Clone the Repository
git clone https://github.com/Peterson-Muriuki/LendIQ.git
cd LendIQ

2. Create and Activate a Virtual Environment
python -m venv venv
venv\Scripts\activate        # Windows
# or
source venv/bin/activate    # Mac/Linux

3. Install Dependencies
pip install -r requirements.txt

4. Generate Synthetic Data
python data/generate_data.py

5. Train the Credit Scoring Model
python models/train_credit_model.py

6. Launch the Dashboard
streamlit run dashboard/app.py

Academic Foundation

LendIQ incorporates methodologies from the WorldQuant University
MSc in Financial Engineering, including:

Financial Markets

Time-Series Analysis & Econometrics

Machine Learning

Portfolio Theory

Derivatives & Risk Management

Optimization Methods

Deep Learning for Finance

Roadmap

Hybrid credit scoring model

Black–Litterman optimization

Deep learning scoring (TabNet / MLP)

API version of scoring & pricing engine

Full Docker deployment

Automated hyperparameter tuning (Optuna)

Author

Peterson Muriuki
Email: pitmuriuki@gmail.com

GitHub: https://github.com/Peterson-Muriuki

LinkedIn: https://www.linkedin.com/in/peterson-muriuki-5857aaa9/

License

This project is licensed under the MIT License.

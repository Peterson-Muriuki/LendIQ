# 💳 LendIQ — AI-Powered Credit Risk Platform

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-ML-orange.svg)
![CVXPY](https://img.shields.io/badge/CVXPY-Optimization-purple.svg)
![Plotly](https://img.shields.io/badge/Plotly-Visualizations-lightgrey.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-yellow.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

An end-to-end **AI credit risk assessment & portfolio optimization platform**, built using:

- Machine Learning (XGBoost + LightGBM)
- Portfolio Optimization (CVXPY)
- Risk Management (VaR, CVaR, Stress Testing)
- Streamlit for interactive dashboards


## 🚀 Live Demo  
🔗 https://peterson-muriuki-lendiq.streamlit.app


## ⭐ Features

- **AI Credit Scoring** → XGBoost model with **82% AUC-ROC**
- **Dynamic Loan Pricing** based on applicant risk
- **Portfolio Optimization** using MPT + CVXPY
- **Risk Management Tools** (VaR, CVaR, scenario stress tests)
- **Explainable AI** using SHAP values
- **Interactive Streamlit Dashboard** with Plotly visuals

## 📊 Key Results

- **82% AUC** on synthetic credit dataset  
- **30% reduction** in expected default risk  
- **15% increase** in approval rates  
- **< 500ms** real-time scoring latency  

## 🛠 Tech Stack

- **Python**, **Pandas**, **NumPy**
- **XGBoost**, **LightGBM**, **Scikit-Learn**
- **CVXPY** for optimization
- **Plotly** visualizations
- **Streamlit** dashboard

## 📁 Project Structure

LendIQ/
│── data/ # Synthetic dataset scripts
│── models/ # ML models + optimization engines
│── dashboard/ # Streamlit application
│── figures/ # Charts & visuals
└── requirements.txt # Dependencies

## ⚙️ Installation & Setup

```bash
# Clone repo
git clone https://github.com/Peterson-Muriuki/LendIQ.git
cd LendIQ

# Create virtual environment
python -m venv venv
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Generate synthetic data
python data/generate_data.py

# Train ML model
python models/train_credit_model.py

# Launch dashboard
streamlit run dashboard/app.py
👤 Author

Peterson Muriuki
📧 Email: pitmuriuki@gmail.com

🐙 GitHub: https://github.com/Peterson-Muriuki

🔗 LinkedIn: https://www.linkedin.com/in/peterson-muriuki-5857aaa9/

📄 License
MIT License

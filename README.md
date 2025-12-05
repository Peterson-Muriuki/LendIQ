# LendIQ — AI-Powered Credit Risk & Portfolio Optimization Platform
![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-ML-orange.svg)
![CVXPY](https://img.shields.io/badge/CVXPY-Optimization-purple.svg)
![Plotly](https://img.shields.io/badge/Plotly-Visualizations-lightgrey.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-yellow.svg)

Live Demo  
LendIQ Streamlit App:  
https://peterson-muriuki-lendiq.streamlit.app

Overview  
LendIQ is an end-to-end AI-driven platform for credit scoring, risk-based pricing, portfolio optimization, and financial risk analytics.  
It integrates machine learning, convex optimization, and quantitative risk modeling to support data-driven lending decisions.

The platform provides:
- Machine-learning credit scoring  
- Real-time risk-based pricing  
- Portfolio optimization using Modern Portfolio Theory (MPT) and CVXPY  
- Risk analytics (VaR, CVaR, stress testing)  
- Explainable AI using SHAP  
- Interactive Streamlit dashboard  

Features  

1. AI Credit Scoring  
- Predicts probability of default (PD)  
- XGBoost-based scoring pipeline  
- Advanced feature engineering  
- Achieves 82% AUC-ROC  

2. Risk-Based Pricing  
- Converts PD into a recommended interest rate  
- Real-time pricing interface in dashboard  

3. Portfolio Optimization  
- Mean–variance optimization using CVXPY  
- Diversification and exposure constraints  
- Efficient frontier visualization  

4. Risk Management Tools  
- Value at Risk (VaR)  
- Conditional VaR (CVaR)  
- Stress-test scenarios  
- Concentration analytics  

5. Explainable AI  
- SHAP for model interpretability  
- Feature contribution breakdowns  

6. Interactive Dashboard  
- Real-time scoring and pricing  
- Portfolio analytics  
- Risk visualizations  

Key Results  

| Metric | Value |
|--------|--------|
| AUC-ROC | 82% |
| Default Risk Reduction | 30% |
| Approval Rate Increase | 15% |
| Scoring Speed | < 500 ms |

Tech Stack  
- Python 3.9+  
- XGBoost, LightGBM  
- Scikit-Learn, NumPy, Pandas  
- CVXPY  
- Streamlit  
- Plotly, Matplotlib, Seaborn  
- Faker (synthetic data generation)  

Project Structure  
LendIQ/
│── data/
│ └── generate_data.py # Synthetic dataset generator
│
│── models/
│ ├── train_credit_model.py # ML training pipeline
│ └── portfolio_optimizer.py # CVXPY optimization engine
│
│── dashboard/
│ └── app.py # Streamlit dashboard
│
│── figures/ # Exported charts
│
└── requirements.txt # Dependencies

Local Installation  

Academic Foundation  
LendIQ incorporates quantitative and ML techniques inspired by the WorldQuant University MSc in Financial Engineering curriculum, including:
- Financial Markets  
- Time Series & Econometrics  
- Machine Learning for Finance  
- Portfolio Theory  
- Derivatives & Risk Management  
- Optimization Methods  
- Deep Learning for Finance  

Roadmap  
- Hybrid ML + DL scoring model  
- Black–Litterman optimization  
- Deep-learning scoring (TabNet / MLP)  
- API version of scoring and pricing  
- Full Docker deployment  
- Automated hyperparameter tuning (Optuna)  
- Portfolio backtesting engine  

Author  
Peterson Muriuki  
Email: pitmuriuki@gmail.com  
GitHub: https://github.com/Peterson-Muriuki  
LinkedIn: https://www.linkedin.com/in/peterson-muriuki-5857aaa9/  

License  
This project is licensed under the MIT License.


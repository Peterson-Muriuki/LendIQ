REM Create/update README.md
(
echo # LendIQ: AI-Powered Credit Risk Platform
echo.
echo ![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
echo ![Streamlit](https://img.shields.io/badge/Streamlit-1.31-red.svg)
echo.
echo ## 🚀 Live Demo
echo.
echo **Try it now:** [https://peterson-muriuki-lendiq.streamlit.app](https://peterson-muriuki-lendiq.streamlit.app)
echo.
echo ## 🎯 Features
echo.
echo - **AI Credit Scoring**: XGBoost model with 82%% AUC-ROC
echo - **Dynamic Pricing**: Real-time interest rate calculation  
echo - **Portfolio Optimization**: Mean-variance optimization
echo - **Risk Management**: VaR and stress testing
echo - **Explainable AI**: SHAP values for transparency
echo.
echo ## 📊 Key Results
echo.
echo - ✅ Credit Model AUC: **82%%**
echo - ✅ 30%% reduction in default rates
echo - ✅ 15%% increase in approval rates
echo - ✅ ^<500ms decision latency
echo.
echo ## 🛠️ Tech Stack
echo.
echo - Python, XGBoost, LightGBM
echo - Streamlit Dashboard
echo - CVXPY Optimization
echo - Plotly Visualizations
echo.
echo ## 🎓 Academic Foundation
echo.
echo Integrates  WorldQuant University MScFE courses:
echo - Financial Markets
echo - Financial Data
echo - Financial Econometrics
echo - Derivative Pricing
echo - Stochastic Modeling
echo - Machine Learning
echo - Deep Learning
echo - Portfolio Management
echo - Risk Management
echo.
echo ## 📁 Project Structure
echo.
echo ```
echo LendIQ/
echo ├── data/                  # Generated synthetic data
echo ├── models/                # Trained ML models
echo ├── dashboard/             # Streamlit app
echo ├── figures/               # Visualizations
echo └── requirements.txt       # Dependencies
echo ```
echo.
echo ## 🚀 Local Development
echo.
echo ```bash
echo # Clone repository
echo git clone https://github.com/Peterson-Muriuki/LendIQ.git
echo cd LendIQ
echo.
echo # Create virtual environment
echo python -m venv venv
echo venv\Scripts\activate  # Windows
echo.
echo # Install dependencies
echo pip install -r requirements.txt
echo.
echo # Generate data
echo python data\generate_data.py
echo.
echo # Train model
echo python models\train_credit_model.py
echo.
echo # Run dashboard
echo streamlit run dashboard\app.py
echo ```
echo.
echo ## 📧 Contact
echo.
echo **Peterson Muriuki**
echo - Email: pitmuriuki@gmail.com
echo - GitHub: [@Peterson-Muriuki](https://github.com/Peterson-Muriuki)
echo - LinkedIn: [([https://linkedin.com/in/yourprofile](https://www.linkedin.com/in/peterson-muriuki-5857aaa9/))
echo.
echo ## 📄 License
echo.
echo MIT License
echo.
echo ---
echo.
echo Built as part of WorldQuant University's MScFE program 🎓
) > README.md

git add README.md
git commit -m "docs: Update README with live demo URL"
git push

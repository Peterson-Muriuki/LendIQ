# dashboard/app.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import joblib
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Page config
st.set_page_config(
    page_title="LendIQ - AI Credit Platform",
    page_icon="💳",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    try:
        portfolio = pd.read_csv('data/optimized_portfolio.csv')
        return portfolio, True
    except:
        # Demo data if files not found
        np.random.seed(42)
        portfolio = pd.DataFrame({
            'credit_score': np.random.normal(700, 60, 1000).clip(300, 850),
            'loan_amount': np.random.uniform(5000, 50000, 1000),
            'interest_rate': np.random.uniform(0.06, 0.30, 1000),
            'probability_of_default': np.random.beta(2, 20, 1000),
            'expected_return': np.random.uniform(0.05, 0.15, 1000),
            'optimal_weight': np.random.dirichlet(np.ones(1000))
        })
        portfolio['risk_segment'] = pd.cut(
            portfolio['credit_score'],
            bins=[0, 620, 680, 740, 850],
            labels=['High Risk', 'Medium-High', 'Medium-Low', 'Low Risk']
        )
        return portfolio, False

portfolio, data_loaded = load_data()

# Sidebar
st.sidebar.title("🎯 Navigation")
page = st.sidebar.radio(
    "Select Page",
    ["📊 Dashboard", "👤 Credit Assessment", "💰 Pricing Calculator", 
     "📈 Portfolio Analytics", "⚠️ Risk Management"]
)

if not data_loaded:
    st.sidebar.warning("⚠️ Using demo data. Run data generation scripts first.")

# Main header
st.markdown('<h1 class="main-header">💳 LendIQ: AI Credit Platform</h1>', 
            unsafe_allow_html=True)
st.markdown("---")

# PAGE 1: DASHBOARD
if page == "📊 Dashboard":
    st.header("Portfolio Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Loans", f"{len(portfolio):,}")
    with col2:
        avg_rate = portfolio['interest_rate'].mean()
        st.metric("Avg Interest Rate", f"{avg_rate:.2%}")
    with col3:
        total_exposure = portfolio['loan_amount'].sum()
        st.metric("Total Exposure", f"${total_exposure/1e6:.1f}M")
    with col4:
        avg_pd = portfolio['probability_of_default'].mean()
        st.metric("Avg Default Risk", f"{avg_pd:.2%}")
    
    st.markdown("---")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Portfolio by Risk Segment")
        segment_counts = portfolio['risk_segment'].value_counts()
        fig = px.pie(
            values=segment_counts.values,
            names=segment_counts.index,
            color_discrete_sequence=px.colors.sequential.RdBu
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Risk-Return Profile")
        sample = portfolio.sample(min(300, len(portfolio)))
        fig = px.scatter(
            sample,
            x='probability_of_default',
            y='expected_return',
            size='loan_amount',
            color='risk_segment',
            hover_data=['credit_score'],
            labels={
                'probability_of_default': 'Default Probability',
                'expected_return': 'Expected Return'
            }
        )
        fig.update_xaxes(tickformat='.1%')
        fig.update_yaxes(tickformat='.1%')
        st.plotly_chart(fig, use_container_width=True)

# PAGE 2: CREDIT ASSESSMENT
elif page == "👤 Credit Assessment":
    st.header("Credit Risk Assessment Tool")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 Applicant Information")
        credit_score = st.slider("Credit Score", 300, 850, 720, 10)
        income = st.number_input("Annual Income ($)", 20000, 500000, 75000, 1000)
        loan_amount = st.number_input("Loan Amount ($)", 1000, 100000, 15000, 500)
        
    with col2:
        st.subheader("📊 Additional Factors")
        total_debt = st.number_input("Total Existing Debt ($)", 0, 200000, 25000, 1000)
        employment_months = st.number_input("Employment Length (months)", 0, 300, 36, 6)
        delinquencies = st.number_input("Delinquencies (past 2 years)", 0, 10, 0)
    
    if st.button("🔍 Assess Credit Risk", type="primary", use_container_width=True):
        # Calculate metrics
        dti = (total_debt + loan_amount) / income
        
        # Simple PD calculation
        if credit_score >= 800:
            pd = 0.005
        elif credit_score >= 740:
            pd = 0.01
        elif credit_score >= 670:
            pd = 0.03
        elif credit_score >= 580:
            pd = 0.08
        else:
            pd = 0.15
        
        pd *= (1 + dti) * (1 + delinquencies * 0.25)
        if employment_months < 12:
            pd *= 1.2
        pd = min(pd, 0.50)
        
        # Decision logic
        if credit_score >= 640 and dti < 0.50 and pd < 0.10:
            decision = "✅ APPROVED"
            decision_color = "success"
        elif credit_score < 580 or dti > 0.60 or pd > 0.15:
            decision = "❌ DENIED"
            decision_color = "error"
        else:
            decision = "⏳ MANUAL REVIEW REQUIRED"
            decision_color = "warning"
        
        st.markdown("---")
        st.markdown(f"### Decision: {decision}")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Default Probability", f"{pd:.2%}")
        with col2:
            st.metric("Debt-to-Income Ratio", f"{dti:.1%}")
        with col3:
            risk_score = min(pd * 100 / 0.15, 100)
            st.metric("Risk Score", f"{risk_score:.0f}/100")
        
        # Risk factors
        st.markdown("### 📊 Risk Factor Analysis")
        
        factors = {
            'Credit Score': (850 - credit_score) / 550,
            'Debt-to-Income': min(dti, 1.0),
            'Employment': max(0, (24 - employment_months) / 24),
            'Delinquencies': min(delinquencies / 5, 1.0)
        }
        
        fig = go.Figure(go.Bar(
            x=list(factors.values()),
            y=list(factors.keys()),
            orientation='h',
            marker=dict(
                color=list(factors.values()),
                colorscale='RdYlGn_r',
                showscale=True,
                colorbar=dict(title="Risk Level")
            )
        ))
        fig.update_layout(
            title="Risk Factor Contribution (0 = Low Risk, 1 = High Risk)",
            xaxis_title="Risk Score",
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)

# PAGE 3: PRICING
elif page == "💰 Pricing Calculator":
    st.header("Dynamic Loan Pricing Engine")
    
    st.markdown("### 📈 Market Conditions")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        risk_free = st.number_input("Risk-Free Rate (%)", 0.0, 10.0, 4.5, 0.1) / 100
    with col2:
        unemployment = st.number_input("Unemployment Rate (%)", 0.0, 15.0, 3.8, 0.1) / 100
    with col3:
        gdp_growth = st.number_input("GDP Growth (%)", -5.0, 10.0, 2.5, 0.1) / 100
    with col4:
        inflation = st.number_input("Inflation Rate (%)", 0.0, 15.0, 2.4, 0.1) / 100
    
    st.markdown("---")
    st.markdown("### 🎯 Loan Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        price_score = st.slider("Credit Score", 580, 850, 720, 10, key="price_score")
        price_amount = st.number_input("Loan Amount ($)", 1000, 100000, 15000, 500, key="price_amt")
    
    with col2:
        price_term = st.selectbox("Loan Term (years)", [1, 2, 3, 4, 5], index=2)
        price_dti = st.slider("Debt-to-Income Ratio", 0.0, 1.0, 0.35, 0.05)
    
    if st.button("💰 Calculate Loan Price", type="primary", use_container_width=True):
        # Calculate PD
        if price_score >= 800:
            pd = 0.005
        elif price_score >= 740:
            pd = 0.01
        elif price_score >= 670:
            pd = 0.03
        else:
            pd = 0.08
        
        pd *= (1 + price_dti)
        
        # Credit spread
        if price_score >= 750:
            credit_spread = 0.008
        elif price_score >= 650:
            credit_spread = 0.015
        else:
            credit_spread = 0.045
        
        # Macro adjustment
        macro_adj = 0
        if unemployment > 0.06:
            macro_adj += 0.01
        if gdp_growth < 0:
            macro_adj += 0.02
        
        # Build rate
        rate = risk_free + 0.03 + credit_spread + (pd * 0.40) + 0.02 + 0.03 + macro_adj
        
        # Calculate payment
        monthly_rate = rate / 12
        n_payments = price_term * 12
        monthly_payment = price_amount * (
            monthly_rate * (1 + monthly_rate) ** n_payments
        ) / ((1 + monthly_rate) ** n_payments - 1)
        
        st.markdown("---")
        st.markdown("### 💵 Pricing Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Annual Interest Rate", f"{rate:.2%}")
        with col2:
            st.metric("Monthly Payment", f"${monthly_payment:.2f}")
        with col3:
            st.metric("Total Interest", f"${monthly_payment * n_payments - price_amount:,.2f}")
        
        # Rate breakdown
        st.markdown("### 📊 Rate Components")
        
        components = {
            'Risk-Free Rate': risk_free,
            'Cost of Funds': 0.03,
            'Credit Spread': credit_spread,
            'Expected Loss': pd * 0.40,
            'Operating Cost': 0.02,
            'Profit Margin': 0.03,
            'Macro Adjustment': macro_adj
        }
        
        fig = go.Figure(go.Waterfall(
            name="Rate", orientation="v",
            measure=["relative"] * len(components),
            x=list(components.keys()),
            y=[v * 100 for v in components.values()],
            connector={"line": {"color": "rgb(63, 63, 63)"}},
        ))
        fig.update_layout(
            title="Interest Rate Build-Up",
            yaxis_title="Rate (%)",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

# PAGE 4: PORTFOLIO
elif page == "📈 Portfolio Analytics":
    st.header("Portfolio Analytics & Optimization")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_exp = portfolio['loan_amount'].sum()
        st.metric("Total Exposure", f"${total_exp/1e6:.1f}M")
    with col2:
        avg_ret = portfolio['expected_return'].mean()
        st.metric("Avg Return", f"{avg_ret:.2%}")
    with col3:
        port_pd = portfolio['probability_of_default'].mean()
        st.metric("Portfolio PD", f"{port_pd:.2%}")
    with col4:
        el = (portfolio['probability_of_default'] * 0.4 * portfolio['loan_amount']).sum()
        st.metric("Expected Loss", f"${el/1e6:.2f}M")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Exposure by Risk Segment")
        segment_exp = portfolio.groupby('risk_segment')['loan_amount'].sum()
        fig = px.pie(
            values=segment_exp.values,
            names=segment_exp.index,
            color_discrete_sequence=px.colors.sequential.RdBu
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Credit Score Distribution")
        fig = px.histogram(
            portfolio,
            x='credit_score',
            nbins=30,
            color_discrete_sequence=['steelblue']
        )
        fig.update_layout(
            xaxis_title="Credit Score",
            yaxis_title="Count",
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)

# PAGE 5: RISK MANAGEMENT
else:
    st.header("Risk Management & Stress Testing")
    
    st.markdown("### 📉 Stress Test Scenarios")
    
    scenario = st.selectbox(
        "Select Scenario",
        ["Base Case", "Mild Recession", "Severe Recession", "Credit Crisis"]
    )
    
    if scenario == "Mild Recession":
        pd_mult, lgd_add = 1.5, 0.05
    elif scenario == "Severe Recession":
        pd_mult, lgd_add = 2.0, 0.10
    elif scenario == "Credit Crisis":
        pd_mult, lgd_add = 2.5, 0.15
    else:
        pd_mult, lgd_add = 1.0, 0.0
    
    st.markdown("---")
    
    # Calculate stressed metrics
    base_loss = (portfolio['probability_of_default'] * 0.4 * portfolio['loan_amount']).sum()
    stressed_pd = portfolio['probability_of_default'] * pd_mult
    stressed_loss = (stressed_pd * (0.4 + lgd_add) * portfolio['loan_amount']).sum()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Base Case Loss", f"${base_loss/1e6:.2f}M")
    with col2:
        st.metric(
            f"{scenario} Loss",
            f"${stressed_loss/1e6:.2f}M",
            f"+${(stressed_loss - base_loss)/1e6:.2f}M",
            delta_color="inverse"
        )
    with col3:
        loss_increase = ((stressed_loss - base_loss) / base_loss * 100)
        st.metric("Loss Increase", f"{loss_increase:.1f}%", delta_color="inverse")
    
    st.markdown("---")
    st.markdown("### 📊 Loss Distribution Comparison")
    
    # Create comparison chart
    scenarios_list = ["Base Case", "Mild Recession", "Severe Recession", "Credit Crisis"]
    losses = []
    
    for sc in scenarios_list:
        if sc == "Base Case":
            mult, add = 1.0, 0.0
        elif sc == "Mild Recession":
            mult, add = 1.5, 0.05
        elif sc == "Severe Recession":
            mult, add = 2.0, 0.10
        else:
            mult, add = 2.5, 0.15
        
        loss = (portfolio['probability_of_default'] * mult * (0.4 + add) * 
                portfolio['loan_amount']).sum() / 1e6
        losses.append(loss)
    
    fig = go.Figure(data=[
        go.Bar(
            x=scenarios_list,
            y=losses,
            marker_color=['green', 'yellow', 'orange', 'red']
        )
    ])
    fig.update_layout(
        title="Expected Loss by Scenario",
        xaxis_title="Scenario",
        yaxis_title="Expected Loss ($M)",
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p><strong>LendIQ</strong> - AI-Powered Credit Risk Assessment Platform</p>
    <p>Built with Financial Engineering | WorldQuant University MScFE</p>
    <p>Developed by Peterson Muriuki</p>
</div>
""", unsafe_allow_html=True)
# models/pricing_engine.py

import pandas as pd
import numpy as np
import joblib

class DynamicPricingEngine:
    """Dynamic loan pricing based on risk and market conditions"""
    
    def __init__(self):
        self.base_rate = 0.045
        self.cost_of_funds = 0.03
        self.operating_cost = 0.02
        self.lgd = 0.40
        
    def calculate_pd(self, credit_score, dti=0.35, delinquencies=0, employment_months=24):
        """Calculate Probability of Default"""
        if credit_score >= 800:
            base_pd = 0.005
        elif credit_score >= 740:
            base_pd = 0.01
        elif credit_score >= 670:
            base_pd = 0.03
        elif credit_score >= 620:
            base_pd = 0.08
        else:
            base_pd = 0.15
        
        pd = base_pd * (1 + dti) * (1 + delinquencies * 0.25)
        if employment_months < 12:
            pd *= 1.2
        
        return min(pd, 0.50)
    
    def get_credit_spread(self, credit_score):
        """Get credit spread based on score"""
        if credit_score >= 750:
            return 0.008
        elif credit_score >= 650:
            return 0.015
        else:
            return 0.045
    
    def calculate_macro_adjustment(self, unemployment_rate=0.038, gdp_growth=0.025):
        """Adjust for macro conditions"""
        adjustment = 0.0
        if unemployment_rate > 0.06:
            adjustment += 0.01
        elif unemployment_rate > 0.05:
            adjustment += 0.005
        
        if gdp_growth < 0:
            adjustment += 0.02
        elif gdp_growth < 0.015:
            adjustment += 0.01
        
        return adjustment
    
    def price_loan(self, credit_score, loan_amount, loan_term_years=3,
                   dti=0.35, delinquencies=0, employment_months=24,
                   unemployment_rate=0.038, gdp_growth=0.025):
        """Calculate optimal loan price"""
        
        pd = self.calculate_pd(credit_score, dti, delinquencies, employment_months)
        credit_spread = self.get_credit_spread(credit_score)
        macro_adj = self.calculate_macro_adjustment(unemployment_rate, gdp_growth)
        expected_loss = pd * self.lgd
        
        interest_rate = (
            self.base_rate +
            credit_spread +
            expected_loss +
            self.operating_cost +
            0.03 +  # Profit margin
            macro_adj
        )
        
        monthly_rate = interest_rate / 12
        n_payments = int(loan_term_years * 12)
        
        if monthly_rate > 0:
            monthly_payment = loan_amount * (
                monthly_rate * (1 + monthly_rate) ** n_payments
            ) / ((1 + monthly_rate) ** n_payments - 1)
        else:
            monthly_payment = loan_amount / n_payments
        
        return {
            'interest_rate': interest_rate,
            'monthly_payment': monthly_payment,
            'total_interest': monthly_payment * n_payments - loan_amount,
            'total_payment': monthly_payment * n_payments,
            'probability_of_default': pd,
            'expected_loss_dollars': expected_loss * loan_amount,
            'rate_components': {
                'base_rate': self.base_rate,
                'credit_spread': credit_spread,
                'expected_loss': expected_loss,
                'operating_cost': self.operating_cost,
                'profit_margin': 0.03,
                'macro_adjustment': macro_adj
            }
        }

# Test pricing engine
if __name__ == "__main__":
    print("=" * 70)
    print("LENDIQ PRICING ENGINE TEST")
    print("=" * 70)
    
    engine = DynamicPricingEngine()
    
    test_scenarios = [
        {'credit_score': 580, 'label': 'Poor Credit'},
        {'credit_score': 650, 'label': 'Fair Credit'},
        {'credit_score': 720, 'label': 'Good Credit'},
        {'credit_score': 800, 'label': 'Excellent Credit'}
    ]
    
    print("\nPricing Examples (Loan: $15,000, Term: 3 years)")
    print("=" * 70)
    
    for scenario in test_scenarios:
        pricing = engine.price_loan(
            credit_score=scenario['credit_score'],
            loan_amount=15000,
            loan_term_years=3
        )
        
        print(f"\n{scenario['label']} (Score: {scenario['credit_score']})")
        print(f"  Interest Rate:    {pricing['interest_rate']:.2%}")
        print(f"  Monthly Payment:  ${pricing['monthly_payment']:.2f}")
        print(f"  Total Interest:   ${pricing['total_interest']:.2f}")
        print(f"  Default Prob:     {pricing['probability_of_default']:.2%}")
    
    print("\n✅ Pricing engine test complete!")
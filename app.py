
import streamlit as st
import pandas as pd
import joblib
import json
import os
import numpy as np

# --- App Config ---
st.set_page_config(page_title='Supermarket Spending Predictor', layout='wide')
st.title('🛍️ Supermarket Category Spending Predictor')
st.markdown("Enter customer details below to predict spending across **6 product categories**.")

# --- Load Resources ---
FEATURE_FILE = 'feature_columns.json'
TARGETS = [
    'MntWines', 'MntFruits', 'MntMeatProducts', 
    'MntFishProducts', 'MntSweetProducts', 'MntGoldProds'
]
TARGET_EMOJIS = {
    'MntWines': '🍷 Wines', 
    'MntFruits': '🍎 Fruits', 
    'MntMeatProducts': '🥩 Meat',
    'MntFishProducts': '🐟 Fish', 
    'MntSweetProducts': '🍬 Sweets', 
    'MntGoldProds': '🥇 Gold'
}

@st.cache_resource
def load_models():
    models = {}
    for target in TARGETS:
        filename = f"model_{target}.pkl"
        if os.path.exists(filename):
            models[target] = joblib.load(filename)
    return models

models = load_models()
feature_cols = json.load(open(FEATURE_FILE)) if os.path.exists(FEATURE_FILE) else []

if not models or not feature_cols:
    st.error("❌ Error: Models or feature configuration file missing.")
    st.stop()

# --- Sidebar Inputs ---
st.sidebar.header("Customer Profile")

def user_input_features():
    # Inputs exactly matching your requested variables
    Income = st.sidebar.number_input('Annual Income ($)', min_value=0, value=50000, step=500)
    Age = st.sidebar.slider('Age', 18, 90, 45)
    Recency = st.sidebar.slider('Recency (Days since purchase)', 0, 100, 49)
    Teenhome = st.sidebar.selectbox('Teenagers at Home', [0, 1, 2])
    
    st.sidebar.subheader("Purchase Behaviors")
    NumCatalogPurchases = st.sidebar.number_input('Catalog Purchases', 0, 30, 5)
    NumWebPurchases = st.sidebar.number_input('Web Purchases', 0, 30, 5)
    NumStorePurchases = st.sidebar.number_input('Store Purchases', 0, 30, 5)
    NumWebVisitsMonth = st.sidebar.number_input('Web Visits/Month', 0, 30, 5)
    NumDealsPurchases = st.sidebar.number_input('Deals Purchased', 0, 30, 2)

    # Create DataFrame in exact training order
    data = {
        'NumCatalogPurchases': NumCatalogPurchases,
        'Income': Income,
        'NumWebPurchases': NumWebPurchases,
        'Age': Age,
        'Recency': Recency,
        'NumStorePurchases': NumStorePurchases,
        'Teenhome': Teenhome,
        'NumWebVisitsMonth': NumWebVisitsMonth,
        'NumDealsPurchases': NumDealsPurchases
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# --- Main App Layout ---
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Input Data")
    st.dataframe(input_df.T) # Transpose for better view

with col2:
    if st.button('🔮 Predict Category Spending', type='primary'):
        st.subheader("Predicted Spending Breakdown")
        
        # Grid layout for results
        metrics_cols = st.columns(3)
        
        results = {}
        total_spending = 0
        
        # Loop through targets and predict
        for i, target in enumerate(TARGETS):
            if target in models:
                pred = models[target].predict(input_df)[0]
                results[TARGET_EMOJIS[target]] = pred
                total_spending += pred
                
                # Display card
                with metrics_cols[i % 3]:
                    st.metric(label=TARGET_EMOJIS[target], value=f"${pred:,.2f}")
        
        st.divider()
        st.success(f"### 💰 Total Predicted Spending: ${total_spending:,.2f}")
        
        # Visual Chart
        st.subheader("Wallet Share Analysis")
        chart_df = pd.DataFrame.from_dict(results, orient='index', columns=['Amount'])
        st.bar_chart(chart_df)


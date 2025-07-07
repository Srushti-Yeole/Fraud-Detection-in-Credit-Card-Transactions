import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Credit Card Fraud Dashboard", layout="wide")
st.title("Credit Card Fraud Detection Dashboard")

# Load data
data_file = "creditcard.csv"
@st.cache_data
def load_data():
    df = pd.read_csv(data_file)
    return df

df = load_data()

# Show basic dataset statistics
st.header("Dataset Overview")
st.write(f"**Shape:** {df.shape[0]} rows, {df.shape[1]} columns")
st.write("**Columns:**", list(df.columns))
st.write("**First 5 rows:**")
st.dataframe(df.head())
st.write("**Summary statistics:**")
st.dataframe(df.describe())

# Bar chart of fraud vs. non-fraud
st.header("Fraud vs. Non-Fraud Transactions")
if 'Class' in df.columns:
    class_counts = df['Class'].value_counts().sort_index()
    fig, ax = plt.subplots()
    ax.bar(['Non-Fraud', 'Fraud'], class_counts, color=['#4caf50', '#e74c3c'])
    ax.set_ylabel('Number of Transactions')
    ax.set_title('Transaction Class Distribution')
    st.pyplot(fig)
else:
    st.warning("No 'Class' column found in the dataset.")

# Correlation heatmap
st.header("Correlation Heatmap")
fig2, ax2 = plt.subplots(figsize=(12, 8))
corr = df.corr()
sns.heatmap(corr, cmap='coolwarm', center=0, ax=ax2)
ax2.set_title('Feature Correlation Heatmap')
st.pyplot(fig2)

# Histogram of fraud transactions over time (if Time exists)
if 'Time' in df.columns and 'Class' in df.columns:
    st.header("Fraud Transactions Over Time")
    fraud_times = df[df['Class'] == 1]['Time']
    fig3, ax3 = plt.subplots()
    ax3.hist(fraud_times, bins=50, color='#e74c3c', alpha=0.7)
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Number of Fraud Transactions')
    ax3.set_title('Fraud Transactions Over Time')
    st.pyplot(fig3) 
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
 
st.set_page_config(page_title="Golden Batch Analyzer", layout="wide")
 
st.title("🏭 Manufacturing Golden Batch Analyzer")
st.write("Upload your batch data and I will analyze it!")
 
uploaded_file = st.file_uploader("Upload batch data", type=["csv", "xlsx", "xls"])
 
if uploaded_file is not None:
    # Read file - handle both CSV and Excel
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith(('.xls', '.xlsx')):
        df = pd.read_excel(uploaded_file)
    st.write("✅ File uploaded!")
    st.write(f"Total rows: {len(df)}")
    st.write(f"Total batches: {df['batch_id'].nunique()}")
    st.subheader("📋 Data Preview")
    st.dataframe(df.head(20))
    batch_summary = df.groupby('batch_id').agg({
        'temperature': 'mean',
        'pressure': 'mean',
        'ph': 'mean',
        'cycle_time': 'mean',
        'yield': 'mean',
        'impurity_percent': 'mean'
    }).reset_index()
    def score_batch(row):
        yield_score = (row['yield'] / 100) * 25
        cycle_time_score = max(0, (150 - row['cycle_time']) / 30 * 25)
        impurity_score = max(0, (5 - row['impurity_percent']) / 5 * 50)
        return yield_score + cycle_time_score + impurity_score
    batch_summary['score'] = batch_summary.apply(score_batch, axis=1)
    batch_summary['status'] = batch_summary['score'].apply(lambda x: 'GOLDEN' if x >= 85 else 'NON-GOLDEN')
    st.subheader("🎯 Analysis Results")
    golden = len(batch_summary[batch_summary['status'] == 'GOLDEN'])
    non_golden = len(batch_summary[batch_summary['status'] == 'NON-GOLDEN'])
    col1, col2, col3 = st.columns(3)
    col1.metric("✅ Golden Batches", golden)
    col2.metric("❌ Non-Golden Batches", non_golden)
    col3.metric("Success Rate", f"{(golden/(golden+non_golden)*100):.1f}%")
    golden_batches = batch_summary[batch_summary['status'] == 'GOLDEN']
    if len(golden_batches) > 0:
        st.subheader("🏆 Golden Batch Operating Window")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info(f"🌡️ Temperature\n{golden_batches['temperature'].min():.1f}°C - {golden_batches['temperature'].max():.1f}°C")
        with col2:
            st.info(f"🔧 Pressure\n{golden_batches['pressure'].min():.2f} - {golden_batches['pressure'].max():.2f} bar")
        with col3:
            st.info(f"⚗️ pH\n{golden_batches['ph'].min():.1f} - {golden_batches['ph'].max():.1f}")
    st.subheader("🔍 Critical Parameters (Ranked by Importance)")
    X = batch_summary[['temperature', 'pressure', 'ph', 'cycle_time']]
    y = (batch_summary['status'] == 'GOLDEN').astype(int)
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X, y)
    importance = pd.DataFrame({
        'Parameter': ['Temperature', 'Pressure', 'pH', 'Cycle Time'],
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    st.dataframe(importance, use_container_width=True)
    st.subheader("📊 Good vs Bad Batch Comparison")
    good_batches = batch_summary[batch_summary['status'] == 'GOLDEN']
    bad_batches = batch_summary[batch_summary['status'] == 'NON-GOLDEN']
    if len(good_batches) > 0 and len(bad_batches) > 0:
        comparison_data = {
            'Parameter': ['Temperature', 'Pressure', 'pH', 'Cycle Time', 'Yield', 'Impurity'],
            'Golden Avg': [
                good_batches['temperature'].mean(),
                good_batches['pressure'].mean(),
                good_batches['ph'].mean(),
                good_batches['cycle_time'].mean(),
                good_batches['yield'].mean(),
                good_batches['impurity_percent'].mean()
            ],
            'Non-Golden Avg': [
                bad_batches['temperature'].mean(),
                bad_batches['pressure'].mean(),
                bad_batches['ph'].mean(),
                bad_batches['cycle_time'].mean(),
                bad_batches['yield'].mean(),
                bad_batches['impurity_percent'].mean()
            ]
        }
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True)
    st.subheader("📊 All Batches")
    st.dataframe(batch_summary, use_container_width=True)
    st.subheader("📄 Summary")
    st.write(f"""
    **Total Batches:** {len(batch_summary)}
    **Golden Batches:** {golden} ({(golden/len(batch_summary)*100):.1f}%)
    **Non-Golden Batches:** {non_golden} ({(non_golden/len(batch_summary)*100):.1f}%)
    **Most Critical Parameter:** {importance.iloc[0]['Parameter']}
    """)
else:
    st.info("👈 Please upload a CSV or Excel file to analyze")

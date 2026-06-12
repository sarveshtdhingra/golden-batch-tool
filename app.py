import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import plotly.graph_objects as go
import plotly.express as px
import google.generativeai as genai
import os
 
st.set_page_config(page_title="Golden Batch Analyzer", layout="wide", initial_sidebar_state="expanded")
 
def get_gemini_model():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None
    genai.configure(api_key=api_key)
    return genai.GenerativeModel('gemini-1.5-flash')
 
GOLDEN_COLOR = "#2ecc71"
NON_GOLDEN_COLOR = "#e74c3c"
PRIMARY_COLOR = "#3498db"
SECONDARY_COLOR = "#f39c12"
 
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 20px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        color: white;
        margin-bottom: 20px;
    }
    .main-header h1 {
        margin: 0;
        font-size: 2.5em;
    }
    .main-header p {
        margin: 10px 0 0 0;
        font-size: 1.1em;
    }
</style>
<div class="main-header">
<h1>Manufacturing Golden Batch Analyzer</h1>
<p>AI-Powered Batch Quality Analysis</p>
</div>
    """, unsafe_allow_html=True)
 
st.sidebar.markdown("## Dashboard Settings")
 
uploaded_file = st.sidebar.file_uploader("Upload Batch Data", type=["csv", "xlsx", "xls"])
 
if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Please upload a CSV or Excel file")
            df = None
    except Exception as e:
        st.error(f"Error reading file: {e}")
        df = None
    if df is not None:
        st.sidebar.success("File loaded successfully!")
        st.session_state.df = df
        st.session_state.file_loaded = True
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
        st.session_state.batch_summary = batch_summary
        golden_batches = batch_summary[batch_summary['status'] == 'GOLDEN']
        non_golden_batches = batch_summary[batch_summary['status'] == 'NON-GOLDEN']
        X = batch_summary[['temperature', 'pressure', 'ph', 'cycle_time']]
        y = (batch_summary['status'] == 'GOLDEN').astype(int)
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X, y)
        importance = pd.DataFrame({
            'Parameter': ['Temperature', 'Pressure', 'pH', 'Cycle Time'],
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        st.session_state.importance = importance
        st.session_state.golden_batches = golden_batches
        st.session_state.non_golden_batches = non_golden_batches
        tab1, tab2, tab3 = st.tabs(["Dashboard", "AI Chat Assistant", "Data Explorer"])
        with tab1:
            st.markdown("## Key Metrics")
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("Golden Batches", len(golden_batches))
            with col2:
                st.metric("Non-Golden Batches", len(non_golden_batches))
            with col3:
                success_rate = (len(golden_batches) / len(batch_summary) * 100)
                st.metric("Success Rate", f"{success_rate:.1f}%")
            with col4:
                avg_yield = batch_summary['yield'].mean()
                st.metric("Avg Yield", f"{avg_yield:.1f}%")
            with col5:
                avg_impurity = batch_summary['impurity_percent'].mean()
                st.metric("Avg Impurity", f"{avg_impurity:.2f}%")
            st.divider()
            st.markdown("## Batch Status Overview")
            col1, col2 = st.columns(2)
            with col1:
                status_counts = batch_summary['status'].value_counts()
                fig_pie = go.Figure(data=[go.Pie(
                    labels=status_counts.index,
                    values=status_counts.values,
                    marker=dict(colors=[GOLDEN_COLOR, NON_GOLDEN_COLOR])
                )])
                fig_pie.update_layout(title="Batch Status Distribution", height=400)
                st.plotly_chart(fig_pie, use_container_width=True)
            with col2:
                fig_score = go.Figure(data=[
                    go.Histogram(x=batch_summary['score'], nbinsx=15, marker=dict(color=PRIMARY_COLOR))
                ])
                fig_score.add_vline(x=85, line_dash="dash", line_color="red")
                fig_score.update_layout(title="Quality Score Distribution", height=400)
                st.plotly_chart(fig_score, use_container_width=True)
            st.divider()
            st.markdown("## Golden Batch Operating Window")
            if len(golden_batches) > 0:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.info(f"Temperature: {golden_batches['temperature'].min():.1f}-{golden_batches['temperature'].max():.1f}C")
                with col2:
                    st.info(f"Pressure: {golden_batches['pressure'].min():.2f}-{golden_batches['pressure'].max():.2f} bar")
                with col3:
                    st.info(f"pH: {golden_batches['ph'].min():.1f}-{golden_batches['ph'].max():.1f}")
            st.divider()
            st.markdown("## Critical Parameters")
            col1, col2 = st.columns(2)
            with col1:
                fig_importance = px.bar(importance, x='Parameter', y='Importance', color='Importance')
                fig_importance.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig_importance, use_container_width=True)
            with col2:
                st.markdown("### Top Parameters")
                for idx, (_, row) in enumerate(importance.iterrows(), 1):
                    st.markdown(f"**{idx}. {row['Parameter']}**: {row['Importance']:.1%}")
            st.divider()
            st.markdown("## Comparison: Golden vs Non-Golden")
            if len(golden_batches) > 0 and len(non_golden_batches) > 0:
                comparison_data = {
                    'Parameter': ['Temperature', 'Pressure', 'pH', 'Cycle Time', 'Yield', 'Impurity'],
                    'Golden': [
                        golden_batches['temperature'].mean(),
                        golden_batches['pressure'].mean(),
                        golden_batches['ph'].mean(),
                        golden_batches['cycle_time'].mean(),
                        golden_batches['yield'].mean(),
                        golden_batches['impurity_percent'].mean()
                    ],
                    'Non-Golden': [
                        non_golden_batches['temperature'].mean(),
                        non_golden_batches['pressure'].mean(),
                        non_golden_batches['ph'].mean(),
                        non_golden_batches['cycle_time'].mean(),
                        non_golden_batches['yield'].mean(),
                        non_golden_batches['impurity_percent'].mean()
                    ]
                }
                comparison_df = pd.DataFrame(comparison_data)
                fig_comparison = go.Figure(data=[
                    go.Bar(x=comparison_df['Parameter'], y=comparison_df['Golden'], name='Golden'),
                    go.Bar(x=comparison_df['Parameter'], y=comparison_df['Non-Golden'], name='Non-Golden')
                ])
                fig_comparison.update_layout(barmode='group', height=400)
                st.plotly_chart(fig_comparison, use_container_width=True)
                st.dataframe(comparison_df, use_container_width=True)
        with tab2:
            st.markdown("## AI Batch Analysis Chat")
            if "chat_history" not in st.session_state:
                st.session_state.chat_history = []
            for message in st.session_state.chat_history:
                if message["role"] == "user":
                    st.chat_message("user").write(message["content"])
                else:
                    st.chat_message("assistant").write(message["content"])
            user_input = st.chat_input("Ask about a batch...")
            if user_input:
                st.session_state.chat_history.append({"role": "user", "content": user_input})
                batch_data_str = batch_summary.to_string()
                if len(golden_batches) > 0:
                    golden_params = f"Temp: {golden_batches['temperature'].min():.1f}-{golden_batches['temperature'].max():.1f}C, Pressure: {golden_batches['pressure'].min():.2f}-{golden_batches['pressure'].max():.2f} bar"
                else:
                    golden_params = "No golden batches"
                system_prompt = f"""You are a manufacturing engineer.
 
GOLDEN BATCH PARAMETERS: {golden_params}
 
BATCH DATA: {batch_data_str}
 
CRITICAL PARAMETERS: {importance.to_string()}
 
Answer questions about batches concisely."""
                try:
                    model = get_gemini_model()
                    if model is None:
                        error_msg = "ERROR: GEMINI_API_KEY not set in Render environment variables"
                        st.session_state.chat_history.append({"role": "assistant", "content": error_msg})
                        st.rerun()
                    else:
                        response = model.generate_content(f"{system_prompt}\n\nUser: {user_input}", generation_config=genai.types.GenerationConfig(max_output_tokens=1024))
                        assistant_message = response.text
                        st.session_state.chat_history.append({"role": "assistant", "content": assistant_message})
                        st.rerun()
                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    st.session_state.chat_history.append({"role": "assistant", "content": error_msg})
                    st.rerun()
            st.markdown("### Example Questions")
            st.markdown("- What's wrong with batch B002?")
            st.markdown("- Why did batch B004 fail?")
        with tab3:
            st.markdown("## All Batches")
            batch_display = batch_summary.copy()
            batch_display['Status'] = batch_display['status'].apply(lambda x: 'GOLDEN' if x == 'GOLDEN' else 'NON-GOLDEN')
            batch_display = batch_display.round(2)
            st.dataframe(batch_display, use_container_width=True, hide_index=True)
            st.divider()
            st.markdown("## Batch Details")
            selected_batch = st.selectbox("Select batch:", batch_summary['batch_id'].unique())
            if selected_batch:
                batch_data = batch_summary[batch_summary['batch_id'] == selected_batch].iloc[0]
                batch_full_data = df[df['batch_id'] == selected_batch]
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Status", batch_data['status'])
                with col2:
                    st.metric("Score", f"{batch_data['score']:.1f}")
                with col3:
                    st.metric("Yield", f"{batch_data['yield']:.1f}%")
                with col4:
                    st.metric("Impurity", f"{batch_data['impurity_percent']:.2f}%")
                st.markdown(f"### Trends for {selected_batch}")
                st.dataframe(batch_full_data, use_container_width=True, hide_index=True)
 
else:
    st.info("Upload a batch data file to get started!")

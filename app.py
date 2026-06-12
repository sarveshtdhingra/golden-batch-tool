import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import plotly.graph_objects as go
import plotly.express as px
import google.generativeai as genai
 
st.set_page_config(page_title="Golden Batch Analyzer", layout="wide", initial_sidebar_state="expanded")
 
# Initialize Gemini
@st.cache_resource
def get_gemini_model():
    api_key = st.secrets.get("GEMINI_API_KEY") or st.secrets["gemini_api_key"]
    genai.configure(api_key=api_key)
    return genai.GenerativeModel('gemini-1.5-flash')
 
# Custom theme colors
GOLDEN_COLOR = "#2ecc71"
NON_GOLDEN_COLOR = "#e74c3c"
PRIMARY_COLOR = "#3498db"
SECONDARY_COLOR = "#f39c12"
 
# ==================== HEADER ====================
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
<h1>🏭 Manufacturing Golden Batch Analyzer</h1>
<p>AI-Powered Batch Quality Analysis & Optimization</p>
</div>
    """, unsafe_allow_html=True)
 
# ==================== SIDEBAR ====================
st.sidebar.markdown("## 📊 Dashboard Settings")
 
uploaded_file = st.sidebar.file_uploader("📤 Upload Batch Data", type=["csv", "xlsx", "xls"])
 
# Store data in session state for chat access
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
        st.sidebar.success("✅ File loaded successfully!")
        # Store in session state
        st.session_state.df = df
        st.session_state.file_loaded = True
        # ==================== BATCH PROCESSING ====================
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
        # Train model for importance
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
        # ==================== TABS ====================
        tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "💬 AI Chat Assistant", "📋 Data Explorer"])
        # ==================== TAB 1: DASHBOARD ====================
        with tab1:
            st.markdown("## 📈 Key Metrics")
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric(
                    "✅ Golden Batches",
                    len(golden_batches),
                    delta=f"{len(golden_batches)}/{len(batch_summary)}",
                    delta_color="off"
                )
            with col2:
                st.metric(
                    "❌ Non-Golden Batches",
                    len(non_golden_batches),
                    delta=f"{len(non_golden_batches)}/{len(batch_summary)}",
                    delta_color="off"
                )
            with col3:
                success_rate = (len(golden_batches) / len(batch_summary) * 100)
                st.metric(
                    "📊 Success Rate",
                    f"{success_rate:.1f}%",
                    delta=f"{success_rate:.1f}% Golden"
                )
            with col4:
                avg_yield = batch_summary['yield'].mean()
                st.metric(
                    "📉 Avg Yield",
                    f"{avg_yield:.1f}%",
                    delta=f"Target: 95%"
                )
            with col5:
                avg_impurity = batch_summary['impurity_percent'].mean()
                st.metric(
                    "🧪 Avg Impurity",
                    f"{avg_impurity:.2f}%",
                    delta=f"Target: <2%"
                )
            st.divider()
            st.markdown("## 🎯 Batch Status Overview")
            col1, col2 = st.columns(2)
            with col1:
                status_counts = batch_summary['status'].value_counts()
                fig_pie = go.Figure(data=[go.Pie(
                    labels=status_counts.index,
                    values=status_counts.values,
                    marker=dict(colors=[GOLDEN_COLOR, NON_GOLDEN_COLOR]),
                    textposition='inside',
                    textinfo='label+percent',
                    hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
                )])
                fig_pie.update_layout(
                    title="Batch Status Distribution",
                    height=400,
                    showlegend=True
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            with col2:
                fig_score = go.Figure(data=[
                    go.Histogram(
                        x=batch_summary['score'],
                        nbinsx=15,
                        marker=dict(color=PRIMARY_COLOR),
                        hovertemplate='Score: %{x:.1f}<br>Count: %{y}<extra></extra>'
                    )
                ])
                fig_score.add_vline(x=85, line_dash="dash", line_color="red", annotation_text="Golden Threshold (85)")
                fig_score.update_layout(
                    title="Batch Quality Score Distribution",
                    xaxis_title="Quality Score",
                    yaxis_title="Number of Batches",
                    height=400,
                    hovermode='x unified'
                )
                st.plotly_chart(fig_score, use_container_width=True)
            st.divider()
            st.markdown("## 🏆 Golden Batch Operating Window")
            if len(golden_batches) > 0:
                golden_temp_min = golden_batches['temperature'].min()
                golden_temp_max = golden_batches['temperature'].max()
                golden_pressure_min = golden_batches['pressure'].min()
                golden_pressure_max = golden_batches['pressure'].max()
                golden_ph_min = golden_batches['ph'].min()
                golden_ph_max = golden_batches['ph'].max()
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.info(
                        f"""
                        🌡️ **Temperature**
                        **Range:** {golden_temp_min:.1f}°C - {golden_temp_max:.1f}°C
                        **Average:** {golden_batches['temperature'].mean():.1f}°C
                        """
                    )
                with col2:
                    st.info(
                        f"""
                        🔧 **Pressure**
                        **Range:** {golden_pressure_min:.2f} - {golden_pressure_max:.2f} bar
                        **Average:** {golden_batches['pressure'].mean():.2f} bar
                        """
                    )
                with col3:
                    st.info(
                        f"""
                        ⚗️ **pH**
                        **Range:** {golden_ph_min:.1f} - {golden_ph_max:.1f}
                        **Average:** {golden_batches['ph'].mean():.1f}
                        """
                    )
            st.divider()
            st.markdown("## 🔍 Critical Parameters Analysis")
            col1, col2 = st.columns(2)
            with col1:
                fig_importance = px.bar(
                    importance,
                    x='Parameter',
                    y='Importance',
                    color='Importance',
                    title='Parameter Importance Ranking',
                    labels={'Importance': 'Importance Score'},
                    color_continuous_scale='Viridis',
                    text='Importance'
                )
                fig_importance.update_traces(texttemplate='%{text:.3f}', textposition='outside')
                fig_importance.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig_importance, use_container_width=True)
            with col2:
                st.markdown("### Top Critical Parameters")
                for idx, (_, row) in enumerate(importance.iterrows(), 1):
                    st.markdown(
                        f"""
                        **{idx}. {row['Parameter']}**
                        Importance: `{row['Importance']:.1%}`
                        """
                    )
            st.divider()
            st.markdown("## 📊 Golden vs Non-Golden Batch Comparison")
            if len(golden_batches) > 0 and len(non_golden_batches) > 0:
                comparison_data = {
                    'Parameter': ['Temperature (°C)', 'Pressure (bar)', 'pH', 'Cycle Time (min)', 'Yield (%)', 'Impurity (%)'],
                    'Golden Avg': [
                        golden_batches['temperature'].mean(),
                        golden_batches['pressure'].mean(),
                        golden_batches['ph'].mean(),
                        golden_batches['cycle_time'].mean(),
                        golden_batches['yield'].mean(),
                        golden_batches['impurity_percent'].mean()
                    ],
                    'Non-Golden Avg': [
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
                    go.Bar(x=comparison_df['Parameter'], y=comparison_df['Golden Avg'], name='Golden Batch', marker_color=GOLDEN_COLOR),
                    go.Bar(x=comparison_df['Parameter'], y=comparison_df['Non-Golden Avg'], name='Non-Golden Batch', marker_color=NON_GOLDEN_COLOR)
                ])
                fig_comparison.update_layout(
                    barmode='group',
                    title='Golden vs Non-Golden Batch Parameters',
                    xaxis_title='Parameter',
                    yaxis_title='Value',
                    height=400,
                    hovermode='x unified'
                )
                st.plotly_chart(fig_comparison, use_container_width=True)
                st.dataframe(comparison_df, use_container_width=True)
        # ==================== TAB 2: AI CHAT ASSISTANT ====================
        with tab2:
            st.markdown("## 💬 AI Batch Analysis Chat")
            st.markdown("Ask questions about your batches and get AI-powered insights!")
            # Initialize chat history
            if "chat_history" not in st.session_state:
                st.session_state.chat_history = []
            # Display chat history
            for message in st.session_state.chat_history:
                if message["role"] == "user":
                    st.chat_message("user").write(message["content"])
                else:
                    st.chat_message("assistant").write(message["content"])
            # Chat input
            user_input = st.chat_input("Ask about a batch... (e.g., 'What's wrong with batch B001?')")
            if user_input:
                st.session_state.chat_history.append({"role": "user", "content": user_input})
                # Prepare context for Gemini
                batch_data_str = batch_summary.to_string()
                if len(golden_batches) > 0:
                    golden_temp_min = golden_batches['temperature'].min()
                    golden_temp_max = golden_batches['temperature'].max()
                    golden_pressure_min = golden_batches['pressure'].min()
                    golden_pressure_max = golden_batches['pressure'].max()
                    golden_ph_min = golden_batches['ph'].min()
                    golden_ph_max = golden_batches['ph'].max()
                    golden_params = f"Temperature: {golden_temp_min:.1f}-{golden_temp_max:.1f}°C, Pressure: {golden_pressure_min:.2f}-{golden_pressure_max:.2f} bar, pH: {golden_ph_min:.1f}-{golden_ph_max:.1f}"
                else:
                    golden_params = "No golden batches available"
                system_prompt = f"""You are an expert manufacturing process engineer and quality control specialist.
 
You have access to batch production data and need to provide detailed analysis of batch performance.
 
GOLDEN BATCH PARAMETERS:
{golden_params}
 
BATCH DATA SUMMARY:
{batch_data_str}
 
CRITICAL PARAMETERS RANKING:
{importance.to_string()}
 
When a user asks about a specific batch:
1. Identify which parameters were outside the golden batch range
2. Explain the impact of each out-of-range parameter
3. Suggest what went wrong and why
4. Provide recommendations for improvement
5. Reference specific values from the data
 
Be specific, technical, and actionable in your responses."""
                # Call Gemini API
                try:
                    model = get_gemini_model()
                    response = model.generate_content(
                        f"{system_prompt}\n\nUser question: {user_input}",
                        generation_config=genai.types.GenerationConfig(
                            max_output_tokens=1024,
                        )
                    )
                    assistant_message = response.text
                    st.session_state.chat_history.append({"role": "assistant", "content": assistant_message})
                    st.rerun()
                except Exception as e:
                    error_msg = f"Error: {str(e)}\n\nMake sure your GEMINI_API_KEY is set correctly in Render environment variables."
                    st.session_state.chat_history.append({"role": "assistant", "content": error_msg})
                    st.rerun()
            # Example questions
            st.divider()
            st.markdown("### 💡 Example Questions")
            st.markdown("""
            - "What's wrong with batch B002?"
            - "Why did batch B004 fail?"
            - "Which batches had temperature issues?"
            - "Compare batch B001 with batch B005"
            - "What parameters caused the low yield in batch B010?"
            """)
        # ==================== TAB 3: DATA EXPLORER ====================
        with tab3:
            st.markdown("## 📋 All Batches Summary")
            batch_display = batch_summary.copy()
            batch_display['Status'] = batch_display['status'].apply(lambda x: '🟢 GOLDEN' if x == 'GOLDEN' else '🔴 NON-GOLDEN')
            batch_display = batch_display.round(2)
            st.dataframe(
                batch_display[['batch_id', 'Status', 'temperature', 'pressure', 'ph', 'cycle_time', 'yield', 'impurity_percent', 'score']],
                use_container_width=True,
                hide_index=True
            )
            st.divider()
            st.markdown("## 📊 Detailed Batch Analysis")
            selected_batch = st.selectbox(
                "Select a batch to view details:",
                batch_summary['batch_id'].unique(),
                key="batch_detail_selector"
            )
            if selected_batch:
                batch_data = batch_summary[batch_summary['batch_id'] == selected_batch].iloc[0]
                batch_full_data = df[df['batch_id'] == selected_batch]
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Batch Status", "🟢 GOLDEN" if batch_data['status'] == 'GOLDEN' else "🔴 NON-GOLDEN")
                with col2:
                    st.metric("Quality Score", f"{batch_data['score']:.1f}/100")
                with col3:
                    st.metric("Yield", f"{batch_data['yield']:.1f}%")
                with col4:
                    st.metric("Impurity", f"{batch_data['impurity_percent']:.2f}%")
                st.markdown(f"### Parameter Trends for {selected_batch}")
                fig_trend = go.Figure()
                fig_trend.add_trace(go.Scatter(
                    y=batch_full_data['temperature'],
                    name='Temperature (°C)',
                    mode='lines+markers',
                    line=dict(color='#e74c3c', width=2),
                    marker=dict(size=6)
                ))
                if len(golden_batches) > 0:
                    fig_trend.add_hline(
                        y=golden_batches['temperature'].mean(),
                        line_dash="dash",
                        line_color="green",
                        annotation_text="Golden Avg"
                    )
                fig_trend.update_layout(
                    title=f"Temperature Trend",
                    xaxis_title="Reading #",
                    yaxis_title="Temperature (°C)",
                    height=400,
                    hovermode='x unified'
                )
                st.plotly_chart(fig_trend, use_container_width=True)
                # Show all raw data for this batch
                st.markdown("### Raw Data")
                st.dataframe(batch_full_data, use_container_width=True)
 
else:
    st.info("👈 **Please upload a batch data file to get started!**")
    st.markdown("""
    ### 📋 Expected File Format
    Your CSV or Excel file should contain:
    - `batch_id` - Batch identifier
    - `temperature` - Process temperature
    - `pressure` - Process pressure
    - `ph` - pH level
    - `cycle_time` - Batch cycle time
    - `yield` - Batch yield percentage
    - `impurity_percent` - Impurity percentage
    """)

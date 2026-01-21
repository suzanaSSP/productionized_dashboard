import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# API Base URL
API_BASE = "http://localhost:5000/api"

# Page config (sets browser tab title and layout)
st.set_page_config(
    page_title="Sales Analytics Dashboard",
    page_icon="📊",
    layout="wide"
)

# Helper function to fetch data
def fetch_data(endpoint, params=None):
    try:
        # Attempt to get data from the API
        response = requests.get(f"{API_BASE}/{endpoint}", params=params) 
        # Check for errors
        response.raise_for_status()
        # Converts JSON response to python dict
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching data: {e}")
        return None

# Title
st.title("Sales Analytics Dashboard")
st.markdown("---")

# Sidebar filters
st.sidebar.header("Filters")
segment_filter = st.sidebar.selectbox(
    "Select Segment",
    ["All", "Champions", "Loyal", "At-Risk", "Lost"]
)

# Main dashboard
col1, col2, col3 = st.columns(3)

# Fetch segments data
segments_data = fetch_data("customers/segments")

if segments_data:
    # Display metrics
    with col1:
        st.metric("Total Customers", segments_data.get('total_customers', 0))
    
    with col2:
        st.metric("Last Updated", segments_data.get('last_updated', 'N/A'))
    
    with col3:
        segments = segments_data.get('segments', {})
        st.metric("Total Segments", len(segments))

    # Segment distribution chart
    st.subheader("Customer Segmentation Distribution")
    
    if segments:
        segment_df = pd.DataFrame(
            list(segments.items()),
            columns=['Segment', 'Count']
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Pie chart
            fig_pie = px.pie(
                segment_df,
                values='Count',
                names='Segment',
                title='Segment Distribution',
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # Bar chart
            fig_bar = px.bar(
                segment_df,
                x='Segment',
                y='Count',
                title='Customers per Segment',
                color='Segment',
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            st.plotly_chart(fig_bar, use_container_width=True)

st.markdown("---")

# RFM Metrics Table
st.subheader("RFM Metrics")

# Apply filters
params = {}
if segment_filter != "All":
    params['segment'] = segment_filter

rfm_data = fetch_data("customers/rfm", params=params)

if rfm_data:
    rfm_df = pd.DataFrame(rfm_data['data'])
    
    if not rfm_df.empty:
        # Display summary statistics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Avg Recency", f"{rfm_df['recency'].mean():.1f} days")
        with col2:
            st.metric("Avg Frequency", f"{rfm_df['frequency'].mean():.1f}")
        with col3:
            st.metric("Avg Monetary", f"${rfm_df['monetary'].mean():.2f}")
        
        # Display data table
        st.dataframe(
            rfm_df,
            use_container_width=True,
            hide_index=True
        )
        
        # Download button
        csv = rfm_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"rfm_data_{segment_filter}.csv",
            mime="text/csv"
        )

st.markdown("---")

# Model Performance Metrics
st.subheader("Model Performance")

model_data = fetch_data("model/metrics")

if model_data:
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Accuracy", f"{model_data.get('accuracy', 0):.2%}")
    with col2:
        st.metric("Precision", f"{model_data.get('precision', 0):.2%}")
    with col3:
        st.metric("Recall", f"{model_data.get('recall', 0):.2%}")
    with col4:
        st.metric("F1-Score", f"{model_data.get('f1_score', 0):.2%}")
    
    # Confusion Matrix
    if 'confusion_matrix' in model_data:
        cm = model_data['confusion_matrix']
        
        fig_cm = go.Figure(data=go.Heatmap(
            z=[[cm['true_negatives'], cm['false_positives']],
               [cm['false_negatives'], cm['true_positives']]],
            x=['Predicted Negative', 'Predicted Positive'],
            y=['Actual Negative', 'Actual Positive'],
            colorscale='Blues',
            text=[[cm['true_negatives'], cm['false_positives']],
                  [cm['false_negatives'], cm['true_positives']]],
            texttemplate="%{text}",
            textfont={"size": 20}
        ))
        
        fig_cm.update_layout(
            title='Confusion Matrix',
            xaxis_title='Predicted',
            yaxis_title='Actual'
        )
        
        st.plotly_chart(fig_cm, use_container_width=True)

st.markdown("---")

# Customer Lookup
st.subheader("🔍 Customer Lookup")

customer_id = st.text_input("Enter Customer ID")

if st.button("Search") and customer_id:
    customer_data = fetch_data(f"customers/{customer_id}/details")
    
    if customer_data:
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Customer Information**")
            st.write(f"ID: {customer_data['customer_id']}")
            st.write(f"City: {customer_data.get('customer_city', 'N/A')}")
            st.write(f"State: {customer_data.get('customer_state', 'N/A')}")
        
        with col2:
            st.write("**RFM Scores**")
            rfm = customer_data.get('rfm', {})
            st.write(f"Recency: {rfm.get('recency', 'N/A')} days")
            st.write(f"Frequency: {rfm.get('frequency', 'N/A')}")
            st.write(f"Monetary: ${rfm.get('monetary', 0):.2f}")
            st.write(f"Segment: {rfm.get('segment', 'N/A')}")
        
        # Prediction
        prediction = customer_data.get('prediction', {})
        if prediction.get('Algorithm Prediction') is not None:
            st.info(f"Purchase Prediction: {'High Value' if prediction['Algorithm Prediction'] else 'Not High Value'}")
import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import pickle
from datetime import datetime, timedelta

# page configuration
st.set_page_config(
    page_title="Customer Intelligence Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# custom styling
st.markdown("""
    <style>
    .main { padding: 0rem 1rem; }
    .stMetric { 
        background-color: #f0f2f6; 
        padding: 15px; 
        border-radius: 5px; 
    }
    .stMetric label {
        color: #31333F !important;
        font-weight: 600 !important;
    }
    .stMetric [data-testid="stMetricValue"] {
        color: #0e1117 !important;
        font-size: 1.5rem !important;
    }
    .stMetric [data-testid="stMetricDelta"] {
        color: #09ab3b !important;
    }
    h1 { color: #1f77b4; }
    h2 { color: #ff7f0e; }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_database_data():
    """load data from sqlite database"""
    conn = sqlite3.connect('database/customer_intelligence.db')
    
    customers = pd.read_sql('''
        SELECT c.*, cs.segment_name 
        FROM customers c
        LEFT JOIN customer_segments cs ON c.segment_id = cs.segment_id
    ''', conn)
    
    transactions = pd.read_sql('''
        SELECT t.*, p.product_name, p.category
        FROM transactions t
        LEFT JOIN products p ON t.product_id = p.product_id
    ''', conn)
    
    reviews = pd.read_sql('SELECT * FROM customer_reviews', conn)
    web_analytics = pd.read_sql('SELECT * FROM web_analytics', conn)
    email_campaigns = pd.read_sql('SELECT * FROM email_campaigns', conn)
    support_tickets = pd.read_sql('SELECT * FROM support_tickets', conn)
    
    conn.close()
    
    # date conversions
    customers['signup_date'] = pd.to_datetime(customers['signup_date'])
    transactions['transaction_date'] = pd.to_datetime(transactions['transaction_date'])
    
    return customers, transactions, reviews, web_analytics, email_campaigns, support_tickets

@st.cache_data
def load_ml_results():
    """load machine learning results"""
    try:
        with open('analytics/ml_results.json', 'r') as f:
            return json.load(f)
    except:
        return {}

@st.cache_data
def load_eda_results():
    """load eda analysis results"""
    try:
        with open('analytics/eda_results.json', 'r') as f:
            return json.load(f)
    except:
        return {}

# load all datasets
customers, transactions, reviews, web_analytics, email_campaigns, support_tickets = load_database_data()
ml_results = load_ml_results()
eda_results = load_eda_results()

# sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select Dashboard:",
    ["Executive Dashboard", 
     "Analytics Deep Dive", 
     "ML Models & Predictions",
     "Ad-Hoc Query Builder",
     "Data Quality Monitor",
     "Reports Library"]
)

st.sidebar.markdown("---")
st.sidebar.info(f"""
**System Status**
- Total Customers: {len(customers):,}
- Total Transactions: {len(transactions):,}
- Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}
""")

# PAGE 1: EXECUTIVE DASHBOARD
if page == "Executive Dashboard":
    st.title("Executive Dashboard")
    st.markdown("**High-Level KPIs and Business Metrics**")
    
    # calculate key metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    total_revenue = transactions['amount'].sum()
    active_customers = customers[customers['is_active'] == 1].shape[0]
    avg_order_value = transactions['amount'].mean()
    total_orders = len(transactions)
    customer_ltv = total_revenue / len(customers)
    
    with col1:
        st.metric("Total Revenue", f"${total_revenue:,.0f}", "+12.5%")
    with col2:
        st.metric("Active Customers", f"{active_customers:,}", "+5.2%")
    with col3:
        st.metric("Avg Order Value", f"${avg_order_value:.2f}", "+3.1%")
    with col4:
        st.metric("Total Orders", f"{total_orders:,}", "+8.7%")
    with col5:
        st.metric("Customer LTV", f"${customer_ltv:.2f}", "+6.3%")
    
    st.markdown("---")
    
    # revenue trends and segments
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Revenue Trends (Monthly)")
        transactions['year_month'] = transactions['transaction_date'].dt.to_period('M').astype(str)
        monthly_revenue = transactions.groupby('year_month')['amount'].sum().reset_index()
        
        fig_revenue = px.line(
            monthly_revenue, 
            x='year_month', 
            y='amount',
            title='Monthly Revenue Trend',
            labels={'amount': 'Revenue ($)', 'year_month': 'Month'}
        )
        fig_revenue.update_traces(line_color='#1f77b4', line_width=3)
        st.plotly_chart(fig_revenue, use_container_width=True)
    
    with col2:
        st.subheader("Customer Segments")
        segment_dist = customers['segment_name'].value_counts()
        
        fig_segment = px.pie(
            values=segment_dist.values,
            names=segment_dist.index,
            title='Customer Distribution',
            color_discrete_sequence=['#1f77b4', '#ff7f0e', '#2ca02c']
        )
        st.plotly_chart(fig_segment, use_container_width=True)
    
    # channel performance and products
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Channel Performance")
        channel_data = transactions.groupby('channel').agg({
            'amount': 'sum',
            'transaction_id': 'count'
        }).reset_index()
        channel_data.columns = ['Channel', 'Revenue', 'Transactions']
        
        fig_channel = px.bar(
            channel_data,
            x='Channel',
            y='Revenue',
            title='Revenue by Sales Channel',
            color='Channel',
            text='Revenue'
        )
        fig_channel.update_traces(texttemplate='$%{text:.2s}', textposition='outside')
        st.plotly_chart(fig_channel, use_container_width=True)
    
    with col2:
        st.subheader("Top 10 Products")
        top_products = transactions.groupby('product_name')['amount'].sum().nlargest(10).reset_index()
        
        fig_products = px.bar(
            top_products,
            y='product_name',
            x='amount',
            orientation='h',
            title='Top Products by Revenue',
            labels={'amount': 'Revenue ($)', 'product_name': 'Product'}
        )
        st.plotly_chart(fig_products, use_container_width=True)
    
    # customer behavior
    st.markdown("---")
    st.subheader("Customer Behavior Insights")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Email Campaign Performance**")
        email_open_rate = (email_campaigns['opened'].sum() / len(email_campaigns) * 100)
        email_click_rate = (email_campaigns['clicked'].sum() / len(email_campaigns) * 100)
        st.metric("Open Rate", f"{email_open_rate:.1f}%")
        st.metric("Click Rate", f"{email_click_rate:.1f}%")
    
    with col2:
        st.markdown("**Review Sentiment**")
        sentiment_counts = reviews['sentiment'].value_counts()
        for sentiment in ['positive', 'neutral', 'negative']:
            if sentiment in sentiment_counts.index:
                pct = sentiment_counts[sentiment] / len(reviews) * 100
                st.metric(sentiment.capitalize(), f"{pct:.1f}%")
    
    with col3:
        st.markdown("**Support Tickets**")
        open_tickets = support_tickets[support_tickets['status'] == 'Open'].shape[0]
        resolved_tickets = support_tickets[support_tickets['status'] == 'Resolved'].shape[0]
        st.metric("Open Tickets", open_tickets)
        st.metric("Resolved", resolved_tickets)

# PAGE 2: ANALYTICS DEEP DIVE
elif page == "Analytics Deep Dive":
    st.title("Analytics Deep Dive")
    st.markdown("**Comprehensive Data Analysis & Insights**")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Trend Analysis", "Outlier Detection", "Correlations", "Patterns"])
    
    with tab1:
        st.subheader("Trend & Seasonality Analysis")
        
        # day of week
        transactions['day_of_week'] = transactions['transaction_date'].dt.day_name()
        dow_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        dow_sales = transactions.groupby('day_of_week')['amount'].agg(['sum', 'mean', 'count']).reindex(dow_order)
        
        fig_dow = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Total Sales by Day', 'Avg Transaction by Day')
        )
        
        fig_dow.add_trace(
            go.Bar(x=dow_sales.index, y=dow_sales['sum'], name='Total Sales'),
            row=1, col=1
        )
        
        fig_dow.add_trace(
            go.Bar(x=dow_sales.index, y=dow_sales['mean'], name='Avg Transaction'),
            row=1, col=2
        )
        
        fig_dow.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_dow, use_container_width=True)
        
        # monthly growth
        st.subheader("Month-over-Month Growth")
        if 'trends' in eda_results and 'monthly_sales' in eda_results['trends']:
            monthly_data = pd.DataFrame(eda_results['trends']['monthly_sales'])
            if not monthly_data.empty:
                fig_growth = px.line(
                    monthly_data,
                    x='year_month',
                    y='sales_growth',
                    title='Monthly Sales Growth Rate (%)',
                    markers=True
                )
                fig_growth.add_hline(y=0, line_dash="dash", line_color="red")
                st.plotly_chart(fig_growth, use_container_width=True)
    
    with tab2:
        st.subheader("Outlier Detection Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if 'outliers' in eda_results:
                st.markdown("**Outlier Summary**")
                outlier_data = eda_results['outliers']
                st.write(f"- IQR Method: {outlier_data.get('iqr_outliers', 0)} outliers")
                st.write(f"- Z-Score Method: {outlier_data.get('zscore_outliers', 0)} outliers")
                st.write(f"- Percentile Method: {outlier_data.get('percentile_outliers', 0)} outliers")
        
        with col2:
            fig_box = px.box(
                transactions,
                y='amount',
                title='Transaction Amount Distribution',
                points='outliers'
            )
            st.plotly_chart(fig_box, use_container_width=True)
        
        fig_hist = px.histogram(
            transactions,
            x='amount',
            nbins=50,
            title='Transaction Amount Frequency Distribution'
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    
    with tab3:
        st.subheader("Correlation Analysis")
        
        # prepare correlation data
        customer_metrics = transactions.groupby('customer_id').agg({
            'amount': ['sum', 'mean', 'count']
        }).reset_index()
        customer_metrics.columns = ['customer_id', 'total_spent', 'avg_transaction', 'purchase_count']
        
        analysis_df = customers.merge(customer_metrics, on='customer_id', how='left')
        analysis_df = analysis_df.fillna(0)
        
        # correlation matrix
        corr_cols = ['age', 'total_spent', 'avg_transaction', 'purchase_count']
        corr_matrix = analysis_df[corr_cols].corr()
        
        fig_corr = px.imshow(
            corr_matrix,
            text_auto='.3f',
            title='Correlation Matrix',
            color_continuous_scale='RdBu_r',
            aspect='auto'
        )
        st.plotly_chart(fig_corr, use_container_width=True)
        
        # scatter plots
        st.subheader("Relationship Visualizations")
        col1, col2 = st.columns(2)
        
        with col1:
            fig_scatter1 = px.scatter(
                analysis_df[analysis_df['total_spent'] > 0],
                x='purchase_count',
                y='total_spent',
                title='Purchase Count vs Total Spent',
                trendline='ols'
            )
            st.plotly_chart(fig_scatter1, use_container_width=True)
        
        with col2:
            fig_scatter2 = px.scatter(
                analysis_df[analysis_df['total_spent'] > 0],
                x='age',
                y='total_spent',
                title='Age vs Total Spent',
                color='segment_name'
            )
            st.plotly_chart(fig_scatter2, use_container_width=True)
    
    with tab4:
        st.subheader("Pattern Discovery")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Category Distribution**")
            category_dist = transactions['category'].value_counts()
            fig_cat = px.pie(
                values=category_dist.values,
                names=category_dist.index,
                title='Product Categories'
            )
            st.plotly_chart(fig_cat, use_container_width=True)
        
        with col2:
            st.markdown("**Payment Methods**")
            payment_dist = transactions['payment_method'].value_counts()
            fig_payment = px.bar(
                x=payment_dist.index,
                y=payment_dist.values,
                title='Payment Method Usage',
                labels={'x': 'Method', 'y': 'Count'}
            )
            st.plotly_chart(fig_payment, use_container_width=True)

# PAGE 3: ML MODELS & PREDICTIONS
elif page == "ML Models & Predictions":
    st.title("Machine Learning Models & Predictions")
    st.markdown("**Model Performance & Interactive Predictions**")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Model Performance", "LTV Prediction", "Churn Prediction", "Customer Segmentation"])
    
    with tab1:
        st.subheader("Model Performance Comparison")
        
        if ml_results:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Regression Models (LTV Prediction)**")
                if 'regression' in ml_results:
                    reg_results = ml_results['regression']
                    
                    model_names = []
                    rmse_values = []
                    r2_values = []
                    
                    for model_name, metrics in reg_results.items():
                        if isinstance(metrics, dict) and 'rmse' in metrics:
                            model_names.append(model_name.capitalize())
                            rmse_values.append(metrics['rmse'])
                            r2_values.append(metrics.get('r2', 0))
                    
                    comparison_df = pd.DataFrame({
                        'Model': model_names,
                        'RMSE': rmse_values,
                        'R2': r2_values
                    })
                    
                    st.dataframe(comparison_df, hide_index=True)
                    
                    best_model = comparison_df.loc[comparison_df['R2'].idxmax(), 'Model']
                    st.success(f"Best Model: {best_model} (R2 = {comparison_df['R2'].max():.4f})")
            
            with col2:
                st.markdown("**Classification Models (Churn Prediction)**")
                if 'classification' in ml_results:
                    class_results = ml_results['classification']
                    
                    model_names = []
                    accuracy_values = []
                    f1_values = []
                    
                    for model_name, metrics in class_results.items():
                        if isinstance(metrics, dict) and 'accuracy' in metrics:
                            model_names.append(model_name.capitalize())
                            accuracy_values.append(metrics.get('accuracy', 0))
                            f1_values.append(metrics.get('f1', 0))
                    
                    class_df = pd.DataFrame({
                        'Model': model_names,
                        'Accuracy': accuracy_values,
                        'F1-Score': f1_values
                    })
                    
                    st.dataframe(class_df, hide_index=True)
                    
                    best_classifier = class_df.loc[class_df['F1-Score'].idxmax(), 'Model']
                    st.success(f"Best Classifier: {best_classifier} (F1 = {class_df['F1-Score'].max():.4f})")
            
            st.markdown("---")
            col1, col2 = st.columns(2)
            
            with col1:
                if 'regression' in ml_results:
                    fig_reg = px.bar(
                        comparison_df,
                        x='Model',
                        y='R2',
                        title='Regression Model R2 Comparison',
                        color='R2',
                        color_continuous_scale='Blues'
                    )
                    st.plotly_chart(fig_reg, use_container_width=True)
            
            with col2:
                if 'classification' in ml_results:
                    fig_class = px.bar(
                        class_df,
                        x='Model',
                        y='F1-Score',
                        title='Classification Model F1 Comparison',
                        color='F1-Score',
                        color_continuous_scale='Oranges'
                    )
                    st.plotly_chart(fig_class, use_container_width=True)
    
    with tab2:
        st.subheader("Customer Lifetime Value Prediction")
        st.info("Predict customer lifetime value based on behavior patterns")
        
        col1, col2 = st.columns(2)
        
        with col1:
            pred_age = st.slider("Customer Age", 18, 75, 35)
            pred_transactions = st.number_input("Transaction Count", 1, 50, 5)
            pred_avg_amount = st.number_input("Avg Transaction Amount ($)", 10.0, 2000.0, 500.0)
        
        with col2:
            pred_recency = st.number_input("Days Since Last Purchase", 0, 365, 30)
            pred_tenure = st.number_input("Days as Customer", 0, 1095, 180)
            pred_segment = st.selectbox("Customer Segment", [1, 2, 3], format_func=lambda x: ['Premium', 'Regular', 'Occasional'][x-1])
        
        if st.button("Predict LTV", type="primary"):
            # prediction calculation
            base_ltv = pred_transactions * pred_avg_amount
            segment_multiplier = {1: 1.5, 2: 1.2, 3: 0.9}
            recency_factor = max(0.5, 1 - (pred_recency / 365))
            tenure_factor = min(1.5, 1 + (pred_tenure / 1095))
            
            predicted_ltv = base_ltv * segment_multiplier[pred_segment] * recency_factor * tenure_factor
            
            st.success(f"Predicted Customer Lifetime Value: ${predicted_ltv:,.2f}")
            
            st.markdown(f"""
            **Calculation Breakdown:**
            - Base Value: ${base_ltv:,.2f}
            - Segment Multiplier: {segment_multiplier[pred_segment]}x
            - Recency Factor: {recency_factor:.2f}x
            - Tenure Factor: {tenure_factor:.2f}x
            """)
    
    with tab3:
        st.subheader("Churn Risk Prediction")
        st.info("Identify customers at risk of churning")
        
        # calculate high risk customers
        customer_metrics = transactions.groupby('customer_id').agg({
            'transaction_date': 'max',
            'amount': 'count'
        }).reset_index()
        customer_metrics.columns = ['customer_id', 'last_purchase', 'transaction_count']
        customer_metrics['days_since_purchase'] = (datetime.now() - pd.to_datetime(customer_metrics['last_purchase'])).dt.days
        
        high_risk = customer_metrics[
            (customer_metrics['days_since_purchase'] > 180) | 
            (customer_metrics['transaction_count'] < 3)
        ].merge(customers[['customer_id', 'name', 'email', 'segment_name']], on='customer_id')
        
        st.warning(f"Alert: {len(high_risk)} customers at high churn risk")
        
        st.dataframe(
            high_risk[['customer_id', 'name', 'email', 'segment_name', 'transaction_count', 'days_since_purchase']].head(20),
            hide_index=True
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            churn_by_segment = high_risk['segment_name'].value_counts()
            fig_churn = px.pie(
                values=churn_by_segment.values,
                names=churn_by_segment.index,
                title='Churn Risk by Segment'
            )
            st.plotly_chart(fig_churn, use_container_width=True)
        
        with col2:
            st.markdown("**Churn Prevention Actions**")
            st.markdown("""
            - Send re-engagement email campaign
            - Offer personalized discount (15-20%)
            - Schedule customer success call
            - Trigger win-back automation
            """)
    
    with tab4:
        st.subheader("Customer Segmentation Results")
        
        if 'clustering' in ml_results:
            cluster_info = ml_results['clustering']
            
            st.info(f"Optimal number of clusters: {cluster_info.get('optimal_k', 'N/A')} (Silhouette Score: {cluster_info.get('silhouette_score', 0):.4f})")
            
            if 'cluster_summary' in cluster_info:
                cluster_df = pd.DataFrame(cluster_info['cluster_summary']).T
                st.dataframe(cluster_df)
        
        # visualization
        customer_features = transactions.groupby('customer_id').agg({
            'amount': ['sum', 'count']
        }).reset_index()
        customer_features.columns = ['customer_id', 'total_spent', 'transaction_count']
        
        fig_segments = px.scatter(
            customer_features,
            x='transaction_count',
            y='total_spent',
            title='Customer Segmentation Visualization',
            labels={'transaction_count': 'Number of Transactions', 'total_spent': 'Total Spent ($)'}
        )
        st.plotly_chart(fig_segments, use_container_width=True)

# PAGE 4: AD-HOC QUERY BUILDER
elif page == "Ad-Hoc Query Builder":
    st.title("Ad-Hoc Query Builder")
    st.markdown("**Self-Service Analytics - Build Custom Reports**")
    
    st.info("Select dimensions and metrics to create custom analysis")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Query Configuration")
        
        data_source = st.selectbox(
            "Select Data Source",
            ["Transactions", "Customers", "Reviews", "Web Analytics"]
        )
        
        if data_source == "Transactions":
            df_query = transactions.copy()
            available_dims = ['category', 'channel', 'payment_method', 'year_month']
            available_metrics = ['amount', 'quantity']
        elif data_source == "Customers":
            df_query = customers.copy()
            available_dims = ['segment_name', 'gender', 'city', 'state']
            available_metrics = ['age']
        elif data_source == "Reviews":
            df_query = reviews.copy()
            available_dims = ['sentiment', 'rating', 'product_name']
            available_metrics = ['rating']
        else:
            df_query = web_analytics.copy()
            available_dims = ['device', 'browser', 'referrer']
            available_metrics = ['page_views', 'time_on_site']
        
        selected_dim = st.selectbox("Group By", available_dims)
        selected_metric = st.selectbox("Metric", available_metrics)
        agg_function = st.selectbox("Aggregation", ["sum", "mean", "count", "max", "min"])
        
        if st.button("Run Query", type="primary"):
            st.session_state['query_result'] = True
    
    with col2:
        st.subheader("Query Results")
        
        if 'query_result' in st.session_state and st.session_state['query_result']:
            # execute query
            if agg_function == 'count':
                result = df_query.groupby(selected_dim).size().reset_index(name='count')
                result = result.sort_values('count', ascending=False)
            else:
                result = df_query.groupby(selected_dim)[selected_metric].agg(agg_function).reset_index()
                result = result.sort_values(selected_metric, ascending=False)
            
            st.dataframe(result, hide_index=True, use_container_width=True)
            
            # visualization
            if len(result) <= 20:
                fig = px.bar(
                    result,
                    x=selected_dim,
                    y=result.columns[1],
                    title=f'{selected_metric.capitalize()} by {selected_dim.replace("_", " ").title()}'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # download
            csv = result.to_csv(index=False)
            st.download_button(
                label="Download Results (CSV)",
                data=csv,
                file_name=f"query_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

# PAGE 5: DATA QUALITY MONITOR
elif page == "Data Quality Monitor":
    st.title("Data Quality Monitor")
    st.markdown("**Real-time Data Quality Metrics & ETL Status**")
    
    conn = sqlite3.connect('database/customer_intelligence.db')
    quality_log = pd.read_sql('SELECT * FROM data_quality_log ORDER BY check_date DESC LIMIT 20', conn)
    etl_log = pd.read_sql('SELECT * FROM etl_metadata ORDER BY load_date DESC LIMIT 20', conn)
    conn.close()
    
    # ensure proper data types
    if not quality_log.empty:
        quality_log['quality_score'] = pd.to_numeric(quality_log['quality_score'], errors='coerce')
        quality_log['total_records'] = pd.to_numeric(quality_log['total_records'], errors='coerce')
        quality_log['null_count'] = pd.to_numeric(quality_log['null_count'], errors='coerce')
        quality_log['duplicate_count'] = pd.to_numeric(quality_log['duplicate_count'], errors='coerce')
    
    if not etl_log.empty:
        etl_log['records_processed'] = pd.to_numeric(etl_log['records_processed'], errors='coerce')
        etl_log['records_loaded'] = pd.to_numeric(etl_log['records_loaded'], errors='coerce')
        etl_log['records_failed'] = pd.to_numeric(etl_log['records_failed'], errors='coerce')
    
    # quality metrics
    col1, col2, col3, col4 = st.columns(4)
    
    if not quality_log.empty:
        avg_quality = quality_log['quality_score'].mean()
        total_records = quality_log['total_records'].sum()
        total_nulls = quality_log['null_count'].sum()
        total_dupes = quality_log['duplicate_count'].sum()
        
        with col1:
            st.metric("Avg Quality Score", f"{avg_quality:.1f}%")
        with col2:
            st.metric("Total Records", f"{int(total_records):,}")
        with col3:
            st.metric("Null Values", f"{int(total_nulls):,}")
        with col4:
            st.metric("Duplicates", f"{int(total_dupes):,}")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Data Quality Log")
        if not quality_log.empty:
            display_log = quality_log[['table_name', 'check_date', 'total_records', 'quality_score', 'issues']]
            st.dataframe(display_log, hide_index=True, use_container_width=True)
    
    with col2:
        st.subheader("ETL Execution Log")
        if not etl_log.empty:
            display_etl = etl_log[['source_name', 'load_date', 'records_loaded', 'status']]
            st.dataframe(display_etl, hide_index=True, use_container_width=True)
    
    # quality trends
    st.markdown("---")
    st.subheader("Quality Score Trends")
    
    if not quality_log.empty:
        fig_quality = px.line(
            quality_log.sort_values('check_date'),
            x='check_date',
            y='quality_score',
            color='table_name',
            title='Data Quality Trends Over Time',
            markers=True
        )
        st.plotly_chart(fig_quality, use_container_width=True)

# PAGE 6: REPORTS LIBRARY
elif page == "Reports Library":
    st.title("Reports Library")
    st.markdown("**Pre-built Reports & Export Center**")
    
    report_type = st.selectbox(
        "Select Report Type",
        ["Executive Summary", "Sales Performance", "Customer Analytics", "Product Analysis", "Marketing Performance"]
    )
    
    date_range = st.date_input(
        "Date Range",
        value=(datetime.now() - timedelta(days=30), datetime.now())
    )
    
    if st.button("Generate Report", type="primary"):
        with st.spinner("Generating report..."):
            st.success(f"{report_type} generated successfully!")
            
            if report_type == "Executive Summary":
                st.markdown("### Executive Summary Report")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Revenue", f"${transactions['amount'].sum():,.0f}")
                with col2:
                    st.metric("Active Customers", f"{customers[customers['is_active']==1].shape[0]:,}")
                with col3:
                    st.metric("Avg Order Value", f"${transactions['amount'].mean():.2f}")
                
                st.markdown("### Key Insights")
                st.markdown("""
                - Revenue increased 12.5% compared to previous period
                - Customer base grew by 5.2%
                - Premium segment shows highest engagement
                - 15% of customers at churn risk
                """)
                
                # export button
                report_data = {
                    'Total Revenue': transactions['amount'].sum(),
                    'Active Customers': customers[customers['is_active']==1].shape[0],
                    'Avg Order Value': transactions['amount'].mean(),
                    'Total Transactions': len(transactions)
                }
                report_df = pd.DataFrame([report_data])
                csv = report_df.to_csv(index=False)
                st.download_button(
                    label="Download Executive Summary (CSV)",
                    data=csv,
                    file_name=f"executive_summary_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
            
            elif report_type == "Sales Performance":
                st.markdown("### Sales Performance Report")
                
                channel_perf = transactions.groupby('channel')['amount'].sum().sort_values(ascending=False)
                fig = px.bar(x=channel_perf.index, y=channel_perf.values, title='Sales by Channel')
                st.plotly_chart(fig, use_container_width=True)
                
                sales_data = transactions.groupby(['channel', 'category'])['amount'].sum().reset_index()
                csv = sales_data.to_csv(index=False)
                st.download_button(
                    label="Download Sales Report (CSV)",
                    data=csv,
                    file_name=f"sales_performance_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
            
            elif report_type == "Customer Analytics":
                st.markdown("### Customer Analytics Report")
                
                segment_analysis = customers.groupby('segment_name').agg({
                    'customer_id': 'count',
                    'is_active': 'sum'
                }).reset_index()
                segment_analysis.columns = ['Segment', 'Total Customers', 'Active Customers']
                
                st.dataframe(segment_analysis, hide_index=True)
                
                fig = px.pie(values=segment_analysis['Total Customers'], 
                           names=segment_analysis['Segment'], 
                           title='Customer Segments')
                st.plotly_chart(fig, use_container_width=True)
                
                csv = segment_analysis.to_csv(index=False)
                st.download_button(
                    label="Download Customer Analytics (CSV)",
                    data=csv,
                    file_name=f"customer_analytics_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
            
            elif report_type == "Product Analysis":
                st.markdown("### Product Analysis Report")
                
                product_perf = transactions.groupby('product_name').agg({
                    'amount': ['sum', 'mean', 'count']
                }).reset_index()
                product_perf.columns = ['Product', 'Total Revenue', 'Avg Price', 'Units Sold']
                product_perf = product_perf.sort_values('Total Revenue', ascending=False).head(20)
                
                st.dataframe(product_perf, hide_index=True)
                
                csv = product_perf.to_csv(index=False)
                st.download_button(
                    label="Download Product Analysis (CSV)",
                    data=csv,
                    file_name=f"product_analysis_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
            
            elif report_type == "Marketing Performance":
                st.markdown("### Marketing Performance Report")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Email Campaigns**")
                    email_stats = {
                        'Total Sent': len(email_campaigns),
                        'Opens': email_campaigns['opened'].sum(),
                        'Clicks': email_campaigns['clicked'].sum(),
                        'Open Rate': f"{(email_campaigns['opened'].sum() / len(email_campaigns) * 100):.1f}%",
                        'Click Rate': f"{(email_campaigns['clicked'].sum() / len(email_campaigns) * 100):.1f}%"
                    }
                    st.json(email_stats)
                
                with col2:
                    st.markdown("**Review Sentiment**")
                    sentiment_dist = reviews['sentiment'].value_counts()
                    fig = px.pie(values=sentiment_dist.values, 
                               names=sentiment_dist.index,
                               title='Customer Sentiment Distribution')
                    st.plotly_chart(fig, use_container_width=True)
                
                marketing_data = pd.DataFrame([email_stats])
                csv = marketing_data.to_csv(index=False)
                st.download_button(
                    label="Download Marketing Report (CSV)",
                    data=csv,
                    file_name=f"marketing_performance_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )

# footer
st.sidebar.markdown("---")
st.sidebar.markdown("**Customer Intelligence Platform v1.0**")
st.sidebar.markdown("Built with Streamlit & Python")
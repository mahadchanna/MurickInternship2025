import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🛍️ Customer Segmentation Dashboard</h1>', unsafe_allow_html=True)
st.markdown("### 📊 E-Commerce Customer Analysis & Targeting Platform")

# Sidebar
with st.sidebar:
    st.title("⚙️ Settings")
    uploaded_file = st.file_uploader("Upload Excel File", type=['xlsx'])
    st.markdown("---")
    num_clusters = st.slider("Number of Clusters", min_value=3, max_value=8, value=4)
    st.markdown("---")
    st.markdown("### 📖 About")
    st.info("""
    This dashboard uses K-Means clustering and RFM analysis to segment customers based on:
    - **Recency**: Days since last purchase
    - **Frequency**: Number of orders
    - **Monetary**: Total spend
    """)

@st.cache_data
def load_data(file):
    return pd.read_excel(file)

@st.cache_data
def process_data(df):
    df_clean = df.copy()
    df_clean = df_clean[df_clean['CustomerID'].notna()]
    df_clean = df_clean[~df_clean['InvoiceNo'].astype(str).str.startswith('C')]
    df_clean = df_clean[(df_clean['Quantity'] > 0) & (df_clean['UnitPrice'] > 0)]
    df_clean = df_clean[(df_clean['Quantity'] < 10000) & (df_clean['UnitPrice'] < 1000)]
    df_clean['TotalPrice'] = df_clean['Quantity'] * df_clean['UnitPrice']
    df_clean['InvoiceDate'] = pd.to_datetime(df_clean['InvoiceDate'])
    return df_clean

@st.cache_data
def calculate_rfm(df_clean):
    snapshot_date = df_clean['InvoiceDate'].max() + timedelta(days=1)
    rfm = df_clean.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
        'InvoiceNo': 'nunique',
        'TotalPrice': 'sum'
    }).reset_index()
    rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']
    rfm['AvgOrderValue'] = rfm['Monetary'] / rfm['Frequency']
    return rfm

@st.cache_data
def perform_clustering(rfm, n_clusters):
    rfm_scaled = rfm.copy()
    rfm_scaled['Monetary_log'] = np.log1p(rfm_scaled['Monetary'])
    rfm_scaled['Frequency_sqrt'] = np.sqrt(rfm_scaled['Frequency'])
    features = ['Recency', 'Frequency_sqrt', 'Monetary_log']
    scaler = StandardScaler()
    rfm_scaled_array = scaler.fit_transform(rfm_scaled[features])
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    rfm['Cluster'] = kmeans.fit_predict(rfm_scaled_array)
    pca = PCA(n_components=2)
    rfm_pca = pca.fit_transform(rfm_scaled_array)
    rfm['PCA1'] = rfm_pca[:, 0]
    rfm['PCA2'] = rfm_pca[:, 1]
    
    def name_segment(row):
        if row['Recency'] <= rfm['Recency'].quantile(0.25) and row['Monetary'] >= rfm['Monetary'].quantile(0.75):
            return 'Champions'
        elif row['Recency'] <= rfm['Recency'].quantile(0.5) and row['Frequency'] >= rfm['Frequency'].quantile(0.5):
            return 'Loyal Customers'
        elif row['Recency'] > rfm['Recency'].quantile(0.75) and row['Frequency'] <= rfm['Frequency'].quantile(0.5):
            return 'At Risk'
        elif row['Recency'] > rfm['Recency'].quantile(0.75):
            return 'Lost Customers'
        else:
            return f'Cluster {row["Cluster"]}'
    
    rfm['Segment'] = rfm.apply(name_segment, axis=1)
    return rfm, pca.explained_variance_ratio_

# Main app logic
if uploaded_file is not None:
    try:
        with st.spinner('Loading data...'):
            df = load_data(uploaded_file)
            df_clean = process_data(df)
            rfm = calculate_rfm(df_clean)
            rfm, pca_variance = perform_clustering(rfm, num_clusters)
        
        st.success(f'✅ Processed {len(df_clean):,} transactions from {rfm.shape[0]:,} customers!')
        
        # Overview Metrics
        st.markdown("## 📈 Key Metrics")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Customers", f"{len(rfm):,}")
        with col2:
            st.metric("Avg Order Value", f"£{rfm['AvgOrderValue'].mean():.2f}")
        with col3:
            st.metric("Total Revenue", f"£{rfm['Monetary'].sum():,.2f}")
        with col4:
            st.metric("Avg Frequency", f"{rfm['Frequency'].mean():.1f}")
        
        st.markdown("---")
        
        # Tabs
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Segments Overview", "🎯 Segment Details", "📉 Visualizations", "💡 Recommendations"])
        
        with tab1:
            st.markdown("## 🎯 Customer Segments Distribution")
            col1, col2 = st.columns([1, 1])
            
            with col1:
                segment_counts = rfm['Segment'].value_counts()
                fig_pie = px.pie(values=segment_counts.values, names=segment_counts.index,
                                title="Customer Distribution by Segment", hole=0.4,
                                color_discrete_sequence=px.colors.qualitative.Set3)
                fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with col2:
                segment_revenue = rfm.groupby('Segment')['Monetary'].sum().sort_values(ascending=False)
                fig_bar = px.bar(x=segment_revenue.index, y=segment_revenue.values,
                                title="Total Revenue by Segment",
                                labels={'x': 'Segment', 'y': 'Revenue (£)'},
                                color=segment_revenue.values, color_continuous_scale='Viridis')
                st.plotly_chart(fig_bar, use_container_width=True)
            
            st.markdown("### 📋 Segment Summary Statistics")
            segment_summary = rfm.groupby('Segment').agg({
                'CustomerID': 'count', 'Recency': 'mean', 'Frequency': 'mean',
                'Monetary': 'mean', 'AvgOrderValue': 'mean'
            }).round(2)
            segment_summary.columns = ['Customer Count', 'Avg Recency (days)', 
                                      'Avg Frequency', 'Avg Monetary (£)', 'Avg Order Value (£)']
            st.dataframe(segment_summary, use_container_width=True)
        
        with tab2:
            st.markdown("## 🔍 Detailed Segment Analysis")
            selected_segment = st.selectbox("Select Segment to Analyze", rfm['Segment'].unique())
            segment_data = rfm[rfm['Segment'] == selected_segment]
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Customers", f"{len(segment_data):,}")
            with col2:
                st.metric("Avg Recency", f"{segment_data['Recency'].mean():.0f} days")
            with col3:
                st.metric("Avg Frequency", f"{segment_data['Frequency'].mean():.1f}")
            with col4:
                st.metric("Avg Monetary", f"£{segment_data['Monetary'].mean():.2f}")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                fig_r = px.histogram(segment_data, x='Recency', title='Recency Distribution',
                                    color_discrete_sequence=['#3498db'])
                st.plotly_chart(fig_r, use_container_width=True)
            with col2:
                fig_f = px.histogram(segment_data, x='Frequency', title='Frequency Distribution',
                                    color_discrete_sequence=['#2ecc71'])
                st.plotly_chart(fig_f, use_container_width=True)
            with col3:
                fig_m = px.histogram(segment_data, x='Monetary', title='Monetary Distribution',
                                    color_discrete_sequence=['#e74c3c'])
                st.plotly_chart(fig_m, use_container_width=True)
        
        with tab3:
            st.markdown("## 📊 Advanced Visualizations")
            fig_3d = px.scatter_3d(rfm, x='Recency', y='Frequency', z='Monetary',
                                  color='Segment', size='AvgOrderValue',
                                  title='3D Customer Segmentation',
                                  color_discrete_sequence=px.colors.qualitative.Set2, height=600)
            st.plotly_chart(fig_3d, use_container_width=True)
            
            col1, col2 = st.columns(2)
            with col1:
                fig_pca = px.scatter(rfm, x='PCA1', y='PCA2', color='Segment',
                                    title=f'PCA Visualization (Variance: {pca_variance.sum():.2%})',
                                    color_discrete_sequence=px.colors.qualitative.Set3)
                st.plotly_chart(fig_pca, use_container_width=True)
            
            with col2:
                segment_means = rfm.groupby('Segment')[['Recency', 'Frequency', 'Monetary']].mean()
                for col in segment_means.columns:
                    segment_means[col] = ((segment_means[col] - segment_means[col].min()) / 
                                         (segment_means[col].max() - segment_means[col].min()) * 100)
                segment_means['Recency'] = 100 - segment_means['Recency']
                
                fig_radar = go.Figure()
                for segment in segment_means.index:
                    fig_radar.add_trace(go.Scatterpolar(
                        r=segment_means.loc[segment].values.tolist() + [segment_means.loc[segment].values[0]],
                        theta=['Recency', 'Frequency', 'Monetary', 'Recency'],
                        fill='toself', name=segment
                    ))
                fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                                       title='Segment Comparison - Radar Chart', height=500)
                st.plotly_chart(fig_radar, use_container_width=True)
        
        with tab4:
            st.markdown("## 💡 Marketing Recommendations")
            recommendations = {
                'Champions': ['🏆', 'Best customers - Recent buyers, frequent orders, high spend', 
                             ['Reward with VIP programs', 'Early access to new products', 
                              'Request reviews and referrals', 'Upsell premium products'], '#27ae60'],
                'Loyal Customers': ['⭐', 'Regular customers with consistent purchasing behavior',
                                   ['Loyalty rewards and points', 'Personalized recommendations',
                                    'Cross-sell complementary products', 'Excellent customer service'], '#3498db'],
                'At Risk': ['⚠️', 'Previously good customers who haven\'t purchased recently',
                           ['Reactivation campaigns', 'Special discounts (10-20% off)',
                            'Gather feedback', 'Remind them of loyalty benefits'], '#f39c12'],
                'Lost Customers': ['❌', 'Haven\'t purchased in a long time',
                                  ['Win-back campaigns (25-30% off)', 'Survey why they left',
                                   'Showcase new products', 'Limited-time offers'], '#e74c3c']
            }
            
            for segment, info in recommendations.items():
                if segment in rfm['Segment'].values:
                    count = len(rfm[rfm['Segment'] == segment])
                    revenue = rfm[rfm['Segment'] == segment]['Monetary'].sum()
                    st.markdown(f"""<div style="background-color: {info[3]}; padding: 1.5rem; 
                                border-radius: 0.8rem; color: white; margin: 1rem 0;">
                                <h3>{info[0]} {segment}</h3>
                                <p><strong>{count:,} customers</strong> | <strong>£{revenue:,.2f} revenue</strong></p>
                                <p>{info[1]}</p></div>""", unsafe_allow_html=True)
                    st.markdown("**Recommended Actions:**")
                    for strategy in info[2]:
                        st.markdown(f"- {strategy}")
                    st.markdown("---")
            
            st.markdown("### 📥 Export Data")
            col1, col2 = st.columns(2)
            with col1:
                csv = rfm.to_csv(index=False).encode('utf-8')
                st.download_button("Download Customer Segments", csv, "customer_segments.csv", "text/csv")
            with col2:
                summary_csv = segment_summary.to_csv().encode('utf-8')
                st.download_button("Download Segment Summary", summary_csv, "segment_summary.csv", "text/csv")
    
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        st.info("Please ensure your file has the correct columns: InvoiceNo, CustomerID, Quantity, UnitPrice, InvoiceDate")
else:
    st.info("👆 Please upload an Excel file to begin analysis")
    st.markdown("""
    ### 📚 How to Use:
    1. Upload your **Online Retail.xlsx** file using the sidebar
    2. Adjust the number of clusters if needed
    3. Explore the different tabs for insights
    4. Download segmented customer data
    
    ### 📊 Expected Data Format:
    - **InvoiceNo**: Transaction ID
    - **CustomerID**: Unique customer identifier
    - **Quantity**: Number of items purchased
    - **UnitPrice**: Price per unit
    - **InvoiceDate**: Date of transaction
    """)

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import traceback
import random
from PIL import Image
import os

# Page configuration
st.set_page_config(
    page_title="Spare Parts Demand Forecast",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        padding: 1rem 0;
        border-bottom: 3px solid #e0e0e0;
    }
    
    .kpi-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        backdrop-filter: blur(4px);
        border: 1px solid rgba(255, 255, 255, 0.18);
        margin-bottom: 1rem;
    }
    
    .kpi-value {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .kpi-label {
        font-size: 0.9rem;
        opacity: 0.9;
        font-weight: 500;
    }
    
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #2c3e50;
        margin: 2rem 0 1rem 0;
        padding-left: 0.5rem;
        border-left: 4px solid #3498db;
    }
    
    .upload-section {
        background: #f8f9fa;
        padding: 2rem;
        border-radius: 10px;
        border: 2px dashed #dee2e6;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .stMetric {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    
    .time-series-container {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    
    .refresh-button {
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

def create_kpi_card(value, label, color="#667eea"):
    """Create a styled KPI card"""
    return f"""
    <div class="kpi-container" style="background: linear-gradient(135deg, {color} 0%, #764ba2 100%);">
        <div class="kpi-value">{value}</div>
        <div class="kpi-label">{label}</div>
    </div>
    """
def get_data_path():
    """Get correct data path for both local and cloud deployment"""
    if os.path.exists('demo-ui/data'):
        # Running from repo root (Streamlit Cloud)
        return 'demo-ui/data/'
    elif os.path.exists('data'):
        # Running from demo-ui folder (Local)
        return 'data/'
    else:
        # Fallback
        return 'data/'

def get_ref_path():
    """Get correct ref path for both local and cloud deployment"""
    if os.path.exists('demo-ui/ref'):
        return 'demo-ui/ref/'
    elif os.path.exists('ref'):
        return 'ref/'
    else:
        return 'ref/'

@st.cache_data
def load_and_process_data():
    """Load and process the CSV data from background"""
    try:
        data_path = get_data_path()
        df = pd.read_csv(f'{data_path}daikins_spare_parts_proc_cons_melted_merged_meta_combined.csv')
        df['MONTH_YEAR'] = pd.to_datetime(df['MONTH_YEAR'])
        df['Classification_Revised'] = df.apply(
            lambda row: row['Classification'] if row['Cons_Demand_TD'] == 'Exists' else "No Demand",
            axis=1
        )
        return df
    except FileNotFoundError:
        st.error("❌ Data file not found. Please ensure 'data/source.csv' exists in the project directory.")
        return None
    except Exception as e:
        st.error(f"❌ Error loading data file: {str(e)}")
        return None

def calculate_kpis(df):
    """Calculate all KPIs from the filtered data"""
    # Filter data where Cons_Demand_TD = 'Exists'
    filtered_df = df[df['Cons_Demand_TD'] == 'Exists'].copy()
    
    category_dist = filtered_df.groupby("CategoryABC_x").agg(count=("PartCode", "nunique"))
    print(category_dist)
    kpis = {
        'total_parts': filtered_df['PartCode'].nunique(),
        'category_dist': category_dist,
        'unique_makes': filtered_df['Make_consumption'].nunique(),
        'unique_machines': filtered_df['CommonMachineName_consumption'].nunique(),
        'unique_lines': filtered_df['Line_consumption'].nunique(),
        'price_data': filtered_df[filtered_df['MONTH_YEAR'] == '01-01-2022']['EachPartPrice_consumption'].dropna()
    }
    
    return kpis, filtered_df

def create_category_distribution_chart(category_dist):
    """Create category distribution pie chart"""
    fig = px.pie(
        values=category_dist['count'],
        names=category_dist.index,
        title="Category ABC Distribution",
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    fig.update_traces(
        textposition='inside',
        textinfo='percent+label',
        hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
    )
    fig.update_layout(
        showlegend=True,
        height=400,
        font=dict(size=12),
        title_x=0.5
    )
    return fig

def create_price_histogram(price_data):
    """Create price distribution histogram"""
    fig = px.histogram(
        x=price_data,
        nbins=20,
        title="Price Distribution",
        labels={'x': 'Price (₹)', 'y': 'Frequency'},
        color_discrete_sequence=['#3498db']
    )
    fig.update_layout(
        showlegend=False,
        height=400,
        xaxis_title="Price (₹)",
        yaxis_title="Number of Parts",
        title_x=0.5
    )
    fig.update_traces(
        hovertemplate='Price Range: ₹%{x}<br>Count: %{y}<extra></extra>'
    )
    return fig

def create_classification_bar_chart(df):
    """Create bar chart for Classification_Revised distribution"""
    classification_counts = df.groupby('Classification_Revised')['PartCode'].nunique().reset_index()
    classification_counts.columns = ['Classification_Revised', 'Unique_PartCode_Count']
    
    fig = px.bar(
        classification_counts,
        x='Classification_Revised',
        y='Unique_PartCode_Count',
        title="Distribution of Parts by Classification",
        labels={'Classification_Revised': 'Classification', 'Unique_PartCode_Count': 'Unique PartCode Count'},
        color='Classification_Revised',
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    
    # Add data labels on bars
    fig.update_traces(
        texttemplate='%{y}',
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>Count: %{y}<extra></extra>'
    )
    
    fig.update_layout(
        height=500,
        showlegend=False,
        title_x=0.5,
        xaxis_title="Classification",
        yaxis_title="Unique PartCode Count"
    )
    
    return fig

def get_random_partcode_by_classification(df, classification):
    """Get a random PartCode for a given classification"""
    parts_in_class = df[df['Classification_Revised'] == classification]['PartCode'].unique()
    return random.choice(parts_in_class) if len(parts_in_class) > 0 else None

def create_time_series_plot(df, partcode, classification):
    """Create time series plot for a specific PartCode"""
    part_data = df[df['PartCode'] == partcode].copy()
    part_data = part_data.sort_values('MONTH_YEAR')
    
    fig = px.line(
        part_data,
        x='MONTH_YEAR',
        y='CONSUMPTION',
        title=f'{classification} - PartCode: {partcode}',
        labels={'MONTH_YEAR': 'Month-Year', 'CONSUMPTION': 'Consumption'},
        markers=True
    )
    
    # Add data labels
    fig.update_traces(
        mode='lines+markers+text',
        text=part_data['CONSUMPTION'],
        textposition='top center',
        textfont_size=10,
        line=dict(width=3),
        marker=dict(size=8)
    )
    
    fig.update_layout(
        height=350,
        title_x=0.5,
        showlegend=False,
        xaxis_title="Month-Year",
        yaxis_title="Consumption"
    )
    
    return fig

# Initialize session state for random part codes
if 'random_parts' not in st.session_state:
    st.session_state.random_parts = {}

def login_interface():
    """Simple login interface"""
    st.markdown('<h1 class="main-header">🔐 Spare Parts Demand Forecasting Demo - Evomaton</h1>', unsafe_allow_html=True)
    
    # Create a centered login form
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
       
        
        st.markdown("### Welcome! Please login to view the demo")
        
        # Login form
        with st.form("login_form"):
            username = st.text_input("Username", placeholder="Enter your username")
            password = st.text_input("Password", type="password", placeholder="Enter your password")
            
            submitted = st.form_submit_button("🚀 Login", use_container_width=True)
            
            if submitted:
                # Simple authentication (replace with your actual authentication logic)
                if authenticate_user(username, password):
                    st.session_state.authenticated = True
                    st.session_state.username = username
                    st.success("Login successful! Redirecting...")
                    st.rerun()
                else:
                    st.error("❌ Invalid username or password. Please try again.")
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Demo credentials info
        # st.info("""
        # **Demo Credentials:**
        # - Username: `admin` | Password: `admin123`
        # - Username: `demo` | Password: `demo123`
        # - Username: `user` | Password: `password`
        # """)

def authenticate_user(username, password):
    username = st.secrets["login"]["username"]
    password = st.secrets["login"]["password"]
    # Demo credentials dictionary
    valid_credentials = {
        username: password,

    }
    
    return username in valid_credentials and valid_credentials[username] == password

def logout():
    """Logout function"""
    st.session_state.authenticated = False
    st.session_state.username = ""
    st.rerun()



# Main dashboard
def main():
        

    # Initialize session state
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'username' not in st.session_state:
        st.session_state.username = ""
    
    # Show login interface if not authenticated
    if not st.session_state.authenticated:
        login_interface()
        return
    
    # Add logout button in sidebar when authenticated
    with st.sidebar:
        st.markdown(f"👋 Welcome, **{st.session_state.username}**!")
        if st.button("🚪 Logout"):
            logout()
        st.markdown("---")

    
    st.markdown('<h1 class="main-header">🔧 Spare Parts Demand Forecast Demo</h1>', unsafe_allow_html=True)
    
    st.write("🔍 **File System Debug:**")

    try:
        data_path = get_data_path()
        files = os.listdir(data_path)
        st.success(f"✅ Found data folder with {len(files)} files: {files}")
    except:
        st.error("❌ Cannot access data folder")
        st.write("Current directory contents:", os.listdir('.'))
    # Load data from background
    with st.spinner('Loading spare parts data...'):
        df = load_and_process_data()
    
    if df is not None:
        try:
            kpis, filtered_df = calculate_kpis(df)
            
            # Display key statistics
            st.markdown('<h2 class="section-header">📊 Key Statistics </h2>', unsafe_allow_html=True)
            
            # KPI Cards Row 1
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(
                    create_kpi_card(
                        f"{kpis['total_parts']:,}",
                        "Total Unique Parts",
                        "#e74c3c"
                    ),
                    unsafe_allow_html=True
                )
            
            with col2:
                st.markdown(
                    create_kpi_card(
                        f"{kpis['unique_makes']:,}",
                        "Unique Makes",
                        "#27ae60"
                    ),
                    unsafe_allow_html=True
                )
            
            with col3:
                st.markdown(
                    create_kpi_card(
                        f"{kpis['unique_machines']:,}",
                        "Unique Machines",
                        "#f39c12"
                    ),
                    unsafe_allow_html=True
                )
            
            # KPI Cards Row 2
            col4, col5, col6 = st.columns(3)
            
            with col4:
                st.markdown(
                    create_kpi_card(
                        f"{kpis['unique_lines']:,}",
                        "Production Lines",
                        "#9b59b6"
                    ),
                    unsafe_allow_html=True
                )
            
            with col5:
                avg_price = kpis['price_data'].mean()
                st.markdown(
                    create_kpi_card(
                        f"₹{avg_price:.2f}",
                        "Average Part Price",
                        "#34495e"
                    ),
                    unsafe_allow_html=True
                )
            
            with col6:
                max_price = kpis['price_data'].max()
                st.markdown(
                    create_kpi_card(
                        f"₹{max_price:.2f}",
                        "Highest Part Price",
                        "#e67e22"
                    ),
                    unsafe_allow_html=True
                )
            
            # Charts Section
            st.markdown('<h2 class="section-header">📈 Key Visualizations</h2>', unsafe_allow_html=True)
            
            col_chart1, col_chart2 = st.columns(2)
            
            with col_chart1:
                # Category distribution pie chart
                category_fig = create_category_distribution_chart(kpis['category_dist'])
                st.plotly_chart(category_fig, use_container_width=True)
            
            with col_chart2:
                # Price distribution histogram
                price_fig = create_price_histogram(kpis['price_data'])
                st.plotly_chart(price_fig, use_container_width=True)
            
            # Summary insights
            st.markdown('<h2 class="section-header">💡 Key Insights</h2>', unsafe_allow_html=True)
            
            insights_col1, insights_col2 = st.columns(2)
            
            with insights_col1:
                st.info(f"""
                **Inventory Overview:**
                - Managing {kpis['total_parts']:,} unique spare parts
                - Sourced from {kpis['unique_makes']} different manufacturers
                - Supporting {kpis['unique_machines']} different machines
                """)
            
            with insights_col2:
                dominant_category = kpis['category_dist'].index[2]
                category_pct = (kpis['category_dist']['count'].iloc[2] / kpis['category_dist']['count'].sum()) * 100
                st.success(f"""
                **Category Analysis:**
                - Category {dominant_category} dominates with {category_pct:.1f}% of parts
                - Price range: ₹{kpis['price_data'].min():.2f} - ₹{kpis['price_data'].max():.2f}
                - Median price: ₹{kpis['price_data'].median():.2f}
                """)
            
            # NEW SECTION: Demand Classification Analysis
            st.markdown('<h2 class="section-header">🎯 Demand Classification</h2>', unsafe_allow_html=True)
            
            # Reference image (if exists)
            try:
                ref_path = get_ref_path()
                if os.path.exists(f'{ref_path}demand_classification.png'):
                    ref_image = Image.open(f'{ref_path}demand_classification.png')
                    st.image(ref_image, caption="Demand Classification Reference", use_column_width=True)
            except:
                st.info(f"Reference image not found at '{ref_path}demand_classification.png'")
            
            # Classification bar chart
            st.markdown('<h3 class="section-header">📊 Classification Distribution</h3>', unsafe_allow_html=True)
            classification_fig = create_classification_bar_chart(df)
            st.plotly_chart(classification_fig, use_container_width=True)
            
            # Time Series Section
            st.markdown('<h2 class="section-header">📈 Sample for each Classification</h2>', unsafe_allow_html=True)
            
            # Get unique classifications
            classifications = df['Classification_Revised'].unique()
            classifications = [c for c in classifications if c != 'No Demand']  # Exclude No Demand for time series
            
            # Create 2x2 grid for time series plots
            if len(classifications) >= 4:
                # First row
                col1, col2 = st.columns(2)
                
                # Initialize random parts if not exists
                for i, classification in enumerate(classifications[:4]):
                    if f'part_{i}' not in st.session_state.random_parts:
                        st.session_state.random_parts[f'part_{i}'] = get_random_partcode_by_classification(df, classification)
                
                with col1:
                    classification = classifications[0]
                    st.markdown(f'<div class="time-series-container">', unsafe_allow_html=True)
                    if st.button(f"🔄 Refresh {classification}", key=f"refresh_0", help="Get random part from this classification"):
                        st.session_state.random_parts['part_0'] = get_random_partcode_by_classification(df, classification)
                    
                    if st.session_state.random_parts['part_0']:
                        fig = create_time_series_plot(df, st.session_state.random_parts['part_0'], classification)
                        st.plotly_chart(fig, use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col2:
                    classification = classifications[1]
                    st.markdown(f'<div class="time-series-container">', unsafe_allow_html=True)
                    if st.button(f"🔄 Refresh {classification}", key=f"refresh_1", help="Get random part from this classification"):
                        st.session_state.random_parts['part_1'] = get_random_partcode_by_classification(df, classification)
                    
                    if st.session_state.random_parts['part_1']:
                        fig = create_time_series_plot(df, st.session_state.random_parts['part_1'], classification)
                        st.plotly_chart(fig, use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Second row
                col3, col4 = st.columns(2)
                
                with col3:
                    classification = classifications[2]
                    st.markdown(f'<div class="time-series-container">', unsafe_allow_html=True)
                    if st.button(f"🔄 Refresh {classification}", key=f"refresh_2", help="Get random part from this classification"):
                        st.session_state.random_parts['part_2'] = get_random_partcode_by_classification(df, classification)
                    
                    if st.session_state.random_parts['part_2']:
                        fig = create_time_series_plot(df, st.session_state.random_parts['part_2'], classification)
                        st.plotly_chart(fig, use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col4:
                    classification = classifications[3]
                    st.markdown(f'<div class="time-series-container">', unsafe_allow_html=True)
                    if st.button(f"🔄 Refresh {classification}", key=f"refresh_3", help="Get random part from this classification"):
                        st.session_state.random_parts['part_3'] = get_random_partcode_by_classification(df, classification)
                    
                    if st.session_state.random_parts['part_3']:
                        fig = create_time_series_plot(df, st.session_state.random_parts['part_3'], classification)
                        st.plotly_chart(fig, use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
            
            else:
                st.warning(f"Only {len(classifications)} classifications available for time series plots.")
            
            # NEW SECTION: Modelling Approaches
            st.markdown('<h2 class="section-header"> Modelling Approaches Comparison</h2>', unsafe_allow_html=True)
            
            # Reference image (if exists)
            try:
                ref_path = get_ref_path()
                if os.path.exists(f'{ref_path}spare_part_modelling_approaches.png'):
                    ref_image = Image.open(f'{ref_path}spare_part_modelling_approaches.png')
                    st.image(ref_image, caption="Demand Classification Reference", use_column_width=True)
            except:
                st.info(f"Reference image not found at '{ref_path}spare_part_modelling_approaches.png'")

            
            # NEW SECTION: Modelling Approaches
            st.markdown('<h2 class="section-header"> Two Stage ML Modelling Approach</h2>', unsafe_allow_html=True)
            
            # Reference image (if exists)
            try:
                ref_path = get_ref_path()
                if os.path.exists(f'{ref_path}ML_Approach.png'):
                    ref_image = Image.open(f'{ref_path}ML_Approach.png')
                    st.image(ref_image, caption="Demand Classification Reference", use_column_width=True)
            except:
                st.info(f"Reference image not found at '{ref_path}ML_Approach.png'")

            # NEW SECTION: Training vs Prediction Analysis
            st.markdown('<h2 class="section-header"> Training vs Prediction</h2>', unsafe_allow_html=True)

            @st.cache_data
            def load_training_data():
                """Load training data"""
                try:
                    data_path = get_data_path()
                    return pd.read_csv(f'{data_path}daikins_spare_parts_demand_prod_merged_r2_meta.csv')
                except:
                    return None

            @st.cache_data
            def load_prediction_data():
                """Load prediction data"""
                try:
                    data_path = get_data_path()
                    return pd.read_csv(f'{data_path}daikins_spare_parts_final_long_croston.csv')
                except:
                    return None

            def get_common_partcodes(df1, df2):
                """Get PartCodes that exist in both datasets"""
                if df1 is not None and df2 is not None:
                    common_parts = set(df1['PartCode'].unique()).intersection(set(df2['PartCode'].unique()))
                    return list(common_parts)
                return []

            def create_training_plot(df, partcode):
                """Create training time series plot with dual y-axis"""
                if df is None or partcode not in df['PartCode'].values:
                    return None
                
                part_data = df[df['PartCode'] == partcode].copy()
                part_data['MONTH_YEAR'] = pd.to_datetime(part_data['MONTH_YEAR'])
                
                # Filter for Jan 2022 to Dec 2023
                part_data = part_data[
                    (part_data['MONTH_YEAR'] >= '2022-01-01') & 
                    (part_data['MONTH_YEAR'] <= '2023-12-31')
                ].sort_values('MONTH_YEAR')
                
                if part_data.empty:
                    return None
                
                # Create subplot with secondary y-axis
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                
                # Add consumption trace
                fig.add_trace(
                    go.Scatter(
                        x=part_data['MONTH_YEAR'],
                        y=part_data['CONSUMPTION'],
                        mode='lines+markers+text',
                        text=part_data['CONSUMPTION'],
                        textposition='top center',
                        name='Consumption',
                        line=dict(color='#3498db', width=3),
                        marker=dict(size=8)
                    ),
                    secondary_y=False,
                )
                
                # Add breakdown time trace
                fig.add_trace(
                    go.Scatter(
                        x=part_data['MONTH_YEAR'],
                        y=part_data['Breakdown_Time'],
                        mode='lines+markers+text',
                        text=part_data['Breakdown_Time'],
                        textposition='bottom center',
                        name='Breakdown Time',
                        line=dict(color='#e74c3c', width=3),
                        marker=dict(size=8)
                    ),
                    secondary_y=True,
                )
                
                # Set x-axis title
                fig.update_xaxes(title_text="Month-Year")
                
                # Set y-axes titles
                fig.update_yaxes(title_text="Consumption", secondary_y=False)
                fig.update_yaxes(title_text="Breakdown Time", secondary_y=True)
                
                fig.update_layout(
                    title=f"Past - Training<br>PartCode: {partcode}",
                    height=400,
                    title_x=0.5,
                    showlegend=True,
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )
                
                return fig

            def create_prediction_plot(df, partcode):
                """Create prediction time series plot"""
                if df is None or partcode not in df['PartCode'].values:
                    return None
                
                part_data = df[df['PartCode'] == partcode].copy()
                part_data['MONTH_YEAR'] = pd.to_datetime(part_data['MONTH_YEAR'])
                
                # Filter for Jan 2024 to Dec 2024
                part_data = part_data[
                    (part_data['MONTH_YEAR'] >= '2024-01-01') & 
                    (part_data['MONTH_YEAR'] <= '2024-12-31')
                ].sort_values('MONTH_YEAR')
                
                if part_data.empty:
                    return None
                
                # Ceil Croston_Pred to nearest integer
                part_data['Croston_Pred_Ceil'] = np.ceil(part_data['Croston_Pred'])
                
                fig = go.Figure()
                
                # Add actual demand trace
                fig.add_trace(
                    go.Scatter(
                        x=part_data['MONTH_YEAR'],
                        y=part_data['CONSUMPTION_eval'],
                        mode='lines+markers+text',
                        text=part_data['CONSUMPTION_eval'],
                        textposition='top center',
                        name='Actual Demand',
                        line=dict(color='#2c3e50', width=3),
                        marker=dict(size=8)
                    )
                )
                
                # Add forecast trace
                fig.add_trace(
                    go.Scatter(
                        x=part_data['MONTH_YEAR'],
                        y=part_data['forecast'],
                        mode='lines+markers+text',
                        text=part_data['forecast'],
                        textposition='middle right',
                        name='Forecast',
                        line=dict(color='#27ae60', width=3),
                        marker=dict(size=8)
                    )
                )
                
                # Add Croston prediction trace (ceiled)
                fig.add_trace(
                    go.Scatter(
                        x=part_data['MONTH_YEAR'],
                        y=part_data['Croston_Pred_Ceil'],
                        mode='lines+markers+text',
                        text=part_data['Croston_Pred_Ceil'],
                        textposition='bottom center',
                        name='Croston Prediction',
                        line=dict(color='#f39c12', width=3),
                        marker=dict(size=8)
                    )
                )
                
                fig.update_layout(
                    title=f"Future - Prediction<br>PartCode: {partcode}",
                    height=400,
                    title_x=0.5,
                    xaxis_title="Month-Year",
                    yaxis_title="Values",
                    showlegend=True,
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )
                
                return fig

            # Load both datasets
            training_df = load_training_data()
            prediction_df = load_prediction_data()

            # Get common PartCodes
            common_parts = get_common_partcodes(training_df, prediction_df)
            data_path = get_data_path()
            parts_list = pd.read_csv(f"{data_path}parts.txt", header=None)[0].tolist()
            print(parts_list)
            common_parts = parts_list

            if common_parts:
                # Initialize common random part if not exists
                if 'common_random_part' not in st.session_state:
                    st.session_state.common_random_part = random.choice(common_parts)
                
                # Common refresh button
                col_refresh = st.columns([1, 2, 1])
                with col_refresh[1]:
                    if st.button("🔄 Refresh Training & Prediction", key="refresh_common", help="Get random part for both plots"):
                        st.session_state.common_random_part = random.choice(common_parts)
                
                # Create two columns for the plots
                col_train, col_pred = st.columns(2)
                
                with col_train:
                    st.markdown('<div class="time-series-container">', unsafe_allow_html=True)
                    training_fig = create_training_plot(training_df, st.session_state.common_random_part)
                    if training_fig:
                        st.plotly_chart(training_fig, use_container_width=True)
                    else:
                        st.warning("No training data available for selected PartCode")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col_pred:
                    st.markdown('<div class="time-series-container">', unsafe_allow_html=True)
                    prediction_fig = create_prediction_plot(prediction_df, st.session_state.common_random_part)
                    if prediction_fig:
                        st.plotly_chart(prediction_fig, use_container_width=True)
                    else:
                        st.warning("No prediction data available for selected PartCode")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Show current PartCode
                st.info(f"📋 Currently displaying data for PartCode: **{st.session_state.common_random_part}**")
                
            else:
                st.error("No common PartCodes found between training and prediction datasets, or datasets could not be loaded.")
                st.info("Please ensure both data files exist:\n- data/daikins_spare_parts_demand_prod_merged_r2_meta.csv\n- data/daikins_spare_parts_final_long_croston.csv")
                    
        except Exception as e:
            st.error(f"Error processing data: {str(e)}")
            st.code(traceback.format_exc(), language="python")
            st.info("Please ensure your CSV file contains all required columns: PartCode, Description_consumption, Make_consumption, CategoryABC_x, MinStock, MaxStock, CommonMachineName_consumption, Line_consumption, EachPartPrice_consumption, MONTH_YEAR, CONSUMPTION, Classification, ADI, CV2, Cons_Demand_TD, ZeroDemandProportion")
    
    else:
        st.error("📁 Unable to load data. Please ensure 'data/source.csv' exists in your project directory.")
        st.markdown("### Project Structure Should Be:")
        st.code("""
project_folder/
├── spare_parts_dashboard.py
├── ref/
│   └── demand_classification.png
└── data/
    └── source.csv
        """)
        
        # Show expected columns
        st.markdown("### Expected CSV Columns:")
        expected_cols = [
            "PartCode", "Description_consumption", "Make_consumption", "CategoryABC_x",
            "MinStock", "MaxStock", "CommonMachineName_consumption", "Line_consumption",
            "EachPartPrice_consumption", "MONTH_YEAR", "CONSUMPTION", "Classification",
            "ADI", "CV2", "Cons_Demand_TD", "ZeroDemandProportion"
        ]
        
        cols_display = st.columns(4)
        for i, col in enumerate(expected_cols):
            with cols_display[i % 4]:
                st.code(col)

if __name__ == "__main__":
    main()
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Ensure the page stretches to full screen width
st.set_page_config(layout="wide", page_title="Restaurant Growth Potential")

@st.cache_data
def load_data():
    # Load the processed data outputted by our model
    df = pd.read_csv("Restaurant_Strategic_Classification.csv")
    return df

df = load_data()

# --- SIDEBAR FILTERS ---
st.sidebar.header("Filter Criteria")
subregion = st.sidebar.multiselect("Subregion", df['Subregion'].unique(), default=df['Subregion'].unique())
cuisine = st.sidebar.multiselect("Cuisine", df['CuisineType'].unique(), default=df['CuisineType'].unique())
segment = st.sidebar.multiselect("Segment", df['Segment'].unique(), default=df['Segment'].unique())

filtered_df = df[(df['Subregion'].isin(subregion)) & 
                 (df['CuisineType'].isin(cuisine)) & 
                 (df['Segment'].isin(segment))]

st.title("SkyCity Auckland: Restaurant Growth & Classification")
st.markdown("Assess the structural readiness and strategic mapping of local restaurants.")

# --- SCORECARDS ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Restaurants", len(filtered_df))
col2.metric("Avg Growth Potential Index (GPI)", round(filtered_df['GPI'].mean(), 1))
col3.metric("Ready to Expand (>75 GPI)", len(filtered_df[filtered_df['GPI'] >= 75]))
col4.metric("Avg Net Profit", f"${filtered_df['TotalNetProfit'].mean():,.2f}")

st.divider()

# --- VISUALIZATION MODULES ---
col_map, col_strat = st.columns((2, 1))

with col_map:
    st.subheader("Restaurant Archetype Cluster Map (PCA Projection)")
    fig_cluster = px.scatter(
        filtered_df, x='PCA1', y='PCA2', color='ClusterLabel',
        hover_name='RestaurantName', 
        hover_data=['GPI', 'StrategicRecommendation', 'MonthlyOrders'],
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    fig_cluster.update_layout(plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_cluster, use_container_width=True)

with col_strat:
    st.subheader("Strategy Recommendation Split")
    fig_strategy = px.histogram(
        filtered_df, y='StrategicRecommendation', color='ClusterLabel', 
        orientation='h', color_discrete_sequence=px.colors.qualitative.Pastel
    ).update_yaxes(categoryorder='total ascending')
    fig_strategy.update_layout(plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_strategy, use_container_width=True)

# --- RADAR CHART & DRILL DOWN ---
st.subheader("Deep Dive: Feature Contribution Radar")
selected_rest = st.selectbox("Select a Restaurant to review its operational makeup:", filtered_df['RestaurantName'].sort_values().unique())

if selected_rest:
    rest_data = filtered_df[filtered_df['RestaurantName'] == selected_rest].iloc[0]
    categories = ['Volume Generation', 'Cost Efficiency', 'Channel Independence', 'Logistics Scalability']
    
    # Normalizing relative to max for radar visual representation
    r_values = [
        rest_data['MonthlyOrders'] / df['MonthlyOrders'].max(),
        1 - (rest_data['COGSRate'] + rest_data['OPEXRate']),
        1 - (rest_data['UE_share'] + rest_data['DD_share']),
        rest_data['DeliveryRadiusKM'] / df['DeliveryRadiusKM'].max()
    ]
    
    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=r_values,
        theta=categories,
        fill='toself',
        name=selected_rest,
        line=dict(color='indigo')
    ))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])), 
        showlegend=False,
        height=400
    )
    st.plotly_chart(fig_radar, use_container_width=False)

# --- DATA TABLE ---
st.subheader("Restaurant Data Master List")
st.dataframe(filtered_df[['RestaurantName', 'ClusterLabel', 'GPI', 'StrategicRecommendation', 'Subregion', 'TotalNetProfit']].sort_values(by="GPI", ascending=False), use_container_width=True)
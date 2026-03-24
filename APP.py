import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
from lightgbm import LGBMClassifier

# --- PAGE CONFIG ---
st.set_page_config(page_title="Rainfall Analytics Dashboard", layout="wide")

st.title("🌧️ Rainfall Prediction & Analytics Dashboard")

# --- FUNCTIONS ---
def feature_engineering(df):
    df = df.copy()
    df['temp_range'] = df['maxtemp'] - df['mintemp']
    df['dew_depression'] = df['temparature'] - df['dewpoint']
    if 'humidity' in df.columns and 'cloud' in df.columns:
        df['humidity_cloud_interaction'] = df['humidity'] * df['cloud']
    return df

# --- SIDEBAR: FILE UPLOADS ---
st.sidebar.header("Data Source")
train_file = st.sidebar.file_uploader("Upload train.csv for Analysis", type="csv")

if train_file:
    df = pd.read_csv(train_file)
    df = feature_engineering(df)
    
    # --- INTERACTIVE DATA EXPLORATION ---
    st.header("📊 Interactive Visualizations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Distribution Analysis")
        # User selects which feature to look at
        viz_feature = st.selectbox("Select Feature for Distribution", 
                                   ['temparature', 'humidity', 'maxtemp', 'mintemp', 'windspeed', 'temp_range'])
        
        # Choice of Plot Type
        plot_type = st.radio("Chart Type", ["Box Plot", "Violin Plot", "Histogram"])
        
        if plot_type == "Box Plot":
            fig = px.box(df, x="rainfall", y=viz_feature, color="rainfall", points="all",
                         title=f"{viz_feature} vs Rainfall")
        elif plot_type == "Violin Plot":
            fig = px.violin(df, x="rainfall", y=viz_feature, color="rainfall", box=True,
                           title=f"{viz_feature} Density")
        else:
            fig = px.histogram(df, x=viz_feature, color="rainfall", barmode="overlay",
                               title=f"{viz_feature} Distribution")
        
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Feature Correlations")
        # Bar chart of correlation with target
        corr = df.corr()['rainfall'].sort_values(ascending=False).drop('rainfall')
        fig_corr = px.bar(corr, x=corr.index, y=corr.values, 
                          labels={'index': 'Feature', 'y': 'Correlation Score'},
                          title="What drives Rainfall? (Correlation)",
                          color=corr.values, color_continuous_scale='RdBu')
        st.plotly_chart(fig_corr, use_container_width=True)

    # --- SIMULTANEOUS MULTI-VARIABLE ANALYSIS ---
    st.header("🔍 Bi-Variate Analysis")
    c1, c2, c3 = st.columns(3)
    x_axis = c1.selectbox("X Axis", df.columns, index=2)
    y_axis = c2.selectbox("Y Axis", df.columns, index=3)
    size_bubble = c3.selectbox("Bubble Size (Optional)", [None] + list(df.columns))

    fig_scatter = px.scatter(df, x=x_axis, y=y_axis, color="rainfall", 
                             size=size_bubble, hover_data=['id'],
                             title=f"{x_axis} vs {y_axis}")
    st.plotly_chart(fig_scatter, use_container_width=True)

    # --- HEATMAP ---
    if st.checkbox("Show Correlation Heatmap"):
        st.subheader("Full Correlation Matrix")
        df_corr = df.corr()
        fig_heat = px.imshow(df_corr, text_auto=True, aspect="auto", 
                             color_continuous_scale='Viridis',
                             title="Interactive Heatmap")
        st.plotly_chart(fig_heat, use_container_width=True)

else:
    st.info("👋 Welcome! Please upload your `train.csv` in the sidebar to visualize the weather patterns.")

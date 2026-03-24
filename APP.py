import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score

# --- CONFIG ---
st.set_page_config(page_title="Rainfall ML Dashboard", layout="wide")
st.title("🌧️ Advanced Rainfall Prediction Hub")

# --- 1. FEATURE ENGINEERING (From Notebook) ---
def apply_logic(df):
    """Derived from notebook feature engineering steps"""
    df = df.copy()
    if 'maxtemp' in df.columns and 'mintemp' in df.columns:
        df['temp_range'] = df['maxtemp'] - df['mintemp']
    if 'temparature' in df.columns and 'dewpoint' in df.columns:
        df['dew_depression'] = df['temparature'] - df['dewpoint']
    if 'humidity' in df.columns and 'cloud' in df.columns:
        df['humidity_cloud_interaction'] = df['humidity'] * df['cloud']
    return df

# --- 2. MODELING ENGINE ---
def run_ml_pipeline(train_df, model_type):
    train_df = apply_logic(train_df)
    features = [c for c in train_df.columns if c not in ['id', 'rainfall']]
    X = train_df[features]
    y = train_df['rainfall']
    
    # Scaling is required for SVM and Logistic Regression
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if model_type == "LGBM (Recommended)":
        model = LGBMClassifier(n_estimators=100, learning_rate=0.05, verbosity=-1)
    elif model_type == "Logistic Regression":
        model = LogisticRegression()
    elif model_type == "SVM":
        model = SVC(probability=True)
    elif model_type == "Naive Bayes":
        model = GaussianNB()
        
    model.fit(X_scaled, y)
    return model, features, scaler

# --- SIDEBAR: FILE UPLOADS ---
st.sidebar.header("📁 Step 1: Data Upload")
train_file = st.sidebar.file_uploader("Upload train.csv", type="csv")
test_file = st.sidebar.file_uploader("Upload test.csv", type="csv")

if train_file and test_file:
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)
    
    # --- STEP 2: ML CONFIG ---
    st.sidebar.header("🤖 Step 2: Model Selection")
    chosen_model = st.sidebar.selectbox("Choose Algorithm", 
        ["LGBM (Recommended)", "Logistic Regression", "SVM", "Naive Bayes"])
    
    # --- DATA OPERATIONS (Dynamic Filters) ---
    st.header("⚙️ Simultaneous Operations & Analysis")
    st.markdown("Adjust these filters to update all charts and predictions instantly.")
    
    op_col1, op_col2, op_col3 = st.columns(3)
    temp_min = op_col1.slider("Filter Min Temperature", float(train_df['mintemp'].min()), float(train_df['mintemp'].max()))
    cloud_max = op_col2.slider("Max Cloud Cover", float(train_df['cloud'].min()), float(train_df['cloud'].max()))
    target_view = op_col3.selectbox("Visual Focus", ["All", "Only Rain (1)", "Only No Rain (0)"])
    
    # Filter Data Simultaneously
    filtered_df = train_df[(train_df['mintemp'] >= temp_min) & (train_df['cloud'] <= cloud_max)]
    if target_view != "All":
        val = 1 if target_view == "Only Rain (1)" else 0
        filtered_df = filtered_df[filtered_df['rainfall'] == val]

    # --- SIMULTANEOUS CHARTS ---
    st.write(f"Showing {len(filtered_df)} matching records")
    c1, c2 = st.columns(2)
    
    with c1:
        st.plotly_chart(px.box(filtered_df, x="rainfall", y="humidity", color="rainfall", title="Humidity Distribution"), use_container_width=True)
        st.plotly_chart(px.histogram(filtered_df, x="windspeed", title="Wind Speed Frequency"), use_container_width=True)
    
    with c2:
        st.plotly_chart(px.scatter(filtered_df, x="maxtemp", y="dewpoint", color="rainfall", title="Temp vs Dewpoint"), use_container_width=True)
        st.plotly_chart(px.bar(filtered_df.groupby('rainfall').size().reset_index(name='count'), x='rainfall', y='count', title="Rainfall Count"), use_container_width=True)

    # --- PREDICTION EXECUTION ---
    if st.button("🚀 Run Prediction Pipeline"):
        model, feat_cols, scaler = run_ml_pipeline(train_df, chosen_model)
        
        # Prepare test data
        test_processed = apply_logic(test_df)
        test_processed['winddirection'] = test_processed['winddirection'].fillna(train_df['winddirection'].median())
        
        X_test_scaled = scaler.transform(test_processed[feat_cols])
        preds = model.predict_proba(X_test_scaled)[:, 1]
        
        res = pd.DataFrame({'id': test_df['id'], 'rainfall_prob': preds})
        st.success(f"Predictions complete using {chosen_model}!")
        st.dataframe(res.head(20))
        
        st.download_button("Download CSV", res.to_csv(index=False), "output.csv", "text/csv")

else:
    st.info("Please upload your CSV files to activate the dashboard.")

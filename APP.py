import streamlit as st
import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="Rainfall Predictor", layout="wide")

st.title("🌧️ Rainfall Binary Prediction App")
st.markdown("""
This app predicts the probability of rainfall based on weather parameters using the logic from your Kaggle notebook.
""")

# --- FUNCTIONS ---
def feature_engineering(df):
    """Applies the feature engineering steps found in the notebook."""
    df = df.copy()
    # Logic extracted from notebook cells 8 and 9
    df['temp_range'] = df['maxtemp'] - df['mintemp']
    df['dew_depression'] = df['temparature'] - df['dewpoint']
    if 'humidity' in df.columns and 'cloud' in df.columns:
        df['humidity_cloud_interaction'] = df['humidity'] * df['cloud']
    return df

@st.cache_data
def train_model(train_df):
    """Trains the LightGBM model on the uploaded training data."""
    train_df = feature_engineering(train_df)
    features = [c for c in train_df.columns if c not in ['id', 'rainfall']]
    X = train_df[features]
    y = train_df['rainfall']
    
    model = LGBMClassifier(
        n_estimators=1000, 
        learning_rate=0.03, 
        max_depth=6, 
        verbosity=-1,
        random_state=42
    )
    model.fit(X, y)
    return model, features

# --- SIDEBAR: FILE UPLOADS ---
st.sidebar.header("Upload Datasets")
train_file = st.sidebar.file_uploader("Upload train.csv", type="csv")
test_file = st.sidebar.file_uploader("Upload test.csv", type="csv")

if train_file and test_file:
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)
    
    st.success("Files uploaded successfully!")
    
    # --- TRAINING ---
    with st.status("Training model..."):
        model, features = train_model(train_df)
    st.write("### Model Training Complete")
    st.write(f"Features used: `{', '.join(features)}`")

    # --- PREDICTION ---
    st.write("### Predictions")
    if st.button("Generate Predictions"):
        # Preprocess test data
        test_processed = feature_engineering(test_df)
        
        # Handle missing winddirection as per notebook logic
        if 'winddirection' in test_processed.columns:
            median_val = train_df['winddirection'].median()
            test_processed['winddirection'] = test_processed['winddirection'].fillna(median_val)
            
        # Predict probabilities
        probs = model.predict_proba(test_processed[features])[:, 1]
        
        submission = pd.DataFrame({
            'id': test_df['id'] if 'id' in test_df.columns else range(len(probs)),
            'rainfall': probs
        })
        
        st.dataframe(submission.head(10))
        
        # --- DOWNLOAD ---
        csv = submission.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download final_submission.csv",
            data=csv,
            file_name='final_submission.csv',
            mime='text/csv',
        )
else:
    st.info("Please upload both `train.csv` and `test.csv` in the sidebar to begin.")

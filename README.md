# 🌧️ Advanced Rainfall Prediction Hub

A high-performance machine learning dashboard designed to process meteorological attributes and accurately predict binary rainfall occurrences. Built on top of **Streamlit** and powered by multiple classification models (including LightGBM), this analytical environment features interactive, simultaneous data filtration and dynamic feature engineering pipelines based on custom Kaggle dataset training modules.

## 🚀 Live Demo
🔗 **Explore the live prediction system here:** [Rainfall Prediction ML Dashboard](https://rainfall-prediction-gefbfjnmza7fw5y2rtlxth.streamlit.app/)

---

## 📌 Problem Statement & Objectives
### The Challenge
Predicting precipitation patterns from raw local meteorological telemetry requires accounting for complex non-linear relationships across microclimatic shifts. Standard predictive solutions often fail to handle interactive variable spikes (like humidity-cloud patterns) or lack a transparent interface for real-time exploratory data testing.

### System Goals
* **Automated Feature Engineering:** Derive critical meteorological indicators such as thermal distribution gaps (`temp_range`), dew point differentials, and humidity-cloud interactions directly on payload initialization.
* **Algorithmic Versatility:** Provide an operational framework to quickly swap, train, and contrast diverse statistical paradigms—ranging from parametric models to tree-based gradient boosted engines.
* **Reactive Visualization Engine:** Equip weather analysts with a synchronized filtering system that adjusts statistical graphs instantly across dynamic sample partitions.

---

## ✨ System Features
* **Multi-Classifier Pipeline Backend:** Integrated support for **LightGBM**, **Logistic Regression**, **Support Vector Machines (SVM)**, and **Gaussian Naive Bayes**.
* **Simultaneous Operations Dashboard:** Interactive sliders modify threshold limits globally, re-rendering analytical charts (`Plotly Express`) over dynamic data partitions without crashing state engines.
* **Production-Ready Test Pipeline:** Gracefully handles missing historical entries (such as median imputation for missing wind directions) before executing predictions on hidden test populations and exporting them via standard flat files (`output.csv`).

---

## 🛠️ Tech Stack & Dependencies
* **Dashboard Interface:** Streamlit (Layout-Optimized Wide View Configuration)
* **Data Visualizations:** Plotly Express (Box Plots, Histograms, Scatter Arrays, Bar Quantifications)
* **Feature Processing:** Scikit-Learn (`StandardScaler`)
* **Machine Learning Estimators:** LightGBM (`LGBMClassifier`), Scikit-Learn (`LogisticRegression`, `SVC`, `GaussianNB`)
* **Data Wrangling Matrix:** Pandas, NumPy

---

## 📂 Project Structure
```text
├── APP.py                                                         # Streamlit layout configuration and execution logic
├── Kaggle_Completion_Binary_Prediction_with_a_Rainfall_Dataset.ipynb # Exploratory Notebook detailing feature pipeline designs
├── train.csv                                                      # Labeled training dataset holding meteorological features
├── test.csv                                                       # Unlabeled validation data used for deployment inference
├── requirements.txt                                               # Production dependencies checklist
└── README.md                                                      # Current project documentation

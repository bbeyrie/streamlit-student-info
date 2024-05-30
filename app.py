# app.py

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import time

# Load data
@st.cache_data
def load_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    data = pd.read_csv(url, sep=';')
    return data

# Display the data
def display_data(data):
    st.write("### Red Wine Quality Data")
    st.dataframe(data)

# Display data summary
def display_summary(data):
    st.write("### Data Summary")
    st.write(data.describe())

# Display data visualization
def display_visualizations(data):
    st.write("### Data Visualizations")
    
    st.write("#### Correlation Heatmap")
    plt.figure(figsize=(10, 6))
    sns.heatmap(data.corr(), annot=True, fmt=".2f", cmap="coolwarm")
    st.pyplot(plt)
    
    st.write("#### Distribution of Quality")
    plt.figure(figsize=(10, 6))
    sns.countplot(x='quality', data=data, palette="viridis")
    st.pyplot(plt)
    
    st.write("#### Alcohol Content vs. Quality")
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='quality', y='alcohol', data=data, palette="viridis")
    st.pyplot(plt)

# Train a prediction model with grid search
def train_model(data, model_type):
    st.write("### Train a Prediction Model")
    
    X = data.drop('quality', axis=1)
    y = data['quality']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    if model_type == "RandomForest":
        model = RandomForestClassifier(random_state=42)
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_features': ['auto', 'sqrt', 'log2']
        }
    elif model_type == "GradientBoosting":
        model = GradientBoostingClassifier(random_state=42)
        param_grid = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 4, 5]
        }
    elif model_type == "KNeighbors":
        model = KNeighborsClassifier()
        param_grid = {
            'n_neighbors': [3, 5, 7, 9],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan', 'minkowski']
        }

    progress_bar = st.progress(0)
    for i in range(100):
        time.sleep(0.05)
        progress_bar.progress(i + 1)

    with st.spinner('Training the model, please wait...'):
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, return_train_score=True)
        grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    st.write(f"Best Model Parameters: {grid_search.best_params_}")
    st.write(f"Model Accuracy: {accuracy:.2f}")
    
    results = pd.DataFrame(grid_search.cv_results_)
    st.write("### Grid Search Results")
    st.dataframe(results[['params', 'mean_test_score', 'std_test_score', 'rank_test_score']])
    
    return best_model

# Streamlit app layout
def main():
    st.title("Red Wine Quality Analysis")

    data = load_data()
    
    display_data(data)
    display_summary(data)
    display_visualizations(data)
    
    model_type = st.selectbox("Select Model Type", ("RandomForest", "GradientBoosting", "KNeighbors"))
    
    if st.button("Train Model"):
        train_model(data, model_type)

if __name__ == "__main__":
    main()

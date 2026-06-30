import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, classification_report)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from imblearn.over_sampling import SMOTE
import joblib
import logging
import os

# 
# Setup logging
logging.basicConfig(filename='predictions.log', level=logging.INFO,
                    format='%(asctime)s - %(message)s')

# 
# Page config
st.set_page_config(page_title="Play Tennis Prediction", layout="wide")

# 
# Cached data loading
@st.cache_data
def load_data():
    return pd.read_csv("play-tennis.csv")

df = load_data()

# 
# Global variables
ordinal_mappings = {
    'temp': {'Cool': 0, 'Mild': 1, 'Hot': 2},
    'humidity': {'Normal': 0, 'High': 1}
}
nominal_features = ['outlook', 'wind']

# 
# Sidebar: Prediction form
with st.sidebar:
    st.title("Play Tennis Prediction")
    st.markdown("---")
    st.markdown("### Make a Prediction")
    with st.form("prediction_form"):
        outlook = st.selectbox("Outlook", ['Sunny', 'Overcast', 'Rain'])
        temp = st.selectbox("Temperature", ['Cool', 'Mild', 'Hot'])
        humidity = st.selectbox("Humidity", ['Normal', 'High'])
        wind = st.selectbox("Wind", ['Weak', 'Strong'])
        submitted = st.form_submit_button("Predict")

    if 'history' not in st.session_state:
        st.session_state.history = []

# 
# Main title
st.title("Play Tennis Prediction App")


# 
# Run training only once per session (or on rerun it will be re-executed, but data is small)
if 'pipeline_ready' not in st.session_state:
    # Progress placeholder
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Step 0: Raw data
    with st.expander("Data Overview", expanded=False):
        st.dataframe(df.head())

    # Step 1: Clean duplicates, drop unnecessary columns
    initial_rows = len(df)
    df = df.drop_duplicates()
    if 'day' in df.columns:
        df.drop(columns=['day'], inplace=True)
    progress_bar.progress(5)
    status_text.text("Data cleaned...")

    # Step 2: Handle missing values
    cat_cols = df.select_dtypes(include='object').columns
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "Unknown")
    progress_bar.progress(10)

    # Step 3: Handle categorical data – encode target, define features
    le = LabelEncoder()
    y = le.fit_transform(df['play'])
    X = df[['outlook', 'temp', 'humidity', 'wind']]

    # Step 4: Outliers – none (all categorical)
    progress_bar.progress(15)

    # Step 5: X and y defined
    progress_bar.progress(20)

    # Step 6: Split train/val/test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.1765, random_state=42, stratify=y_temp)
    progress_bar.progress(25)

    # Manually encode ordinal features
    for col, mapping in ordinal_mappings.items():
        X_train[col] = X_train[col].map(mapping)
        X_val[col] = X_val[col].map(mapping)
        X_test[col] = X_test[col].map(mapping)

    # Preprocessor: one-hot encode nominal, keep ordinal as is
    preprocessor = ColumnTransformer(
        transformers=[
            ('nominal', Pipeline([
                ('impute', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ]), nominal_features)
        ],
        remainder='passthrough'
    )

    # Step 7: Scaling
    scaler = StandardScaler()
    X_train_prep = preprocessor.fit_transform(X_train)
    X_val_prep = preprocessor.transform(X_val)
    X_test_prep = preprocessor.transform(X_test)

    X_train_scaled = scaler.fit_transform(X_train_prep)
    X_val_scaled = scaler.transform(X_val_prep)
    X_test_scaled = scaler.transform(X_test_prep)
    progress_bar.progress(30)

    # Step 8: Imbalance handling
    unique, counts = np.unique(y_train, return_counts=True)
    imbalance_ratio = min(counts) / max(counts) if max(counts) > 0 else 1
    if len(unique) > 1 and imbalance_ratio < 0.8:
        smote = SMOTE(random_state=42)
        X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)
        smote_applied = True
    else:
        X_train_res, y_train_res = X_train_scaled, y_train
        smote_applied = False
    progress_bar.progress(40)

    # Step 9: Model evaluation
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'K-Nearest Neighbors': KNeighborsClassifier(),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'SVM': SVC(probability=True, random_state=42),
        'Naive Bayes': GaussianNB()
    }
    results = []
    for name, model in models.items():
        model.fit(X_train_res, y_train_res)
        y_pred = model.predict(X_val_scaled)
        acc = accuracy_score(y_val, y_pred)
        prec = precision_score(y_val, y_pred, average='binary', zero_division=0)
        rec = recall_score(y_val, y_pred, average='binary', zero_division=0)
        f1 = f1_score(y_val, y_pred, average='binary', zero_division=0)
        try:
            roc = roc_auc_score(y_val, model.predict_proba(X_val_scaled)[:, 1])
        except:
            roc = 0.0
        results.append([name, acc, prec, rec, f1, roc])

    results_df = pd.DataFrame(results, columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1', 'ROC AUC'])
    progress_bar.progress(60)

    # Step 10: Best model selection (by F1)
    best_row = results_df.loc[results_df['F1'].idxmax()]
    best_model_name = best_row['Model']

    # Step 11: Hyperparameter tuning
    param_grids = {
        'Logistic Regression': {'C': [0.1, 1, 10]},
        'K-Nearest Neighbors': {'n_neighbors': [3, 5, 7]},
        'Decision Tree': {'max_depth': [None, 5, 10]},
        'Random Forest': {'n_estimators': [50, 100], 'max_depth': [None, 5]},
        'SVM': {'C': [0.1, 1, 10], 'kernel': ['rbf', 'linear']},
        'Naive Bayes': {}
    }
    if best_model_name in param_grids and param_grids[best_model_name]:
        grid = GridSearchCV(models[best_model_name], param_grids[best_model_name],
                            cv=3, scoring='f1', n_jobs=-1, refit=True)
        grid.fit(X_train_res, y_train_res)
        best_estimator = grid.best_estimator_
    else:
        best_estimator = models[best_model_name]
    progress_bar.progress(75)

    # Step 12: Retrain on train+val
    X_trainval = np.vstack((X_train_scaled, X_val_scaled))
    y_trainval = np.hstack((y_train, y_val))
    if smote_applied:
        smote_comb = SMOTE(random_state=42)
        X_trainval_res, y_trainval_res = smote_comb.fit_resample(X_trainval, y_trainval)
    else:
        X_trainval_res, y_trainval_res = X_trainval, y_trainval

    final_model = best_estimator.__class__(**best_estimator.get_params())
    final_model.fit(X_trainval_res, y_trainval_res)
    progress_bar.progress(85)

    # Step 13: Final test evaluation
    y_test_pred = final_model.predict(X_test_scaled)
    test_report = classification_report(y_test, y_test_pred, target_names=le.classes_, output_dict=True)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    # Step 14: Build final pipeline (preprocessor + scaler + classifier)
    full_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('scaler', scaler),
        ('classifier', final_model)
    ])

    # Step 15: Save model
    joblib.dump(full_pipeline, "model.pkl")
    progress_bar.progress(95)

    # Store everything in session state
    st.session_state.pipeline_ready = True
    st.session_state.full_pipeline = full_pipeline
    st.session_state.le = le
    st.session_state.results_df = results_df
    st.session_state.best_model_name = best_model_name
    st.session_state.best_estimator = best_estimator
    st.session_state.test_report = test_report
    st.session_state.test_accuracy = test_accuracy
    st.session_state.smote_applied = smote_applied
    st.session_state.X_train_res = X_train_res  # for feature importance if needed

    progress_bar.progress(100)
    status_text.text("Model training complete!")

# 
# Display training summary in expanders
if st.session_state.pipeline_ready:
    with st.expander("Model Performance", expanded=True):
        col1, col2, col3 = st.columns(3)
        col1.metric("Best Model", st.session_state.best_model_name)
        col2.metric("Test Accuracy", f"{st.session_state.test_accuracy:.3f}")
        col3.metric("Validation F1 (best)", f"{st.session_state.results_df['F1'].max():.3f}")

        st.markdown("**Validation Scores (all models)**")
        st.dataframe(st.session_state.results_df.style.highlight_max(subset=['F1'], color='lightgreen'))

    with st.expander("Detailed Steps", expanded=False):
        st.markdown("**Step 8: Imbalance**")
        if st.session_state.smote_applied:
            st.write("SMOTE was applied to the training data.")
        else:
            st.write("No severe imbalance detected, SMOTE skipped.")

        st.markdown("**Step 13: Final Test Classification Report**")
        st.dataframe(pd.DataFrame(st.session_state.test_report).transpose())

        st.markdown("**Step 15: Model saved as `model.pkl`**")

# 
# Handle prediction from sidebar
if submitted:
    errors = []
    if outlook not in ['Sunny', 'Overcast', 'Rain']:
        errors.append("Outlook must be Sunny, Overcast, or Rain.")
    if temp not in ordinal_mappings['temp']:
        errors.append("Temperature must be Cool, Mild, or Hot.")
    if humidity not in ordinal_mappings['humidity']:
        errors.append("Humidity must be Normal or High.")
    if wind not in ['Weak', 'Strong']:
        errors.append("Wind must be Weak or Strong.")

    if errors:
        for err in errors:
            st.sidebar.error(err)
    else:
        # Inference function
        input_df = pd.DataFrame([[outlook, temp, humidity, wind]],
                                columns=['outlook', 'temp', 'humidity', 'wind'])
        input_df['temp'] = input_df['temp'].map(ordinal_mappings['temp'])
        input_df['humidity'] = input_df['humidity'].map(ordinal_mappings['humidity'])

        pipeline = st.session_state.full_pipeline
        le = st.session_state.le
        pred = pipeline.predict(input_df)
        proba = pipeline.predict_proba(input_df)[0]  # array of both class probabilities
        class_labels = le.classes_

        # Build result dict
        prediction_result = {
            'Outlook': outlook,
            'Temperature': temp,
            'Humidity': humidity,
            'Wind': wind,
            'Predicted': le.inverse_transform(pred)[0],
            'Probability_Yes': proba[1],
            'Probability_No': proba[0]
        }
        st.session_state.history.append(prediction_result)

        # Log prediction
        logging.info(
            f"Input: {outlook}, {temp}, {humidity}, {wind} -> "
            f"Pred: {prediction_result['Predicted']} (Yes={proba[1]:.3f}, No={proba[0]:.3f})"
        )

        # Display prediction result in main area
        st.markdown("---")
        st.markdown("## Latest Prediction")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Input**")
            st.write(f"Outlook: {outlook}")
            st.write(f"Temperature: {temp}")
            st.write(f"Humidity: {humidity}")
            st.write(f"Wind: {wind}")

        with col2:
            st.markdown("**Result**")
            st.metric("Predicted", prediction_result['Predicted'])
            st.write(f"Confidence (Yes): {proba[1]:.3f}")
            st.write(f"Confidence (No): {proba[0]:.3f}")

            # Simple reasoning (based on probabilities)
            if proba[1] > 0.7:
                st.success("Strong indication for Play = Yes.")
            elif proba[1] > 0.5:
                st.info("Moderate chance of Play = Yes.")
            elif proba[0] > 0.7:
                st.error("Strong indication for Play = No.")
            else:
                st.warning("Uncertain prediction; model is not confident.")

        # Show probability bar chart
        st.markdown("**Class Probabilities**")
        st.bar_chart(pd.DataFrame({'Class': class_labels, 'Probability': proba}).set_index('Class'))

    # Display prediction history (optional)
    if st.session_state.history:
        with st.sidebar.expander("Prediction History"):
            hist_df = pd.DataFrame(st.session_state.history)
            st.dataframe(hist_df.tail(5))


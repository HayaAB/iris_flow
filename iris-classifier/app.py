# app.py
import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.datasets import load_iris

st.set_page_config(page_title="Iris Classifier", page_icon="🌸", layout="centered")

# --- تحميل البيانات (للقيم الدنيا/العظمى والأسماء)
@st.cache_data(show_spinner=False)
def load_iris_df():
    iris = load_iris(as_frame=True)
    return iris

iris = load_iris_df()
feature_names = iris.feature_names
target_names = iris.target_names
df = iris.frame

# --- تحميل الموديل (كمورد مشترك)
@st.cache_resource
def load_model():
    return joblib.load("models/iris_pipeline.joblib")

model = load_model()

st.title("🌸 Iris Classifier — Streamlit")
st.write("Enter the following values ​​and continue the prediction:")

# تحضير حدود سليدرات من البيانات نفسها
mins = df[feature_names].min()
maxs = df[feature_names].max()
defaults = df[feature_names].median()

col1, col2 = st.columns(2)
with col1:
    sl = st.slider(feature_names[0], float(mins[0]), float(maxs[0]), float(defaults[0]), 0.1)
    pl = st.slider(feature_names[2], float(mins[2]), float(maxs[2]), float(defaults[2]), 0.1)
with col2:
    sw = st.slider(feature_names[1], float(mins[1]), float(maxs[1]), float(defaults[1]), 0.1)
    pw = st.slider(feature_names[3], float(mins[3]), float(maxs[3]), float(defaults[3]), 0.1)

if st.button("🔮 prediction"):
    X = np.array([[sl, sw, pl, pw]])
    y_pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0]

    st.success(f"Expected type: **{target_names[y_pred]}**")
    st.write("Probabilities for each category:")
    prob_df = pd.DataFrame({"class": target_names, "probability": proba})
    st.bar_chart(prob_df.set_index("class"))
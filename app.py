import streamlit as st
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(
    page_title="Hospital Disease Prediction System",
    layout="wide",
    page_icon="🏥"
)

# -------------------------
# CUSTOM STYLE (Hospital Theme)
# -------------------------
st.markdown("""
<style>
body {
    background-color: #f4f8fb;
}
.section {
    padding: 15px;
    border-radius: 10px;
    background-color: #ffffff;
    margin-bottom: 20px;
    box-shadow: 0px 2px 8px rgba(0,0,0,0.1);
}
.title {
    font-size: 30px;
    font-weight: bold;
    color: #2c7be5;
}
</style>
""", unsafe_allow_html=True)

# -------------------------
# LOAD DATA
# -------------------------
data = pd.read_csv("dataset.csv")

X = data.drop("disease", axis=1)
y = data["disease"]

# -------------------------
# TRAIN MODEL
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier()
model.fit(X_train, y_train)

# -------------------------
# HEADER
# -------------------------
st.markdown('<p class="title">🏥 Hospital Disease Prediction System</p>', unsafe_allow_html=True)

# -------------------------
# PATIENT INFO
# -------------------------
st.markdown('<div class="section">', unsafe_allow_html=True)
st.subheader("👤 Patient Information")

col1, col2, col3 = st.columns(3)

with col1:
    name = st.text_input("Patient Name")
with col2:
    age = st.number_input("Age", 1, 100)
with col3:
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])

st.markdown('</div>', unsafe_allow_html=True)

# -------------------------
# SYMPTOMS SECTION
# -------------------------
st.markdown('<div class="section">', unsafe_allow_html=True)
st.subheader("🩺 Symptoms")

def yes_no(label):
    return st.radio(label, ["No", "Yes"], horizontal=True)

col1, col2 = st.columns(2)

with col1:
    fever = yes_no("Fever")
    cough = yes_no("Cough")

with col2:
    fatigue = yes_no("Fatigue")
    headache = yes_no("Headache")

st.markdown('</div>', unsafe_allow_html=True)

# Convert Yes/No to 0/1
def convert(value):
    return 1 if value == "Yes" else 0

input_data = np.array([[
    convert(fever),
    convert(cough),
    convert(fatigue),
    convert(headache)
]])

# -------------------------
# PREDICTION BUTTON
# -------------------------
if st.button("🧾 Generate Diagnosis Report"):

    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]
    classes = model.classes_

    # -------------------------
    # RESULT SECTION
    # -------------------------
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.subheader("📋 Diagnosis Report")

    st.write(f"**Patient Name:** {name}")
    st.write(f"**Age:** {age}")
    st.write(f"**Gender:** {gender}")

    st.success(f"🧠 Predicted Disease: **{prediction}**")

    # Confidence
    confidence = max(probabilities) * 100
    st.info(f"Confidence Level: {confidence:.2f}%")

    # -------------------------
    # PROBABILITY TABLE
    # -------------------------
    prob_df = pd.DataFrame({
        "Disease": classes,
        "Probability (%)": [round(p * 100, 2) for p in probabilities]
    })

    st.subheader("📊 Probability Analysis")
    st.dataframe(prob_df)

    st.bar_chart(prob_df.set_index("Disease"))

    # -------------------------
    # BASIC MEDICAL NOTE
    # -------------------------
    st.warning("""
    ⚠️ This is a machine learning prediction and not a medical diagnosis.
    Please consult a qualified doctor for confirmation.
    """)

    st.markdown('</div>', unsafe_allow_html=True)
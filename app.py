import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import csv

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="BankPredict AI",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CUSTOM CSS — MODERN REAL-WORLD APP
# =========================================================

st.markdown("""
<style>
    /* ---------- App shell ---------- */
    .stApp {
        background:
            radial-gradient(circle at 0% 0%, rgba(37, 99, 235, .08), transparent 28%),
            radial-gradient(circle at 100% 10%, rgba(14, 165, 233, .07), transparent 25%),
            #f5f7fb;
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2.5rem;
        max-width: 1450px;
    }

    /* ---------- Sidebar ---------- */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0b1220 0%, #111827 55%, #172033 100%);
        border-right: 1px solid rgba(255,255,255,.08);
    }

    section[data-testid="stSidebar"] * {
        color: #f8fafc;
    }

    section[data-testid="stSidebar"] .stRadio label {
        padding: 7px 8px;
        border-radius: 9px;
    }

    .sidebar-brand {
        padding: 8px 2px 18px 2px;
    }

    .sidebar-brand .brand-icon {
        font-size: 30px;
        margin-bottom: 4px;
    }

    .sidebar-brand h2 {
        margin: 0;
        font-size: 22px;
        letter-spacing: -.4px;
    }

    .sidebar-brand p {
        margin: 4px 0 0 0;
        color: #94a3b8;
        font-size: 12px;
    }

    .creator-card {
        margin-top: 18px;
        padding: 14px;
        border: 1px solid rgba(255,255,255,.10);
        background: rgba(255,255,255,.05);
        border-radius: 14px;
    }

    .creator-card .label {
        color: #94a3b8;
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: .8px;
    }

    .creator-card .name {
        margin-top: 3px;
        font-size: 15px;
        font-weight: 700;
    }

    /* ---------- Hero ---------- */
    .hero {
        position: relative;
        overflow: hidden;
        background: linear-gradient(135deg, #0b1220 0%, #172554 55%, #1d4ed8 100%);
        padding: 34px 38px;
        border-radius: 24px;
        color: white;
        margin-bottom: 26px;
        box-shadow: 0 18px 45px rgba(15, 23, 42, .16);
        border: 1px solid rgba(255,255,255,.08);
    }

    .hero:after {
        content: "";
        position: absolute;
        width: 220px;
        height: 220px;
        right: -55px;
        top: -85px;
        border-radius: 50%;
        background: rgba(255,255,255,.08);
    }

    .hero .eyebrow {
        display: inline-block;
        padding: 6px 10px;
        border-radius: 999px;
        background: rgba(255,255,255,.10);
        border: 1px solid rgba(255,255,255,.12);
        color: #dbeafe;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-bottom: 10px;
    }

    .hero h1 {
        font-size: 42px;
        line-height: 1.05;
        margin: 0 0 9px 0;
        font-weight: 800;
        letter-spacing: -1.5px;
    }

    .hero p {
        color: #dbeafe;
        font-size: 16px;
        margin: 0;
        max-width: 720px;
        line-height: 1.55;
    }

    /* ---------- Cards / sections ---------- */
    .section-title {
        font-size: 25px;
        font-weight: 800;
        color: #0f172a;
        margin-top: 14px;
        margin-bottom: 15px;
        letter-spacing: -.5px;
    }

    .section-subtitle {
        color: #64748b;
        margin-top: -7px;
        margin-bottom: 18px;
        font-size: 14px;
    }

    .info-card, .prediction-card {
        background: rgba(255,255,255,.96);
        border: 1px solid #e2e8f0;
        border-radius: 18px;
        padding: 22px;
        margin-bottom: 18px;
        box-shadow: 0 8px 25px rgba(15, 23, 42, .055);
    }

    .prediction-card {
        padding: 28px;
    }

    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 16px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 6px 20px rgba(15, 23, 42, .05);
    }

    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid #e2e8f0;
        padding: 16px 18px;
        border-radius: 16px;
        box-shadow: 0 6px 20px rgba(15, 23, 42, .045);
    }

    div[data-testid="stMetricLabel"] {
        color: #64748b !important;
    }

    div[data-testid="stMetricValue"] {
        color: #0f172a !important;
        font-weight: 800;
    }

    /* ---------- Inputs ---------- */
    div[data-baseweb="select"] > div,
    div[data-testid="stNumberInput"] input {
        border-radius: 10px;
    }

    .stTextInput input:focus,
    .stNumberInput input:focus {
        border-color: #2563eb;
    }

    /* ---------- Buttons ---------- */
    div.stButton > button,
    div[data-testid="stFormSubmitButton"] button {
        border-radius: 11px;
        min-height: 46px;
        font-weight: 750;
        border: 0;
        transition: transform .15s ease, box-shadow .15s ease;
    }

    div[data-testid="stFormSubmitButton"] button {
        background: linear-gradient(135deg, #2563eb, #1d4ed8);
        color: white;
        box-shadow: 0 8px 20px rgba(37, 99, 235, .25);
    }

    div[data-testid="stFormSubmitButton"] button:hover {
        transform: translateY(-1px);
        box-shadow: 0 11px 25px rgba(37, 99, 235, .32);
    }

    /* ---------- Results ---------- */
    .result-yes, .result-no {
        padding: 20px;
        border-radius: 15px;
        font-size: 17px;
        font-weight: 750;
        text-align: center;
        margin-bottom: 10px;
    }

    .result-yes {
        background: linear-gradient(135deg, #ecfdf5, #d1fae5);
        border: 1px solid #86efac;
        color: #065f46;
    }

    .result-no {
        background: linear-gradient(135deg, #fff7ed, #ffedd5);
        border: 1px solid #fdba74;
        color: #9a3412;
    }

    .small-muted {
        color: #64748b;
        font-size: 13px;
    }

    .creator-footer {
        text-align: center;
        padding: 18px 0 4px;
        color: #64748b;
        font-size: 13px;
    }

    .creator-footer strong {
        color: #1d4ed8;
    }

    /* ---------- Streamlit cleanup ---------- */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header[data-testid="stHeader"] {
        background: transparent;
    }
</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD DATA / MODEL
# =========================================================

@st.cache_data
def load_data():
    df = pd.read_csv(
        "TASK3 data.csv",
        sep=";",
        quoting=csv.QUOTE_NONE,
        engine="python"
    )

    df.columns = df.columns.str.replace('"', '', regex=False)

    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].str.replace('"', '', regex=False).str.strip()

    return df


@st.cache_resource
def load_model():
    return joblib.load("random_forest_model.pkl")


try:
    df = load_data()
    model = load_model()
    app_ready = True
except Exception as e:
    app_ready = False
    st.error("Application could not load the dataset/model.")
    st.code(str(e))
    st.info(
        "Keep TASK3 data.csv and random_forest_model.pkl in the same folder as app.py."
    )
    st.stop()


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <div class="brand-icon">🏦</div>
        <h2>BankPredict AI</h2>
        <p>Customer Subscription Intelligence</p>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    page = st.radio(
        "Workspace",
        [
            "Overview",
            "Customer Prediction",
            "Customer Analytics",
            "Model Performance"
        ],
        label_visibility="collapsed"
    )

    st.divider()

    st.caption("MODEL")
    st.write("Random Forest Classifier")
    st.caption("BANK MARKETING DATASET")
    st.write(f"{len(df):,} customer records")

    st.markdown("""
    <div class="creator-card">
        <div class="label">Created by</div>
        <div class="name">Shaurya Pandey</div>
    </div>
    """, unsafe_allow_html=True)


# =========================================================
# HEADER
# =========================================================

st.markdown("""
<div class="hero">
    <div class="eyebrow">Banking Intelligence Platform</div>
    <h1>BankPredict AI</h1>
    <p>AI-powered customer subscription prediction using a Random Forest model, with analytics and model-performance insights in one dashboard.</p>
</div>
""", unsafe_allow_html=True)


# =========================================================
# OVERVIEW
# =========================================================

if page == "Overview":

    st.markdown('<div class="section-title">Business Overview</div><div class="section-subtitle">A quick view of customer activity and subscription patterns.</div>', unsafe_allow_html=True)

    total = len(df)
    subscribed = int((df["y"].str.lower() == "yes").sum())
    not_subscribed = int((df["y"].str.lower() == "no").sum())
    subscription_rate = (subscribed / total * 100) if total else 0

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric("Customers", f"{total:,}")

    with c2:
        st.metric("Subscribed", f"{subscribed:,}")

    with c3:
        st.metric("Not Subscribed", f"{not_subscribed:,}")

    with c4:
        st.metric("Subscription Rate", f"{subscription_rate:.1f}%")

    st.markdown("<br>", unsafe_allow_html=True)

    left, right = st.columns([1.05, 1])

    with left:
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        st.subheader("Subscription Overview")

        fig, ax = plt.subplots(figsize=(6, 3.8))
        counts = df["y"].value_counts()
        sns.barplot(x=counts.index, y=counts.values, ax=ax)
        ax.set_xlabel("Subscription")
        ax.set_ylabel("Customers")
        ax.set_title("Customer Subscription Distribution")
        sns.despine()
        st.pyplot(fig, width="stretch")
        plt.close(fig)

        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        st.subheader("Customer Profile")

        fig, ax = plt.subplots(figsize=(6, 3.8))
        sns.histplot(df["age"], bins=25, kde=True, ax=ax)
        ax.set_xlabel("Age")
        ax.set_ylabel("Customers")
        ax.set_title("Age Distribution")
        sns.despine()
        st.pyplot(fig, width="stretch")
        plt.close(fig)

        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-title">Dataset Preview</div>', unsafe_allow_html=True)
    st.dataframe(df.head(10), width="stretch", hide_index=True)


# =========================================================
# CUSTOMER PREDICTION
# =========================================================

elif page == "Customer Prediction":

    st.markdown(
        '<div class="section-title">Customer Subscription Assessment</div><div class="section-subtitle">Enter customer and campaign details to generate an AI-assisted prediction.</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Enter customer and campaign information. The trained Random Forest model "
        "will estimate the likelihood of term-deposit subscription."
    )

    st.markdown('<div class="prediction-card">', unsafe_allow_html=True)

    with st.form("prediction_form"):

        st.markdown("### Customer Profile")

        c1, c2, c3 = st.columns(3)

        with c1:
            age = st.number_input(
                "Age",
                min_value=18,
                max_value=100,
                value=35
            )

            job = st.selectbox(
                "Job",
                sorted(df["job"].dropna().unique())
            )

            marital = st.selectbox(
                "Marital Status",
                sorted(df["marital"].dropna().unique())
            )

        with c2:
            education = st.selectbox(
                "Education",
                sorted(df["education"].dropna().unique())
            )

            default = st.selectbox(
                "Credit Default",
                sorted(df["default"].dropna().unique())
            )

            balance = st.number_input(
                "Account Balance",
                value=1000
            )

        with c3:
            housing = st.selectbox(
                "Housing Loan",
                sorted(df["housing"].dropna().unique())
            )

            loan = st.selectbox(
                "Personal Loan",
                sorted(df["loan"].dropna().unique())
            )

            contact = st.selectbox(
                "Contact Method",
                sorted(df["contact"].dropna().unique())
            )

        st.divider()

        st.markdown("### Campaign Information")

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            day = st.number_input(
                "Contact Day",
                min_value=1,
                max_value=31,
                value=15
            )

        with c2:
            month = st.selectbox(
                "Contact Month",
                sorted(df["month"].dropna().unique())
            )

        with c3:
            duration = st.number_input(
                "Call Duration",
                min_value=0,
                value=200
            )

        with c4:
            campaign = st.number_input(
                "Campaign Contacts",
                min_value=1,
                value=1
            )

        c1, c2 = st.columns(2)

        with c1:
            pdays = st.number_input(
                "Days Since Previous Contact",
                value=-1
            )

        with c2:
            previous = st.number_input(
                "Previous Contacts",
                min_value=0,
                value=0
            )

        poutcome = st.selectbox(
            "Previous Campaign Outcome",
            sorted(df["poutcome"].dropna().unique())
        )

        submitted = st.form_submit_button(
            "Predict Customer Outcome",
            width="stretch",
            type="primary"
        )

    st.markdown("</div>", unsafe_allow_html=True)

    if submitted:

        customer = pd.DataFrame({
            "age": [age],
            "job": [job],
            "marital": [marital],
            "education": [education],
            "default": [default],
            "balance": [balance],
            "housing": [housing],
            "loan": [loan],
            "contact": [contact],
            "day": [day],
            "month": [month],
            "duration": [duration],
            "campaign": [campaign],
            "pdays": [pdays],
            "previous": [previous],
            "poutcome": [poutcome]
        })

        try:
            prediction = model.predict(customer)[0]

            probability = None
            if hasattr(model, "predict_proba"):
                probability = model.predict_proba(customer)[0][1]

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("### Prediction Result")

            if prediction == 1:
                st.markdown(
                    '<div class="result-yes">✓ High likelihood of term-deposit subscription</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    '<div class="result-no">! Low likelihood of term-deposit subscription</div>',
                    unsafe_allow_html=True
                )

            if probability is not None:
                st.progress(float(probability))
                st.caption(
                    f"Estimated subscription probability: {probability * 100:.1f}%"
                )

        except Exception as e:
            st.error("Prediction failed.")
            st.code(str(e))


# =========================================================
# CUSTOMER ANALYTICS
# =========================================================

elif page == "Customer Analytics":

    st.markdown(
        '<div class="section-title">Customer Analytics</div><div class="section-subtitle">Explore relationships between customer attributes and term-deposit subscriptions.</div>',
        unsafe_allow_html=True
    )

    analysis = st.selectbox(
        "Choose analysis",
        [
            "Job & Subscription",
            "Education & Subscription",
            "Housing Loan & Subscription",
            "Previous Campaign Outcome",
            "Balance Distribution"
        ]
    )

    if analysis == "Job & Subscription":

        fig, ax = plt.subplots(figsize=(11, 6))
        sns.countplot(
            data=df,
            y="job",
            hue="y",
            order=df["job"].value_counts().index,
            ax=ax
        )
        ax.set_title("Job vs Term Deposit Subscription")
        ax.set_xlabel("Customers")
        ax.set_ylabel("Job")
        sns.despine()
        st.pyplot(fig, width="stretch")
        plt.close(fig)

    elif analysis == "Education & Subscription":

        fig, ax = plt.subplots(figsize=(9, 5))
        sns.countplot(
            data=df,
            x="education",
            hue="y",
            ax=ax
        )
        ax.set_title("Education vs Term Deposit Subscription")
        ax.set_xlabel("Education")
        ax.set_ylabel("Customers")
        sns.despine()
        st.pyplot(fig, width="stretch")
        plt.close(fig)

    elif analysis == "Housing Loan & Subscription":

        fig, ax = plt.subplots(figsize=(8, 5))
        sns.countplot(
            data=df,
            x="housing",
            hue="y",
            ax=ax
        )
        ax.set_title("Housing Loan vs Subscription")
        ax.set_xlabel("Housing Loan")
        ax.set_ylabel("Customers")
        sns.despine()
        st.pyplot(fig, width="stretch")
        plt.close(fig)

    elif analysis == "Previous Campaign Outcome":

        fig, ax = plt.subplots(figsize=(9, 5))
        sns.countplot(
            data=df,
            x="poutcome",
            hue="y",
            ax=ax
        )
        ax.set_title("Previous Campaign Outcome vs Subscription")
        ax.set_xlabel("Previous Outcome")
        ax.set_ylabel("Customers")
        sns.despine()
        st.pyplot(fig, width="stretch")
        plt.close(fig)

    elif analysis == "Balance Distribution":

        fig, ax = plt.subplots(figsize=(10, 5))
        sns.histplot(
            data=df,
            x="balance",
            bins=40,
            kde=True,
            ax=ax
        )
        ax.set_title("Customer Account Balance")
        ax.set_xlabel("Balance")
        ax.set_ylabel("Customers")
        sns.despine()
        st.pyplot(fig, width="stretch")
        plt.close(fig)


# =========================================================
# MODEL PERFORMANCE
# =========================================================

elif page == "Model Performance":

    st.markdown(
        '<div class="section-title">Model Performance</div><div class="section-subtitle">Evaluate the Random Forest model using test-set metrics and classification results.</div>',
        unsafe_allow_html=True
    )

    try:
        x_test = joblib.load("x_test.pkl")
        y_test = joblib.load("y_test.pkl")

        y_pred = model.predict(x_test)

        accuracy = accuracy_score(y_test, y_pred)

        st.metric(
            "Random Forest Accuracy",
            f"{accuracy * 100:.2f}%"
        )

        st.markdown("<br>", unsafe_allow_html=True)

        left, right = st.columns(2)

        with left:

            st.subheader("Confusion Matrix")

            cm = confusion_matrix(y_test, y_pred)

            fig, ax = plt.subplots(figsize=(6, 5))

            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=["No", "Yes"],
                yticklabels=["No", "Yes"],
                ax=ax
            )

            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")

            st.pyplot(fig, width="stretch")
            plt.close(fig)

        with right:

            st.subheader("Classification Report")

            report = classification_report(
                y_test,
                y_pred,
                output_dict=True
            )

            report_df = pd.DataFrame(report).transpose()

            st.dataframe(
                report_df.round(3),
                width="stretch"
            )

    except FileNotFoundError:

        st.warning(
            "x_test.pkl or y_test.pkl not found. "
            "Save them from the notebook first."
        )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.markdown("""
<div class="creator-footer">
    BankPredict AI • Random Forest Classification • Bank Marketing Dataset<br>
    Created by <strong>Shaurya Pandey</strong>
</div>
""", unsafe_allow_html=True)
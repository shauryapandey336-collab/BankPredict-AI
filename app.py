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
    page_icon="🌳",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CUSTOM CSS — MODERN REAL-WORLD APP
# =========================================================

st.markdown("""
<style>
    .stApp {
        background: #f6f8fb;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1400px;
    }

    .hero {
        background: linear-gradient(135deg, #111827 0%, #1f2937 55%, #374151 100%);
        padding: 34px 38px;
        border-radius: 20px;
        color: white;
        margin-bottom: 26px;
        box-shadow: 0 12px 35px rgba(15, 23, 42, .12);
    }

    .hero h1 {
        font-size: 38px;
        margin: 0 0 8px 0;
        font-weight: 750;
        letter-spacing: -1px;
    }

    .hero p {
        color: #d1d5db;
        font-size: 16px;
        margin: 0;
    }

    .metric-card {
        background: white;
        padding: 22px;
        border-radius: 16px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 5px 18px rgba(15, 23, 42, .05);
    }

    .metric-label {
        color: #6b7280;
        font-size: 13px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: .5px;
    }

    .metric-value {
        color: #111827;
        font-size: 28px;
        font-weight: 750;
        margin-top: 5px;
    }

    .section-title {
        font-size: 24px;
        font-weight: 750;
        color: #111827;
        margin-top: 15px;
        margin-bottom: 14px;
    }

    .info-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 16px;
        padding: 22px;
        margin-bottom: 18px;
    }

    .prediction-card {
        background: white;
        border-radius: 20px;
        border: 1px solid #e5e7eb;
        padding: 30px;
        box-shadow: 0 8px 25px rgba(15, 23, 42, .06);
    }

    .result-yes {
        background: #ecfdf5;
        border: 1px solid #a7f3d0;
        color: #065f46;
        padding: 22px;
        border-radius: 15px;
        font-size: 18px;
        font-weight: 650;
        text-align: center;
    }

    .result-no {
        background: #fff7ed;
        border: 1px solid #fed7aa;
        color: #9a3412;
        padding: 22px;
        border-radius: 15px;
        font-size: 18px;
        font-weight: 650;
        text-align: center;
    }

    .small-muted {
        color: #6b7280;
        font-size: 14px;
    }

    section[data-testid="stSidebar"] {
        background: #111827;
    }

    section[data-testid="stSidebar"] * {
        color: #f9fafb;
    }

    div.stButton > button {
        border-radius: 10px;
        min-height: 44px;
        font-weight: 650;
    }

    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid #e5e7eb;
        padding: 14px;
        border-radius: 14px;
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
    st.markdown("## 🌳 BankPredict AI")
    st.caption("Customer Subscription Intelligence")
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


# =========================================================
# HEADER
# =========================================================

st.markdown("""
<div class="hero">
    <h1>BankPredict AI</h1>
    <p>AI-powered customer subscription prediction using Random Forest.</p>
</div>
""", unsafe_allow_html=True)


# =========================================================
# OVERVIEW
# =========================================================

if page == "Overview":

    st.markdown('<div class="section-title">Business Overview</div>', unsafe_allow_html=True)

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
        st.pyplot(fig, use_container_width=True)
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
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-title">Dataset Preview</div>', unsafe_allow_html=True)
    st.dataframe(df.head(10), use_container_width=True, hide_index=True)


# =========================================================
# CUSTOMER PREDICTION
# =========================================================

elif page == "Customer Prediction":

    st.markdown(
        '<div class="section-title">Customer Subscription Assessment</div>',
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
            use_container_width=True,
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
        '<div class="section-title">Customer Analytics</div>',
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
        st.pyplot(fig, use_container_width=True)
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
        st.pyplot(fig, use_container_width=True)
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
        st.pyplot(fig, use_container_width=True)
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
        st.pyplot(fig, use_container_width=True)
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
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)


# =========================================================
# MODEL PERFORMANCE
# =========================================================

elif page == "Model Performance":

    st.markdown(
        '<div class="section-title">Model Performance</div>',
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

            st.pyplot(fig, use_container_width=True)
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
                use_container_width=True
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

st.caption(
    "BankPredict AI • Random Forest Classification • "
    "Bank Marketing Dataset"
)
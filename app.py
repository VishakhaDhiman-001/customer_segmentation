import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Customer Segmentation",
    page_icon="🛍️",
    layout="wide"
)

# -----------------------------
# Load Saved Models
# -----------------------------
@st.cache_resource
def load_files():
    with open("Kmeans_model (2).pkl", "rb") as f:
        Kmeans_model = joblib.load(f)

    with open("scaler (2).pkl", "rb") as f:
        scaler = joblib.load(f)

    with open("cluster_mapping (1).pkl", "rb") as f:
        cluster_mapping = joblib.load(f)

    return Kmeans_model, scaler, cluster_mapping


try:
    Kmeans_model, scaler, cluster_mapping = load_files()
except Exception as e:
    st.error(f"Unable to load model files: {e}")
    st.stop()


# -----------------------------
# Sample Data for Sidebar Charts
# -----------------------------
# Replace this sample dataset with your actual dataset
# to display real customer segmentation statistics.

@st.cache_data
def load_sample_data():
    np.random.seed(42)

    data = pd.DataFrame({
        "Age": np.random.randint(18, 70, 200),
        "Annual Income (k$)": np.random.randint(15, 140, 200),
        "Spending Score (1-100)": np.random.randint(1, 100, 200)
    })

    return data


df = load_sample_data()

# Predict segments for the sample records
try:
    scaled_df = scaler.transform(df)
    df["Cluster"] = Kmeans_model.predict(scaled_df)
except Exception:
    df["Cluster"] = 0


# -----------------------------
# Sidebar Dashboard
# -----------------------------
with st.sidebar:
    st.title("📊 Analytics Dashboard")
    st.caption("Customer Segmentation Overview")

    st.metric("Total Sample Customers", len(df))
    st.metric("Number of Clusters", 4)
    st.metric("Average Age", f"{df['Age'].mean():.1f} years")
    st.metric(
        "Average Income",
        f"${df['Annual Income (k$)'].mean():.1f}k"
    )

    st.divider()

    # Cluster distribution graph
    st.subheader("Cluster Distribution")

    cluster_counts = (
        df["Cluster"]
        .value_counts()
        .sort_index()
    )

    fig1, ax1 = plt.subplots(figsize=(4, 3))
    ax1.bar(
        [f"Cluster {c}" for c in cluster_counts.index],
        cluster_counts.values,
        color=["#4F46E5", "#06B6D4", "#F59E0B", "#10B981"][
            :len(cluster_counts)
        ]
    )
    ax1.set_ylabel("Customers")
    ax1.tick_params(axis="x", rotation=30)
    fig1.tight_layout()

    st.pyplot(fig1)
    plt.close(fig1)

    st.divider()

    # Age distribution
    st.subheader("Age Distribution")

    fig2, ax2 = plt.subplots(figsize=(4, 2.5))
    ax2.hist(
        df["Age"],
        bins=12,
        color="#8B5CF6",
        edgecolor="white"
    )
    ax2.set_xlabel("Age")
    ax2.set_ylabel("Customers")
    fig2.tight_layout()

    st.pyplot(fig2)
    plt.close(fig2)

    st.divider()

    # Income vs Spending
    st.subheader("Income vs Spending")

    fig3, ax3 = plt.subplots(figsize=(4, 3))

    scatter = ax3.scatter(
        df["Annual Income (k$)"],
        df["Spending Score (1-100)"],
        c=df["Cluster"],
        cmap="viridis",
        alpha=0.7
    )

    ax3.set_xlabel("Income (k$)")
    ax3.set_ylabel("Spending Score")
    fig3.tight_layout()

    st.pyplot(fig3)
    plt.close(fig3)

    st.caption("Charts currently use illustrative sample data.")


# -----------------------------
# Main Page
# -----------------------------
st.title("🛍️ Customer Segmentation")
st.write(
    "Predict customer segments using age, annual income, "
    "and spending score."
)

st.markdown("---")

# Top-level overview
col1, col2, col3 = st.columns(3)

col1.info("👥 **4 Customer Clusters**")
col2.info("💰 **Income-Based Analysis**")
col3.info("🎯 **Spending Behavior**")

st.subheader("Enter Customer Details")

with st.form("customer_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input(
            "Age",
            min_value=1,
            max_value=100,
            value=30
        )

    with col2:
        income = st.number_input(
            "Annual Income (k$)",
            min_value=0.0,
            max_value=500.0,
            value=50.0,
            step=1.0
        )

    with col3:
        spending_score = st.slider(
            "Spending Score (1-100)",
            min_value=1,
            max_value=100,
            value=50
        )

    submitted = st.form_submit_button(
        "🔍 Predict Customer Segment",
        use_container_width=True
    )


# -----------------------------
# Prediction Result
# -----------------------------
if submitted:
    customer = pd.DataFrame(
        [[age, income, spending_score]],
        columns=[
            "Age",
            "Annual Income (k$)",
            "Spending Score (1-100)"
        ]
    )

    try:
        scaled_customer = scaler.transform(customer)
        prediction = Kmeans_model.predict(scaled_customer)[0]

        # Handle integer and string dictionary keys
        segment = f"Cluster {prediction}"

        if isinstance(cluster_mapping, dict):
            segment = cluster_mapping.get(
                prediction,
                cluster_mapping.get(
                    str(prediction),
                    segment
                )
            )

        st.markdown("---")
        st.subheader("🎉 Prediction Result")

        st.success(f"Predicted Segment: {segment}")

        c1, c2, c3 = st.columns(3)
        c1.metric("Age", f"{age} years")
        c2.metric("Annual Income", f"${income:.1f}k")
        c3.metric("Spending Score", spending_score)

        st.caption(f"Model cluster ID: {prediction}")

        # Visualize this customer's profile
        st.subheader("Customer Profile")

        profile = pd.DataFrame({
            "Feature": [
                "Age",
                "Annual Income (k$)",
                "Spending Score (1-100)"
            ],
            "Value": [age, income, spending_score]
        })

        st.bar_chart(
            profile.set_index("Feature"),
            horizontal=True
        )

    except Exception as e:
        st.error(f"Prediction failed: {e}")
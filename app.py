
import io
import warnings

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="RetailIQ | Customer Segmentation",
    page_icon="🛍️",
    layout="wide"
)

st.markdown("""
<style>
.main {background-color: #FAF9F6;}
h1, h2, h3 {color: #244F78;}
[data-testid="stMetric"] {
    background: white;
    border: 1px solid #E5E7EB;
    padding: 16px;
    border-radius: 12px;
}
.stButton > button {
    background-color: #244F78;
    color: white;
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)


def normalize_columns(df):
    aliases = {
        "invoiceno": "InvoiceNo",
        "invoice": "InvoiceNo",
        "invoicenumber": "InvoiceNo",
        "quantity": "Quantity",
        "qty": "Quantity",
        "invoicedate": "InvoiceDate",
        "transactiondate": "InvoiceDate",
        "date": "InvoiceDate",
        "unitprice": "UnitPrice",
        "price": "UnitPrice",
        "customerid": "CustomerID",
        "customer": "CustomerID",
        "description": "Description",
        "stockcode": "StockCode",
        "country": "Country"
    }

    df.columns = [
        str(c).strip() for c in df.columns
    ]

    rename = {}

    for col in df.columns:
        key = (
            col.lower()
            .replace(" ", "")
            .replace("_", "")
        )

        if key in aliases:
            rename[col] = aliases[key]

    return df.rename(columns=rename)


@st.cache_data(show_spinner=False)
def read_dataset(file_bytes, filename):
    if filename.lower().endswith(".csv"):
        try:
            return pd.read_csv(
                io.BytesIO(file_bytes),
                encoding="utf-8"
            )
        except UnicodeDecodeError:
            return pd.read_csv(
                io.BytesIO(file_bytes),
                encoding="latin1"
            )

    return pd.read_excel(io.BytesIO(file_bytes))


def prepare_rfm(raw):
    df = normalize_columns(raw.copy())

    required = [
        "InvoiceNo",
        "Quantity",
        "InvoiceDate",
        "UnitPrice",
        "CustomerID"
    ]

    missing = [
        c for c in required if c not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing)
        )

    before = len(df)

    df["InvoiceDate"] = pd.to_datetime(
        df["InvoiceDate"], errors="coerce"
    )
    df["Quantity"] = pd.to_numeric(
        df["Quantity"], errors="coerce"
    )
    df["UnitPrice"] = pd.to_numeric(
        df["UnitPrice"], errors="coerce"
    )
    df["CustomerID"] = pd.to_numeric(
        df["CustomerID"], errors="coerce"
    )
    df["InvoiceNo"] = df["InvoiceNo"].astype(str)

    df = df.dropna(
        subset=[
            "InvoiceDate", "CustomerID",
            "Quantity", "UnitPrice"
        ]
    )

    df = df[
        (df["Quantity"] > 0)
        & (df["UnitPrice"] > 0)
        & (~df["InvoiceNo"].str.upper().str.startswith("C"))
    ]

    df = df.drop_duplicates()
    df["TotalAmount"] = (
        df["Quantity"] * df["UnitPrice"]
    )

    if df.empty:
        raise ValueError(
            "No valid transactions remain after cleaning."
        )

    reference_date = (
        df["InvoiceDate"].max()
        + pd.Timedelta(days=1)
    )

    rfm = df.groupby("CustomerID").agg(
        Recency=(
            "InvoiceDate",
            lambda x: (reference_date - x.max()).days
        ),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("TotalAmount", "sum")
    ).reset_index()

    rfm = rfm.replace(
        [np.inf, -np.inf], np.nan
    ).dropna(
        subset=["Recency", "Frequency", "Monetary"]
    )

    if len(rfm) < 3:
        raise ValueError(
            "At least three customers are needed."
        )

    return df, rfm, before, reference_date


def build_personas(profile):
    result = profile.copy()

    overall_r = result["Average_Recency"].mean()
    overall_f = result["Average_Frequency"].mean()
    overall_m = result["Average_Monetary"].mean()

    names = []
    actions = []

    for _, row in result.iterrows():
        r = row["Average_Recency"]
        f = row["Average_Frequency"]
        m = row["Average_Monetary"]

        if (
            r <= overall_r
            and f >= overall_f
            and m >= overall_m
        ):
            name = "Loyal High-Value Customers"
            action = (
                "Offer VIP rewards, loyalty benefits, "
                "and early access to new products."
            )

        elif r > overall_r * 1.5:
            name = "At-Risk / Inactive Customers"
            action = (
                "Run targeted win-back campaigns "
                "and test limited-time return offers."
            )

        elif f >= overall_f and r <= overall_r:
            name = "Loyal Regular Customers"
            action = (
                "Encourage repeat purchases with "
                "loyalty points and relevant bundles."
            )

        elif m >= overall_m and r <= overall_r:
            name = "Recent Big Spenders"
            action = (
                "Use personalized recommendations "
                "and complementary product offers."
            )

        elif f < overall_f and m < overall_m:
            name = "Low-Value / Occasional Customers"
            action = (
                "Send personalized recommendations "
                "and targeted return-purchase offers."
            )

        else:
            name = "Potential Growth Customers"
            action = (
                "Use relevant promotions to encourage "
                "more frequent purchases."
            )

        names.append(name)
        actions.append(action)

    result["Marketing_Persona"] = names
    result["Recommended_Action"] = actions

    return result


# Sidebar
with st.sidebar:
    st.title("🛍️ RetailIQ")
    st.caption("Retail Customer Analytics")
    st.divider()

    uploaded = st.file_uploader(
        "Upload retail transaction data",
        type=["csv", "xlsx", "xls"]
    )

    k_choice = st.slider(
        "Number of clusters (K)",
        min_value=2,
        max_value=8,
        value=2
    )

    st.caption(
        "Select K manually or use the model-selection "
        "analysis to compare possible values."
    )

    st.divider()
    st.caption(
        "Required columns: InvoiceNo, Quantity, "
        "InvoiceDate, UnitPrice, CustomerID."
    )


st.title("RetailIQ — Customer Segmentation")
st.write(
    "Discover customer groups, understand purchasing "
    "behaviour, and choose targeted marketing actions."
)

if uploaded is None:
    st.info(
        "Upload your Online Retail CSV or Excel file "
        "using the sidebar to begin."
    )

    st.subheader("How it works")

    a, b, c = st.columns(3)
    a.metric("1", "RFM Analysis")
    b.metric("2", "Machine Learning")
    c.metric("3", "Marketing Personas")

    st.markdown("""
    **Project workflow**

    1. Upload and clean transaction data.
    2. Calculate Recency, Frequency, and Monetary value.
    3. Scale the features and compare cluster counts.
    4. Run K-Means and Hierarchical Clustering.
    5. Explore visualizations, customer personas, and actions.
    """)
    st.stop()


try:
    raw = read_dataset(
        uploaded.getvalue(), uploaded.name
    )
    transactions, rfm, rows_before, reference_date = (
        prepare_rfm(raw)
    )
except Exception as exc:
    st.error(f"Could not process the dataset: {exc}")
    st.stop()


st.success("Dataset loaded and cleaned successfully.")

m1, m2, m3, m4 = st.columns(4)

m1.metric("Original rows", f"{rows_before:,}")
m2.metric("Valid transactions", f"{len(transactions):,}")
m3.metric("Unique customers", f"{len(rfm):,}")
m4.metric("Chosen clusters", k_choice)

st.caption(
    f"RFM reference date: {reference_date.date()}"
)

with st.expander("Preview cleaned transaction data"):
    st.dataframe(
        transactions.head(20),
        use_container_width=True
    )

with st.expander("Preview customer RFM data"):
    st.dataframe(
        rfm.head(20),
        use_container_width=True
    )


# Transform and scale
features = rfm[
    ["Recency", "Frequency", "Monetary"]
].copy()

log_features = np.log1p(features)

scaler = StandardScaler()
scaled = scaler.fit_transform(log_features)

# Select K using a sample for silhouette scoring when necessary
max_k = min(8, len(rfm) - 1)
candidate_ks = range(2, max_k + 1)

if k_choice > max_k:
    st.error("Not enough customers for the selected K.")
    st.stop()

inertias = []
silhouettes = []

for k in candidate_ks:
    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )
    labels = model.fit_predict(scaled)

    inertias.append(model.inertia_)

    score = silhouette_score(
        scaled,
        labels,
        sample_size=min(2000, len(rfm)),
        random_state=42
    )
    silhouettes.append(score)


# Main clustering model
kmeans = KMeans(
    n_clusters=k_choice,
    random_state=42,
    n_init=10
)
rfm["KMeans_Cluster"] = kmeans.fit_predict(scaled)

kmeans_score = silhouette_score(
    scaled,
    rfm["KMeans_Cluster"],
    sample_size=min(2000, len(rfm)),
    random_state=42
)

# Hierarchical clustering comparison
with st.spinner("Running hierarchical clustering..."):
    hierarchical = AgglomerativeClustering(
        n_clusters=k_choice,
        linkage="ward"
    )
    rfm["Hierarchical_Cluster"] = (
        hierarchical.fit_predict(scaled)
    )

hierarchical_score = silhouette_score(
    scaled,
    rfm["Hierarchical_Cluster"],
    sample_size=min(2000, len(rfm)),
    random_state=42
)


# Profiles
profile = rfm.groupby("KMeans_Cluster").agg(
    Customers=("CustomerID", "count"),
    Average_Recency=("Recency", "mean"),
    Average_Frequency=("Frequency", "mean"),
    Average_Monetary=("Monetary", "mean"),
    Total_Revenue=("Monetary", "sum")
).reset_index()

profile["Customer_Percentage"] = (
    profile["Customers"] / len(rfm) * 100
)

profile = build_personas(profile)

persona_map = dict(zip(
    profile["KMeans_Cluster"],
    profile["Marketing_Persona"]
))

action_map = dict(zip(
    profile["KMeans_Cluster"],
    profile["Recommended_Action"]
))

rfm["Marketing_Persona"] = (
    rfm["KMeans_Cluster"].map(persona_map)
)
rfm["Recommended_Action"] = (
    rfm["KMeans_Cluster"].map(action_map)
)


# Tabs
overview, segments, models, customers, downloads = st.tabs([
    "📊 Overview",
    "👥 Customer Personas",
    "🧠 Model Analysis",
    "🔎 Customer Explorer",
    "⬇️ Downloads"
])


with overview:
    st.subheader("Customer segmentation overview")

    c1, c2 = st.columns(2)

    with c1:
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.barplot(
            data=profile,
            x="KMeans_Cluster",
            y="Customers",
            ax=ax
        )
        ax.set_title("Customers per cluster")
        ax.set_xlabel("Cluster")
        ax.set_ylabel("Number of customers")
        st.pyplot(fig)
        plt.close(fig)

    with c2:
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.barplot(
            data=profile,
            x="KMeans_Cluster",
            y="Average_Monetary",
            ax=ax
        )
        ax.set_title("Average spending per cluster")
        ax.set_xlabel("Cluster")
        ax.set_ylabel("Average monetary value")
        st.pyplot(fig)
        plt.close(fig)

    st.subheader("Customer RFM profile")
    st.dataframe(
        profile.round(2),
        use_container_width=True,
        hide_index=True
    )


with segments:
    st.subheader("Marketing personas")

    for _, row in profile.iterrows():
        with st.container(border=True):
            st.markdown(
                f"### {row['Marketing_Persona']}"
            )

            a, b, c = st.columns(3)
            a.metric(
                "Customers",
                f"{int(row['Customers']):,}"
            )
            b.metric(
                "Customer share",
                f"{row['Customer_Percentage']:.1f}%"
            )
            c.metric(
                "Average spending",
                f"{row['Average_Monetary']:,.2f}"
            )

            st.write(
                f"**Typical recency:** "
                f"{row['Average_Recency']:.1f} days"
            )
            st.write(
                f"**Typical frequency:** "
                f"{row['Average_Frequency']:.2f} invoices"
            )
            st.write(
                f"**Marketing action:** "
                f"{row['Recommended_Action']}"
            )

    st.subheader("One-page persona sheet")

    persona_sheet = profile[[
        "KMeans_Cluster",
        "Customers",
        "Customer_Percentage",
        "Average_Recency",
        "Average_Frequency",
        "Average_Monetary",
        "Marketing_Persona",
        "Recommended_Action"
    ]].copy()

    persona_sheet.columns = [
        "Cluster",
        "Customers",
        "Customer %",
        "Avg Recency (days)",
        "Avg Frequency",
        "Avg Spending",
        "Marketing Persona",
        "Recommended Marketing Action"
    ]

    st.dataframe(
        persona_sheet.round(2),
        use_container_width=True,
        hide_index=True
    )


with models:
    st.subheader("Elbow method")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(
        list(candidate_ks),
        inertias,
        marker="o"
    )
    ax.set_xlabel("Number of clusters (K)")
    ax.set_ylabel("Inertia")
    ax.set_title("Elbow Method")
    ax.grid(alpha=0.3)
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Silhouette score")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(
        list(candidate_ks),
        silhouettes,
        marker="o"
    )
    ax.set_xlabel("Number of clusters (K)")
    ax.set_ylabel("Silhouette score")
    ax.set_title("Silhouette Scores by K")
    ax.grid(alpha=0.3)
    st.pyplot(fig)
    plt.close(fig)

    best_k = list(candidate_ks)[
        int(np.argmax(silhouettes))
    ]

    st.info(
        f"Best K among tested values by silhouette score: "
        f"{best_k}. Current selected K: {k_choice}."
    )

    st.subheader("Algorithm comparison")

    comparison = pd.DataFrame({
        "Algorithm": [
            "K-Means",
            "Hierarchical Clustering"
        ],
        "Clusters": [k_choice, k_choice],
        "Silhouette Score": [
            kmeans_score,
            hierarchical_score
        ]
    })

    st.dataframe(
        comparison.round(4),
        use_container_width=True,
        hide_index=True
    )

    st.subheader("PCA cluster visualization")

    pca = PCA(n_components=2)
    coords = pca.fit_transform(scaled)

    plot_df = pd.DataFrame({
        "PCA1": coords[:, 0],
        "PCA2": coords[:, 1],
        "Cluster": rfm["KMeans_Cluster"].astype(str).values
    })

    fig, ax = plt.subplots(figsize=(9, 6))
    sns.scatterplot(
        data=plot_df,
        x="PCA1",
        y="PCA2",
        hue="Cluster",
        palette="tab10",
        s=35,
        alpha=0.7,
        ax=ax
    )
    ax.set_title("Customer Segments Using PCA")
    st.pyplot(fig)
    plt.close(fig)

    st.caption(
        "PCA explained variance: "
        f"{pca.explained_variance_ratio_.sum() * 100:.2f}% "
        "for the first two components."
    )

    st.subheader("Hierarchical clustering dendrogram")

    # Use a subset to keep the dendrogram readable and memory manageable.
    sample_n = min(250, len(rfm))
    rng = np.random.default_rng(42)
    sample_idx = rng.choice(
        len(rfm), size=sample_n, replace=False
    )
    sample_scaled = scaled[sample_idx]

    linked = linkage(sample_scaled, method="ward")

    fig, ax = plt.subplots(figsize=(12, 5))
    dendrogram(
        linked,
        truncate_mode="lastp",
        p=25,
        ax=ax
    )
    ax.set_title(
        f"Hierarchical Dendrogram "
        f"(sample of {sample_n} customers)"
    )
    ax.set_xlabel("Customer groups")
    ax.set_ylabel("Distance")
    st.pyplot(fig)
    plt.close(fig)


with customers:
    st.subheader("Search customer records")

    query = st.text_input(
        "Search by Customer ID"
    )

    filtered = rfm.copy()

    if query.strip():
        filtered = filtered[
            filtered["CustomerID"]
            .astype(str)
            .str.contains(query.strip(), regex=False)
        ]

    selected_cluster = st.selectbox(
        "Filter by cluster",
        ["All"] + [
            str(c) for c in sorted(
                rfm["KMeans_Cluster"].unique()
            )
        ]
    )

    if selected_cluster != "All":
        filtered = filtered[
            filtered["KMeans_Cluster"]
            == int(selected_cluster)
        ]

    st.dataframe(
        filtered[
            [
                "CustomerID", "Recency", "Frequency",
                "Monetary", "KMeans_Cluster",
                "Marketing_Persona", "Recommended_Action"
            ]
        ].round(2),
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "Persona labels describe the overall cluster and "
        "may not perfectly describe every individual customer."
    )


with downloads:
    st.subheader("Download project results")

    st.download_button(
        "Download customer segments CSV",
        data=rfm.to_csv(index=False).encode("utf-8"),
        file_name="customer_segments.csv",
        mime="text/csv"
    )

    st.download_button(
        "Download one-page persona sheet CSV",
        data=persona_sheet.to_csv(index=False).encode("utf-8"),
        file_name="customer_persona_sheet.csv",
        mime="text/csv"
    )

    st.download_button(
        "Download clustering comparison CSV",
        data=comparison.to_csv(index=False).encode("utf-8"),
        file_name="clustering_comparison.csv",
        mime="text/csv"
    )

    st.caption(
        "Download the persona sheet and use it in your "
        "project report or presentation."
    )


st.divider()
st.caption(
    "RetailIQ | Customer Segmentation for a Retail Chain | "
    "Academic model-making project"
)

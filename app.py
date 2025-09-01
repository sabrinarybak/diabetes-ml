import io
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import requests

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

# Klassificeringsmodeller
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

# Metrics
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve,
    classification_report
)

st.set_page_config(page_title="Diabetes ML – end-to-end", layout="wide")
st.title("🏥 Diabetes (Pima) – end-to-end ML i Streamlit")
st.caption("Ladda dataset → EDA → Träna modeller → Utvärdera → Exportera → Prediktion")

# ------------------------------------------------------------
# 1) LADDA DATA
# ------------------------------------------------------------
st.sidebar.header("1) Data")

SAMPLE_URL = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"  # Pima-format
DATA_PATH = "diabetes.csv"

col1, col2 = st.columns(2)
with col1:
    file = st.file_uploader("Ladda upp CSV (valfritt)", type=["csv"])
with col2:
    if st.button("⬇️ Ladda exempeldataset (Pima)", type="secondary"):
        try:
            r = requests.get(SAMPLE_URL, timeout=15)
            r.raise_for_status()
            with open(DATA_PATH, "wb") as f:
                f.write(r.content)
            st.success("Exempeldataset sparat som diabetes.csv i projektmappen.")
        except Exception as e:
            st.error(f"Kunde inte ladda exempeldataset: {e}")

@st.cache_data
def load_df(uploaded_file, local_path):
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    if os.path.exists(local_path):
        return pd.read_csv(local_path)
    return None

df = load_df(file, DATA_PATH)

if df is None:
    st.info("Ladda upp en CSV eller klicka på 'Ladda exempeldataset (Pima)'.")
    st.stop()

st.subheader("Förhandsvisning av data")
st.dataframe(df.head(), use_container_width=True)

st.write("**Form:**", df.shape)
st.write("**Kolumner:**", list(df.columns))

# Standard-Pima har featurekolumner + 'Outcome' som mål (0/1)
if "Outcome" in df.columns:
    target_col = "Outcome"
else:
    # Försök hitta en binär kolumn automatiskt
    possible = [c for c in df.columns if df[c].nunique() == 2]
    target_col = possible[0] if possible else st.selectbox("Välj målvariabel (binär)", df.columns)

feature_cols = [c for c in df.columns if c != target_col]

# ------------------------------------------------------------
# 2) EDA
# ------------------------------------------------------------
st.subheader("🔎 EDA & Statistik")

with st.expander("Saknade värden & dtyper"):
    st.write(df.isna().sum().sort_values(ascending=False))
    st.write(df.dtypes)

num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = [c for c in feature_cols if c not in num_cols]

with st.expander("Histogram (numeriska)"):
    for col in num_cols:
        fig, ax = plt.subplots()
        ax.hist(df[col].dropna(), bins=30)
        ax.set_title(f"Histogram – {col}")
        st.pyplot(fig, clear_figure=True)

with st.expander("Korrelationsmatris"):
    if len(num_cols) > 1:
        corr = df[num_cols].corr(numeric_only=True)
        fig, ax = plt.subplots(figsize=(6,5))
        sns.heatmap(corr, ax=ax)
        ax.set_title("Korrelationsmatris")
        st.pyplot(fig, clear_figure=True)

with st.expander("PCA (2D) för visualisering"):
    if len(num_cols) >= 2:
        pca = PCA(n_components=2)
        pcs = pca.fit_transform(df[num_cols].dropna())
        fig, ax = plt.subplots()
        ax.scatter(pcs[:,0], pcs[:,1], s=12)
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
        ax.set_title("PCA – 2D plot")
        st.pyplot(fig, clear_figure=True)

# ------------------------------------------------------------
# 3) MODELLERING (Klassificering)
# ------------------------------------------------------------
st.subheader("⚙️ Modellering – Klassificering")

X = df[feature_cols]
y = df[target_col]

# Ta bort rader där y saknas
valid_idx = y.notna()
X = X[valid_idx]
y = y[valid_idx]

# Preprocess (impute + scale + one-hot)
num_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
cat_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", num_transformer, [c for c in feature_cols if c in num_cols]),
    ("cat", cat_transformer, [c for c in feature_cols if c in cat_cols])
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model_name = st.selectbox("Välj algoritm", [
    "LogisticRegression", "RandomForestClassifier", "SVC (RBF)", "MLPClassifier"
])

if model_name == "LogisticRegression":
    model = LogisticRegression(max_iter=1000)
elif model_name == "RandomForestClassifier":
    model = RandomForestClassifier(n_estimators=300, random_state=42)
elif model_name == "SVC (RBF)":
    model = SVC(probability=True)
else:
    model = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=400, random_state=42)

clf = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])

if st.button("Träna modell", type="primary"):
    clf.fit(X_train, y_train)
    st.success("Modell tränad!")

    # --------------------------------------------------------
    # 4) UTVÄRDERING
    # --------------------------------------------------------
    st.subheader("📊 Utvärdering")
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted")
    st.write({"accuracy": acc, "precision": prec, "recall": rec, "f1": f1})

    # Confusion Matrix
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred)).plot(ax=ax, colorbar=False)
    ax.set_title("Confusion Matrix")
    st.pyplot(fig, clear_figure=True)

    # ROC endast om binär
    try:
        y_proba = clf.predict_proba(X_test)
        if y_proba.shape[1] == 2:
            fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1], pos_label=list(np.unique(y_test))[1])
            auc = roc_auc_score(y_test, y_proba[:, 1])
            fig2, ax2 = plt.subplots()
            ax2.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
            ax2.plot([0,1],[0,1], linestyle="--")
            ax2.set_title("ROC-kurva"); ax2.set_xlabel("FPR"); ax2.set_ylabel("TPR")
            ax2.legend()
            st.pyplot(fig2, clear_figure=True)
    except Exception:
        pass

    st.code(classification_report(y_test, y_pred, zero_division=0))

    # Spara tränad pipeline
    buffer = io.BytesIO()
    joblib.dump(clf, buffer)
    buffer.seek(0)
    st.download_button("💾 Ladda ner modell (.joblib)", data=buffer, file_name="diabetes_pipeline.joblib")

# ------------------------------------------------------------
# 5) SNABBPREDIKTION
# ------------------------------------------------------------
st.subheader("🔮 Snabbprediktion (ladda modell eller använd senaste)")
uploaded_model = st.file_uploader("Ladda upp .joblib (valfritt)", type=["joblib"])
if uploaded_model is not None:
    clf = joblib.load(uploaded_model)
    st.success("Uppladdad modell laddad.")

if 'clf' in locals():
    with st.form("predict_form"):
        st.caption("Fyll i feature-värden – lämna tomt för NaN")
        inputs = {}
        for col in feature_cols:
            default = ""
            val = st.text_input(col, value=default)
            try:
                inputs[col] = float(val) if val != "" else np.nan
            except Exception:
                inputs[col] = val if val != "" else None
        submitted = st.form_submit_button("Predict")
    if submitted:
        X_inf = pd.DataFrame([inputs])
        pred = clf.predict(X_inf)[0]
        st.success(f"Prediktion (Outcome): **{pred}**")
import io
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import requests

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve,
    classification_report, silhouette_score
)

# Klassificeringsmodeller
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

# XGBoost (valfritt)
try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False

# UMAP (för visualisering)
try:
    import umap.umap_ as umap
    UMAP_AVAILABLE = True
except Exception:
    UMAP_AVAILABLE = False

st.set_page_config(page_title="Diabetes ML – end-to-end (UMAP + GridSearch)", layout="wide")
st.title("🏥 Diabetes (Pima) – end-to-end ML i Streamlit")
st.caption("Data → EDA → Supervised → Model Comparison (GridSearchCV) → Unsupervised (UMAP + Clustering) → Export → Prediction")

# ------------------------------------------------------------
# 1) LADDA DATA
# ------------------------------------------------------------
st.sidebar.header("1) Data")

SAMPLE_URL = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"  # Pima-format
DATA_PATH = "diabetes.csv"

c1, c2 = st.columns(2)
with c1:
    file = st.file_uploader("Ladda upp CSV (valfritt)", type=["csv"])
with c2:
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

# Standard-Pima har featurekolumner + 'Outcome' (0/1)
target_col = "Outcome" if "Outcome" in df.columns else st.selectbox("Välj målvariabel (binär)", df.columns)
feature_cols = [c for c in df.columns if c != target_col]

num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = [c for c in feature_cols if c not in num_cols]

# Gör train/test en gång och återanvänd i alla tabs
valid_idx = df[target_col].notna()
X = df.loc[valid_idx, feature_cols]
y = df.loc[valid_idx, target_col]

test_size = st.sidebar.slider("Teststorlek", 0.1, 0.4, 0.2, step=0.05)
random_state = st.sidebar.number_input("Random state", value=42)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=random_state, stratify=y
)

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

# ------------------------------------------------------------
# TABS
# ------------------------------------------------------------
tab_eda, tab_sup, tab_gs, tab_unsup, tab_pred = st.tabs([
    "EDA", "Supervised (Train & Evaluate)", "Model Comparison (GridSearchCV)", "Unsupervised (UMAP + Clustering)", "Snabbprediktion"
])

# =============================
# EDA TAB
# =============================
with tab_eda:
    st.subheader("🔎 EDA & Statistik")
    st.write("**Form:**", df.shape)
    st.write("**Kolumner:**", list(df.columns))

    with st.expander("Saknade värden & dtyper"):
        st.write(df.isna().sum().sort_values(ascending=False))
        st.write(df.dtypes)

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

    with st.expander("UMAP (2D) för visualisering"):
        if UMAP_AVAILABLE and len(num_cols) >= 2:
            n_neighbors = st.slider("n_neighbors", 5, 50, 15)
            min_dist = st.slider("min_dist", 0.0, 0.99, 0.1)
            reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=random_state)
            Xu = df[num_cols].dropna()
            emb = reducer.fit_transform(Xu)
            fig, ax = plt.subplots()
            ax.scatter(emb[:,0], emb[:,1], s=12)
            ax.set_title("UMAP – 2D embedding")
            ax.set_xlabel("UMAP1"); ax.set_ylabel("UMAP2")
            st.pyplot(fig, clear_figure=True)
        elif not UMAP_AVAILABLE:
            st.info("UMAP är inte installerat. Lägg till `umap-learn` i requirements (det ingår redan i den här mallen).")

# =============================
# SUPERVISED TAB
# =============================
with tab_sup:
    st.subheader("⚙️ Modellering – Klassificering")
    use_balanced = st.checkbox("Använd class_weight='balanced' (hantera ev. obalans)", value=True)

    model_name = st.selectbox("Välj algoritm", [
        "LogisticRegression", "RandomForestClassifier", "SVC (RBF)", "MLPClassifier"
    ])

    if model_name == "LogisticRegression":
        model = LogisticRegression(max_iter=1000, class_weight="balanced" if use_balanced else None)
    elif model_name == "RandomForestClassifier":
        model = RandomForestClassifier(n_estimators=300, random_state=random_state, class_weight="balanced" if use_balanced else None)
    elif model_name == "SVC (RBF)":
        model = SVC(probability=True, class_weight="balanced" if use_balanced else None)
    else:
        model = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=400, random_state=random_state)

    clf = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])

    if st.button("Träna vald modell", type="primary"):
        clf.fit(X_train, y_train)
        st.session_state["clf"] = clf  # Spara fitted modell
        st.success("Modell tränad!")

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

        # ROC (binär)
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

        # Spara modell
        buffer = io.BytesIO()
        joblib.dump(clf, buffer)
        buffer.seek(0)
        st.download_button("💾 Ladda ner modell (.joblib)", data=buffer, file_name="diabetes_pipeline.joblib")

# =============================
# GRID SEARCH TAB
# =============================
with tab_gs:
    st.subheader("🔎 Model Comparison – GridSearchCV")

    # Scoring
    is_binary = y.nunique() == 2
    scoring_choice = st.selectbox("Välj scoring", ["accuracy", "f1_weighted"] + (["roc_auc"] if is_binary else []))
    n_splits = st.slider("CV folds", 3, 10, 5)
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # Kandidater
    candidates = {}

    # Logistic Regression
    candidates["LogisticRegression"] = (LogisticRegression(max_iter=2000, class_weight="balanced"), {
        "model__C": [0.1, 1.0, 10.0],
        "model__solver": ["liblinear", "lbfgs"]
    })

    # Random Forest
    candidates["RandomForestClassifier"] = (RandomForestClassifier(random_state=random_state, class_weight="balanced"), {
        "model__n_estimators": [200, 500],
        "model__max_depth": [None, 5, 10]
    })

    # SVC
    candidates["SVC"] = (SVC(probability=True, class_weight="balanced"), {
        "model__C": [0.1, 1, 10],
        "model__gamma": ["scale"],
        "model__kernel": ["rbf"]
    })

    # MLP
    candidates["MLPClassifier"] = (MLPClassifier(max_iter=600, random_state=random_state), {
        "model__hidden_layer_sizes": [(64,32), (128,64)],
        "model__alpha": [0.0001, 0.001]
    })

    # XGB
    if XGB_AVAILABLE:
        candidates["XGBClassifier"] = (XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=random_state,
            tree_method="hist"
        ), {
            "model__n_estimators": [300, 600],
            "model__max_depth": [3, 5],
            "model__learning_rate": [0.05, 0.1],
            "model__subsample": [0.8, 1.0]
        })
    else:
        st.info("XGBoost inte tillgängligt i miljön. Installera `xgboost` (ingår i requirements) för att aktivera.")

    results_rows = []
    best_overall = None
    best_overall_score = -np.inf
    best_overall_name = None

    run_search = st.button("Kör GridSearchCV på alla modeller", type="primary")
    if run_search:
        for name, (estimator, grid) in candidates.items():
            pipe = Pipeline(steps=[("preprocess", preprocessor), ("model", estimator)])
            gs = GridSearchCV(
                pipe,
                param_grid=grid,
                scoring=scoring_choice,
                cv=kf,
                n_jobs=-1,
                refit=True,
                return_train_score=False,
            )
            with st.spinner(f"Kör GridSearch för {name}..."):
                gs.fit(X_train, y_train)

            # Hämta score på valideringen (mean cv)
            mean_cv = gs.best_score_
            results_rows.append({
                "model": name,
                "best_cv_score": mean_cv,
                "best_params": gs.best_params_
            })

            # Håll koll på bästa
            if mean_cv > best_overall_score:
                best_overall_score = mean_cv
                best_overall = gs.best_estimator_
                best_overall_name = name

        # Visa sammanställning
        res_df = pd.DataFrame(results_rows).sort_values(by="best_cv_score", ascending=False)
        st.dataframe(res_df, use_container_width=True)

        # Utvärdera bästa modellen på testset
        if best_overall is not None:
            st.success(f"Bästa modell enligt CV: **{best_overall_name}** (score={best_overall_score:.4f})")
            y_pred = best_overall.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
            rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
            f1 = f1_score(y_test, y_pred, average="weighted")
            st.write({"TEST_accuracy": acc, "TEST_precision": prec, "TEST_recall": rec, "TEST_f1": f1})

            # Confusion matrix
            fig, ax = plt.subplots()
            ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred)).plot(ax=ax, colorbar=False)
            ax.set_title("Confusion Matrix (best model)")
            st.pyplot(fig, clear_figure=True)

            # ROC
            try:
                y_proba = best_overall.predict_proba(X_test)
                if y_proba.shape[1] == 2 and is_binary:
                    fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1], pos_label=list(np.unique(y_test))[1])
                    auc = roc_auc_score(y_test, y_proba[:, 1])
                    fig2, ax2 = plt.subplots()
                    ax2.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
                    ax2.plot([0,1],[0,1], linestyle="--")
                    ax2.set_title("ROC-kurva (best model)"); ax2.set_xlabel("FPR"); ax2.set_ylabel("TPR")
                    ax2.legend()
                    st.pyplot(fig2, clear_figure=True)
            except Exception:
                pass

            # Spara & sätt som aktiv modell för Pred-fliken
            st.session_state["clf"] = best_overall
            buffer = io.BytesIO()
            joblib.dump(best_overall, buffer)
            buffer.seek(0)
            st.download_button("💾 Ladda ner bästa modell (.joblib)", data=buffer, file_name="best_diabetes_pipeline.joblib")

# =============================
# UNSUPERVISED TAB
# =============================
with tab_unsup:
    st.subheader("🌀 Unsupervised – UMAP + Klustring")

    if len(num_cols) < 2:
        st.info("Behöver minst två numeriska kolumner för klustring/UMAP.")
    else:
        # Förbered numerisk matris
        imputer = SimpleImputer(strategy="median")
        scaler = StandardScaler()
        Xu = imputer.fit_transform(df[num_cols])
        Xu = scaler.fit_transform(Xu)

        # Metodval
        method = st.selectbox("Välj metod", ["KMeans", "DBSCAN"])
        if method == "KMeans":
            k = st.slider("Antal kluster (k)", 2, 10, 3)
            km = KMeans(n_clusters=k, n_init="auto", random_state=random_state)
            labels = km.fit_predict(Xu)
        else:
            eps = st.slider("eps", 0.1, 5.0, 0.5)
            min_samples = st.slider("min_samples", 3, 20, 5)
            db = DBSCAN(eps=eps, min_samples=min_samples)
            labels = db.fit_predict(Xu)

        st.write("Klusteretiketter (första 50):", labels[:50])

        # Silhouette score (n>1 kluster och inte bara -1)
        try:
            if len(set(labels)) > 1 and not (set(labels) == {-1}):
                sil = silhouette_score(Xu, labels)
                st.write({"silhouette_score": sil})
        except Exception:
            pass

        # UMAP-proj om möjligt, annars PCA
        if UMAP_AVAILABLE:
            n_neighbors = st.slider("UMAP n_neighbors", 5, 50, 15)
            min_dist = st.slider("UMAP min_dist", 0.0, 0.99, 0.1)
            reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=random_state)
            emb = reducer.fit_transform(Xu)
            fig, ax = plt.subplots()
            scatter = ax.scatter(emb[:,0], emb[:,1], c=labels, s=12)
            ax.set_title("UMAP – färgad av kluster")
            ax.set_xlabel("UMAP1"); ax.set_ylabel("UMAP2")
            st.pyplot(fig, clear_figure=True)
        else:
            pca = PCA(n_components=2)
            pcs = pca.fit_transform(Xu)
            fig, ax = plt.subplots()
            scatter = ax.scatter(pcs[:,0], pcs[:,1], c=labels, s=12)
            ax.set_title("PCA – färgad av kluster")
            ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
            st.pyplot(fig, clear_figure=True)

        # Jämför mot Outcome om finns (visualisering)
        if target_col in df.columns and UMAP_AVAILABLE:
            yv = df[target_col].values[:len(labels)]
            try:
                reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=random_state)
                emb = reducer.fit_transform(Xu)
                fig, ax = plt.subplots()
                ax.scatter(emb[:,0], emb[:,1], c=yv, s=12)
                ax.set_title("UMAP – färgad av Outcome (0/1)")
                ax.set_xlabel("UMAP1"); ax.set_ylabel("UMAP2")
                st.pyplot(fig, clear_figure=True)
            except Exception:
                pass

# =============================
# PREDICTION TAB (SAFE)
# =============================
with tab_pred:
    st.subheader("🔮 Snabbprediktion")
    uploaded_model = st.file_uploader("Ladda upp .joblib (valfritt)", type=["joblib"])
    if uploaded_model is not None:
        loaded = joblib.load(uploaded_model)
        st.session_state["clf"] = loaded
        st.success("Uppladdad modell laddad och aktiverad.")

    clf_ready = st.session_state.get("clf", None)

    if clf_ready is None:
        st.warning("Ingen tränad modell hittad ännu. Träna en modell i 'Supervised' eller kör 'Model Comparison' – eller ladda upp en .joblib här.")
    else:
        # Om modellen har predict_proba kan vi använda tröskel
        has_proba = hasattr(clf_ready, "predict_proba")
        if has_proba:
            threshold = st.slider("Besluts-tröskel för klass '1' (diabetes)", 0.1, 0.9, 0.5, 0.01)
        with st.form("predict_form"):
            st.caption("Fyll i feature-värden – lämna tomt för NaN")
            inputs = {}
            for col in feature_cols:
                val = st.text_input(col, value="")
                try:
                    inputs[col] = float(val) if val != "" else np.nan
                except Exception:
                    inputs[col] = val if val != "" else None
            submitted = st.form_submit_button("Predict")

        if submitted:
            X_inf = pd.DataFrame([inputs])
            if has_proba:
                proba_vec = clf_ready.predict_proba(X_inf)[0]
                p1 = float(proba_vec[1])
                pred_label = 1 if p1 >= threshold else 0
                label_text = "Diabetes (1)" if pred_label == 1 else "Ingen diabetes (0)"
                st.success(f"P(1=diabetes) = {p1:.3f} | tröskel = {threshold:.2f} → **{label_text}**")
            else:
                pred = clf_ready.predict(X_inf)[0]
                label_text = "Diabetes (1)" if int(pred) == 1 else "Ingen diabetes (0)"
                st.success(f"Prediktion: **{label_text}** (modell utan predict_proba)")

# Fotnot
st.caption("⚠️ Detta är en demomodell på ett öppet dataset – inte medicinsk rådgivning.")
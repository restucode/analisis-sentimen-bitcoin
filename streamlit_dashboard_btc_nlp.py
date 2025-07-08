# streamlit_dashboard_btc_nlp.py

import streamlit as st
import os
import pandas as pd
import string
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    accuracy_score, precision_score, recall_score, f1_score
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud
from collections import Counter

st.set_page_config(page_title="Bitcoin Sentiment Dashboard", layout="wide")

# === Tampilkan isi requirements.txt di sidebar
st.sidebar.markdown("---")
if os.path.exists("requirements.txt"):
    with open("requirements.txt") as f:
        reqs = f.read()
    st.sidebar.text_area("requirements.txt", reqs, height=200)
    st.sidebar.warning("Install requirements terlebih dulu via terminal:\npip install -r requirements.txt")
else:
    st.sidebar.info("File requirements.txt tidak ditemukan.")

st.title("🪙 Bitcoin Dataset Sentiment Dashboard")
st.caption("Responsive dashboard | Classification, Balancing, Feature Visualizations, and Model Comparison")

@st.cache_data
def load_data(input_file):
    df = pd.read_csv(input_file)
    return df

input_file = st.sidebar.text_input(
    "Input CSV file", "bitcoin2225_pelabelan_embedding.csv"
)
result_df = load_data(input_file)
emb_cols = [col for col in result_df.columns if col.startswith('emb_')]

# Feature matrix & label
X = result_df[emb_cols].values
y = result_df['label'].values

# --- Split train-test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# --- SMOTE
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
X_smote, y_smote = smote.fit_resample(X, y)

# =========== SIDEBAR SNAPSHOT =============
with st.sidebar:
    st.header("Data Snapshot & Info")
    st.write("**Contoh Data**")
    st.dataframe(result_df.head(), use_container_width=True)
    st.write("**Fitur Embedding**", f"(jumlah: {len(emb_cols)})")
    st.write(emb_cols[:10])
    st.markdown("---")
    st.write(f"**Total Data:** {len(X)}")
    st.write(f"**Train/Test:** {len(X_train)}/{len(X_test)}")
    st.write(f"**Proporsi:** {len(X_train)/len(X):.2f} : {len(X_test)/len(X):.2f}")
    st.markdown("---")
    st.write("**Distribusi Label Data Lengkap**")
    st.write(pd.Series(y).value_counts())
    st.write("**Distribusi Label Data Latih**")
    st.write(pd.Series(y_train).value_counts())
    st.write("**Distribusi Label Latih - SMOTE**")
    st.write(pd.Series(y_train_smote).value_counts())

st.subheader("Distribusi Label pada Data Lengkap, Latih, Sesudah SMOTE, dan Uji")
cols = st.columns(4)
cols[0].metric('Full Data', f"{len(y)}", "")
cols[1].metric('Train', f"{len(y_train)}", "")
cols[2].metric('SMOTE Train', f"{len(y_train_smote)}", "")
cols[3].metric('Test', f"{len(y_test)}", "")

dist_df = pd.DataFrame({
    "Full Data": pd.Series(y).value_counts().sort_index(),
    "Train": pd.Series(y_train).value_counts().sort_index(),
    "SMOTE Train": pd.Series(y_train_smote).value_counts().sort_index(),
    "Test": pd.Series(y_test).value_counts().sort_index()
})
st.dataframe(dist_df, use_container_width=True)

# =========== t-SNE 2D Visualisasi ============
st.subheader("t-SNE Visualisasi Embedded Feature (sebelum & sesudah SMOTE)")
col1, col2 = st.columns(2)
def plot_tsne(X, y, title):
    idx = np.arange(len(X))
    if len(X) > 2000:  # agar TSNE cepat
        idx = np.random.choice(len(X), 2000, replace=False)
        X, y = X[idx], y[idx]
    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=2, random_state=42)
    X_emb = tsne.fit_transform(X)
    fig, ax = plt.subplots(figsize=(5,4))
    scatter = ax.scatter(X_emb[:,0], X_emb[:,1], c=y, cmap='viridis', alpha=0.6)
    plt.title(title)
    plt.axis("off")
    return fig

with col1:
    st.pyplot(plot_tsne(X, y, "t-SNE Sebelum SMOTE"))
with col2:
    st.pyplot(plot_tsne(X_smote, y_smote, "t-SNE Sesudah SMOTE"))

# =========== Barchart Distribusi Label ===========
st.subheader("Distribusi Label Sebelum/Sesudah SMOTE (Full Data)")
fig, ax = plt.subplots(figsize=(7,4))
dist_df2 = pd.DataFrame({"Sebelum SMOTE": pd.Series(y).value_counts().sort_index(),
                         "Sesudah SMOTE": pd.Series(y_smote).value_counts().sort_index()})
dist_df2.plot(kind='bar', ax=ax)
plt.title("Distribusi Label Full Dataset: Sebelum dan Sesudah SMOTE")
plt.xlabel("Label")
plt.ylabel("Jumlah Data")
st.pyplot(fig)

# ========== Word Frequency & Wordcloud ==========
if "teks" in result_df.columns:
    st.subheader("10 Kata Sentimen Positif/Negatif Paling Sering & Wordcloud")
    pos_texts = result_df[result_df['label'] == 1]['teks'].dropna().astype(str)
    neg_texts = result_df[result_df['label'] == 0]['teks'].dropna().astype(str)
    cv = CountVectorizer(stop_words='english')
    pos_corpus = " ".join(pos_texts.tolist()).lower().translate(str.maketrans('', '', string.punctuation))
    neg_corpus = " ".join(neg_texts.tolist()).lower().translate(str.maketrans('', '', string.punctuation))

    X_pos = cv.fit_transform([pos_corpus])
    pos_word_freq = dict(zip(cv.get_feature_names_out(), X_pos.toarray()[0]))
    pos_common = Counter(pos_word_freq).most_common(10)
    X_neg = cv.fit_transform([neg_corpus])
    neg_word_freq = dict(zip(cv.get_feature_names_out(), X_neg.toarray()[0]))
    neg_common = Counter(neg_word_freq).most_common(10)

    wc_pos = WordCloud(width=400, height=200, background_color="white")
    wc_neg = WordCloud(width=400, height=200, background_color="black", colormap="Reds")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.write("Top 10 Kata Positif")
        st.bar_chart(pd.DataFrame(dict(pos_common), index=['count']).T)
        st.image(wc_pos.generate(pos_corpus).to_array(), caption="Wordcloud Positif", use_column_width=True)
    with col2:
        st.write("Top 10 Kata Negatif")
        st.bar_chart(pd.DataFrame(dict(neg_common), index=['count']).T)
        st.image(wc_neg.generate(neg_corpus).to_array(), caption="Wordcloud Negatif", use_column_width=True)
else:
    st.warning("Kolom `teks` tidak ada pada data, fitur wordcloud tidak akan muncul.")

# ============ MODELLING ===============
st.header("Model Comparison with SMOTE")
best_rf_params = {'n_estimators': 200, 'max_depth': 10, 'min_samples_split': 2}
best_xgb_params = {'n_estimators': 200, 'max_depth': 3, 'learning_rate': 0.1}

rf_pipeline = Pipeline([
    ('smote', SMOTE(random_state=42)),
    ('clf', RandomForestClassifier(random_state=42, **best_rf_params))
])
xgb_pipeline = Pipeline([
    ('smote', SMOTE(random_state=42)),
    ('clf', XGBClassifier(eval_metric='logloss', use_label_encoder=False, random_state=42, **best_xgb_params))
])

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
rf_pipeline.fit(X_train, y_train)
xgb_pipeline.fit(X_train, y_train)

def metric_scores(y_true, y_pred):
    return {
        'Akurasi': accuracy_score(y_true, y_pred),
        'Presisi': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'F1 Score': f1_score(y_true, y_pred, zero_division=0)
    }

def show_cm(model, X, y, name):
    y_pred = model.predict(X)
    fig, ax = plt.subplots(figsize=(3,3))
    ConfusionMatrixDisplay.from_predictions(y, y_pred, ax=ax, cmap='Blues', colorbar=False)
    ax.set_title(f"Confusion Matrix: {name}")
    st.pyplot(fig)

# Evaluasi di data uji
st.subheader("Evaluasi di Data Test")
col1, col2 = st.columns(2)
with col1:
    st.write("**Random Forest (SMOTE Pipeline)**")
    y_pred_rf = rf_pipeline.predict(X_test)
    st.text(classification_report(y_test, y_pred_rf, zero_division=0))
    show_cm(rf_pipeline, X_test, y_test, "Random Forest + SMOTE")
with col2:
    st.write("**XGBoost (SMOTE Pipeline)**")
    y_pred_xgb = xgb_pipeline.predict(X_test)
    st.text(classification_report(y_test, y_pred_xgb, zero_division=0))
    show_cm(xgb_pipeline, X_test, y_test, "XGBoost + SMOTE")

# 10-Fold CV pada data train
st.subheader("10-Fold Stratified Cross-validation di Train Set")
rf_cv_scores = cross_val_score(rf_pipeline, X_train, y_train, scoring='accuracy', cv=cv, n_jobs=-1)
xgb_cv_scores = cross_val_score(xgb_pipeline, X_train, y_train, scoring='accuracy', cv=cv, n_jobs=-1)
st.write(f"Random Forest mean acc: **{np.mean(rf_cv_scores):.4f}** ± {np.std(rf_cv_scores):.4f}")
st.write(f"XGBoost mean acc: **{np.mean(xgb_cv_scores):.4f}** ± {np.std(xgb_cv_scores):.4f}")

# Perbandingan sebelum/sesudah SMOTE
def simple_model_scores(X_tr, y_tr):
    rf = RandomForestClassifier(random_state=42, **best_rf_params)
    xgb = XGBClassifier(eval_metric='logloss', use_label_encoder=False, random_state=42, **best_xgb_params)
    rf_scores = cross_val_score(rf, X_tr, y_tr, cv=cv, scoring='accuracy', n_jobs=-1)
    xgb_scores = cross_val_score(xgb, X_tr, y_tr, cv=cv, scoring='accuracy', n_jobs=-1)
    return rf_scores, xgb_scores

rf_acc_before, xgb_acc_before = simple_model_scores(X_train, y_train)

st.subheader("Akurasi Random Forest & XGBoost Sebelum/Sesudah SMOTE")
barchar_df = pd.DataFrame({
    "Sebelum SMOTE": [rf_acc_before.mean(), xgb_acc_before.mean()],
    "Sesudah SMOTE": [rf_cv_scores.mean(), xgb_cv_scores.mean()]
}, index=["Random Forest", "XGBoost"])
st.bar_chart(barchar_df)

# Tabel evaluasi komplit (semua metrik, test set)
rf_before = RandomForestClassifier(random_state=42, **best_rf_params)
xgb_before = XGBClassifier(eval_metric='logloss', use_label_encoder=False, random_state=42, **best_xgb_params)
rf_before.fit(X_train, y_train)
xgb_before.fit(X_train, y_train)
scores_df = pd.DataFrame({
    "Akurasi": [
        accuracy_score(y_test, rf_before.predict(X_test)),
        accuracy_score(y_test, rf_pipeline.predict(X_test)),
        accuracy_score(y_test, xgb_before.predict(X_test)),
        accuracy_score(y_test, xgb_pipeline.predict(X_test))
    ],
    "Presisi": [
        precision_score(y_test, rf_before.predict(X_test), zero_division=0),
        precision_score(y_test, rf_pipeline.predict(X_test), zero_division=0),
        precision_score(y_test, xgb_before.predict(X_test), zero_division=0),
        precision_score(y_test, xgb_pipeline.predict(X_test), zero_division=0)
    ],
    "Recall": [
        recall_score(y_test, rf_before.predict(X_test), zero_division=0),
        recall_score(y_test, rf_pipeline.predict(X_test), zero_division=0),
        recall_score(y_test, xgb_before.predict(X_test), zero_division=0),
        recall_score(y_test, xgb_pipeline.predict(X_test), zero_division=0)
    ],
    "F1 Score": [
        f1_score(y_test, rf_before.predict(X_test), zero_division=0),
        f1_score(y_test, rf_pipeline.predict(X_test), zero_division=0),
        f1_score(y_test, xgb_before.predict(X_test), zero_division=0),
        f1_score(y_test, xgb_pipeline.predict(X_test), zero_division=0)
    ]
}, index=[
    "Random Forest Sebelum SMOTE", "Random Forest Sesudah SMOTE",
    "XGBoost Sebelum SMOTE", "XGBoost Sesudah SMOTE"
])
st.dataframe(scores_df.style.highlight_max(axis=0), use_container_width=True)

st.markdown("---")
st.info("Dashboard siap digunakan di berbagai device! \
Install requirements terlebih dahulu via terminal: \
pip install -r requirements.txt, lalu jalankan: \
streamlit run streamlit_dashboard_btc_nlp.py")
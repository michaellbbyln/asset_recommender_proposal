import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =========================
# Konfigurasi
# =========================
DATA_CONTRACT = "Data Kontrak Final.xlsx"
TOP_HISTORICAL = 5
TOP_ASSET = 5

# =========================
# Helpers
# =========================


ALIAS = {
    "strategyone": {"strategy one","strategyone"},
    "avocado": {"b2b","port-f","port f"},
    "port f": {"b2b","port-f","portf"},
    "fifgroup card 2.0": {"fgc","fif card","fifgroupcard"},
    "web fifgroup": {"corporate website","corporate website sprint","website fifgroup"},
    "fims (fiduciary management system)": {"fims","fiduciary management system"},
    "web eform": {"eform"},
    "imove": {"imove","i-move","imove mobile"},
    "action": {"action","actionrest","actionreport","actionmrest"},
    "fmsales": {"fm sales","fmsales","visum","ufi"},
    "microfinancing microapps - fincor": {"microfinancing"},
    "e-Recruitment":{"RISE"},
    "digital leads":{"DBLM"},
    "dealer information system":{"DIS"}
}

def apply_alias(s: str) -> str:
    if not isinstance(s, str):
        return s

    text = s.lower()
    for target, aliases in ALIAS.items():
        for alias in aliases:
            # cek apakah alias ada di teks
            if alias.lower() in text:
                text = text.replace(alias.lower(), target.lower())
    return text

def clean_text(s: str) -> str:
    if pd.isna(s):
        return ""
    s = re.sub(r"[^\w\s]", " ", str(s).lower())
    return re.sub(r"\s+", " ", s).strip()

def load_contract_data(path: str):
    df = pd.read_excel(path)

    # Buang aset dengan tanda kurung ()
    # df = df[~df["ASSET_NAME"].astype(str).str.contains(r"\(", regex=True)]

    # Buang data dengan IS_ACTIVE = 0
    if "IS_ACTIVE" in df.columns:
        df = df[df["IS_ACTIVE"] != 0]

    # Ambil YEAR dari CREATION_DATE
    if "CREATION_DATE" in df.columns:
        df["YEAR"] = pd.to_datetime(df["CREATION_DATE"], errors="coerce").dt.year

    # df["CONTRACT_NAME_CLEAN"] = df["CONTRACT_NAME"].apply(clean_text)
    # df["CONTRACT_NAME_CLEAN"] = df["CONTRACT_NAME"].apply(apply_alias)
    df["CONTRACT_NAME_CLEAN"] = (
    df["CONTRACT_NAME"]
    .apply(clean_text)   # step 1: bersihkan teks
    .apply(apply_alias)  # step 2: mapping alias
)

    return df.reset_index(drop=True)

# =========================
# Load data kontrak
# =========================
contracts = load_contract_data(DATA_CONTRACT)

# Build TF-IDF index
vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(3,5), min_df=1)
M_contracts = vectorizer.fit_transform(contracts["CONTRACT_NAME_CLEAN"])

# =========================
# Recommend Function
# =========================
def recommend_contract(title: str, top_h=TOP_HISTORICAL, top_a=TOP_ASSET):
    # ===== Bersihkan dan mapping alias =====
    query = clean_text(title)
    query = apply_alias(query)  # tambahkan ini
    
    # 🔧 cari aset yg aliasnya cocok
    asset_match = None
    for main_asset, aliases in ALIAS.items():
        if any(alias.lower() in query for alias in aliases) or main_asset in query:
            asset_match = main_asset
            break

    if asset_match:
    # handle NaN dan non-string
        subset = contracts[
        contracts["ASSET_NAME"]
        .astype(str)                    # ubah semua jadi string
        .str.lower()
        .fillna("")                     # isi NaN jadi string kosong
        .str.contains(asset_match.lower(), na=False)  # ignore NaN
    ]
    else:
        subset = contracts.copy()

    
    # lanjut TF-IDF di subset itu aja
    q_vec = vectorizer.transform([query])
    cos = cosine_similarity(q_vec, M_contracts).ravel()
    subset["SIMILARITY"] = cos[:len(subset)]
    
    grouped = (
        subset.groupby("CONTRACT_NAME")
        .agg({
            "ASSET_NAME": lambda x: list(x),
            "YEAR": "max",
            "SIMILARITY": "max"
        })
        .reset_index()
        .sort_values(by="SIMILARITY", ascending=False)
        .head(top_h)
    )

    exploded = grouped.explode("ASSET_NAME")
    asset_scores = (
        exploded.groupby("ASSET_NAME")
        .agg({"SIMILARITY":"max","YEAR":"first"})
        .reset_index()
        .sort_values("SIMILARITY", ascending=False)
    )
    rec_assets = asset_scores.head(top_a)

    return grouped, rec_assets


# =========================
# Streamlit App
# =========================
st.set_page_config(page_title="Contract Mapping Recommender", layout="wide")
st.title("📑 Contract Mapping Recommender (IT P&G)")

title = st.text_input("Masukkan Judul Kontrak")
top_h = st.slider("Jumlah kontrak historis yang ditampilkan", 1, 10, 5)
top_a = st.slider("Jumlah rekomendasi aset", 1, 10, 5)

if st.button("Cari Rekomendasi"):
    if title.strip() == "":
        st.warning("Mohon masukkan judul kontrak.")
    else:
        hist, rec_assets = recommend_contract(title, top_h, top_a)

        st.write("### 🔎 Layer 1: Kontrak Historis Mirip")
        for _, row in hist.iterrows():
            with st.expander(f"{row['CONTRACT_NAME']} (jumlah aset: {len(row['ASSET_NAME'])})"):
                df_assets = pd.DataFrame({
                    "ASSET_NAME": row["ASSET_NAME"],
                })
                df_assets["YEAR"] = row["YEAR"]
                st.dataframe(df_assets[["ASSET_NAME","YEAR"]]) # sama semua
                #df_assets["SIMILARITY"] = row["SIMILARITY"] # sama semua
                # st.dataframe(df_assets)

        st.write("### 🔮 Layer 2: Rekomendasi Aset")
        # st.dataframe(rec_assets)
        st.dataframe(rec_assets[["ASSET_NAME","YEAR"]]) 





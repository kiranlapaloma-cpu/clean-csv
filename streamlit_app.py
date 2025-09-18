# streamlit_app.py
import io
import pandas as pd
import streamlit as st

st.set_page_config(page_title="CSV Factory", page_icon="🧰", layout="wide")

st.title("🧰 CSV Factory — single-file build")
st.caption("Upload → preview → clean → download. No extra files required.")

with st.sidebar:
    st.header("Settings")
    keep_index = st.checkbox("Keep original index in download", value=False)
    strip_whitespace = st.checkbox("Strip leading/trailing whitespace", value=True)
    drop_empty_cols = st.checkbox("Drop completely empty columns", value=True)
    drop_empty_rows = st.checkbox("Drop completely empty rows", value=True)
    dedupe_rows = st.checkbox("Drop duplicate rows", value=True)

st.markdown("#### 1) Upload a file")
uploaded = st.file_uploader(
    "CSV/TSV/TXT or Excel",
    type=["csv", "tsv", "txt", "xlsx", "xls"],
    help="Delimited text (csv/tsv/txt) or Excel files."
)

@st.cache_data(show_spinner=False)
def load_file(file_bytes: bytes, name: str) -> pd.DataFrame:
    name = (name or "").lower()
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(io.BytesIO(file_bytes))

    # Try to auto-detect delimiter for text files; fall back to comma
    try:
        return pd.read_csv(io.BytesIO(file_bytes), sep=None, engine="python")
    except Exception:
        return pd.read_csv(io.BytesIO(file_bytes), sep=",")

def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if strip_whitespace:
        # strip strings
        out = out.applymap(lambda x: x.strip() if isinstance(x, str) else x)
        # strip column names too
        out.columns = [c.strip() if isinstance(c, str) else c for c in out.columns]

    if drop_empty_cols:
        out = out.dropna(axis=1, how="all")
    if drop_empty_rows:
        out = out.dropna(axis=0, how="all")
    if dedupe_rows:
        out = out.drop_duplicates()

    return out

if uploaded is None:
    st.info("👆 Upload a CSV/TSV/TXT or Excel file to begin.")
else:
    # Read raw bytes once so we can re-use
    file_bytes = uploaded.read()

    try:
        df_raw = load_file(file_bytes, uploaded.name)
    except Exception as e:
        st.error(f"Failed to read file: {e}")
        st.stop()

    st.markdown("#### 2) Preview")
    st.write("**Detected shape:**", df_raw.shape)
    st.dataframe(df_raw.head(100), use_container_width=True)

    st.markdown("#### 3) Clean")
    df_clean = clean_df(df_raw)
    st.write("**Cleaned shape:**", df_clean.shape)
    st.dataframe(df_clean.head(100), use_container_width=True)

    st.markdown("#### 4) Download")
    csv_bytes = df_clean.to_csv(index=keep_index).encode("utf-8")
    st.download_button(
        "⬇️ Download Clean CSV",
        data=csv_bytes,
        file_name="clean.csv",
        mime="text/csv"
    )

st.divider()
with st.expander("About / Notes"):
    st.write(
        "This is a minimal, single-file Streamlit app. "
        "Extend it with your domain logic (e.g., sectional metrics, FSP/SPI)."
    )

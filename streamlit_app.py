# streamlit_app.py
import io
import pandas as pd
import streamlit as st

st.set_page_config(page_title="CSV Factory", page_icon="🧰", layout="wide")

st.title("🧰 CSV Factory — upload or paste")
st.caption("Upload → preview → clean → download — or paste a table directly (CSV/TSV/Excel clipboard).")

# ---------------- Sidebar options ----------------
with st.sidebar:
    st.header("Settings")
    keep_index = st.checkbox("Keep original index in download", value=False)
    strip_whitespace = st.checkbox("Strip leading/trailing whitespace", value=True)
    drop_empty_cols = st.checkbox("Drop completely empty columns", value=True)
    drop_empty_rows = st.checkbox("Drop completely empty rows", value=True)
    dedupe_rows = st.checkbox("Drop duplicate rows", value=True)

# ---------------- Helpers ----------------
@st.cache_data(show_spinner=False)
def load_file(file_bytes: bytes, name: str) -> pd.DataFrame:
    """Load CSV/TSV/TXT or Excel from bytes (auto-detect delimiter for text)."""
    name = (name or "").lower()
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(io.BytesIO(file_bytes))
    # Try to detect delimiter automatically; fall back to comma
    try:
        return pd.read_csv(io.BytesIO(file_bytes), sep=None, engine="python")
    except Exception:
        return pd.read_csv(io.BytesIO(file_bytes), sep=",")

@st.cache_data(show_spinner=False)
def parse_pasted(text: str) -> pd.DataFrame:
    """
    Parse pasted table text.
    Handles:
      - CSV (comma-separated)
      - TSV (tab-separated, e.g., Excel/Sheets clipboard)
      - Semicolon-separated (common in EU locales)
      - Auto-detection via engine='python'
    """
    s = text.strip()
    # Fast path: try Python engine with auto-infer
    try:
        return pd.read_csv(io.StringIO(s), sep=None, engine="python")
    except Exception:
        pass

    # Try common delimiters explicitly
    for sep in ["\t", ",", ";", "|"]:
        try:
            df = pd.read_csv(io.StringIO(s), sep=sep)
            # Heuristic: if only one column but separator present, keep trying
            if df.shape[1] == 1 and sep in s:
                continue
            return df
        except Exception:
            continue

    # Final fallback: single-column
    return pd.DataFrame({"text": s.splitlines()})

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

# ---------------- Input section ----------------
st.markdown("#### 1) Input your data")

input_mode = st.radio("Choose input method:", ["Upload file", "Paste table"], horizontal=True)
df_raw = None

if input_mode == "Upload file":
    uploaded = st.file_uploader(
        "CSV/TSV/TXT or Excel",
        type=["csv", "tsv", "txt", "xlsx", "xls"],
        help="Upload a delimited text file or an Excel workbook."
    )
    if uploaded is not None:
        try:
            df_raw = load_file(uploaded.read(), uploaded.name)
        except Exception as e:
            st.error(f"Failed to read file: {e}")

else:
    st.write("Paste directly from Excel/Sheets or any CSV/TSV text:")
    example = "col1\tcol2\tcol3\n1\t2\t3\n4\t5\t6"
    pasted = st.text_area(
        "Paste your table here (we auto-detect commas, tabs, semicolons, etc.)",
        placeholder=example,
        height=220,
    )
    if pasted.strip():
        try:
            df_raw = parse_pasted(pasted)
        except Exception as e:
            st.error(f"Could not parse pasted data: {e}")

# ---------------- Preview / Clean / Download ----------------
if df_raw is None:
    st.info("👆 Upload a file **or** paste a table to continue.")
else:
    st.markdown("#### 2) Preview")
    st.write("**Detected shape:**", df_raw.shape)
    st.dataframe(df_raw.head(200), use_container_width=True)

    st.markdown("#### 3) Clean")
    df_clean = clean_df(df_raw)
    st.write("**Cleaned shape:**", df_clean.shape)
    st.dataframe(df_clean.head(200), use_container_width=True)

    st.markdown("#### 4) Download")
    csv_bytes = df_clean.to_csv(index=keep_index).encode("utf-8")
    st.download_button(
        "⬇️ Download Clean CSV",
        data=csv_bytes,
        file_name="clean.csv",
        mime="text/csv"
    )

st.divider()
with st.expander("Tips & Notes"):
    st.write(
        "- Pasting from Excel typically produces **tab-separated** text — we detect that automatically.\n"
        "- If a column looks 'merged', your data might use a different delimiter; paste again and we’ll auto-detect.\n"
        "- You can extend this app with domain logic (e.g., sectional metrics, FSP/SPI) right below the cleaning step."
    )

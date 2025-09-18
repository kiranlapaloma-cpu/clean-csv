# streamlit_app.py
import io
import re
import pandas as pd
import streamlit as st

st.set_page_config(page_title="CSV Factory — Paste or Upload", page_icon="🧰", layout="wide")
st.title("🧰 CSV Factory — upload or paste")
st.caption("Upload / paste → parse → clean → sort → download")

# ---------------- Sidebar: cleaning toggles ----------------
with st.sidebar:
    st.header("Cleaning")
    keep_index = st.checkbox("Keep index in download", value=False)
    strip_whitespace = st.checkbox("Strip whitespace (cells & headers)", value=True)
    drop_empty_cols = st.checkbox("Drop empty columns", value=True)
    drop_empty_rows = st.checkbox("Drop empty rows", value=True)
    dedupe_rows = st.checkbox("Drop duplicate rows", value=True)
    st.markdown("---")
    st.caption("Tip: If your pasted table looks 'all-in-one-column', set the correct delimiter below.")

# ---------------- Helpers ----------------
def parse_duration_to_seconds(val):
    """
    Parse race-style times: 158.37, 1:58.37, 91:00.09, 12:34:56.78 -> seconds (float)
    Returns None if not parseable.
    """
    if pd.isna(val):
        return None
    s = str(val).strip()
    if not s or s.lower() in {"nr", "nan", "none", "-"}:
        return None
    # plain number -> try float
    if re.fullmatch(r"\d+(\.\d+)?", s):
        try:
            return float(s)
        except Exception:
            return None
    # split on :
    parts = s.split(":")
    try:
        if len(parts) == 2:  # MM:SS(.fff)
            mm = float(parts[0])
            ss = float(parts[1])
            return mm * 60 + ss
        if len(parts) == 3:  # HH:MM:SS(.fff)
            hh = float(parts[0]); mm = float(parts[1]); ss = float(parts[2])
            return hh * 3600 + mm * 60 + ss
    except Exception:
        return None
    return None

def tidy_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if strip_whitespace:
        out = out.applymap(lambda x: x.strip() if isinstance(x, str) else x)
        out.columns = [c.strip() if isinstance(c, str) else c for c in out.columns]
    if drop_empty_cols:
        out = out.dropna(axis=1, how="all")
    if drop_empty_rows:
        out = out.dropna(axis=0, how="all")
    if dedupe_rows:
        out = out.drop_duplicates()
    return out

@st.cache_data(show_spinner=False)
def read_file(file_bytes: bytes, name: str) -> pd.DataFrame:
    name = (name or "").lower()
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(io.BytesIO(file_bytes))
    try:
        return pd.read_csv(io.BytesIO(file_bytes), sep=None, engine="python")
    except Exception:
        return pd.read_csv(io.BytesIO(file_bytes), sep=",")

@st.cache_data(show_spinner=False)
def parse_text(text: str, sep_choice: str, header_row: int | None):
    s = text.strip("\n\r\t ")
    if not s:
        return pd.DataFrame()
    # map UI choice to actual sep
    sep_map = {
        "Auto-detect": None,
        "Comma ,": ",",
        "Tab \\t": "\t",
        "Semicolon ;": ";",
        "Pipe |": "|",
        "Whitespace (any)": r"\s+",
    }
    sep = sep_map[sep_choice]
    # header handling
    header = header_row if header_row is not None else "infer"
    if sep is None:
        # try auto first
        try:
            return pd.read_csv(io.StringIO(s), sep=None, engine="python", header=header)
        except Exception:
            # common fallbacks
            for sep_try in ["\t", ",", ";", "|", r"\s+"]:
                try:
                    return pd.read_csv(io.StringIO(s), sep=sep_try, engine="python", header=header)
                except Exception:
                    continue
            # last resort: one-column
            return pd.read_csv(io.StringIO(s), header=None, names=["text"])
    else:
        return pd.read_csv(io.StringIO(s), sep=sep, engine="python", header=header)

# ---------------- Input ----------------
st.markdown("#### 1) Input your data")
mode = st.radio("Choose input method:", ["Upload file", "Paste table"], horizontal=True)

df_raw = None
if mode == "Upload file":
    up = st.file_uploader("CSV/TSV/TXT or Excel", type=["csv", "tsv", "txt", "xlsx", "xls"])
    if up is not None:
        try:
            df_raw = read_file(up.read(), up.name)
        except Exception as e:
            st.error(f"Read error: {e}")
else:
    cols = st.columns([2, 1, 1])
    with cols[0]:
        txt = st.text_area("Paste table data (Excel/Sheets clipboard or CSV/TSV text)",
                           height=220,
                           placeholder="col1\tcol2\tcol3\n1\t2\t3\n4\t5\t6")
    with cols[1]:
        sep_choice = st.selectbox("Delimiter", ["Auto-detect", "Tab \\t", "Comma ,", "Semicolon ;", "Pipe |", "Whitespace (any)"])
    with cols[2]:
        header_use = st.selectbox("Header row", ["Infer", "Row 1", "No header"])
        header_row = None if header_use == "Infer" else (0 if header_use == "Row 1" else None)
        if header_use == "No header":
            header_row = None
    if txt.strip():
        try:
            df_raw = parse_text(txt, sep_choice, header_row if header_use != "No header" else None)
            if header_use == "No header":
                # assign generic headers
                df_raw.columns = [f"col_{i+1}" for i in range(df_raw.shape[1])]
        except Exception as e:
            st.error(f"Parse error: {e}")

if df_raw is None or df_raw.empty:
    st.info("👆 Upload a file **or** paste a table, then adjust delimiter/header if needed.")
    st.stop()

# ---------------- Preview ----------------
st.markdown("#### 2) Preview")
st.write("**Shape:**", df_raw.shape)
st.dataframe(df_raw.head(250), use_container_width=True)

# ---------------- Clean ----------------
st.markdown("#### 3) Clean")
df_clean = tidy_df(df_raw)
st.write("**After cleaning — shape:**", df_clean.shape)
st.dataframe(df_clean.head(250), use_container_width=True)

# ---------------- Sort & Tidy ----------------
st.markdown("#### 4) Sort & tidy")
if df_clean.shape[1] > 0:
    sort_col = st.selectbox("Column to sort by", list(df_clean.columns))
    treat_as = st.selectbox("Treat selected column as", ["Text", "Number", "Duration (mm:ss or hh:mm:ss)"])
    ascending = st.toggle("Ascending", value=True)

    sort_series = df_clean[sort_col]
    if treat_as == "Number":
        key_series = pd.to_numeric(sort_series.replace({"NR": None, "nr": None, "-": None}), errors="coerce")
    elif treat_as == "Duration (mm:ss or hh:mm:ss)":
        key_series = sort_series.map(parse_duration_to_seconds)
    else:
        key_series = sort_series.astype(str)

    df_sorted = df_clean.copy()
    df_sorted["_sort_key"] = key_series
    df_sorted = df_sorted.sort_values(by="_sort_key", ascending=ascending, kind="mergesort").drop(columns=["_sort_key"])
else:
    df_sorted = df_clean

st.dataframe(df_sorted.head(250), use_container_width=True)

# ---------------- Download ----------------
st.markdown("#### 5) Download")
csv_bytes = df_sorted.to_csv(index=keep_index).encode("utf-8")
st.download_button("⬇️ Download CSV", data=csv_bytes, file_name="clean.csv", mime="text/csv")

st.divider()
with st.expander("Tips"):
    st.write(
        "- If everything lands in one column, set **Delimiter** to the real one (Excel paste is usually **Tab**).\n"
        "- Use **Header row** if your first row contains column names.\n"
        "- For race times like `158.37`, `1:58.37`, or `91:00.09`, choose **Duration** in *Sort & tidy*."
    )

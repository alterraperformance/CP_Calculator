from __future__ import annotations

import pandas as pd
import streamlit as st

@st.cache_data(show_spinner=False)
def load_csv(file) -> pd.DataFrame:
    '''Load a CSV from a Streamlit UploadedFile (or any file-like object).
    Cached so switching pages doesn't keep re-reading the same file.
    '''
    return pd.read_csv(file)

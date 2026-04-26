"""
Kanji Hub — Giao diện Web bằng Streamlit
Chạy local: streamlit run streamlit_app.py
Deploy   : streamlit.io/cloud
"""

import os
import sys
import re
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed

import streamlit as st
import streamlit.components.v1 as _components
import streamlit.components.v1 as _components

# --- Path setup ---
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

try:
    from kanji_lookup import (
        get_kanji_info, search_by_viet, search_by_viet_gemini,
        MANUAL_VI, N4_VI, MNN_N4_EXTRA, MNN_N5, N3_VI, N2_VI, N1_VI,
        get_gemini_key, set_gemini_key, GeminiQuotaError, AIQuotaError,
        lookup_kanji_gemini, lookup_kanji_openrouter,
        get_ai_provider, set_ai_provider,
        get_openrouter_key, set_openrouter_key,
        analyze_kanji_ai,
    )
    from pdf_generator import generate_pdf, generate_vocab_table_pdf
    from vocab_lessons import VOCAB_LESSONS
except Exception as _import_err:
    import traceback
    st.error(f"**Import Error:** {_import_err}")
    st.code(traceback.format_exc())
    st.stop()


# --- CJK helpers ---
def extract_kanji(text):
    seen = set()
    result = []
    for ch in text:
        cp = ord(ch)
        if (0x3000 <= cp <= 0x9FFF or 0xF900 <= cp <= 0xFAFF
                or 0x20000 <= cp <= 0x2A6DF) and ch not in seen:
            seen.add(ch)
            result.append(ch)
    return result


def has_cjk(text):
    for ch in text:
        cp = ord(ch)
        if 0x3000 <= cp <= 0x9FFF or 0xF900 <= cp <= 0xFAFF or 0x20000 <= cp <= 0x2A6DF:
            return True
    return False


# --- Page config ---
st.set_page_config(page_title="Kanji Hub", page_icon="✍️",
                   layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Serif+JP:wght@400;700;900&display=swap');

/* ── Bảng màu Nhật Bản sáng ──
   Washi (giấy kem): #f7f2e8
   Sumi (mực): #1a1209
   Urushi đỏ: #c0392b
   Kincha vàng: #b8902a
   Asagi xanh nhạt: #4a7a8a
*/

/* ── Reset & Base ── */
[data-testid="stAppViewContainer"] {
  background: #f7f2e8;
  background-image:
    repeating-linear-gradient(0deg, transparent, transparent 39px, rgba(184,144,42,.07) 39px, rgba(184,144,42,.07) 40px),
    repeating-linear-gradient(90deg, transparent, transparent 39px, rgba(184,144,42,.07) 39px, rgba(184,144,42,.07) 40px);
}
.main .block-container { max-width: 900px; padding: 1rem 1.5rem 3rem; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
  background: #f0e8d8;
  border-right: 2px solid #c8a45a66;
}
[data-testid="stSidebar"] * { color: #3a2a1a !important; }
[data-testid="stSidebar"] input { background: #fff !important; color: #1a1209 !important; }

/* ── App header / Logo ── */
.app-header {
  text-align: center; padding: 1.8rem 0 1.2rem;
  background: linear-gradient(160deg, #fff8f0 0%, #fdf4e8 50%, #fff8f0 100%);
  border-radius: 6px; margin-bottom: 1.2rem;
  border-top: 4px solid #c0392b;
  border-bottom: 1px solid #c8a45a66;
  box-shadow: 0 2px 12px rgba(192,57,43,.08);
  position: relative; overflow: hidden;
}
.app-header::before {
  content: "山 川 花 月 雪 風 龍 鳥 竹 梅 松 波";
  position: absolute; top: 8px; left: 50%; transform: translateX(-50%);
  font-size: .6rem; color: rgba(184,144,42,.18); letter-spacing: 10px;
  white-space: nowrap; pointer-events: none; font-family: 'Noto Serif JP', serif;
}
.logo-seal {
  display: inline-flex; align-items: center; justify-content: center;
  width: 76px; height: 76px; border-radius: 50%;
  background: radial-gradient(circle at 38% 35%, #c94040, #7a0000);
  border: 3px solid #e05050;
  box-shadow: 0 0 0 5px rgba(192,57,43,.12), 0 6px 20px rgba(139,0,0,.3);
  font-size: 2.2rem; color: #fff8f0; font-weight: 900;
  margin-bottom: 12px; line-height: 1;
  font-family: 'Noto Serif JP', serif;
}
.logo-title {
  font-size: 2rem; font-weight: 900; color: #1a1209;
  margin: 0; letter-spacing: 6px;
  font-family: Georgia, 'Noto Serif JP', serif;
}
.logo-jp { color: #c0392b; font-size: .95rem; letter-spacing: 8px;
  display: block; margin-top: 4px; font-family: 'Noto Serif JP', serif; }
.logo-sub { color: #9a8a70; font-size: .78rem; margin: 8px 0 0; letter-spacing: 2px; }

/* ── Tab radio — pill style ── */
div[data-testid="stRadio"] > label,
div[data-testid="stRadio"] span[data-testid="stWidgetLabel"] { display: none; }
div[data-testid="stRadio"],
div[data-testid="stRadio"] > div,
div[data-testid="stRadio"] > div[role="radiogroup"] {
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
}
div[data-testid="stRadio"] > div[role="radiogroup"] {
  display: flex; flex-direction: row; gap: 8px;
  padding: 6px 0; overflow: visible;
}
/* Base label */
div[data-testid="stRadio"] > div[role="radiogroup"] > label {
  display: flex !important; align-items: center !important; justify-content: center !important;
  flex: 1 !important;
  background: #ffffff !important;
  border: 2.5px solid #e0d4be !important;
  border-radius: 50px !important;
  padding: 10px 18px !important;
  font-weight: 700 !important; cursor: pointer !important;
  transition: all .18s ease !important;
  white-space: nowrap !important; font-size: .88rem !important; letter-spacing: .2px !important;
  box-shadow: 0 2px 8px rgba(0,0,0,.07) !important;
}
/* Tab 1 — Tra Kanji: đỏ */
div[data-testid="stRadio"] > div[role="radiogroup"] > label:nth-child(1),
div[data-testid="stRadio"] > div[role="radiogroup"] > label:nth-child(1) * {
  border-color: #e84040 !important; color: #c0392b !important;
}
/* Tab 2 — Lộ trình: xanh dương */
div[data-testid="stRadio"] > div[role="radiogroup"] > label:nth-child(2),
div[data-testid="stRadio"] > div[role="radiogroup"] > label:nth-child(2) * {
  border-color: #3a7bd5 !important; color: #2c5f9e !important;
}
/* Tab 3 — Từ Vựng: xanh lá */
div[data-testid="stRadio"] > div[role="radiogroup"] > label:nth-child(3),
div[data-testid="stRadio"] > div[role="radiogroup"] > label:nth-child(3) * {
  border-color: #27ae60 !important; color: #1e8449 !important;
}
/* Active: fill solid */
div[data-testid="stRadio"] > div[role="radiogroup"] > label:nth-child(1):has(input:checked) {
  background: #e84040 !important; border-color: #e84040 !important;
  box-shadow: 0 4px 14px rgba(232,64,64,.30) !important; transform: translateY(-1px) !important;
}
div[data-testid="stRadio"] > div[role="radiogroup"] > label:nth-child(1):has(input:checked) * { color: #fff !important; }
div[data-testid="stRadio"] > div[role="radiogroup"] > label:nth-child(2):has(input:checked) {
  background: #3a7bd5 !important; border-color: #3a7bd5 !important;
  box-shadow: 0 4px 14px rgba(58,123,213,.30) !important; transform: translateY(-1px) !important;
}
div[data-testid="stRadio"] > div[role="radiogroup"] > label:nth-child(2):has(input:checked) * { color: #fff !important; }
div[data-testid="stRadio"] > div[role="radiogroup"] > label:nth-child(3):has(input:checked) {
  background: #27ae60 !important; border-color: #27ae60 !important;
  box-shadow: 0 4px 14px rgba(39,174,96,.30) !important; transform: translateY(-1px) !important;
}
div[data-testid="stRadio"] > div[role="radiogroup"] > label:nth-child(3):has(input:checked) * { color: #fff !important; }
/* Hover inactive */
div[data-testid="stRadio"] > div[role="radiogroup"] > label:nth-child(1):hover:not(:has(input:checked)) { background: #fff5f4 !important; transform: translateY(-1px) !important; }
div[data-testid="stRadio"] > div[role="radiogroup"] > label:nth-child(2):hover:not(:has(input:checked)) { background: #f0f5ff !important; transform: translateY(-1px) !important; }
div[data-testid="stRadio"] > div[role="radiogroup"] > label:nth-child(3):hover:not(:has(input:checked)) { background: #f0faf4 !important; transform: translateY(-1px) !important; }
div[data-testid="stRadio"] > div[role="radiogroup"] > label input[type="radio"] { display: none; }
div[data-testid="stRadio"] > div[role="radiogroup"] > label > div:first-child { display: none; }

/* ── Compat ── */
.card-box { display:none; }

/* ── Status badges ── */
.rc-badge { font-size:.65rem; font-weight:700; letter-spacing:.5px; border-radius:3px; padding:2px 8px; white-space:nowrap; }
.tag-db   { background: #e8f5ee; color: #2d6e4a; border: 1px solid #a0d4b8; }
.tag-ai   { background: #f0ecfa; color: #6040a0; border: 1px solid #c0a8e8; }
.tag-jisho{ background: #fdf5e0; color: #8a6010; border: 1px solid #d4a840; }
.tag-miss { background: #fdecea; color: #a02020; border: 1px solid #e0a0a0; }

/* ── Section titles ── */
.sec-title {
  font-size: 1.2rem; font-weight: 800; color: #1a1209;
  border-left: 4px solid #c0392b; padding-left: 12px; margin-bottom: 14px;
  font-family: 'Noto Serif JP', serif; letter-spacing: 1px;
}

/* ── Vocab word card ── */
.vocab-card {
  background: #ffffff; border: 1px solid #e0d4be;
  border-bottom: 2px solid #c0392b55;
  border-radius: 4px; padding: 12px 14px; margin-bottom: 8px;
  box-shadow: 0 1px 4px rgba(0,0,0,.06); transition: box-shadow .2s;
}
.vocab-card:hover { box-shadow: 0 3px 12px rgba(192,57,43,.1); }
.vocab-word   { font-size: 1.4rem; font-weight: 900; color: #1a1209; font-family: 'Noto Serif JP', serif; }
.vocab-kana   { font-size: .84rem; color: #b8902a; }
.vocab-hanviet { font-size: .76rem; color: #9a8a6a; font-style: italic; }
.vocab-meaning { font-size: .92rem; color: #3a2a1a; margin-top: 4px; }
.vocab-example { font-size: .8rem; color: #8a7a6a; font-style: italic; margin-top: 2px; }

/* ── Buttons ── */
/* Primary — đỏ urushi gradient, full pill */
button[data-testid="baseButton-primary"] {
  background: linear-gradient(135deg, #e84040 0%, #b8200e 100%) !important;
  border: none !important;
  color: #fff !important;
  border-radius: 50px !important;
  font-weight: 700 !important;
  font-size: .92rem !important;
  letter-spacing: .4px !important;
  padding: 0.55rem 1.6rem !important;
  box-shadow: 0 4px 14px rgba(192,57,43,.32), 0 1px 3px rgba(0,0,0,.08) !important;
  transition: all .2s cubic-bezier(.4,0,.2,1) !important;
}
button[data-testid="baseButton-primary"]:hover {
  background: linear-gradient(135deg, #ff5555 0%, #c8300e 100%) !important;
  box-shadow: 0 6px 20px rgba(192,57,43,.42), 0 2px 6px rgba(0,0,0,.1) !important;
  transform: translateY(-2px) !important;
}
button[data-testid="baseButton-primary"]:active {
  transform: translateY(0) !important;
  box-shadow: 0 2px 8px rgba(192,57,43,.25) !important;
}

/* Secondary — pill outline vàng nhạt */
button[data-testid="baseButton-secondary"] {
  background: #fff !important;
  border: 2px solid #e8d5a8 !important;
  color: #8a5a20 !important;
  border-radius: 50px !important;
  font-weight: 600 !important;
  font-size: .9rem !important;
  padding: 0.5rem 1.4rem !important;
  box-shadow: 0 2px 8px rgba(0,0,0,.06) !important;
  transition: all .2s cubic-bezier(.4,0,.2,1) !important;
}
button[data-testid="baseButton-secondary"]:hover {
  background: #fff5ea !important;
  border-color: #e84040 !important;
  color: #c0392b !important;
  box-shadow: 0 4px 14px rgba(192,57,43,.14) !important;
  transform: translateY(-1px) !important;
}

/* Download button wrapper — xóa nền đen mặc định */
[data-testid="stDownloadButton"] > div,
[data-testid="stDownloadButton"] > button > div {
  background: transparent !important;
}

/* Download button — pill outline xanh lá nhạt */
a[data-testid="stDownloadButton-downloadButton"],
button[data-testid="stDownloadButton-downloadButton"] {
  background: #fff !important;
  border: 2px solid #b8d4b0 !important;
  color: #3a6a30 !important;
  border-radius: 50px !important;
  font-weight: 600 !important;
  padding: 0.5rem 1.4rem !important;
  box-shadow: 0 2px 8px rgba(0,0,0,.06) !important;
  transition: all .2s cubic-bezier(.4,0,.2,1) !important;
}
a[data-testid="stDownloadButton-downloadButton"]:hover,
button[data-testid="stDownloadButton-downloadButton"]:hover {
  background: #f0faf0 !important;
  border-color: #4a8a40 !important;
  color: #2a5a20 !important;
  box-shadow: 0 4px 14px rgba(74,138,64,.18) !important;
  transform: translateY(-1px) !important;
}

/* ── Input, selectbox ── */
[data-testid="stTextInput"] > div,
[data-testid="stTextInput"] div[data-baseweb="input"],
[data-testid="stTextInput"] div[data-baseweb="base-input"] {
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
  padding: 0 !important;
  outline: none !important;
}
[data-testid="stTextInput"] input {
  background: #ffffff !important;
  border: 2px solid #ddd0be !important;
  border-radius: 14px !important;
  color: #3a2a1a !important;
  padding: 0.65rem 1.2rem !important;
  font-size: 1rem !important;
  box-shadow: 0 2px 10px rgba(0,0,0,.06) !important;
  transition: border-color .18s, box-shadow .18s !important;
  outline: none !important;
}
[data-testid="stTextInput"] input:focus,
[data-testid="stTextInput"] input:focus-visible {
  border-color: #e84040 !important;
  box-shadow: 0 0 0 3px rgba(232,64,64,.15), 0 2px 10px rgba(0,0,0,.06) !important;
  outline: none !important;
}
[data-testid="stTextInput"] div[data-baseweb="input"]:focus-within,
[data-testid="stTextInput"] div[data-baseweb="base-input"]:focus-within {
  border: none !important;
  box-shadow: none !important;
  outline: none !important;
}
[data-testid="stTextInput"] input::placeholder { color: #b8a898 !important; }

/* Selectbox control (baseweb) */
div[data-baseweb="select"] > div:first-child {
  background: #ffffff !important;
  border: 1.5px solid #e0d0be !important;
  border-radius: 50px !important;
  color: #3a2a1a !important;
}
div[data-baseweb="select"],
div[data-baseweb="select"] *,
div[data-baseweb="select"] input,
div[data-baseweb="select"] [data-testid="stSelectboxValue"],
div[data-baseweb="select"] span { color: #3a2a1a !important; background: transparent !important; }

/* Dropdown popup list */
div[data-baseweb="popover"] div[data-baseweb="menu"],
div[data-baseweb="popover"] ul[data-testid="stSelectboxVirtualDropdown"] {
  background: #fff !important;
  border: 1px solid #e0d0be !important;
  border-radius: 12px !important;
  box-shadow: 0 8px 24px rgba(0,0,0,.12) !important;
}
div[data-baseweb="option"] {
  background: #fff !important; color: #3a2a1a !important;
}
div[data-baseweb="option"]:hover,
div[data-baseweb="option"][aria-selected="true"] {
  background: #fff5f0 !important; color: #c0392b !important;
}

/* Sidebar selectbox */
[data-testid="stSidebar"] div[data-baseweb="select"] > div:first-child {
  background: #fff !important; border: 1.5px solid #d8c8a8 !important;
  border-radius: 50px !important; color: #3a2a1a !important;
}

/* ── Expander ── */
[data-testid="stExpander"] {
  background: #fdf8f0 !important; border: 1px solid #e0d4be !important;
  border-radius: 4px !important;
}
[data-testid="stExpander"] summary { color: #5a4a30 !important; font-size: .88rem; }

/* ── Divider ── */
hr { border-color: #d8c8a8 !important; }

/* ── Metric ── */
[data-testid="stMetric"] {
  background: #ffffff; border: 1px solid #e0d4be;
  border-radius: 4px; padding: 8px 12px !important;
  box-shadow: 0 1px 4px rgba(0,0,0,.05);
}
[data-testid="stMetricValue"] { color: #1a1209 !important; }
[data-testid="stMetricLabel"] { color: #9a8a6a !important; }

/* ── Progress bar ── */
[data-testid="stProgressBar"] > div { background: #c0392b !important; }
[data-testid="stProgressBar"] { background: #e8dcc8 !important; }

/* ── Spinner — nổi bật trên nền sáng ── */
[data-testid="stSpinner"] {
  background: rgba(247,242,232,.92) !important;
  border: 1.5px solid #c0392b33 !important;
  border-radius: 8px !important;
  padding: 12px 20px !important;
  box-shadow: 0 4px 20px rgba(0,0,0,.12) !important;
}
[data-testid="stSpinner"] p {
  color: #c0392b !important;
  font-weight: 700 !important;
  font-size: 1rem !important;
}
[data-testid="stSpinner"] svg {
  color: #c0392b !important;
  stroke: #c0392b !important;
}

/* ── Prog label ── */
.prog-label { font-size: .8rem; color: #9a8a6a; text-align: right; margin-top: -6px; }
.kanji-char { font-size: 3.4rem; font-weight: 900; color: #1a1209;
  text-align: center; line-height: 1.05; font-family: 'Noto Serif JP', serif; }
.kanji-read { font-size: .75rem; color: #b8902a; text-align: center; letter-spacing: 1px; }
.kanji-viet { font-size: 1.1rem; font-weight: 800; color: #1a1209; }
.kanji-mean { color: #5a4a3a; font-size: .9rem; margin-top: 2px; }
.kanji-meo  { color: #3a6a3a; font-style: italic; font-size: .82rem; margin-top: 4px;
  border-left: 2px solid #8ab488; padding-left: 6px; }
.vocab-item { color: #5a4a3a; font-size: .86rem; }

/* ── Hide Streamlit chrome ── */
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }

/* ── Mobile Responsive (≤ 768px) ── */
@media (max-width: 768px) {
  .main .block-container { padding: 0.4rem 0.5rem 2rem !important; max-width: 100% !important; }
  .app-header { padding: 0.8rem 0 0.6rem; margin-bottom: 0.5rem; }
  .logo-seal  { width: 58px; height: 58px; font-size: 1.7rem; margin-bottom: 8px; }
  .logo-title { font-size: 1.5rem; letter-spacing: 4px; }
  .kanji-char { font-size: 2.6rem; }
  div[data-testid="stRadio"] > div[role="radiogroup"] {
    overflow-x: auto !important; flex-wrap: nowrap !important;
    -webkit-overflow-scrolling: touch; scrollbar-width: none;
  }
  div[data-testid="stRadio"] > div[role="radiogroup"]::-webkit-scrollbar { display: none; }
  div[data-testid="stRadio"] > div[role="radiogroup"] > label { padding: 10px 12px !important; font-size: .82rem; }
  button[data-testid="baseButton-primary"],
  button[data-testid="baseButton-secondary"] { min-height: 44px !important; font-size: .92rem !important; border-radius: 50px !important; }
  [data-testid="stMetricValue"] { font-size: 1.1rem !important; }
  [data-testid="stMetricLabel"] { font-size: .7rem !important; }
  .sec-title  { font-size: 1rem; }
  input[type="text"], textarea { font-size: 16px !important; }
  [data-testid="collapsedControl"] { top: 0.4rem !important; }
}
</style>
""", unsafe_allow_html=True)

ALL_DB = {**MANUAL_VI, **N4_VI, **MNN_N4_EXTRA, **MNN_N5, **N3_VI, **N2_VI, **N1_VI}

for _k, _v in {"results": [], "path_results": []}.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


def status_of(info):
    k = info.get("kanji", "")
    if k in ALL_DB:
        return "✓ Có trong DB", "tag-db"
    src = info.get("source", "")
    if src == "gemini":
        return "✨ Gemini AI", "tag-ai"
    if src == "openrouter":
        return "🚀 OpenRouter", "tag-ai"
    if src == "mazii":
        return "🇻🇳 Mazii", "tag-db"
    if info.get("meaning_vi") and info["meaning_vi"] != k:
        return "⚡ Jisho API", "tag-jisho"
    return "✗ Không tìm thấy", "tag-miss"


def gif_url(kanji):
    if not kanji:
        return ""
    h = hex(ord(kanji[0]))[2:].lower()
    return f"https://raw.githubusercontent.com/mistval/kanji_images/master/gifs/{h}.gif"


def make_pdf_bytes(infos, extra_rows=0):
    is_vocab = all(i.get("source") == "json_import" for i in infos)
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        path = f.name
    try:
        if is_vocab:
            generate_vocab_table_pdf(infos, path)
        else:
            generate_pdf(infos, path, extra_rows=extra_rows)
        with open(path, "rb") as f:
            return f.read()
    finally:
        try:
            os.unlink(path)
        except Exception:
            pass


# idx + prefix -> key stable va unique giua cac lan rerun
def render_card(info, idx, prefix):
    import html as _html
    uid     = f"{prefix}_{idx}"
    kanji   = info.get("kanji", "")
    viet    = info.get("viet", "")
    reading = info.get("reading", "")
    meaning = info.get("meaning_vi", "")
    meo     = info.get("meo", "")
    vocab   = info.get("vocab", [])
    meanings_en = info.get("meanings_en", [])
    status_txt, status_cls = status_of(info)

    gif_src = gif_url(kanji) if kanji else ""
    gif_html = (
        f'<img src="{gif_src}" width="100" height="100" title="Thứ tự nét viết"'
        f' style="border-radius:6px;background:#fff;padding:3px;display:block;margin:8px auto 0;box-shadow:0 2px 8px rgba(0,0,0,.3)"'
        f' onerror="this.style.display=\'none\'">'
        if gif_src else ""
    )

    # ── Ý nghĩa section ──
    y_nghia = _html.escape(viet) if viet else ""
    meaning_sub = _html.escape(meaning) if meaning else ""

    # ── Mẹo nhớ section ──
    meo_box = ""
    if meo:
        meo_box = (
            f'<div style="background:#fffbf0;border:1px solid #d4bc8a;border-radius:6px;'
            f'padding:8px 12px;margin-top:8px">'
            f'<div style="font-size:.68rem;font-weight:700;color:#b8902a;letter-spacing:1.5px;margin-bottom:4px">✦ GỢI Ý CÁCH NHỚ</div>'
            f'<div style="color:#3a2a1a;font-size:.88rem;line-height:1.6">{_html.escape(meo)}</div>'
            f'</div>'
        )

    # ── Từ vựng section ──
    vocab_box = ""
    items_src = vocab[:5] if vocab else []
    if items_src:
        rows = ""
        for v in items_src:
            w = _html.escape(str(v[0]))
            r = _html.escape(str(v[1]))
            m = _html.escape(str(v[2]))
            rows += (
                f'<div style="padding:6px 0;border-bottom:1px solid #e0d0b0;display:flex;align-items:baseline;gap:6px">'
                f'<span style="color:#1a3060;font-weight:700;font-size:.95rem">{w}</span>'
                f'<span style="color:#b8902a;font-size:.82rem">（{r}）</span>'
                f'<span style="color:#6a5a4a;font-size:.82rem">— {m}</span>'
                f'</div>'
            )
        vocab_box = (
            f'<div style="margin-top:10px">'
            f'<div style="font-size:.68rem;font-weight:700;color:#c0392b;letter-spacing:1.5px;margin-bottom:4px">'
            f'📚 TỪ VỰNG ({len(items_src)} từ)</div>'
            f'{rows}'
            f'</div>'
        )
    elif meanings_en:
        m_en = _html.escape(" · ".join(meanings_en[:4]))
        vocab_box = (
            f'<div style="margin-top:8px;font-size:.82rem;color:#8a7a6a">{m_en}</div>'
        )

    # ── Dùng Streamlit columns để TTS nằm trong cột trái ──
    st.markdown(
        '<div style="background:#ffffff;border:1px solid #e0d4be;'
        'border-radius:8px;margin-bottom:10px;'
        'box-shadow:0 2px 8px rgba(0,0,0,.08)">',
        unsafe_allow_html=True)

    col_l, col_r = st.columns([1, 3])

    with col_l:
        st.markdown(
            f'<div style="background:#fdf6e8;min-height:100%;padding:16px 10px 8px;'
            f'text-align:center;border-right:1px solid #e0d0b0">'
            f'<div style="font-size:3.4rem;font-weight:900;color:#1a1209;line-height:1;'
            f'font-family:serif">{_html.escape(kanji)}</div>'
            f'<div style="font-size:.75rem;color:#b8902a;letter-spacing:2px;margin-top:6px;'
            f'background:#fff8ee;border:1px solid #d4bc8a;border-radius:20px;'
            f'padding:3px 10px;display:inline-block">{_html.escape(reading) or "—"}</div>'
            f'{gif_html}'
            f'</div>',
            unsafe_allow_html=True)
        speak_text = kanji if kanji else reading
        if speak_text:
            safe = speak_text.replace("\\", "\\\\").replace("'", "\\'").replace('"', '\\"')
            _components.html(f"""
<style>html,body{{margin:0;padding:0;overflow:hidden}}*{{box-sizing:border-box}}</style>
<button onclick="speak()" style="
  background:#fff8ee;border:1px solid #d4bc8a;border-radius:0;
  color:#8a6010;font-size:.88rem;cursor:pointer;padding:8px 0;
  width:100%;font-family:sans-serif;display:block;letter-spacing:.5px;
  position:absolute;bottom:0;left:0;right:0;
" onmouseover="this.style.background='#fdf0d8'"
  onmouseout="this.style.background='#fff8ee'">🔊 Nghe</button>
<script>
function speak(){{
  try{{
    var s=window.parent.speechSynthesis||window.speechSynthesis;
    var U=window.parent.SpeechSynthesisUtterance||window.SpeechSynthesisUtterance;
    s.cancel(); var u=new U('{safe}'); u.lang='ja-JP'; u.rate=0.85; s.speak(u);
  }}catch(e){{}}
}}
</script>
""", height=40, scrolling=False)

    with col_r:
        badge_html = f'<span class="rc-badge {status_cls}" style="margin-left:8px">{_html.escape(status_txt)}</span>'
        y_nghia_block = ""
        if y_nghia or meaning_sub:
            y_nghia_block = (
                f'<div style="margin-bottom:4px">'
                f'<div style="font-size:.65rem;font-weight:700;color:#9a8a6a;letter-spacing:2px;margin-bottom:3px">Ý NGHĨA</div>'
                f'<div style="font-size:1.4rem;font-weight:900;color:#1a1209;font-family:serif">'
                f'{y_nghia}{badge_html}</div>'
                + (f'<div style="font-size:.88rem;color:#5a4a3a;margin-top:2px">{meaning_sub}</div>' if meaning_sub and meaning_sub != y_nghia else "")
                + f'</div>'
            )
        st.markdown(
            f'<div style="padding:14px 16px 10px">'
            f'{y_nghia_block}'
            f'{meo_box}'
            f'{vocab_box}'
            f'</div>',
            unsafe_allow_html=True)
        if viet or meaning:
            res_key     = f"ai_res_{uid}"
            loading_key = f"ai_loading_{uid}"

            # Nút bấm — click lần 1: set loading flag + rerun
            if st.button("🤖 Phân tích AI", key=f"analyze_{uid}", use_container_width=False):
                st.session_state[loading_key] = True
                st.rerun()

            # Lần rerun sau khi click: hiển thị spinner + gọi API
            if st.session_state.get(loading_key):
                with st.spinner("🔄 Đang phân tích AI, vui lòng chờ…"):
                    result = analyze_kanji_ai(kanji)
                st.session_state[res_key] = result
                del st.session_state[loading_key]
                st.rerun()

            if res_key in st.session_state:
                st.markdown(
                    f'<div style="background:#fdf8f0;border:1px solid #e0d4be;'
                    f'border-radius:4px;padding:10px 12px;font-size:.85rem;color:#3a2a1a;margin-top:4px">'
                    f'{st.session_state[res_key]}</div>',
                    unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


def do_lookup(query, search_mode):
    if not query.strip():
        return []
    provider = get_ai_provider()
    use_ai   = "AI" in search_mode

    if has_cjk(query):
        is_csv = any(sep in query for sep in (",", "、", "，"))
        if is_csv:
            seen = set()
            kanji_list = []
            for part in re.split(r"[,、，]", query):
                word = "".join(ch for ch in part.strip() if has_cjk(ch))
                if word and word not in seen:
                    seen.add(word)
                    kanji_list.append(word)
        else:
            kanji_list = extract_kanji(query)

        if not kanji_list:
            return []

        if use_ai and "+" not in search_mode:
            results = []
            for k in kanji_list:
                try:
                    g = None
                    if provider == "gemini":
                        g = lookup_kanji_gemini(k)
                        if not g and get_openrouter_key():
                            g = lookup_kanji_openrouter(k)
                            if g:
                                g["source"] = "openrouter"
                    else:
                        g = lookup_kanji_openrouter(k)
                    info = {"kanji": k, "meanings_en": [], **(g or {})} if g else get_kanji_info(k)
                    results.append(info)
                except AIQuotaError as e:
                    st.error(str(e))
                    return results
            return results

        infos = [None] * len(kanji_list)
        with ThreadPoolExecutor(max_workers=8) as ex:
            fmap = {ex.submit(get_kanji_info, k): i for i, k in enumerate(kanji_list)}
            for fut in as_completed(fmap):
                i = fmap[fut]
                k = kanji_list[i]
                try:
                    info = fut.result()
                    if use_ai and info.get("source") == "jisho":
                        try:
                            g = None
                            if provider == "gemini":
                                g = lookup_kanji_gemini(k)
                                if not g and get_openrouter_key():
                                    g = lookup_kanji_openrouter(k)
                                    if g:
                                        g["source"] = "openrouter"
                            else:
                                g = lookup_kanji_openrouter(k)
                            if g:
                                info = {"kanji": k, "meanings_en": [], **g}
                        except AIQuotaError:
                            pass
                    infos[i] = info
                except Exception:
                    infos[i] = {"kanji": k, "vocab": []}
        return [x for x in infos if x is not None]

    infos = search_by_viet(query)
    if not infos and use_ai:
        try:
            if provider == "gemini":
                infos = search_by_viet_gemini(query)
                if not infos and get_openrouter_key():
                    from kanji_lookup import search_by_viet_openrouter
                    infos = search_by_viet_openrouter(query)
            else:
                from kanji_lookup import search_by_viet_openrouter
                infos = search_by_viet_openrouter(query)
        except Exception as e:
            st.error(str(e))
    return infos or []


# --- Nạp key từ Streamlit Secrets (ưu tiên) ---
def _load_secrets():
    """Đọc key từ st.secrets nếu có, set vào config để các hàm lookup dùng được."""
    try:
        gk = st.secrets.get("GEMINI_API_KEY", "")
        ok = st.secrets.get("OPENROUTER_API_KEY", "")
        pv = st.secrets.get("AI_PROVIDER", "")
        if gk and not get_gemini_key():
            set_gemini_key(gk)
        if ok and not get_openrouter_key():
            set_openrouter_key(ok)
        if pv and get_ai_provider() == "gemini":
            set_ai_provider(pv)
    except Exception:
        pass

_load_secrets()

# --- Sidebar ---
with st.sidebar:
    st.markdown("## ✍️ Kanji Hub")
    st.caption("Ứng dụng học Kanji cho người Việt")
    st.divider()
    st.markdown("### ⚙️ Cài đặt AI")

    # Kiểm tra xem secrets có sẵn không
    _has_secret_gem = bool(st.secrets.get("GEMINI_API_KEY", "")) if hasattr(st, "secrets") else False
    _has_secret_or  = bool(st.secrets.get("OPENROUTER_API_KEY", "")) if hasattr(st, "secrets") else False

    prov_options = ["gemini", "openrouter"]
    cur_prov = get_ai_provider()
    new_prov = st.selectbox("Nhà cung cấp AI", prov_options,
                            index=prov_options.index(cur_prov), key="sb_prov")
    if new_prov != cur_prov:
        set_ai_provider(new_prov)
        st.rerun()

    if _has_secret_gem:
        st.success("✓ Gemini key đã được cấu hình sẵn")
    else:
        gem_key = st.text_input("🔑 Gemini API Key", value=get_gemini_key(),
                                type="password", placeholder="AIzaSy…", key="sb_gkey")

    if _has_secret_or:
        st.success("✓ OpenRouter key đã được cấu hình sẵn")
    else:
        or_key = st.text_input("🔑 OpenRouter API Key", value=get_openrouter_key(),
                               type="password", placeholder="sk-or-…", key="sb_okey")

    if not _has_secret_gem or not _has_secret_or:
        if st.button("💾 Lưu cài đặt", use_container_width=True, key="sb_save", type="primary"):
            if not _has_secret_gem:
                set_gemini_key(gem_key)
            if not _has_secret_or:
                set_openrouter_key(or_key)
            st.success("✓ Đã lưu!")

    st.divider()
    st.markdown("""
**📌 Cách nhập:**
- Kanji: 📌  *山川田*
- Tiếng Việt: 📌  *Học, Tình*
- Danh sách: 📌  *会社,学校*

**🎯 Chế độ tra:**
- **DB** — nhanh, offline
- **DB + AI** — kết hợp
- **AI** — chỉ dùng AI
""")
    st.caption("Kanji Hub v2 • Streamlit Cloud")


# ── CSS cho custom tab radio ──────────────────────────────────────────────────
st.markdown("""
<style>
/* Ẩn label chính của radio widget (tiêu đề "tab") */
div[data-testid="stRadio"] > label,
div[data-testid="stRadio"] span[data-testid="stWidgetLabel"] { display: none; }
/* Nút radio nằm ngang, bỏ circle, styled như tab */
div[data-testid="stRadio"] > div[role="radiogroup"] {
  display: flex; flex-direction: row; gap: 4px;
  background: #181825; border-radius: 12px; padding: 4px 6px;
}
div[data-testid="stRadio"] > div[role="radiogroup"] > label {
  display: flex; align-items: center;
  background: transparent; border-radius: 8px; padding: 6px 18px;
  color: #6c7086 !important; font-weight: 600; cursor: pointer;
  transition: all .15s; white-space: nowrap;
}
div[data-testid="stRadio"] > div[role="radiogroup"] > label:has(input:checked) {
  background: #313244 !important; color: #cdd6f4 !important;
}
div[data-testid="stRadio"] > div[role="radiogroup"] > label:hover {
  color: #cdd6f4 !important;
}
/* Ẩn circle input, giữ text */
div[data-testid="stRadio"] > div[role="radiogroup"] > label input[type="radio"] {
  display: none;
}
div[data-testid="stRadio"] > div[role="radiogroup"] > label > div:first-child {
  display: none;
}
</style>
""", unsafe_allow_html=True)

# ── Custom tabs bằng radio (có thể control bằng session_state) ───────────────
TAB_NAMES = ["🔍 Tra Kanji", "🗺️ Lộ trình học", "📖 Từ Vựng"]
if "tab_radio" not in st.session_state:
    st.session_state["tab_radio"] = TAB_NAMES[0]
# Áp dụng pending tab switch TRƯỚC khi widget render
if "pending_tab" in st.session_state:
    st.session_state["tab_radio"] = st.session_state.pop("pending_tab")

active_tab = st.radio("tab", TAB_NAMES, horizontal=True,
                      key="tab_radio", label_visibility="collapsed")
st.divider()

# === TAB 1 ===
if active_tab == TAB_NAMES[0]:
    st.markdown('''
<div class="app-header">
  <div class="logo-seal">漢</div>
  <div class="logo-title">KANJI HUB</div>
  <span class="logo-jp">漢字 · 学習</span>
  <p class="logo-sub">Tra cứu Kanji Nhật &middot; Nghĩa tiếng Việt &middot; Luyện viết</p>
</div>
''', unsafe_allow_html=True)

    with st.form("search_form"):
        query = st.text_input("q", label_visibility="collapsed",
                              placeholder="Nhập kanji: 山川田… hoặc tiếng Việt: Học…")
        c2, c3 = st.columns([3, 1])
        with c2:
            search_mode = st.selectbox("m", ["DB", "DB + AI", "AI"],
                                       index=1, label_visibility="collapsed")
        with c3:
            submitted = st.form_submit_button("🔍 Tra", use_container_width=True, type="primary")

    if submitted and query:
        with st.spinner("⏳ Đang tra cứu…"):
            st.session_state.results = do_lookup(query, search_mode)

    results = st.session_state.results

    if results:
        valid = [i for i in results if i.get("viet") or i.get("meaning_vi")]
        errs  = [i for i in results if status_of(i)[0].startswith("✗")]

        # Stats row
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Tổng", len(results))
        col_b.metric("Tìm thấy", len(results) - len(errs))
        col_c.metric("Không có", len(errs))

        if valid:
            rows_map = {"1 hàng (6 ô)": 0, "2 hàng (18 ô)": 1,
                        "3 hàng (30 ô)": 2, "4 hàng (42 ô)": 3}
            sel = st.selectbox("Số hàng luyện viết", list(rows_map.keys()), key="sr_rows")
            try:
                pdf_b = make_pdf_bytes(valid, rows_map[sel])
                safe  = "".join(extract_kanji("".join(i["kanji"] for i in valid))[:10]) or "Kanji"
                st.download_button("📄 Tải PDF luyện viết", data=pdf_b,
                                   file_name=f"Kanji_{safe}.pdf",
                                   mime="application/pdf", key="dl_search",
                                   use_container_width=True)
            except Exception as e:
                st.error(f"Lỗi PDF: {e}")

        st.divider()
        for idx, info in enumerate(results):
            render_card(info, idx=idx, prefix="s")
    elif submitted and query:
        st.warning("Không tìm thấy. Thử gõ có dấu hoặc chuyển sang chế độ AI.")

# === TAB 2 ===
elif active_tab == TAB_NAMES[1]:
    st.markdown('<div class="sec-title">🗺️ Lộ trình học Kanji</div>', unsafe_allow_html=True)
    level_data = {
        "N5": list(MNN_N5.keys()), "N4": list(N4_VI.keys()),
        "N3": list(N3_VI.keys()),  "N2": list(N2_VI.keys()),
        "N1": list(N1_VI.keys()),
    }

    lv_cols = st.columns(len(level_data))
    for ci, lv in enumerate(level_data):
        lv_cols[ci].button(lv, key=f"lvbtn_{lv}", use_container_width=True,
                           type="primary" if st.session_state.get("pl_sel_val", "N5") == lv else "secondary",
                           on_click=lambda l=lv: st.session_state.update({"pl_sel_val": l}))

    level = st.session_state.get("pl_sel_val", "N5")

    chunks   = [level_data[level][i:i+10] for i in range(0, len(level_data[level]), 10)]
    prog_key = f"prog_{level}"
    if prog_key not in st.session_state:
        st.session_state[prog_key] = set()
    done_n = len(st.session_state[prog_key])

    # Thống kê + progress
    pct = done_n / len(chunks) if chunks else 0
    st.markdown(f"""
<div style="display:flex;align-items:center;gap:16px;margin:8px 0 4px">
  <span style="color:#3a2a1a;font-size:.95rem">
    <b>{level}</b> · {len(level_data[level])} chữ · {len(chunks)} bài
  </span>
  <span style="color:#2d6e4a;font-size:.85rem">✅ {done_n}/{len(chunks)} hoàn thành</span>
</div>""", unsafe_allow_html=True)
    st.progress(pct)
    for i, chunk in enumerate(chunks):
        lid     = f"{level}_{i+1}"
        res_key = f"path_res_{lid}"
        is_done = lid in st.session_state[prog_key]
        has_res = bool(st.session_state.get(res_key))
        icon    = "✅" if is_done else "📖"
        with st.expander(f"{icon} Bài {i+1} ({len(chunk)} chữ) — {' '.join(chunk)}",
                         expanded=has_res):
            st.markdown(f'<div style="font-size:1.6rem;letter-spacing:3px;color:#1a3060;margin-bottom:10px">{"  ".join(chunk)}</div>',
                        unsafe_allow_html=True)
            b1, b2 = st.columns([1, 1])
            with b1:
                if st.button(f"🔍 Tra bài {i+1}", key=f"tra_{level}_{i}",
                             type="primary", use_container_width=True):
                    st.session_state[prog_key].add(lid)
                    # Xóa kết quả AI cũ từ lần tra trước
                    for k in list(st.session_state.keys()):
                        if k.startswith("ai_res_") or k.startswith("ai_loading_"):
                            del st.session_state[k]
                    with st.spinner("Đang tra…"):
                        res = do_lookup("".join(chunk), "DB + AI")
                    st.session_state[res_key] = res
                    st.session_state.results   = res
                    st.session_state["pending_tab"] = TAB_NAMES[0]  # nhảy về Tra Kanji
                    st.rerun()
            with b2:
                st.markdown('<div style="margin-top:6px">', unsafe_allow_html=True)
                chk = st.checkbox("Đã học xong", value=is_done, key=f"chk_{level}_{i}")
                st.markdown('</div>', unsafe_allow_html=True)
                if chk and not is_done:
                    st.session_state[prog_key].add(lid)
                elif not chk and is_done:
                    st.session_state[prog_key].discard(lid)

            # Hiện kết quả ngay trong expander
            bai_results = st.session_state.get(res_key, [])
            if bai_results:
                st.divider()
                valid_p = [r for r in bai_results if r.get("viet") or r.get("meaning_vi")]
                if valid_p:
                    try:
                        safe_p = "".join(r["kanji"] for r in valid_p[:10])
                        st.download_button("📄 Tải PDF bài này",
                                           data=make_pdf_bytes(valid_p),
                                           file_name=f"Kanji_{safe_p}.pdf",
                                           mime="application/pdf",
                                           key=f"dl_path_{lid}")
                    except Exception:
                        pass
                for idx, info in enumerate(bai_results):
                    render_card(info, idx=idx, prefix=f"p_{lid}")

# === TAB 3 ===
elif active_tab == TAB_NAMES[2]:
    st.markdown('<div class="sec-title">📖 Từ Vựng theo Bài</div>', unsafe_allow_html=True)
    if not VOCAB_LESSONS:
        st.info("Chưa có bài từ vựng nào.")
    else:
        lesson_nums = sorted(VOCAB_LESSONS.keys())
        sel_lesson  = st.selectbox("Chọn bài", lesson_nums,
                                   format_func=lambda n: f"Bài {n}  ({len(VOCAB_LESSONS[n])} từ)",
                                   key="vl_sel")
        words = VOCAB_LESSONS[sel_lesson]
        st.info(f"**Bài {sel_lesson}** — {len(words)} từ vựng")
        # Grid 2 cột
        for row_i in range(0, len(words), 2):
            cols = st.columns(2)
            for ci in range(2):
                wi = row_i + ci
                if wi >= len(words):
                    break
                item = words[wi]
                with cols[ci]:
                    st.markdown(f"""
<div class="vocab-card">
  <div class="vocab-word">{item['word']}</div>
  <div class="vocab-kana">（{item.get('reading', '')}）
    <span class="vocab-hanviet">{item.get('hanviet', '')}</span>
  </div>
  <div class="vocab-meaning">▸ {item.get('meaning', '')}</div>
  {"<div class='vocab-example'>📝 " + item['example'] + "</div>" if item.get('example') else ""}
  {"<div class='vocab-example'>↳ " + item['exampleVi'] + "</div>" if item.get('exampleVi') else ""}
</div>""", unsafe_allow_html=True)
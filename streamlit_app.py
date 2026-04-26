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
/* ── Reset & Base ── */
[data-testid="stAppViewContainer"] { background: #11111b; }
[data-testid="stSidebar"] { background: #181825; border-right: 1px solid #313244; }
[data-testid="stSidebar"] * { color: #cdd6f4 !important; }

/* ── App header ── */
.app-header {
  text-align: center; padding: 1.2rem 0 0.4rem;
  background: linear-gradient(135deg, #1e1e2e 0%, #181825 100%);
  border-radius: 16px; margin-bottom: 1rem;
}
.app-header h1 { font-size: 2.4rem; font-weight: 900; color: #cdd6f4; margin: 0; letter-spacing: 2px; }
.app-header p  { color: #6c7086; font-size: .9rem; margin: 4px 0 0; }

/* ── Kanji card ── */
.card-box {
  background: #1e1e2e;
  border: 1px solid #313244;
  border-radius: 14px;
  padding: 16px 18px;
  margin-bottom: 12px;
  transition: border-color .2s, box-shadow .2s;
}
.card-box:hover { border-color: #585b70; box-shadow: 0 4px 20px rgba(0,0,0,.35); }

/* ── Kanji glyph ── */
.kanji-char {
  font-size: 3.6rem; font-weight: 900; color: #cdd6f4;
  text-align: center; line-height: 1.05; margin-bottom: 2px;
  text-shadow: 0 2px 8px rgba(137,180,250,.25);
}
.kanji-read { font-size: .78rem; color: #89b4fa; text-align: center; letter-spacing: .5px; }

/* ── Info text ── */
.kanji-viet { font-size: 1.15rem; font-weight: 800; color: #cba6f7; }
.kanji-mean { color: #cdd6f4; font-size: .93rem; margin-top: 2px; }
.kanji-meo  { color: #a6e3a1; font-style: italic; font-size: .82rem; margin-top: 4px; }
.vocab-item { color: #f5c2e7; font-size: .86rem; }

/* ── Status badges ── */
.badge {
  display: inline-block; border-radius: 20px;
  padding: 2px 10px; font-size: .7rem; font-weight: 700; letter-spacing: .3px;
}
.tag-db   { background: #1a3d2b; color: #a6e3a1; border: 1px solid #2d6a4f; }
.tag-ai   { background: #2d2046; color: #cba6f7; border: 1px solid #5a3e8a; }
.tag-jisho{ background: #3d2800; color: #fab387; border: 1px solid #7a4f00; }
.tag-miss { background: #3d0f0f; color: #f38ba8; border: 1px solid #7a1f1f; }

/* ── Section titles ── */
.sec-title {
  font-size: 1.3rem; font-weight: 800; color: #cdd6f4;
  border-left: 4px solid #89b4fa; padding-left: 10px; margin-bottom: 12px;
}

/* ── Vocab word card ── */
.vocab-card {
  background: #181825; border: 1px solid #313244;
  border-radius: 10px; padding: 12px 14px; margin-bottom: 8px;
  transition: border-color .2s;
}
.vocab-card:hover { border-color: #585b70; }
.vocab-word  { font-size: 1.5rem; font-weight: 900; color: #cdd6f4; }
.vocab-kana  { font-size: .85rem; color: #89b4fa; }
.vocab-hanviet { font-size: .78rem; color: #a6adc8; font-style: italic; }
.vocab-meaning { font-size: .95rem; color: #cba6f7; margin-top: 4px; }
.vocab-example { font-size: .82rem; color: #6c7086; font-style: italic; margin-top: 2px; }

/* ── Progress bar label ── */
.prog-label { font-size: .8rem; color: #6c7086; text-align: right; margin-top: -6px; }

/* ── Hide Streamlit chrome ── */
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }

/* ── Mobile Responsive (Xiaomi 15 Ultra & phones ≤ 768px) ── */
@media (max-width: 768px) {
  /* Thu gọn padding main area */
  .main .block-container {
    padding: 0.4rem 0.5rem 2rem !important;
    max-width: 100% !important;
  }
  /* Header nhỏ hơn */
  .app-header { padding: 0.6rem 0 0.2rem; margin-bottom: 0.5rem; }
  .app-header h1 { font-size: 1.6rem; letter-spacing: 1px; }
  .app-header p  { font-size: .75rem; }
  /* Kanji glyph trong card */
  .kanji-char { font-size: 2.6rem; }
  /* Card padding nhỏ hơn */
  .card-box { padding: 10px 10px; margin-bottom: 8px; }
  /* Tab radio: scroll ngang thay vì tràn */
  div[data-testid="stRadio"] > div[role="radiogroup"] {
    overflow-x: auto !important;
    flex-wrap: nowrap !important;
    -webkit-overflow-scrolling: touch;
    scrollbar-width: none;
    padding-bottom: 2px;
  }
  div[data-testid="stRadio"] > div[role="radiogroup"]::-webkit-scrollbar { display: none; }
  div[data-testid="stRadio"] > div[role="radiogroup"] > label {
    padding: 8px 14px !important;
    white-space: nowrap;
    font-size: .85rem;
  }
  /* Button touch-friendly */
  button[data-testid="baseButton-primary"],
  button[data-testid="baseButton-secondary"] {
    min-height: 44px !important;
    font-size: .9rem !important;
  }
  /* Metric nhỏ hơn */
  [data-testid="stMetric"] { padding: 4px !important; }
  [data-testid="stMetricValue"] { font-size: 1.1rem !important; }
  [data-testid="stMetricLabel"] { font-size: .7rem !important; }
  /* Vocab card 1 cột */
  .vocab-word  { font-size: 1.25rem; }
  .vocab-meaning { font-size: .88rem; }
  /* Section title */
  .sec-title { font-size: 1.05rem; }
  /* Expander header dễ tap hơn */
  [data-testid="stExpander"] summary { padding: 10px 12px !important; font-size: .88rem; }
  /* Input text lớn hơn để dễ gõ trên mobile */
  input[type="text"], textarea {
    font-size: 16px !important; /* tránh iOS zoom khi focus */
  }
  /* Ẩn sidebar toggle label */
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
    uid     = f"{prefix}_{idx}"
    kanji   = info.get("kanji", "")
    viet    = info.get("viet", "")
    reading = info.get("reading", "")
    meaning = info.get("meaning_vi", "")
    meo     = info.get("meo", "")
    vocab   = info.get("vocab", [])
    status_txt, status_cls = status_of(info)

    st.markdown('<div class="card-box">', unsafe_allow_html=True)
    col_l, col_r = st.columns([1, 3])

    with col_l:
        gif_src = gif_url(kanji) if kanji else ""
        gif_html = (
            f'<img src="{gif_src}" width="80" title="Thứ tự nét viết"'
            f' onerror="this.style.display:none" style="display:block;margin:4px auto 0">'
            if gif_src else ""
        )
        st.markdown(f"""
<div style="text-align:center;line-height:1">
  <div class="kanji-char">{kanji}</div>
  <div class="kanji-read">{reading}</div>
  {gif_html}
</div>""", unsafe_allow_html=True)

        # TTS button chạy JS thực sự qua components iframe
        speak_text = kanji if kanji else reading
        if speak_text:
            safe = speak_text.replace("\\", "\\\\").replace("'", "\\'").replace('"', '\\"')
            _components.html(f"""
<button onclick="speak()" title="Phát âm tiếng Nhật" style="
  background:#2a2a3e;border:1px solid #585b70;border-radius:8px;
  color:#89b4fa;font-size:1.15rem;cursor:pointer;
  padding:3px 0;width:100%;font-family:sans-serif;
">🔊</button>
<script>
function speak() {{
  try {{
    var synth = window.parent.speechSynthesis || window.speechSynthesis;
    var Utt   = window.parent.SpeechSynthesisUtterance || window.SpeechSynthesisUtterance;
    synth.cancel();
    var u = new Utt('{safe}');
    u.lang = 'ja-JP'; u.rate = 0.85;
    synth.speak(u);
  }} catch(e) {{ console.error('TTS', e); }}
}}
</script>
""", height=42)

    with col_r:
        h1, h2 = st.columns([3, 2])
        with h1:
            st.markdown(f'<span class="kanji-viet">{viet}</span>', unsafe_allow_html=True)
        with h2:
            st.markdown(f'<span class="badge {status_cls}">{status_txt}</span>', unsafe_allow_html=True)
        if meaning:
            st.markdown(f'<div class="kanji-mean">📖 {meaning}</div>', unsafe_allow_html=True)
        if meo:
            st.markdown(f'<div class="kanji-meo">💡 {meo}</div>', unsafe_allow_html=True)
        if vocab:
            st.markdown("<hr style='margin:6px 0;border-color:#313244'>", unsafe_allow_html=True)
            for item in vocab[:2]:
                w, r, m = item[0], item[1], item[2]
                st.markdown(f'<div class="vocab-item">• <b>{w}</b>（{r}）— {m}</div>',
                            unsafe_allow_html=True)
        elif info.get("meanings_en"):
            st.markdown("<hr style='margin:6px 0;border-color:#313244'>", unsafe_allow_html=True)
            st.markdown(f'<div class="vocab-item">{" / ".join(info["meanings_en"][:3])}</div>',
                        unsafe_allow_html=True)
        if viet or meaning:
            res_key = f"ai_res_{uid}"
            st.markdown("<div style='margin-top:8px'>", unsafe_allow_html=True)
            if st.button("🤖 Phân tích AI", key=f"analyze_{uid}",
                         use_container_width=False):
                with st.spinner("Đang hỏi AI…"):
                    st.session_state[res_key] = analyze_kanji_ai(kanji)
            if res_key in st.session_state:
                st.markdown(st.session_state[res_key])
            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


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


# --- Sidebar ---
with st.sidebar:
    st.markdown("## ✍️ Kanji Hub")
    st.caption("Ứng dụng học Kanji cho người Việt")
    st.divider()
    st.markdown("### ⚙️ Cài đặt AI")
    prov_options = ["gemini", "openrouter"]
    cur_prov = get_ai_provider()
    new_prov = st.selectbox("Nhà cung cấp AI", prov_options,
                            index=prov_options.index(cur_prov), key="sb_prov")
    if new_prov != cur_prov:
        set_ai_provider(new_prov)
        st.rerun()
    gem_key = st.text_input("🔑 Gemini API Key", value=get_gemini_key(),
                            type="password", placeholder="AIzaSy…", key="sb_gkey")
    or_key  = st.text_input("🔑 OpenRouter API Key", value=get_openrouter_key(),
                            type="password", placeholder="sk-or-…", key="sb_okey")
    if st.button("💾 Lưu cài đặt", use_container_width=True, key="sb_save", type="primary"):
        set_gemini_key(gem_key)
        set_openrouter_key(or_key)
        st.success("✓ Đã lưu!")
    st.divider()
    st.markdown("""
**📌 Cách nhập:**
- Kanji: `山川田`
- Tiếng Việt: `Học`, `Tình`
- Danh sách: `会社,学校`

**🎯 Chế độ tra:**
- `DB` — nhanh, offline
- `DB + AI` — kết hợp
- `AI` — chỉ dùng AI
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
    st.markdown('<div class="app-header"><h1>✍️ Kanji Hub</h1>'
                '<p>Tra cứu Kanji Nhật · Nghĩa tiếng Việt · Luyện viết</p></div>',
                unsafe_allow_html=True)

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
  <span style="color:#cdd6f4;font-size:.95rem">
    <b>{level}</b> · {len(level_data[level])} chữ · {len(chunks)} bài
  </span>
  <span style="color:#a6e3a1;font-size:.85rem">✅ {done_n}/{len(chunks)} hoàn thành</span>
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
            st.markdown(f'<div style="font-size:1.6rem;letter-spacing:3px;color:#cdd6f4;margin-bottom:10px">{"  ".join(chunk)}</div>',
                        unsafe_allow_html=True)
            b1, b2 = st.columns([1, 1])
            with b1:
                if st.button(f"🔍 Tra bài {i+1}", key=f"tra_{level}_{i}",
                             type="primary", use_container_width=True):
                    st.session_state[prog_key].add(lid)
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
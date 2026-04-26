"""
Kanji Hub — Giao diện Web bằng Streamlit
Chạy local: streamlit run streamlit_app.py
Deploy   : streamlit.io/cloud  (kết nối GitHub repo)
"""

import os
import sys
import json
import tempfile
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

import streamlit as st

# ─── Path setup ───────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from kanji_lookup import (
    get_kanji_info, search_by_viet, search_by_viet_gemini,
    MANUAL_VI, N4_VI, MNN_N4_EXTRA, MNN_N5, N3_VI, N2_VI, N1_VI,
    get_gemini_key, set_gemini_key, GeminiQuotaError, AIQuotaError,
    lookup_kanji_gemini, lookup_kanji_openrouter,
    get_ai_provider, set_ai_provider,
    get_openrouter_key, set_openrouter_key,
    extract_kanji, has_cjk,
    analyze_kanji_ai,
)
from pdf_generator import generate_pdf, generate_vocab_table_pdf
from vocab_lessons import VOCAB_LESSONS

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Kanji Hub",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .kanji-char  { font-size: 3rem; font-weight: 900; color: #CDD6F4;
                 text-align: center; line-height: 1.1; margin-bottom: 2px; }
  .kanji-read  { font-size: 0.8rem; color: #89B4FA; text-align: center; }
  .kanji-viet  { font-size: 1.1rem; font-weight: 700; color: #CBA6F7; }
  .kanji-mean  { color: #CDD6F4; font-size: 0.95rem; }
  .kanji-meo   { color: #A6E3A1; font-style: italic; font-size: 0.85rem; }
  .vocab-item  { color: #F5C2E7; font-size: 0.88rem; }
  .tag-db      { color: #2ECC71; font-size: 0.75rem; font-weight: 600; }
  .tag-ai      { color: #A78BFA; font-size: 0.75rem; font-weight: 600; }
  .tag-jisho   { color: #F39C12; font-size: 0.75rem; font-weight: 600; }
  .tag-miss    { color: #E74C3C; font-size: 0.75rem; font-weight: 600; }
  .card-box    { background: #1E1E2E; border: 1px solid #313244;
                 border-radius: 10px; padding: 14px 16px; margin-bottom: 10px; }
  [data-testid="stSidebar"] { background-color: #1A1A2E; }
  div[data-testid="stHorizontalBlock"] { align-items: flex-start; }
</style>
""", unsafe_allow_html=True)

ALL_DB = {**MANUAL_VI, **N4_VI, **MNN_N4_EXTRA, **MNN_N5, **N3_VI, **N2_VI, **N1_VI}


# ─── Session state init ───────────────────────────────────────────────────────
def _init():
    for k, v in {
        "results": [],
        "pdf_bytes": None,
        "progress": {},
        "path_results": [],   # kết quả tra từ lộ trình học
        "path_level": "N5",
    }.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init()


# ─── Helpers ─────────────────────────────────────────────────────────────────
def status_of(info: dict):
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


def generate_pdf_bytes(infos: list, extra_rows: int = 0) -> bytes:
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


def gif_url(kanji: str) -> str:
    if not kanji:
        return ""
    h = hex(ord(kanji[0]))[2:].lower()
    return f"https://raw.githubusercontent.com/mistval/kanji_images/master/gifs/{h}.gif"


# ─── Kanji card ──────────────────────────────────────────────────────────────
def render_card(info: dict):
    kanji   = info.get("kanji", "")
    viet    = info.get("viet", "")
    reading = info.get("reading", "")
    meaning = info.get("meaning_vi", "")
    meo     = info.get("meo", "")
    vocab   = info.get("vocab", [])
    status_txt, status_cls = status_of(info)

    st.markdown('<div class="card-box">', unsafe_allow_html=True)
    col_l, col_r = st.columns([1, 5])

    with col_l:
        st.markdown(f'<div class="kanji-char">{kanji}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="kanji-read">{reading}</div>', unsafe_allow_html=True)
        # Stroke order GIF
        if kanji:
            url = gif_url(kanji)
            st.markdown(
                f'<img src="{url}" width="90" title="Thứ tự nét viết"'
                f' onerror="this.style.display=\'none\'" style="margin-top:6px">',
                unsafe_allow_html=True,
            )

    with col_r:
        hdr_l, hdr_r = st.columns([3, 2])
        with hdr_l:
            st.markdown(f'<span class="kanji-viet">{viet}</span>', unsafe_allow_html=True)
        with hdr_r:
            st.markdown(f'<span class="{status_cls}">{status_txt}</span>', unsafe_allow_html=True)

        if meaning:
            st.markdown(f'<div class="kanji-mean">Nghĩa: {meaning}</div>', unsafe_allow_html=True)
        if meo:
            st.markdown(f'<div class="kanji-meo">💡 {meo}</div>', unsafe_allow_html=True)

        if vocab:
            st.markdown("<hr style='margin:6px 0; border-color:#313244'>", unsafe_allow_html=True)
            for item in vocab[:2]:
                w, r, m = item[0], item[1], item[2]
                st.markdown(f'<div class="vocab-item">• {w}（{r}）— {m}</div>', unsafe_allow_html=True)
        elif info.get("meanings_en"):
            st.markdown("<hr style='margin:6px 0; border-color:#313244'>", unsafe_allow_html=True)
            st.markdown(
                f'<div class="vocab-item">{" / ".join(info["meanings_en"][:3])}</div>',
                unsafe_allow_html=True,
            )

        # AI analysis expander
        if viet or meaning:
            with st.expander("🔍 Phân tích AI chuyên sâu"):
                if st.button("Phân tích ngay", key=f"analyze_{kanji}"):
                    with st.spinner("Đang hỏi AI…"):
                        result = analyze_kanji_ai(kanji)
                    st.markdown(result)

    st.markdown("</div>", unsafe_allow_html=True)


# ─── Lookup logic ─────────────────────────────────────────────────────────────
def do_lookup(query: str, search_mode: str) -> list:
    """Tra cứu và trả về list kết quả."""
    if not query.strip():
        return []

    provider = get_ai_provider()
    use_ai = "AI" in search_mode

    # ── Nhập có CJK → tra kanji ──────────────────────────────────────────
    if has_cjk(query):
        import re
        is_csv = any(sep in query for sep in (',', '、', '，'))
        if is_csv:
            raw_parts = re.split(r'[,、，]', query)
            seen: set[str] = set()
            kanji_list = []
            for part in raw_parts:
                word = ''.join(ch for ch in part.strip() if has_cjk(ch))
                if word and word not in seen:
                    seen.add(word)
                    kanji_list.append(word)
        else:
            kanji_list = extract_kanji(query)

        if not kanji_list:
            return []

        # Chế độ AI thuần
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

        # Chế độ DB / DB + AI
        infos = [None] * len(kanji_list)
        with ThreadPoolExecutor(max_workers=8) as ex:
            future_to_idx = {ex.submit(get_kanji_info, k): i for i, k in enumerate(kanji_list)}
            for fut in as_completed(future_to_idx):
                idx = future_to_idx[fut]
                k = kanji_list[idx]
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
                    infos[idx] = info
                except Exception:
                    infos[idx] = {"kanji": k, "vocab": []}
        return [i for i in infos if i is not None]

    # ── Nhập tiếng Việt → tìm kanji ──────────────────────────────────────
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
        except (AIQuotaError, Exception) as e:
            st.error(str(e))
    return infos or []


# ─── Sidebar: Settings ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Cài đặt AI")

    prov_options = ["gemini", "openrouter"]
    cur_prov = get_ai_provider()
    new_prov = st.selectbox("Nhà cung cấp AI", prov_options,
                            index=prov_options.index(cur_prov))
    if new_prov != cur_prov:
        set_ai_provider(new_prov)
        st.rerun()

    gem_key = st.text_input("Gemini API Key", value=get_gemini_key(),
                             type="password", placeholder="AIzaSy…")
    or_key  = st.text_input("OpenRouter API Key", value=get_openrouter_key(),
                             type="password", placeholder="sk-or-…")

    if st.button("💾 Lưu cài đặt", use_container_width=True):
        set_gemini_key(gem_key)
        set_openrouter_key(or_key)
        st.success("✓ Đã lưu API keys!")

    st.markdown("---")
    st.markdown("**Hướng dẫn nhập:**")
    st.markdown("- Kanji: `山川田`")
    st.markdown("- Tiếng Việt: `Học`, `Tĩnh`")
    st.markdown("- Danh sách: `会社,学校,病院`")
    st.markdown("---")
    st.caption("Kanji Hub • Streamlit Web")


# ─── Main tabs ────────────────────────────────────────────────────────────────
tab_search, tab_path, tab_vocab = st.tabs(["🔍 Tra Kanji", "🗺️ Lộ trình học", "📖 Từ Vựng"])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — TRA KANJI
# ═══════════════════════════════════════════════════════════════════════════════
with tab_search:
    st.markdown("## ✍️ Kanji Hub")

    with st.form("search_form"):
        c_inp, c_mode, c_btn = st.columns([5, 2, 1])
        with c_inp:
            query = st.text_input(
                "query", label_visibility="collapsed",
                placeholder="Nhập kanji: 山川田… hoặc tiếng Việt: Học, Tĩnh…",
            )
        with c_mode:
            search_mode = st.selectbox(
                "mode", ["DB", "DB + AI", "AI"],
                index=1, label_visibility="collapsed",
            )
        with c_btn:
            submitted = st.form_submit_button("🔍 Tra", use_container_width=True)

    if submitted and query:
        with st.spinner("⏳ Đang tra cứu…"):
            st.session_state.results = do_lookup(query, search_mode)
        st.session_state.pdf_bytes = None   # reset cache PDF

    results = st.session_state.results

    if results:
        # ── PDF export ──────────────────────────────────────────────────────
        valid = [i for i in results if i.get("viet") or i.get("meaning_vi")]
        if valid:
            c_rows, c_dl = st.columns([3, 2])
            with c_rows:
                rows_map = {
                    "1 hàng (6 ô)":  0,
                    "2 hàng (18 ô)": 1,
                    "3 hàng (30 ô)": 2,
                    "4 hàng (42 ô)": 3,
                }
                sel_rows  = st.selectbox("Số hàng luyện viết", list(rows_map.keys()))
                extra_rows = rows_map[sel_rows]

            with c_dl:
                st.write("")
                try:
                    pdf_bytes = generate_pdf_bytes(valid, extra_rows)
                    kanji_str = "".join(i["kanji"] for i in valid)
                    safe_name = "".join(extract_kanji(kanji_str)[:10]) or "Kanji"
                    st.download_button(
                        "📄 Tải PDF luyện viết",
                        data=pdf_bytes,
                        file_name=f"Kanji_{safe_name}.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                    )
                except Exception as e:
                    st.error(f"Lỗi tạo PDF: {e}")

        # ── Status ─────────────────────────────────────────────────────────
        errs  = [i for i in results if status_of(i)[0].startswith("✗")]
        found = len(results) - len(errs)
        st.caption(
            f"Tìm thấy **{len(results)}** kanji • "
            f"✓ {found} trong DB • ✗ {len(errs)} không tìm được"
        )
        st.divider()

        # ── Cards ───────────────────────────────────────────────────────────
        for info in results:
            render_card(info)

    elif submitted and query and not results:
        st.warning("Không tìm thấy kết quả. Thử gõ có dấu hoặc chuyển sang chế độ AI.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — LỘ TRÌNH HỌC
# ═══════════════════════════════════════════════════════════════════════════════
with tab_path:
    st.markdown("## 🗺️ Lộ trình học Kanji")

    level_data = {
        "N5": list(MNN_N5.keys()),
        "N4": list(N4_VI.keys()),
        "N3": list(N3_VI.keys()),
        "N2": list(N2_VI.keys()),
        "N1": list(N1_VI.keys()),
    }

    level = st.selectbox("Chọn cấp độ", list(level_data.keys()), key="path_level_sel")
    kanji_all = level_data[level]
    chunks = [kanji_all[i:i+10] for i in range(0, len(kanji_all), 10)]

    prog_key = f"prog_{level}"
    if prog_key not in st.session_state:
        st.session_state[prog_key] = set()

    done_count = len(st.session_state[prog_key])
    st.info(
        f"**{level}** — {len(kanji_all)} chữ — {len(chunks)} bài • "
        f"Đã hoàn thành: {done_count}/{len(chunks)} bài"
    )

    for i, chunk in enumerate(chunks):
        lid = f"{level}_{i+1}"
        is_done = lid in st.session_state[prog_key]
        icon = "✅" if is_done else "📖"
        label = f"{icon} Bài {i+1} ({len(chunk)} chữ) — {'  '.join(chunk)}"

        with st.expander(label, expanded=False):
            st.markdown(
                f'<span style="font-size:1.6rem;letter-spacing:4px">'
                f'{"  ".join(chunk)}</span>',
                unsafe_allow_html=True,
            )
            cb, c2 = st.columns([2, 3])
            with cb:
                if st.button(f"🔍 Tra cứu bài {i+1}", key=f"tra_{level}_{i}"):
                    st.session_state[prog_key].add(lid)
                    with st.spinner("Đang tra cứu…"):
                        st.session_state.path_results = do_lookup("".join(chunk), "DB + AI")
                    st.session_state.results = st.session_state.path_results
            with c2:
                checked = st.checkbox(
                    "Đánh dấu đã học", value=is_done, key=f"chk_{level}_{i}"
                )
                if checked != is_done:
                    if checked:
                        st.session_state[prog_key].add(lid)
                    else:
                        st.session_state[prog_key].discard(lid)

    # Hiện kết quả tra cứu từ lộ trình bên dưới
    path_results = st.session_state.get("path_results", [])
    if path_results:
        st.divider()
        st.markdown("### Kết quả tra cứu")
        valid = [i for i in path_results if i.get("viet") or i.get("meaning_vi")]
        if valid:
            try:
                pdf_b = generate_pdf_bytes(valid)
                safe  = "".join(i["kanji"] for i in valid[:10])
                st.download_button(
                    "📄 Tải PDF bài này",
                    data=pdf_b,
                    file_name=f"Kanji_{safe}.pdf",
                    mime="application/pdf",
                )
            except Exception:
                pass
        for info in path_results:
            render_card(info)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — TỪ VỰNG
# ═══════════════════════════════════════════════════════════════════════════════
with tab_vocab:
    st.markdown("## 📖 Bài Từ Vựng")

    if not VOCAB_LESSONS:
        st.info("Chưa có bài từ vựng nào.")
    else:
        lesson_nums = sorted(VOCAB_LESSONS.keys())

        sel_lesson = st.selectbox(
            "Chọn bài",
            lesson_nums,
            format_func=lambda n: f"Bài {n}  ({len(VOCAB_LESSONS[n])} từ)",
        )

        words = VOCAB_LESSONS[sel_lesson]
        st.info(f"**Bài {sel_lesson}** — {len(words)} từ vựng")

        for idx, item in enumerate(words):
            c1, c2 = st.columns([1, 3])
            with c1:
                st.markdown(
                    f"**{idx+1}. {item['word']}**  \n"
                    f"<span style='color:#89B4FA'>（{item.get('reading','')}）</span>  \n"
                    f"<span style='color:#CBA6F7;font-size:.8rem'>{item.get('hanviet','')}</span>",
                    unsafe_allow_html=True,
                )
            with c2:
                st.markdown(f"▸ {item.get('meaning', '')}")
                if item.get("example"):
                    st.caption(f"Ví dụ: {item['example']}")
                if item.get("exampleVi"):
                    st.caption(f"↳ {item['exampleVi']}")
            st.divider()

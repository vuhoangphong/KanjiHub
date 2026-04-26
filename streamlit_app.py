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
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
  .kanji-char{font-size:3rem;font-weight:900;color:#CDD6F4;text-align:center;line-height:1.1;margin-bottom:2px}
  .kanji-read{font-size:.8rem;color:#89B4FA;text-align:center}
  .kanji-viet{font-size:1.1rem;font-weight:700;color:#CBA6F7}
  .kanji-mean{color:#CDD6F4;font-size:.95rem}
  .kanji-meo{color:#A6E3A1;font-style:italic;font-size:.85rem}
  .vocab-item{color:#F5C2E7;font-size:.88rem}
  .tag-db{color:#2ECC71;font-size:.75rem;font-weight:600}
  .tag-ai{color:#A78BFA;font-size:.75rem;font-weight:600}
  .tag-jisho{color:#F39C12;font-size:.75rem;font-weight:600}
  .tag-miss{color:#E74C3C;font-size:.75rem;font-weight:600}
  .card-box{background:#1E1E2E;border:1px solid #313244;border-radius:10px;padding:14px 16px;margin-bottom:10px}
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
    col_l, col_r = st.columns([1, 5])

    with col_l:
        st.markdown(f'<div class="kanji-char">{kanji}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="kanji-read">{reading}</div>', unsafe_allow_html=True)
        if kanji:
            url = gif_url(kanji)
            st.markdown(
                f'<img src="{url}" width="90" title="Thu tu net viet"'
                f' onerror="this.style.display=\'none\'" style="margin-top:6px">',
                unsafe_allow_html=True)

    with col_r:
        h1, h2 = st.columns([3, 2])
        with h1:
            st.markdown(f'<span class="kanji-viet">{viet}</span>', unsafe_allow_html=True)
        with h2:
            st.markdown(f'<span class="{status_cls}">{status_txt}</span>', unsafe_allow_html=True)
        if meaning:
            st.markdown(f'<div class="kanji-mean">Nghia: {meaning}</div>', unsafe_allow_html=True)
        if meo:
            st.markdown(f'<div class="kanji-meo">💡 {meo}</div>', unsafe_allow_html=True)
        if vocab:
            st.markdown("<hr style='margin:6px 0;border-color:#313244'>", unsafe_allow_html=True)
            for item in vocab[:2]:
                w, r, m = item[0], item[1], item[2]
                st.markdown(f'<div class="vocab-item">• {w}（{r}）— {m}</div>',
                            unsafe_allow_html=True)
        elif info.get("meanings_en"):
            st.markdown("<hr style='margin:6px 0;border-color:#313244'>", unsafe_allow_html=True)
            st.markdown(f'<div class="vocab-item">{" / ".join(info["meanings_en"][:3])}</div>',
                        unsafe_allow_html=True)
        if viet or meaning:
            with st.expander("🔍 Phan tich AI chuyen sau", expanded=False):
                res_key = f"ai_res_{uid}"
                if st.button("Phan tich ngay", key=f"analyze_{uid}"):
                    with st.spinner("Dang hoi AI…"):
                        st.session_state[res_key] = analyze_kanji_ai(kanji)
                if res_key in st.session_state:
                    st.markdown(st.session_state[res_key])

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
    st.markdown("## ⚙️ Cai dat AI")
    prov_options = ["gemini", "openrouter"]
    cur_prov = get_ai_provider()
    new_prov = st.selectbox("Nha cung cap AI", prov_options,
                            index=prov_options.index(cur_prov), key="sb_prov")
    if new_prov != cur_prov:
        set_ai_provider(new_prov)
        st.rerun()
    gem_key = st.text_input("Gemini API Key", value=get_gemini_key(),
                            type="password", placeholder="AIzaSy…", key="sb_gkey")
    or_key  = st.text_input("OpenRouter API Key", value=get_openrouter_key(),
                            type="password", placeholder="sk-or-…", key="sb_okey")
    if st.button("💾 Luu cai dat", use_container_width=True, key="sb_save"):
        set_gemini_key(gem_key)
        set_openrouter_key(or_key)
        st.success("✓ Da luu!")
    st.markdown("---")
    st.markdown("**Cach nhap:**\n- Kanji: `山川田`\n- Tieng Viet: `Hoc`, `Tinh`\n- Danh sach: `会社,学校`")
    st.caption("Kanji Hub • Streamlit Web")


# --- Tabs ---
tab_search, tab_path, tab_vocab = st.tabs(["🔍 Tra Kanji", "🗺️ Lo trinh hoc", "📖 Tu Vung"])

# === TAB 1 ===
with tab_search:
    st.markdown("## ✍️ Kanji Hub")
    with st.form("search_form"):
        c1, c2, c3 = st.columns([5, 2, 1])
        with c1:
            query = st.text_input("q", label_visibility="collapsed",
                                  placeholder="Nhap kanji: 山川田… hoac tieng Viet: Hoc…")
        with c2:
            search_mode = st.selectbox("m", ["DB", "DB + AI", "AI"],
                                       index=1, label_visibility="collapsed")
        with c3:
            submitted = st.form_submit_button("🔍 Tra", use_container_width=True)

    if submitted and query:
        with st.spinner("⏳ Dang tra cuu…"):
            st.session_state.results = do_lookup(query, search_mode)

    results = st.session_state.results

    if results:
        valid = [i for i in results if i.get("viet") or i.get("meaning_vi")]
        if valid:
            r1, r2 = st.columns([3, 2])
            with r1:
                rows_map = {"1 hang (6 o)": 0, "2 hang (18 o)": 1,
                            "3 hang (30 o)": 2, "4 hang (42 o)": 3}
                sel = st.selectbox("So hang luyen viet", list(rows_map.keys()), key="sr_rows")
            with r2:
                st.write("")
                try:
                    pdf_b = make_pdf_bytes(valid, rows_map[sel])
                    safe  = "".join(extract_kanji("".join(i["kanji"] for i in valid))[:10]) or "Kanji"
                    st.download_button("📄 Tai PDF luyen viet", data=pdf_b,
                                       file_name=f"Kanji_{safe}.pdf",
                                       mime="application/pdf", key="dl_search",
                                       use_container_width=True)
                except Exception as e:
                    st.error(f"Loi PDF: {e}")

        errs = [i for i in results if status_of(i)[0].startswith("✗")]
        st.caption(f"**{len(results)}** kanji • ✓ {len(results)-len(errs)} trong DB "
                   f"• ✗ {len(errs)} khong tim duoc")
        st.divider()
        for idx, info in enumerate(results):
            render_card(info, idx=idx, prefix="s")
    elif submitted and query:
        st.warning("Khong tim thay. Thu go co dau hoac chuyen sang che do AI.")

# === TAB 2 ===
with tab_path:
    st.markdown("## 🗺️ Lo trinh hoc Kanji")
    level_data = {
        "N5": list(MNN_N5.keys()), "N4": list(N4_VI.keys()),
        "N3": list(N3_VI.keys()),  "N2": list(N2_VI.keys()),
        "N1": list(N1_VI.keys()),
    }
    level  = st.selectbox("Cap do", list(level_data.keys()), key="pl_sel")
    chunks = [level_data[level][i:i+10] for i in range(0, len(level_data[level]), 10)]
    prog_key = f"prog_{level}"
    if prog_key not in st.session_state:
        st.session_state[prog_key] = set()
    done_n = len(st.session_state[prog_key])
    st.info(f"**{level}** — {len(level_data[level])} chu — {len(chunks)} bai • "
            f"Da hoan thanh: {done_n}/{len(chunks)} bai")

    for i, chunk in enumerate(chunks):
        lid     = f"{level}_{i+1}"
        is_done = lid in st.session_state[prog_key]
        icon    = "✅" if is_done else "📖"
        with st.expander(f"{icon} Bai {i+1} ({len(chunk)} chu) — {'  '.join(chunk)}"):
            st.markdown(f'<span style="font-size:1.5rem;letter-spacing:4px">{"  ".join(chunk)}</span>',
                        unsafe_allow_html=True)
            b1, b2 = st.columns([2, 3])
            with b1:
                if st.button(f"🔍 Tra bai {i+1}", key=f"tra_{level}_{i}"):
                    st.session_state[prog_key].add(lid)
                    with st.spinner("Dang tra…"):
                        st.session_state.path_results = do_lookup("".join(chunk), "DB + AI")
                    st.rerun()
            with b2:
                chk = st.checkbox("Da hoc", value=is_done, key=f"chk_{level}_{i}")
                if chk and not is_done:
                    st.session_state[prog_key].add(lid)
                elif not chk and is_done:
                    st.session_state[prog_key].discard(lid)

    path_results = st.session_state.path_results
    if path_results:
        st.divider()
        st.markdown("### Ket qua tra cuu")
        valid_p = [i for i in path_results if i.get("viet") or i.get("meaning_vi")]
        if valid_p:
            try:
                safe_p = "".join(i["kanji"] for i in valid_p[:10])
                st.download_button("📄 Tai PDF bai nay",
                                   data=make_pdf_bytes(valid_p),
                                   file_name=f"Kanji_{safe_p}.pdf",
                                   mime="application/pdf", key="dl_path")
            except Exception:
                pass
        for idx, info in enumerate(path_results):
            render_card(info, idx=idx, prefix="p")

# === TAB 3 ===
with tab_vocab:
    st.markdown("## 📖 Bai Tu Vung")
    if not VOCAB_LESSONS:
        st.info("Chua co bai tu vung nao.")
    else:
        lesson_nums = sorted(VOCAB_LESSONS.keys())
        sel_lesson  = st.selectbox("Chon bai", lesson_nums,
                                   format_func=lambda n: f"Bai {n}  ({len(VOCAB_LESSONS[n])} tu)",
                                   key="vl_sel")
        words = VOCAB_LESSONS[sel_lesson]
        st.info(f"**Bai {sel_lesson}** — {len(words)} tu vung")
        for idx, item in enumerate(words):
            c1, c2 = st.columns([1, 3])
            with c1:
                st.markdown(
                    f"**{idx+1}. {item['word']}**  \n"
                    f"<span style='color:#89B4FA'>（{item.get('reading','')}）</span>  \n"
                    f"<span style='color:#CBA6F7;font-size:.8rem'>{item.get('hanviet','')}</span>",
                    unsafe_allow_html=True)
            with c2:
                st.markdown(f"▸ {item.get('meaning', '')}")
                if item.get("example"):
                    st.caption(f"Vi du: {item['example']}")
                if item.get("exampleVi"):
                    st.caption(f"↳ {item['exampleVi']}")
            st.divider()
"""
Kanji Hub — Giao diện Web bằng Streamlit
Chạy local: streamlit run streamlit_app.py
Deploy   : streamlit.io/cloud
"""

import os
import sys
import re
import tempfile
import base64
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
        analyze_kanji_ai, lookup_vocab,
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


# ── Flash Card HTML builder ─────────────────────────────────────────────────
_FC_TEMPLATE = """<!DOCTYPE html>
<html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Segoe UI',system-ui,sans-serif;background:transparent;padding:8px 4px 4px;overflow-x:hidden}
/* ── Wrapper centered ── */
.fc-wrap{max-width:680px;margin:0 auto}
/* ── Layout: arrow | scene | arrow ── */
.fc-outer{display:flex;align-items:center;gap:6px;margin-bottom:10px}
.fc-arrow{background:rgba(0,0,0,.07);border:none;border-radius:50%;width:30px;height:30px;font-size:1.1rem;
  cursor:pointer;display:flex;align-items:center;justify-content:center;color:#5a6a7a;
  flex-shrink:0;transition:background .14s}.fc-arrow:hover{background:rgba(0,0,0,.13)}
/* ── 3D Scene ── */
.fc-scene{flex:1;perspective:900px;min-width:0}
.fc-card{position:relative;width:100%;height:360px;transform-style:preserve-3d;
  transition:transform .55s cubic-bezier(.645,.045,.355,1);cursor:pointer;border-radius:12px}
.fc-card.flipped{transform:rotateY(180deg)}
.fc-face{position:absolute;width:100%;height:100%;backface-visibility:hidden;
  -webkit-backface-visibility:hidden;border-radius:12px;display:flex;flex-direction:column;
  align-items:center;justify-content:center;padding:20px 22px 14px;overflow:hidden}
.fc-front{background:#1e2d40}
.fc-back{background:#1e2d40;transform:rotateY(180deg)}
/* star */
.fc-star{position:absolute;top:9px;right:11px;font-size:.95rem;color:#4a6080;
  background:none;border:none;cursor:pointer;padding:2px}.fc-star:hover{color:#c8a45a}
/* front */
.fc-main{font-size:3.6rem;font-weight:700;color:#fff;text-align:center;
  font-family:'Noto Serif JP','Segoe UI',serif;line-height:1.2;margin-bottom:12px}
.fc-sub{font-size:1.25rem;color:#7a95b0;text-align:center}
/* back */
.fc-b-meaning{font-size:2.1rem;font-weight:700;color:#fff;text-align:center;margin-bottom:8px;line-height:1.3}
.fc-b-hv{font-size:1rem;color:#e07878;font-weight:700;letter-spacing:1.5px;margin-bottom:16px}
.fc-b-jp{font-size:1rem;color:#b0c4d8;text-align:center;
  font-family:'Noto Serif JP','Segoe UI',serif;margin-bottom:6px;line-height:1.6}
.fc-b-vi{font-size:.95rem;color:#7a95b0;text-align:center;font-style:italic;line-height:1.6}
/* TTS on face */
.fc-audio{background:none;border:none;color:#5a7898;cursor:pointer;margin-top:10px;
  display:flex;align-items:center;gap:4px;font-size:.8rem;padding:3px 9px;
  border-radius:6px;transition:all .14s}
.fc-audio:hover{color:#8099b0;background:rgba(255,255,255,.07)}
.fc-audio.speaking{color:#60a0e0;animation:pulse .5s ease infinite alternate}
@keyframes pulse{from{transform:scale(1)}to{transform:scale(1.12)}}
/* flash overlay */
.fc-ov{position:absolute;inset:0;pointer-events:none;opacity:0;transition:opacity .18s;border-radius:12px}
.fc-ov.green{background:rgba(39,174,96,.18)}.fc-ov.red{background:rgba(220,53,69,.16)}.fc-ov.show{opacity:1}
/* keyboard shortcut bar */
.fc-kb{position:absolute;bottom:0;left:0;right:0;background:rgba(0,0,0,.22);
  padding:5px 10px;display:flex;align-items:center;gap:5px;flex-wrap:wrap}
.fc-kb-lbl{font-size:.58rem;color:#5a7898;margin-right:2px;flex-shrink:0}
.fc-kb-item{display:flex;align-items:center;gap:3px;margin-right:3px}
.fc-kb-key{background:#243345;border:1px solid #344d60;border-radius:3px;
  padding:1px 4px;font-size:.56rem;color:#7a9ab0;font-weight:600}
.fc-kb-act{font-size:.56rem;color:#5a7898}
/* ── Below-card controls ── */
.fc-controls{display:flex;align-items:center;justify-content:space-between;gap:6px}
.fc-tabs{display:flex;gap:5px}
.fc-tab{background:#3b82f6;color:#fff;border:none;border-radius:6px;
  padding:5px 10px;font-size:.73rem;font-weight:600;cursor:pointer;opacity:.4;transition:opacity .14s}
.fc-tab.active{opacity:1}.fc-tab:hover{opacity:.82}
/* center: ✗ counter ✓ */
.fc-center{display:flex;align-items:center;gap:9px}
.fc-unk-btn,.fc-known-btn{border:none;border-radius:50%;width:32px;height:32px;
  font-size:.9rem;color:#fff;cursor:pointer;display:flex;align-items:center;
  justify-content:center;transition:transform .12s,box-shadow .12s}
.fc-unk-btn{background:#dc3545}.fc-unk-btn:hover{transform:scale(1.09);box-shadow:0 3px 10px rgba(220,53,69,.4)}
.fc-known-btn{background:#28a745}.fc-known-btn:hover{transform:scale(1.09);box-shadow:0 3px 10px rgba(40,167,69,.4)}
.fc-counter{font-size:.82rem;font-weight:700;color:#2a3a4a;min-width:42px;text-align:center}
/* right icons */
.fc-right{display:flex;align-items:center;gap:5px}
.fc-dir-btn{background:#fff;border:1.5px solid #d0d8e0;border-radius:6px;padding:4px 8px;
  font-size:.68rem;color:#2a3a4a;font-weight:600;cursor:pointer;white-space:nowrap;
  display:flex;align-items:center;gap:3px;transition:all .14s}
.fc-dir-btn:hover{background:#f0f5ff;border-color:#3b82f6;color:#3b82f6}
.fc-icon-btn{background:none;border:none;font-size:.95rem;color:#5a6a7a;cursor:pointer;
  padding:4px;border-radius:6px;display:flex;align-items:center;transition:all .14s}
.fc-icon-btn:hover{color:#1a2a3a;background:rgba(0,0,0,.07)}
/* shake */
@keyframes shake{0%,100%{transform:translateX(0)}15%{transform:translateX(-7px)}35%{transform:translateX(6px)}55%{transform:translateX(-4px)}75%{transform:translateX(3px)}}
/* done */
.fc-done{text-align:center;padding:20px 14px;background:#1e2d40;border-radius:12px;color:#fff;max-width:520px;margin:0 auto}
.fc-done-title{font-size:1.25rem;font-weight:900;margin:7px 0 4px}
.fc-done-sub{font-size:.78rem;color:#7a95b0;margin-bottom:14px}
.fc-done-stats{display:inline-flex;gap:16px;background:#162030;border-radius:10px;padding:11px 20px;margin-bottom:14px}
.fc-ds{display:flex;flex-direction:column;align-items:center;gap:2px}
.fc-ds-num{font-size:1.5rem;font-weight:900}.fc-ds-lbl{font-size:.58rem;color:#5a7898;letter-spacing:.5px}
.n-k{color:#28a745}.n-u{color:#dc3545}.n-t{color:#c8a45a}
.fc-done-btn{background:#3b82f6;color:#fff;border:none;border-radius:8px;padding:7px 16px;
  font-size:.78rem;font-weight:700;cursor:pointer;margin:3px;transition:background .14s}
.fc-done-btn:hover{background:#2563eb}.fc-done-btn.sec{background:#2a3c50;color:#7a95b0}
.fc-done-btn.sec:hover{background:#3a4c60;color:#fff}
/* confetti */
.cp{position:fixed;pointer-events:none;animation:cf linear forwards}
@keyframes cf{0%{transform:translateY(-20px) rotate(0deg);opacity:1}100%{transform:translateY(100vh) rotate(720deg);opacity:0}}
</style></head>
<body><div id="root"></div><script>
const DECK=__DECK_JSON__;
let idx=0,flipped=false,known=[],unk=[],done=false,mode='word',dir='jp-vi';
function cur(){return DECK[idx];}
function getFront(d){
  if(mode==='ex') return dir==='jp-vi'?(d.back_ex_jp||d.front):(d.back_ex_vi||d.back_main);
  return dir==='jp-vi'?d.front:d.back_main;
}
function getFrontSub(d){
  if(mode==='ex') return '';
  return dir==='jp-vi'?(d.front_sub||''):(d.back_sub||'');
}
function getBackMain(d){
  if(mode==='ex') return dir==='jp-vi'?(d.back_ex_vi||d.back_main):(d.back_ex_jp||d.front);
  return dir==='jp-vi'?d.back_main:d.front;
}
function getBackSub(d){
  if(mode==='ex') return '';
  return dir==='jp-vi'?(d.back_sub||''):(d.front_sub||'');
}
function render(){
  if(done){renderDone();return;}
  const d=cur(),tot=DECK.length;
  const fTxt=getFront(d),fSub=getFrontSub(d);
  const bMain=getBackMain(d),bSub=getBackSub(d);
  const bExJp=mode==='word'&&dir==='jp-vi'?(d.back_ex_jp||''):'';
  const bExVi=mode==='word'&&dir==='jp-vi'?(d.back_ex_vi||''):'';
  const dirLbl=dir==='jp-vi'?'JP\u2192VI':'VI\u2192JP';
  const jp=d.jp_word||d.front;
  document.getElementById('root').innerHTML=`
  <div class="fc-wrap">
    <div class="fc-outer">
      <button class="fc-arrow" onclick="prevCard()">&#8249;</button>
      <div class="fc-scene">
        <div class="fc-card ${flipped?'flipped':''}" id="card">
          <div class="fc-face fc-front" onclick="doFlip()">
            <div class="fc-ov" id="ov-f"></div>
            <button class="fc-star" onclick="event.stopPropagation()" title="Đánh dấu">&#9733;</button>
            <div class="fc-main">${fTxt}</div>
            ${fSub?`<div class="fc-sub">${fSub}</div>`:''}
            <button class="fc-audio" id="abtn-f" onclick="event.stopPropagation();speakCard()">&#128264; Phát âm</button>
            <div class="fc-kb">
              <span class="fc-kb-lbl">Phím tắt:</span>
              <span class="fc-kb-item"><span class="fc-kb-key">Space</span><span class="fc-kb-act">lật</span></span>
              <span class="fc-kb-item"><span class="fc-kb-key">Z</span><span class="fc-kb-act">biết</span></span>
              <span class="fc-kb-item"><span class="fc-kb-key">X</span><span class="fc-kb-act">chưa biết</span></span>
              <span class="fc-kb-item"><span class="fc-kb-key">R</span><span class="fc-kb-act">nghe audio</span></span>
            </div>
          </div>
          <div class="fc-face fc-back" onclick="doFlip()">
            <div class="fc-ov" id="ov-b"></div>
            <button class="fc-star" onclick="event.stopPropagation()" title="Đánh dấu">&#9733;</button>
            <div class="fc-b-meaning">${bMain}</div>
            ${bSub?`<div class="fc-b-hv">${bSub}</div>`:''}
            ${bExJp?`<div class="fc-b-jp">\u300c${bExJp}\u300d</div>`:''}
            ${bExVi?`<div class="fc-b-vi">${bExVi}</div>`:''}
            <button class="fc-audio" id="abtn-b" onclick="event.stopPropagation();speakCard()">&#128264; Phát âm</button>
            <div class="fc-kb">
              <span class="fc-kb-lbl">Phím tắt:</span>
              <span class="fc-kb-item"><span class="fc-kb-key">Space</span><span class="fc-kb-act">lật</span></span>
              <span class="fc-kb-item"><span class="fc-kb-key">Z</span><span class="fc-kb-act">biết</span></span>
              <span class="fc-kb-item"><span class="fc-kb-key">X</span><span class="fc-kb-act">chưa biết</span></span>
              <span class="fc-kb-item"><span class="fc-kb-key">R</span><span class="fc-kb-act">nghe audio</span></span>
            </div>
          </div>
        </div>
      </div>
      <button class="fc-arrow" onclick="nextCard()">&#8250;</button>
    </div>
    <div class="fc-controls">
      <div class="fc-tabs">
        <button class="fc-tab ${mode==='word'?'active':''}" onclick="setMode('word')">Từ đơn</button>
        <button class="fc-tab ${mode==='ex'?'active':''}" onclick="setMode('ex')">Ví dụ</button>
      </div>
      <div class="fc-center">
        <button class="fc-unk-btn" onclick="markUnk()" title="Chưa biết">&#10005;</button>
        <span class="fc-counter">${idx+1} / ${tot}</span>
        <button class="fc-known-btn" onclick="markKnown()" title="Đã biết">&#10003;</button>
      </div>
      <div class="fc-right">
        <button class="fc-dir-btn" onclick="toggleDir()">&#8652; ${dirLbl}</button>
        <button class="fc-icon-btn" onclick="restartDeck()" title="Làm lại">&#8635;</button>
        <button class="fc-icon-btn" onclick="shuffleDeck()" title="Trộn thẻ">&#128256;</button>
      </div>
    </div>
  </div>`;
}
function doFlip(){
  flipped=!flipped;
  const card=document.getElementById('card');
  if(card) card.classList.toggle('flipped',flipped);
  if(flipped) setTimeout(()=>speakCard(),580);
}
function speakCard(){speak((cur().jp_word||cur().front));}
function speak(text){
  if(!text||!window.speechSynthesis)return;
  window.speechSynthesis.cancel();
  const u=new SpeechSynthesisUtterance(text);
  u.lang='ja-JP';u.rate=0.9;u.pitch=1;
  const vv=window.speechSynthesis.getVoices();
  const jpv=vv.find(v=>v.lang==='ja-JP')||vv.find(v=>v.lang.startsWith('ja'));
  if(jpv)u.voice=jpv;
  ['abtn-f','abtn-b'].forEach(id=>{
    const btn=document.getElementById(id);
    if(btn){btn.classList.add('speaking');u.onend=()=>btn&&btn.classList.remove('speaking');}
  });
  window.speechSynthesis.speak(u);
}
function flash(col){
  ['ov-f','ov-b'].forEach(id=>{
    const o=document.getElementById(id);if(!o)return;
    o.className='fc-ov '+col+' show';
    setTimeout(()=>o&&o.classList.remove('show'),400);
  });
}
function advance(){
  if(idx>=DECK.length-1){done=true;render();}else{idx++;flipped=false;render();}
}
function nextCard(){advance();}
function prevCard(){if(idx>0){idx--;flipped=false;render();}}
function markKnown(){known.push(DECK[idx]);flash('green');setTimeout(advance,380);}
function markUnk(){
  unk.push(DECK[idx]);flash('red');
  const c=document.getElementById('card');
  if(c){c.style.animation='shake .22s ease';setTimeout(()=>c&&(c.style.animation=''),220);}
  setTimeout(advance,460);
}
function setMode(m){mode=m;flipped=false;render();}
function toggleDir(){dir=dir==='jp-vi'?'vi-jp':'jp-vi';flipped=false;render();}
function restartDeck(){idx=0;flipped=false;known=[];unk=[];done=false;render();}
function shuffleDeck(){
  for(let i=DECK.length-1;i>0;i--){const j=Math.floor(Math.random()*(i+1));[DECK[i],DECK[j]]=[DECK[j],DECK[i]];}
  idx=0;flipped=false;render();
}
function renderDone(){
  spawnConfetti();
  const em=known.length===DECK.length?'🎉':known.length>DECK.length/2?'⭐':'💪';
  document.getElementById('root').innerHTML=`
  <div class="fc-done">
    <div style="font-size:2.4rem">${em}</div>
    <div class="fc-done-title">${known.length===DECK.length?'Ho\u00e0n h\u1ea3o! B\u1ea1n thu\u1ed9c t\u1ea5t c\u1ea3!':'\u0110\u00e3 xong b\u1ed9 th\u1ba3!'}</div>
    <div class="fc-done-sub">K\u1ebft qu\u1ea3 phi\u00ean h\u1ecdc</div>
    <div class="fc-done-stats">
      <div class="fc-ds"><span class="fc-ds-num n-k">${known.length}</span><span class="fc-ds-lbl">\u2713 \u0110\u00c3 THU\u1ed8C</span></div>
      <div class="fc-ds"><span class="fc-ds-num n-u">${unk.length}</span><span class="fc-ds-lbl">\u2717 CH\u01afA THU\u1ed8C</span></div>
      <div class="fc-ds"><span class="fc-ds-num n-t">${DECK.length}</span><span class="fc-ds-lbl">T\u1ed4NG S\u1ed0 TH\u1ba2</span></div>
    </div>
    ${unk.length>0?`<button class="fc-done-btn" onclick="reviewUnk()">🔁 \u00d4n l\u1ea1i ${unk.length} th\u1ba3 ch\u01b0a thu\u1ed9c</button>`:''}
    <button class="fc-done-btn sec" onclick="restartDeck()">\u21ba H\u1ecdc l\u1ea1i t\u1eeb \u0111\u1ea7u</button>
  </div>`;
}
function reviewUnk(){
  const tmp=[...unk];DECK.length=0;tmp.forEach(x=>DECK.push(x));
  idx=0;flipped=false;known=[];unk=[];done=false;render();
}
function spawnConfetti(){
  const cols=['#3b82f6','#28a745','#dc3545','#c8a45a','#9b59b6','#e67e22'];
  for(let i=0;i<28;i++){
    const el=document.createElement('div');el.className='cp';
    el.style.cssText=`left:${Math.random()*100}vw;top:-10px;background:${cols[Math.floor(Math.random()*cols.length)]};width:${5+Math.random()*6}px;height:${5+Math.random()*6}px;border-radius:${Math.random()>.5?'50%':'2px'};animation-duration:${1.2+Math.random()*1.8}s;animation-delay:${Math.random()*.6}s`;
    document.body.appendChild(el);el.addEventListener('animationend',()=>el.remove());
  }
}
document.addEventListener('keydown',e=>{
  if(done)return;
  if(e.code==='Space'){e.preventDefault();doFlip();}
  else if(e.key==='z'||e.key==='Z'){markKnown();}
  else if(e.key==='x'||e.key==='X'){markUnk();}
  else if(e.key==='r'||e.key==='R')speakCard();
});
window.speechSynthesis.onvoiceschanged=()=>{};
render();
</script></body></html>"""

def _build_fc_html(deck_json: str) -> str:
    return _FC_TEMPLATE.replace("__DECK_JSON__", deck_json)


# --- Logo ---
_logo_path = os.path.join(_HERE, "static", "logo.png")
_logo_uri = None
if os.path.exists(_logo_path):
    with open(_logo_path, "rb") as _f:
        _logo_uri = "data:image/png;base64," + base64.b64encode(_f.read()).decode()

# --- Page config ---
st.set_page_config(
    page_title="Kanji Hub — Tra Cứu Kanji Nhật, Nghĩa Tiếng Việt, Luyện Viết",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

if _logo_uri:
    st.logo(_logo_uri, icon_image=_logo_uri)

# --- SEO meta tags ---
st.markdown("""
<meta name="description" content="Kanji Hub — Tra cứu Kanji tiếng Nhật nhanh chóng, xem nghĩa tiếng Việt, âm Hán Việt, cách đọc, bảng luyện viết PDF. Hỗ trợ AI Gemini phân tích sâu.">
<meta name="keywords" content="kanji, học tiếng nhật, tra kanji, nghĩa tiếng việt, hán việt, jlpt, n5 n4 n3 n2 n1, luyện viết kanji, kanji hub">
<meta name="author" content="Kanji Hub">
<meta name="robots" content="index, follow">
<link rel="canonical" href="https://kanjihub.streamlit.app/">

<meta property="og:type" content="website">
<meta property="og:url" content="https://kanjihub.streamlit.app/">
<meta property="og:title" content="Kanji Hub — Tra Cứu Kanji Nhật · Nghĩa Tiếng Việt · Luyện Viết">
<meta property="og:description" content="Tra cứu Kanji tiếng Nhật nhanh chóng, xem nghĩa tiếng Việt, âm Hán Việt, cách đọc, bảng luyện viết PDF. Hỗ trợ AI phân tích sâu.">
<meta property="og:image" content="https://kanjihub.streamlit.app/app/static/logo.png">
<meta property="og:locale" content="vi_VN">
<meta property="og:site_name" content="Kanji Hub">

<meta name="twitter:card" content="summary">
<meta name="twitter:title" content="Kanji Hub — Tra Kanji Nhật · Tiếng Việt">
<meta name="twitter:description" content="Tra cứu Kanji tiếng Nhật, nghĩa tiếng Việt, luyện viết PDF. Hỗ trợ AI.">
<meta name="twitter:image" content="https://kanjihub.streamlit.app/app/static/logo.png">

<script type="application/ld+json">
{
  "@context": "https://schema.org",
  "@type": "WebApplication",
  "name": "Kanji Hub",
  "url": "https://kanjihub.streamlit.app/",
  "description": "Tra cứu Kanji tiếng Nhật, nghĩa tiếng Việt, âm Hán Việt, lộ trình học JLPT N5-N1, luyện viết PDF, phân tích AI.",
  "applicationCategory": "EducationApplication",
  "operatingSystem": "Web",
  "inLanguage": ["vi", "ja"],
  "offers": { "@type": "Offer", "price": "0", "priceCurrency": "VND" },
  "author": { "@type": "Organization", "name": "Kanji Hub" },
  "keywords": "kanji, tiếng nhật, jlpt, hán việt, luyện viết, tra cứu"
}
</script>
""", unsafe_allow_html=True)

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
.main .block-container { max-width: 900px; padding: 0 1.5rem 0.5rem; }
[data-testid="stMainBlockContainer"] { padding-top: 0 !important; padding-bottom: 0 !important; }
[data-testid="stMain"] { padding-bottom: 0 !important; }
[data-testid="stAppViewContainer"] > section { padding-bottom: 0 !important; }
[data-testid="stHeader"] { display: none !important; }
footer, [data-testid="stBottom"], [data-testid="stStatusWidget"],
[data-testid="stDecoration"], #stDecoration,
[data-testid="stToolbar"], [data-testid="stDeployButton"],
[data-testid="stAppDeployedOn"], [data-testid="stHostedBadge"],
.viewerBadge_container__r5tak, .viewerBadge_link__qRIco,
a[href="https://streamlit.io/cloud"], a[href*="share.streamlit.io"] { display: none !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
  background: #f0e8d8;
  border-right: 2px solid #c8a45a66;
}
[data-testid="stSidebar"] * { color: #3a2a1a !important; }
[data-testid="stSidebar"] input { background: #fff !important; color: #1a1209 !important; }

/* ── Site Header (sticky top bar) ── */
.site-header {
  display: flex; align-items: center; justify-content: space-between;
  padding: 10px 24px; margin-bottom: 8px;
  background: linear-gradient(135deg, #fff8f0 0%, #fdf4e8 100%);
  border-bottom: 2px solid #c0392b22;
  border-radius: 0 0 12px 12px;
  box-shadow: 0 2px 12px rgba(192,57,43,.07);
}
.site-header-left { display: flex; align-items: center; gap: 12px; }
.site-header-logo { width: 44px; height: 44px; border-radius: 50%; box-shadow: 0 2px 8px rgba(192,57,43,.3); flex-shrink: 0; }
.site-header-name { font-size: 1.25rem; font-weight: 900; color: #1a1209; letter-spacing: 3px; font-family: Georgia, 'Noto Serif JP', serif; }
.site-header-sub  { font-size: .72rem; color: #9a8a70; letter-spacing: 1.5px; display: block; margin-top: 1px; }
.site-header-right { display: flex; gap: 16px; align-items: center; }
.site-header-link { font-size: .78rem; color: #7a5a40; text-decoration: none; letter-spacing: .5px; font-weight: 600; }
.site-header-link:hover { color: #c0392b; }
.site-header-badge {
  font-size: .65rem; background: #c0392b; color: #fff;
  border-radius: 20px; padding: 2px 10px; font-weight: 700; letter-spacing: .5px;
}

/* ── Site Footer ── */
.site-footer {
  margin-top: 16px; padding: 16px 24px 14px;
  background: linear-gradient(160deg, #fff8f0 0%, #fdf4e8 100%);
  border-top: 2px solid #c0392b22;
  border-radius: 12px 12px 0 0;
  text-align: center;
}
.site-footer-logo { display: flex; align-items: center; justify-content: center; gap: 10px; margin-bottom: 10px; }
.site-footer-logo img { width: 36px; height: 36px; border-radius: 50%; box-shadow: 0 2px 6px rgba(192,57,43,.25); }
.site-footer-logo span { font-size: 1.05rem; font-weight: 900; color: #1a1209; letter-spacing: 3px; font-family: Georgia, serif; }
.site-footer-links { display: flex; justify-content: center; gap: 20px; margin: 8px 0 12px; flex-wrap: wrap; }
.site-footer-links a { font-size: .78rem; color: #7a5a40; text-decoration: none; font-weight: 600; }
.site-footer-links a:hover { color: #c0392b; }
.site-footer-divider { border: none; border-top: 1px solid #e0d4be; margin: 10px 0; }
.site-footer-copy { font-size: .72rem; color: #a89a80; letter-spacing: .5px; }
.site-footer-jp { font-size: .8rem; color: #c0392b55; letter-spacing: 6px; margin-top: 6px; font-family: 'Noto Serif JP', serif; }

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

/* ── Hero Section ── */
.hero {
  text-align: center;
  padding: 2.8rem 1rem 2.2rem;
  background: linear-gradient(160deg, #fffdf7 0%, #fdf6e8 50%, #fffaf2 100%);
  border-radius: 4px; margin-bottom: 1.6rem;
  border-top: 4px solid #c0392b;
  box-shadow: 0 2px 24px rgba(192,57,43,.08), inset 0 0 80px rgba(184,144,42,.04);
  position: relative; overflow: hidden;
}
.hero::before {
  content: "禅　道　山　月　風　花　竹　雪　龍　波　松　梅";
  position: absolute; top: 0; left: 0; right: 0; bottom: 0;
  display: flex; align-items: center; justify-content: center;
  font-size: 5rem; color: rgba(184,144,42,.04); letter-spacing: 16px;
  pointer-events: none; font-family: 'Noto Serif JP', serif;
  font-weight: 900; white-space: nowrap; overflow: hidden;
}
.hero::after {
  content: "";
  position: absolute; left: 0; top: 0; bottom: 0; width: 3px;
  background: linear-gradient(180deg, #c0392b 0%, rgba(192,57,43,0) 100%);
}
.hero-jp {
  font-size: .88rem; color: #c0392b; letter-spacing: 10px;
  font-family: 'Noto Serif JP', serif; display: block; margin-bottom: 14px;
  opacity: .85;
}
.hero-title {
  font-size: 2.5rem; font-weight: 900; color: #1a0e06;
  letter-spacing: 1px; line-height: 1.25; margin: 0 0 10px;
  font-family: Georgia, 'Noto Serif JP', serif;
}
.hero-sub {
  font-size: .88rem; color: #7a6050; margin: 6px 0 22px; letter-spacing: 1px;
  font-family: 'Noto Serif JP', serif;
}
.hero-cta {
  display: inline-block;
  background: linear-gradient(135deg, #e84040 0%, #b8200e 100%);
  color: #fff !important; font-weight: 700; font-size: .95rem;
  padding: 12px 32px; border-radius: 50px; letter-spacing: 1px;
  box-shadow: 0 4px 16px rgba(192,57,43,.3); text-decoration: none !important;
  border: none; cursor: pointer; margin-bottom: 6px;
}
.hero-kanji-block {
  display: inline-flex; align-items: stretch; gap: 0;
  background: #fffdf7;
  border: 1px solid #c8a87a;
  border-radius: 2px;
  margin-top: 20px; overflow: hidden;
  box-shadow: 4px 4px 0 rgba(192,57,43,.08), 0 6px 24px rgba(0,0,0,.10);
  max-width: 600px; width: 100%; text-align: left;
}
.hero-kanji-left {
  display: flex; flex-direction: column; align-items: center; justify-content: center;
  background: linear-gradient(170deg, #180808 0%, #2a0e08 100%);
  border-right: 1px solid #c8a87a;
  padding: 22px 26px; min-width: 96px; position: relative;
}
.hero-kanji-left::before {
  content: ''; position: absolute;
  inset: 6px; border: 1px solid rgba(192,57,43,.25); pointer-events: none;
}
.hero-kanji-char {
  font-size: 3.8rem; font-family: 'Noto Serif JP', serif;
  color: #fff8f0; line-height: 1;
  text-shadow: 0 2px 16px rgba(192,57,43,.5), 0 0 40px rgba(255,220,160,.15);
}
.hero-kanji-reading {
  font-size: .68rem; color: #e0a060; font-weight: 700;
  letter-spacing: 2px; margin-top: 8px; font-family: 'Noto Serif JP', serif;
}
.hero-kanji-right {
  display: flex; flex-direction: column; justify-content: center;
  padding: 18px 22px; text-align: left; flex: 1;
}
.hero-kanji-label {
  font-size: .58rem; color: #c0392b; font-weight: 700;
  letter-spacing: 3px; display: block; margin-bottom: 6px;
  font-family: 'Noto Serif JP', serif; text-transform: uppercase;
}
.hero-kanji-mean {
  font-size: 1rem; color: #1a0e06; font-weight: 700;
  margin-bottom: 12px; font-family: 'Noto Serif JP', Georgia, serif;
}
.hero-kanji-quote {
  font-size: .8rem; color: #3a2416;
  font-family: 'Noto Serif JP', Georgia, serif;
  line-height: 1.9; padding: 10px 0 0 14px;
  border-top: 1px solid #d4b896;
  border-left: 3px solid #c0392b;
  font-style: normal;
}
.hero-kanji-quote-author {
  display: block; font-size: .7rem; color: #9a7a5a;
  margin-top: 7px; text-align: right; padding-right: 2px;
  font-family: 'Noto Serif JP', serif; letter-spacing: .5px;
}

/* ── Feature Cards ── */
.feature-grid {
  display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px;
  margin: 0 0 1.4rem;
}
.feature-card {
  background: #fff; border: 1.5px solid #e0d4be;
  border-radius: 12px; padding: 18px 14px 14px;
  text-align: center; transition: all .18s ease;
  box-shadow: 0 2px 8px rgba(0,0,0,.05);
}
.feature-card:hover {
  transform: translateY(-3px);
  box-shadow: 0 6px 20px rgba(192,57,43,.12);
  border-color: #c0392b44;
}
.feature-icon { font-size: 1.8rem; display: block; margin-bottom: 8px; }
.feature-title { font-size: .88rem; font-weight: 700; color: #1a1209; letter-spacing: .5px; }
.feature-desc  { font-size: .72rem; color: #9a8a70; margin-top: 4px; letter-spacing: .3px; }
.feature-badge {
  display: inline-block; font-size: .6rem; font-weight: 700;
  background: #c0392b; color: #fff; border-radius: 10px;
  padding: 1px 8px; margin-top: 5px; letter-spacing: .5px;
}

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
/* Tab 4 — Flash Card: vàng đồng */
div[data-testid="stRadio"] > div[role="radiogroup"] > label:nth-child(4),
div[data-testid="stRadio"] > div[role="radiogroup"] > label:nth-child(4) * {
  border-color: #c8a45a !important; color: #8a6c20 !important;
}
div[data-testid="stRadio"] > div[role="radiogroup"] > label:nth-child(4):has(input:checked) {
  background: #c8a45a !important; border-color: #c8a45a !important;
  box-shadow: 0 4px 14px rgba(200,164,90,.35) !important; transform: translateY(-1px) !important;
}
div[data-testid="stRadio"] > div[role="radiogroup"] > label:nth-child(4):has(input:checked) * { color: #fff !important; }
div[data-testid="stRadio"] > div[role="radiogroup"] > label:nth-child(4):hover:not(:has(input:checked)) { background: #fdf8ec !important; transform: translateY(-1px) !important; }
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
  background: #ffffff !important; border: 1px solid #e0d4be;
  border-bottom: 2px solid #c0392b55;
  border-radius: 4px; padding: 12px 14px; margin-bottom: 8px;
  box-shadow: 0 1px 4px rgba(0,0,0,.06); transition: box-shadow .2s, border-color .3s;
  cursor: pointer; user-select: none;
}
.vocab-card:hover { box-shadow: 0 3px 12px rgba(192,57,43,.15); border-color: #c0392b88; }
.vocab-card.vocab-speaking { border-color: #c0392b !important; box-shadow: 0 0 0 2px rgba(192,57,43,.25) !important; }
.vocab-tts-icon { font-size:.85rem; opacity:.45; margin-left:6px; transition:opacity .2s; vertical-align:middle; cursor:pointer; }
.vocab-tts-icon:hover { opacity:1; }
.vocab-word { cursor: pointer !important; }
.vocab-word:hover { color: #c0392b !important; text-decoration: underline dotted #c0392b99; }
#vocab-toast {
  position:fixed; bottom:28px; left:50%; transform:translateX(-50%) translateY(8px);
  background:#1a1209; color:#f7f2e8; padding:10px 24px 12px;
  border-radius:24px; z-index:9999; pointer-events:none;
  box-shadow:0 4px 20px rgba(0,0,0,.35); opacity:0;
  transition: opacity .22s, transform .22s;
  text-align:center; min-width:160px;
}
#vocab-toast.show { opacity:1; transform:translateX(-50%) translateY(0); }
#vocab-toast .toast-word { font-size:1.6rem; font-family:'Noto Serif JP',serif; font-weight:900; }
#vocab-toast .toast-kana { font-size:.95rem; color:#b8902a; margin-top:2px; }
#vocab-toast .toast-meaning { font-size:.82rem; color:#c8b89a; margin-top:3px; }
.vocab-word   { font-size: 1.4rem !important; font-weight: 900 !important; color: #1a1209 !important; font-family: 'Noto Serif JP', serif !important; }
.vocab-kana   { font-size: .84rem !important; color: #b8902a !important; }
.vocab-hanviet { font-size: .76rem !important; color: #9a8a6a !important; font-style: italic !important; }
.vocab-meaning { font-size: .92rem !important; color: #3a2a1a !important; margin-top: 4px !important; }
.vocab-example { font-size: .8rem !important; color: #8a7a6a !important; font-style: italic !important; margin-top: 2px !important; }
/* Selectbox label trong tab 3 */
[data-testid="stSelectbox"] [data-testid="stWidgetLabel"] p { color: #5a4030 !important; }
/* Vocab card text override trong stMarkdownContainer */
[data-testid="stMarkdownContainer"] .vocab-word { color: #1a1209 !important; }
[data-testid="stMarkdownContainer"] .vocab-kana { color: #b8902a !important; }
[data-testid="stMarkdownContainer"] .vocab-hanviet { color: #9a8a6a !important; }
[data-testid="stMarkdownContainer"] .vocab-meaning { color: #3a2a1a !important; }
[data-testid="stMarkdownContainer"] .vocab-example { color: #8a7a6a !important; }

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
button[data-testid="baseButton-secondary"],
button[kind="secondary"],
[data-testid="stBaseButton-secondary"] {
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
button[data-testid="baseButton-secondary"]:hover,
[data-testid="stBaseButton-secondary"]:hover {
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
[data-testid="stWidgetLabel"],
[data-testid="stWidgetLabel"] p,
[data-testid="stWidgetLabel"] label {
  color: #5a4030 !important;
}
div[data-baseweb="select"] > div:first-child {
  background: #ffffff !important;
  border: 1.5px solid #e0d0be !important;
  border-radius: 50px !important;
  color: #3a2a1a !important;
  box-shadow: none !important;
  outline: none !important;
}
div[data-baseweb="select"] > div:first-child:focus,
div[data-baseweb="select"] > div:first-child:focus-within,
div[data-baseweb="select"] > div:first-child:focus-visible,
div[data-baseweb="select"]:focus-within > div:first-child {
  border-color: #e84040 !important;
  box-shadow: 0 0 0 3px rgba(232,64,64,.15) !important;
  outline: none !important;
}
div[data-baseweb="select"],
div[data-baseweb="select"] *,
div[data-baseweb="select"] input,
div[data-baseweb="select"] [data-testid="stSelectboxValue"],
div[data-baseweb="select"] span { color: #3a2a1a !important; background: transparent !important; }

/* Dropdown popup list */
div[data-baseweb="popover"],
div[data-baseweb="popover"] div[data-baseweb="menu"],
div[data-baseweb="popover"] ul[data-testid="stSelectboxVirtualDropdown"],
div[data-baseweb="popover"] li {
  background: #fff !important;
  border: 1px solid #e0d0be !important;
  border-radius: 12px !important;
  box-shadow: 0 8px 24px rgba(0,0,0,.12) !important;
  color: #3a2a1a !important;
}
div[data-baseweb="option"],
div[data-baseweb="option"] *,
div[data-baseweb="popover"] li,
div[data-baseweb="popover"] li * {
  background: #fff !important;
  color: #3a2a1a !important;
}
div[data-baseweb="option"]:hover,
div[data-baseweb="option"]:hover *,
div[data-baseweb="option"][aria-selected="true"],
div[data-baseweb="option"][aria-selected="true"] * {
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
[data-testid="stToolbar"] { display: none !important; }

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

# ── Site Header ───────────────────────────────────────────────────────────────
_logo_img = f'<img src="{_logo_uri}" class="site-header-logo">' if _logo_uri else '漢'
st.markdown(f"""
<div class="site-header">
  <div class="site-header-left">
    {_logo_img}
    <div>
      <span class="site-header-name">KANJI HUB</span>
      <span class="site-header-sub">漢字 · 学習 · ベトナム語</span>
    </div>
  </div>
  <div class="site-header-right">
    <span class="site-header-badge">JLPT N5→N1</span>
    <span style="font-size:.78rem;color:#9a8a70">Tra Kanji · Lộ trình · Từ vựng · PDF · AI</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Xóa badge "Hosted with Streamlit" bằng JS (CSS không đủ vì inject sau load) ──
_components.html("""<script>
(function() {
  var SELS = [
    '[data-testid="stDecoration"]',
    '[data-testid="stToolbar"]',
    '[data-testid="stDeployButton"]',
    '.viewerBadge_container__r5tak',
    '.viewerBadge_link__qRIco',
    '#stDecoration'
  ];
  function hideAll() {
    SELS.forEach(function(sel) {
      try {
        window.parent.document.querySelectorAll(sel).forEach(function(el) {
          el.style.setProperty('display', 'none', 'important');
        });
      } catch(e) {}
    });
  }
  hideAll();
  var obs = new MutationObserver(hideAll);
  try {
    obs.observe(window.parent.document.body, {childList: true, subtree: true});
  } catch(e) {}
})();
</script>""", height=0, scrolling=False)

# ── Custom tabs bằng radio (có thể control bằng session_state) ───────────────
TAB_NAMES = ["🔍 Tra Kanji", "📝 Tra Từ Vựng", "🗺️ Lộ trình học", "📖 Từ Vựng", "🃏 Flash Card"]

# ── Dialog phân tích AI cho từ vựng ───────────────────────────────────────────
@st.dialog("🤖 Phân tích từ vựng", width="large")
def _vocab_ai_dialog():
    _w  = st.session_state.get("_vl_sel_word", "")
    _rd = st.session_state.get("_vl_sel_reading", "")
    _hv = st.session_state.get("_vl_sel_hanviet", "")
    _mn = st.session_state.get("_vl_sel_meaning", "")
    _ex = st.session_state.get("_vl_sel_example", "")
    _vi = st.session_state.get("_vl_sel_exampleVi", "")
    st.markdown(f"""
<div style="display:flex;align-items:flex-start;gap:18px;margin-bottom:12px">
  <div style="font-family:'Noto Serif JP',serif;font-size:3rem;font-weight:900;
    color:#1a1209;line-height:1;min-width:64px;text-align:center">{_w}</div>
  <div>
    <div style="color:#b8902a;font-size:1rem;margin-bottom:2px">（{_rd}）
      <span style="color:#9a8a6a;font-size:.85rem;font-style:italic">{_hv}</span></div>
    <div style="color:#3a2a1a;font-size:1.05rem;font-weight:700">▸ {_mn}</div>
    {f'<div style="color:#6a5a4a;font-size:.9rem;margin-top:6px">📝 {_ex}</div>' if _ex else ""}
    {f'<div style="color:#8a7a6a;font-size:.85rem;font-style:italic">↳ {_vi}</div>' if _vi else ""}
  </div>
</div>
""", unsafe_allow_html=True)
    st.divider()
    _ai_key = f"_vl_ai_{_w}"
    if _ai_key not in st.session_state:
        with st.spinner("🔄 Đang phân tích AI…"):
            try:
                st.session_state[_ai_key] = analyze_kanji_ai(_w)
            except Exception as _e:
                st.session_state[_ai_key] = f"⚠️ Lỗi phân tích: {_e}"
    st.markdown(
        f'<div style="background:#fdf8f0;border:1px solid #e0d4be;border-radius:8px;'
        f'padding:16px 18px;font-size:.9rem;color:#3a2a1a;line-height:1.75">'
        f'{st.session_state[_ai_key]}</div>',
        unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        if st.button("🔄 Phân tích lại", key="_vl_reanalyze", use_container_width=True):
            if _ai_key in st.session_state:
                del st.session_state[_ai_key]
            st.rerun()
    with c2:
        if st.button("✕ Đóng", key="_vl_close_dlg", use_container_width=True):
            st.session_state.pop("_vl_sel_word", None)
            st.rerun()

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
    # Kanji ngẫu nhiên của ngày
    import random as _random
    _kanji_pool = [
        ("夢", "Mộng", "Giấc mơ, ước mơ"),
        ("心", "Tâm", "Tâm hồn, con tim"),
        ("道", "Đạo", "Con đường, đạo lý"),
        ("力", "Lực", "Sức mạnh, năng lực"),
        ("愛", "Ái", "Tình yêu, yêu thương"),
        ("学", "Học", "Học hỏi, học tập"),
        ("山", "Sơn", "Núi, ngọn núi"),
        ("花", "Hoa", "Hoa, bông hoa"),
        ("月", "Nguyệt", "Mặt trăng, tháng"),
        ("風", "Phong", "Gió, phong cách"),
        ("火", "Hỏa", "Lửa, ngọn lửa"),
        ("水", "Thủy", "Nước, dòng chảy"),
        ("空", "Không", "Bầu trời, khoảng trống"),
        ("光", "Quang", "Ánh sáng, rực rỡ"),
        ("時", "Thời", "Thời gian, thời đại"),
        ("命", "Mệnh", "Sinh mệnh, số phận"),
        ("人", "Nhân", "Con người, nhân loại"),
        ("和", "Hòa", "Hòa bình, hài hòa"),
        ("信", "Tín", "Lòng tin, chữ tín"),
        ("勇", "Dũng", "Dũng cảm, can đảm"),
        ("静", "Tĩnh", "Yên tĩnh, bình thản"),
        ("忍", "Nhẫn", "Nhẫn nại, kiên nhẫn"),
        ("義", "Nghĩa", "Chính nghĩa, đạo nghĩa"),
        ("知", "Tri", "Tri thức, hiểu biết"),
        ("友", "Hữu", "Bạn bè, hữu nghị"),
        ("希", "Hi", "Hi vọng, ước muốn"),
        ("自", "Tự", "Bản thân, tự do"),
        ("真", "Chân", "Sự thật, chân thực"),
        ("誠", "Thành", "Thành thật, chân thành"),
        ("幸", "Hạnh", "Hạnh phúc, may mắn"),
        ("徳", "Đức", "Đức hạnh, phẩm giá"),
        ("礼", "Lễ", "Lễ phép, kính trọng"),
        ("親", "Thân", "Cha mẹ, thân thiết"),
        ("健", "Kiện", "Sức khỏe, lành mạnh"),
        ("福", "Phúc", "Phúc lộc, may mắn"),
        ("善", "Thiện", "Điều tốt, lòng nhân"),
        ("美", "Mỹ", "Vẻ đẹp, tươi đẹp"),
        ("強", "Cường", "Mạnh mẽ, kiên cường"),
        ("努", "Nỗ", "Nỗ lực, cố gắng"),
        ("仁", "Nhân", "Lòng nhân, từ bi"),
        ("敬", "Kính", "Kính trọng, tôn kính"),
        ("恩", "Ân", "Ân nghĩa, biết ơn"),
        ("勤", "Cần", "Cần cù, siêng năng"),
        ("謙", "Khiêm", "Khiêm tốn, nhún nhường"),
    ]
    _kanji_quotes = {
        "夢": ("Người không có ước mơ thì chẳng khác nào đang ngủ. Hãy mơ lớn và dám theo đuổi nó đến cùng.", "Aristotle"),
        "心": ("Nơi nào có tâm, nơi đó có đường. Trái tim chân thật là la bàn dẫn lối khi lý trí mù quáng.", "Tục ngữ Nhật"),
        "道": ("Đường đi không khó vì ngăn sông cách núi, mà khó vì lòng người ngại núi e sông.", "Nguyễn Bá Học"),
        "力": ("Sức mạnh không đến từ thể xác — nó đến từ ý chí bất khuất. Kẻ mạnh nhất là kẻ chiến thắng chính mình.", "Mahatma Gandhi"),
        "愛": ("Yêu thương là cho đi mà không cần nhận lại. Tình yêu thật sự không đo bằng lời nói mà bằng hành động.", "Tục ngữ"),
        "学": ("Học, học nữa, học mãi. Tri thức là thứ duy nhất càng chia sẻ lại càng sinh sôi.", "Lenin"),
        "山": ("Núi càng cao, tầm nhìn càng rộng. Gian khổ càng lớn, ý chí càng bền — đừng sợ leo dốc.", "Tục ngữ"),
        "花": ("Hoa đẹp không chỉ ở sắc, mà còn ở chỗ biết nở đúng lúc. Mỗi người có một mùa xuân riêng.", "Tục ngữ Nhật"),
        "月": ("Dù đêm tối đến đâu, mặt trăng vẫn luôn tỏa sáng trên cao — hy vọng chưa bao giờ tắt hẳn.", "Tục ngữ"),
        "風": ("Gió không để lại vết, nhưng ta biết gió đã qua vì cây rung. Ảnh hưởng của bạn lớn hơn bạn nghĩ.", "Thiền ngữ"),
        "火": ("Lửa thử vàng, gian nan thử sức. Không có áp lực, kim cương chỉ là mảnh than đá.", "Tục ngữ"),
        "水": ("Nước mềm nhưng đục được đá. Sự kiên trì lặng lẽ luôn thắng sức mạnh hung hăng.", "Lão Tử"),
        "空": ("Bầu trời không giới hạn — ước mơ của bạn cũng vậy. Chỉ có tư duy mới là rào cản thật sự.", "Thiền ngữ"),
        "光": ("Thắp một ngọn nến còn hơn ngàn lần nguyền rủa bóng tối. Hãy là ánh sáng bạn muốn thấy.", "Khổng Tử"),
        "時": ("Thời gian là thứ công bằng nhất — ai cũng có 24 giờ. Sự khác biệt nằm ở cách bạn dùng nó.", "Tục ngữ"),
        "命": ("Cuộc đời không phải là những gì xảy ra với bạn, mà là những gì bạn làm với những gì xảy ra.", "Viktor Frankl"),
        "人": ("Con người là thước đo của vạn vật. Hãy đối xử với người khác như bạn muốn được đối xử.", "Protagoras"),
        "和": ("Hòa thuận là gốc rễ của hạnh phúc. Một gia đình hòa thuận, nghìn việc thành công.", "Tục ngữ Nhật"),
        "信": ("Mất tiền mất ít, mất danh dự mất nhiều, mất chữ tín mất tất cả.", "Tục ngữ"),
        "勇": ("Dũng cảm không phải là không sợ hãi — mà là tiến bước dù đang sợ. Hành động mới tạo ra dũng khí.", "Mark Twain"),
        "静": ("Trong tĩnh lặng ta nghe được tiếng lòng. Tâm bình thì thế giới bình.", "Thiền ngữ"),
        "忍": ("Nhẫn một thời khắc nóng giận, tránh trăm ngày hối hận. Kiên nhẫn là mẹ của mọi đức hạnh.", "Tục ngữ Nhật"),
        "義": ("Sống có nghĩa khí, dẫu chết vẫn thơm. Làm điều đúng dù không ai nhìn — đó mới là đạo nghĩa.", "Tục ngữ"),
        "知": ("Biết mình biết người, trăm trận trăm thắng. Nhưng trí tuệ thật sự bắt đầu từ việc thừa nhận sự thiếu hiểu biết.", "Tôn Tử"),
        "友": ("Bạn tốt là người nói thật khi bạn cần nghe, không phải khi bạn muốn nghe.", "Tục ngữ"),
        "希": ("Hy vọng không phải là chiến lược, nhưng không có hy vọng thì không có chiến lược nào đứng vững.", "Napoleon"),
        "自": ("Tự do thật sự không phải là làm những gì bạn muốn, mà là muốn những điều đúng đắn.", "Jean-Paul Sartre"),
        "真": ("Sự thật đôi khi đau, nhưng dối trá còn đau hơn về lâu dài. Hãy sống thật.", "Tục ngữ"),
        "誠": ("Thành thật là nền tảng của mọi mối quan hệ bền vững. Một lời nói thật hơn ngàn lời hoa mỹ.", "Khổng Tử"),
        "幸": ("Hạnh phúc không phải là đích đến — nó là cách ta đi. Tìm niềm vui trong từng khoảnh khắc nhỏ.", "Thiền ngữ"),
        "徳": ("Đức hạnh là phần thưởng riêng của nó. Người có đức tự nhiên được kính trọng dù không cầu.", "Khổng Tử"),
        "礼": ("Lễ phép không tốn tiền nhưng mua được tất cả. Một lời kính trọng có thể mở mọi cánh cửa.", "Tục ngữ"),
        "親": ("Công cha như núi Thái Sơn, nghĩa mẹ như nước trong nguồn. Hãy hiếu thảo khi còn có thể.", "Ca dao Việt Nam"),
        "健": ("Sức khỏe là vàng. Không có sức khỏe, mọi của cải đều vô nghĩa — hãy giữ gìn thân tâm.", "Tục ngữ"),
        "福": ("Phúc không cầu mà đến, họa không muốn mà vào — tất cả đều từ tâm mà ra.", "Thiền ngữ"),
        "善": ("Làm điều thiện không cần người biết — đất trời đã biết. Gieo thiện, gặt thiện.", "Tục ngữ"),
        "美": ("Cái đẹp thật sự không ở bề ngoài mà ở tâm hồn. Người đẹp vì tính nết, không vì son phấn.", "Tục ngữ Nhật"),
        "強": ("Mạnh không phải là không vấp ngã, mà là đứng dậy sau mỗi lần ngã. Người kiên cường không bao giờ bỏ cuộc.", "Tục ngữ"),
        "努": ("Thiên tài là 1% cảm hứng và 99% mồ hôi. Nỗ lực không bao giờ phản bội người kiên trì.", "Thomas Edison"),
        "仁": ("Điều mình không muốn, chớ làm cho người. Nhân nghĩa là gốc rễ của mọi đức hạnh.", "Khổng Tử"),
        "敬": ("Kính già yêu trẻ — đó là đạo làm người. Ai kính trọng người khác sẽ được kính trọng lại.", "Tục ngữ"),
        "恩": ("Ân nghĩa phải nhớ, oán thù nên quên. Người biết ơn là người có trái tim cao thượng.", "Tục ngữ Nhật"),
        "勤": ("Cần cù bù thông minh. Siêng năng là chìa khóa mở mọi cánh cửa thành công.", "Tục ngữ"),
        "謙": ("Bông lúa càng chắc hạt càng cúi đầu. Người tài thật sự không cần khoe — tài năng tự nói lên.", "Tục ngữ"),
    }
    if "_hero_kanji" not in st.session_state:
        st.session_state["_hero_kanji"] = _random.choice(_kanji_pool)
    _k, _hv, _mean = st.session_state["_hero_kanji"]
    _quote, _qauthor = _kanji_quotes.get(_k, ("Mỗi chữ kanji là một cánh cửa dẫn đến văn hóa Nhật Bản.", "KanjiHub"))

    st.markdown(f'''
<div class="hero">
  <span class="hero-jp">日本語を勉強しましょう</span>
  <div class="hero-title">Tra Kanji · Học Tiếng Nhật</div>
  <p class="hero-sub">Nghĩa tiếng Việt · Âm Hán Việt · Lộ trình JLPT · Luyện viết PDF · AI phân tích</p>
  <div class="hero-kanji-block">
    <div class="hero-kanji-left">
      <span class="hero-kanji-char">{_k}</span>
      <span class="hero-kanji-reading">{_hv}</span>
    </div>
    <div class="hero-kanji-right">
      <span class="hero-kanji-label">✦ 今日の漢字 · KANJI HÔM NAY</span>
      <span class="hero-kanji-mean">{_mean}</span>
      <div class="hero-kanji-quote">「{_quote}」<span class="hero-kanji-quote-author">― {_qauthor}</span></div>
    </div>
  </div>
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

# === TAB 1b: Tra Từ Vựng ===
elif active_tab == TAB_NAMES[1]:
    _lv = _logo_uri
    st.markdown(
        f'''<div style="display:flex;align-items:center;gap:16px;margin-bottom:16px">
  {'<img src="' + _lv + '" style="width:56px;height:56px;border-radius:50%;box-shadow:0 3px 12px rgba(192,57,43,.28);flex-shrink:0">' if _lv else ''}
  <div>
    <div style="font-size:1.4rem;font-weight:900;color:#1a1209;letter-spacing:2px;line-height:1.2">📚 Tra Từ Vựng</div>
    <div style="color:#9a8a70;font-size:.8rem;letter-spacing:1px;margin-top:2px">Nguồn: Mazii · Jisho · AI phân tích</div>
  </div>
</div>''', unsafe_allow_html=True)

    with st.form("vocab_search_form"):
        _vc1, _vc2 = st.columns([5, 1])
        with _vc1:
            _vq = st.text_input("Nhập từ", placeholder="Nhập từ tiếng Nhật: 食べる、会議、間に合う…",
                                label_visibility="collapsed", key="vocab_query")
        with _vc2:
            _vsub = st.form_submit_button("🔍 Tra", use_container_width=True, type="primary")

    if _vsub and _vq.strip():
        _vq = _vq.strip()
        st.session_state["vocab_last_query"] = _vq

    _vq_active = st.session_state.get("vocab_last_query", "")
    if _vq_active:
        _vq = _vq_active
        _vcache_key = f"vocab_res_{_vq}"
        if _vsub or _vcache_key not in st.session_state:
            if _vsub:
                with st.spinner(f"Đang tra **{_vq}**…"):
                    st.session_state[_vcache_key] = lookup_vocab(_vq)
        _vr = st.session_state.get(_vcache_key, {})

        if _vr:
            _vsrc = _vr.get("source", "")
            _vsrc_label = {"mazii": "🇻🇳 Mazii", "jisho": "⚡ Jisho", "gemini": "✨ Gemini AI", "openrouter": "🤖 OpenRouter AI"}.get(_vsrc, _vsrc)
            _vreading = _vr.get('reading', '') or _vr.get('word', '')
            import html as _html
            _vreading_safe = _html.escape(_vreading, quote=True)
            _meanings_html = ''.join(f'<div style="color:#3a2a1a;font-size:.95rem;margin-bottom:4px">▸ {_html.escape(m)}</div>' for m in _vr.get('meanings_vi', []))
            st.markdown(f"""
<div style="background:#fff;border:1.5px solid #e0d4be;border-radius:10px;padding:20px 24px 16px;margin-bottom:12px">
  <div style="display:flex;align-items:center;gap:14px;flex-wrap:wrap;margin-bottom:8px">
    <span style="font-family:'Noto Serif JP',serif;font-size:2.6rem;font-weight:900;color:#1a1209">{_html.escape(_vr.get('word',''))}</span>
    <span style="color:#b8902a;font-size:1.1rem">（{_html.escape(_vr.get('reading',''))}）</span>
    <span style="color:#9a8a6a;font-size:.85rem;font-style:italic">{_html.escape(_vr.get('han_viet',''))}</span>
    <span style="background:#e8f4e8;color:#2d6e4a;border:1px solid #b0d4b0;border-radius:12px;
      padding:2px 10px;font-size:.75rem;font-weight:700">{_vsrc_label}</span>
  </div>
  <div style="border-top:1px solid #e0d4be;padding-top:10px">
    {_meanings_html}
  </div>
</div>""", unsafe_allow_html=True)
            # TTS button dùng components.html để JS chạy được (st.markdown block onclick)
            _components.html(f"""
<button onclick="speak()" style="background:#b8902a;color:#fff;border:none;border-radius:20px;
  padding:7px 18px;font-size:.88rem;cursor:pointer;font-family:sans-serif;margin-top:2px">
  🔊 Phát âm
</button>
<script>
function speak(){{
  try{{
    var s=window.parent.speechSynthesis||window.speechSynthesis;
    var U=window.parent.SpeechSynthesisUtterance||window.SpeechSynthesisUtterance;
    s.cancel();
    var u=new U('{_vreading_safe}');
    u.lang='ja-JP'; u.rate=0.85;
    var vv=s.getVoices();
    var jv=vv.find(function(v){{return v.lang==='ja-JP';}});
    if(jv) u.voice=jv;
    s.speak(u);
  }}catch(e){{}}
}}
</script>
""", height=46, scrolling=False)

            if _vr.get("examples"):
                st.markdown("**📝 Câu ví dụ:**")
                for _ex in _vr["examples"]:
                    st.markdown(f"""
<div style="background:#fdf8f0;border-left:3px solid #b8902a;padding:8px 14px;margin-bottom:6px;border-radius:0 6px 6px 0">
  <div style="font-family:'Noto Serif JP',serif;font-size:1rem;color:#1a1209">{_ex.get('sentence','')}</div>
  {f'<div style="font-size:.82rem;color:#b8902a">({_ex.get("reading","")})</div>' if _ex.get('reading') else ''}
  <div style="font-size:.88rem;color:#5a4a3a;font-style:italic">↳ {_ex.get('meaning','')}</div>
</div>""", unsafe_allow_html=True)

            if _vr.get("related"):
                st.markdown("**🔗 Từ liên quan:**")
                _rcols = st.columns(min(len(_vr["related"]), 4))
                for _ri, (_rw, _rr, _rm) in enumerate(_vr["related"]):
                    with _rcols[_ri % len(_rcols)]:
                        st.markdown(f"""
<div style="background:#fff;border:1px solid #e0d4be;border-radius:6px;padding:8px 10px;text-align:center">
  <div style="font-family:'Noto Serif JP',serif;font-size:1.2rem;font-weight:900;color:#1a1209">{_rw}</div>
  <div style="font-size:.75rem;color:#b8902a">{_rr}</div>
  <div style="font-size:.78rem;color:#5a4a3a">{_rm}</div>
</div>""", unsafe_allow_html=True)

            # Phân tích AI thêm
            st.divider()
            _vai_key = f"vocab_ai_{_vq}"
            if st.button("🤖 Phân tích AI chuyên sâu", key="vocab_ai_btn"):
                with st.spinner("Đang phân tích…"):
                    st.session_state[_vai_key] = analyze_kanji_ai(_vq)
            if _vai_key in st.session_state:
                st.markdown(
                    f'<div style="background:#fdf8f0;border:1px solid #e0d4be;border-radius:8px;'
                    f'padding:16px;font-size:.9rem;color:#3a2a1a;line-height:1.75">'
                    f'{st.session_state[_vai_key]}</div>', unsafe_allow_html=True)
        else:
            st.warning(f"Không tìm thấy từ **{_vq}** trong Mazii/Jisho. Thử dùng AI phân tích:")
            _vai_key = f"vocab_ai_{_vq}"
            if st.button("🤖 Phân tích bằng AI", key="vocab_ai_fallback_btn", type="primary"):
                with st.spinner("Đang phân tích…"):
                    st.session_state[_vai_key] = analyze_kanji_ai(_vq)
            if _vai_key in st.session_state:
                st.markdown(
                    f'<div style="background:#fdf8f0;border:1px solid #e0d4be;border-radius:8px;'
                    f'padding:16px;font-size:.9rem;color:#3a2a1a;line-height:1.75">'
                    f'{st.session_state[_vai_key]}</div>', unsafe_allow_html=True)

# === TAB 2: Lộ trình ===
elif active_tab == TAB_NAMES[2]:
    st.markdown(
        f'''
<div style="display:flex;align-items:center;gap:16px;margin-bottom:16px">
  {'<img src="' + _logo_uri + '" style="width:56px;height:56px;border-radius:50%;box-shadow:0 3px 12px rgba(192,57,43,.28);flex-shrink:0">' if _logo_uri else ''}
  <div>
    <div style="font-size:1.4rem;font-weight:900;color:#1a1209;letter-spacing:2px;line-height:1.2">🗺️ Lộ trình học Kanji</div>
    <div style="color:#9a8a70;font-size:.8rem;letter-spacing:1px;margin-top:2px">Chọn cấp độ → tra từng bài</div>
  </div>
</div>''',
        unsafe_allow_html=True
    )
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
elif active_tab == TAB_NAMES[3]:
    st.markdown(
        f'''
<div style="display:flex;align-items:center;gap:16px;margin-bottom:16px">
  {'<img src="' + _logo_uri + '" style="width:56px;height:56px;border-radius:50%;box-shadow:0 3px 12px rgba(192,57,43,.28);flex-shrink:0">' if _logo_uri else ''}
  <div>
    <div style="font-size:1.4rem;font-weight:900;color:#1a1209;letter-spacing:2px;line-height:1.2">📖 Từ Vựng theo Bài</div>
    <div style="color:#9a8a70;font-size:.8rem;letter-spacing:1px;margin-top:2px">Học từ vựng theo chủ đề</div>
  </div>
</div>''',
        unsafe_allow_html=True
    )
    if not VOCAB_LESSONS:
        st.info("Chưa có bài từ vựng nào.")
    else:
        # Mở dialog nếu đã chọn từ để phân tích AI
        if st.session_state.get("_vl_sel_word"):
            _vocab_ai_dialog()

        # ── Thanh tìm kiếm từ vựng ──
        _vl_search = st.text_input(
            "🔍 Tìm từ vựng",
            placeholder="Nhập kanji, hiragana, Hán Việt hoặc nghĩa tiếng Việt…",
            key="vl_search",
            label_visibility="collapsed",
        )

        if _vl_search.strip():
            # Gom tất cả từ + kèm số bài
            _all_words = []
            for _ln, _lwords in VOCAB_LESSONS.items():
                for _it in _lwords:
                    _all_words.append((_ln, _it))
            _q = _vl_search.strip().lower()
            _hits = [
                (_ln, _it) for _ln, _it in _all_words
                if _q in _it.get("word", "").lower()
                or _q in _it.get("reading", "").lower()
                or _q in _it.get("hanviet", "").lower()
                or _q in _it.get("meaning", "").lower()
                or _q in _it.get("example", "").lower()
                or _q in _it.get("exampleVi", "").lower()
            ]
            st.caption(f"🔎 Tìm thấy **{len(_hits)}** từ")
            if _hits:
                for _sri in range(0, len(_hits), 2):
                    _scols = st.columns(2)
                    for _sci in range(2):
                        _si = _sri + _sci
                        if _si >= len(_hits):
                            break
                        _sl, _sit = _hits[_si]
                        _w  = _sit["word"]
                        _rd = _sit.get("reading", "")
                        _hv = _sit.get("hanviet", "")
                        _mn = _sit.get("meaning", "")
                        with _scols[_sci]:
                            st.markdown(f"""
<div class="vocab-card"
     data-word="{_w}"
     data-reading="{_rd}"
     data-meaning="{_mn}">
  <div style="font-size:.7rem;color:#c0392b;font-weight:700;margin-bottom:2px">Bài {_sl}</div>
  <div class="vocab-word">{_w}<span class="vocab-tts-icon">🔊</span></div>
  <div class="vocab-kana">（{_rd}）<span class="vocab-hanviet">{_hv}</span></div>
  <div class="vocab-meaning">▸ {_mn}</div>
  {"<div class='vocab-example'>📝 " + _sit['example'] + "</div>" if _sit.get('example') else ""}
  {"<div class='vocab-example'>↳ " + _sit['exampleVi'] + "</div>" if _sit.get('exampleVi') else ""}
</div>""", unsafe_allow_html=True)
                            if st.button("⠀", key=f"vl_ai_s_{_sl}_{_si}", use_container_width=True):
                                st.session_state["_vl_sel_word"]      = _w
                                st.session_state["_vl_sel_reading"]   = _rd
                                st.session_state["_vl_sel_hanviet"]   = _hv
                                st.session_state["_vl_sel_meaning"]   = _mn
                                st.session_state["_vl_sel_example"]   = _sit.get("example", "")
                                st.session_state["_vl_sel_exampleVi"] = _sit.get("exampleVi", "")
                                st.rerun()
            else:
                st.info("Không tìm thấy từ nào phù hợp.")
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
                        _w  = item['word']
                        _rd = item.get('reading', '')
                        _hv = item.get('hanviet', '')
                        _mn = item.get('meaning', '')
                        st.markdown(f"""
<div class="vocab-card"
     data-word="{_w}"
     data-reading="{_rd}"
     data-meaning="{_mn}">
  <div class="vocab-word">{_w}<span class="vocab-tts-icon">🔊</span></div>
  <div class="vocab-kana">（{_rd}）
    <span class="vocab-hanviet">{_hv}</span>
  </div>
  <div class="vocab-meaning">▸ {_mn}</div>
  {"<div class='vocab-example'>📝 " + item['example'] + "</div>" if item.get('example') else ""}
  {"<div class='vocab-example'>↳ " + item['exampleVi'] + "</div>" if item.get('exampleVi') else ""}
</div>""", unsafe_allow_html=True)
                        # Nút ẩn — JS click vào chữ kanji sẽ kích hoạt nút này
                        if st.button("⠀",
                                     key=f"vl_ai_{sel_lesson}_{wi}",
                                     use_container_width=True):
                            st.session_state["_vl_sel_word"]      = _w
                            st.session_state["_vl_sel_reading"]   = _rd
                            st.session_state["_vl_sel_hanviet"]   = _hv
                            st.session_state["_vl_sel_meaning"]   = _mn
                            st.session_state["_vl_sel_example"]   = item.get('example', '')
                            st.session_state["_vl_sel_exampleVi"] = item.get('exampleVi', '')
                            st.rerun()

        # ── Inject TTS click-to-read script ──
        _components.html("""
<script>
(function(){
  var jaVoice = null;
  function loadVoice(){
    var voices = window.speechSynthesis.getVoices();
    jaVoice = voices.find(function(v){ return v.lang.startsWith('ja'); }) || null;
  }
  window.speechSynthesis.addEventListener('voiceschanged', loadVoice);
  loadVoice();

  function speak(text){
    window.speechSynthesis.cancel();
    var u = new SpeechSynthesisUtterance(text);
    u.lang = 'ja-JP'; u.rate = 0.85;
    if(jaVoice) u.voice = jaVoice;
    window.speechSynthesis.speak(u);
  }

  function showToast(word, kana, meaning){
    var doc = window.parent.document;
    var t = doc.getElementById('vocab-toast');
    if(!t){
      t = doc.createElement('div'); t.id='vocab-toast';
      doc.body.appendChild(t);
    }
    t.innerHTML = '<div class="toast-word">' + word + '</div>' +
                  '<div class="toast-kana">(' + kana + ')</div>' +
                  '<div class="toast-meaning">' + meaning + '</div>';
    t.classList.add('show');
    clearTimeout(t._hide);
    t._hide = setTimeout(function(){ t.classList.remove('show'); }, 2200);
  }

  function attachHandlers(){
    try{
      var doc = window.parent.document;

      // Dùng CSS :has() để ẩn nút trigger trong cột chứa vocab-card
      if(!doc.getElementById('vocab-btn-hide-style')){
        var s = doc.createElement('style');
        s.id = 'vocab-btn-hide-style';
        s.textContent = '[data-testid="stColumn"]:has(.vocab-card) [data-testid="stButton"],' +
                        '[data-testid="stColumn"]:has(.vocab-card) [data-testid="stBaseButton-secondary"]' +
                        '{ display:none !important; }';
        doc.head.appendChild(s);
      }

      doc.querySelectorAll('.vocab-card:not([data-tts])').forEach(function(card){
        card.setAttribute('data-tts','1');
        var word    = card.getAttribute('data-word') || '';
        var reading = card.getAttribute('data-reading') || '';
        var meaning = card.getAttribute('data-meaning') || '';

        // TTS: chỉ click vào icon 🔊
        var icon = card.querySelector('.vocab-tts-icon');
        if(icon){
          icon.addEventListener('click', function(e){
            e.stopPropagation();
            speak(reading || word);
            showToast(word, reading, meaning);
            card.classList.add('vocab-speaking');
            setTimeout(function(){ card.classList.remove('vocab-speaking'); }, 900);
          });
        }

        // AI Popup: click vào chữ kanji → tìm button trong stColumn cha
        var wordEl = card.querySelector('.vocab-word');
        if(wordEl){
          wordEl.addEventListener('click', function(e){
            e.stopPropagation();
            var col = card.closest('[data-testid="stColumn"]');
            if(col){
              var btn = col.querySelector('button');
              if(btn) btn.click();
            }
          });
        }
      });
    }catch(e){}
  }

  attachHandlers();
  var obs = new MutationObserver(attachHandlers);
  try{ obs.observe(window.parent.document.body,{childList:true,subtree:true}); }catch(e){}
})();
</script>
""", height=0, scrolling=False)

# === TAB 4: Flash Card ===
elif active_tab == TAB_NAMES[4]:
    import json as _json, random as _rand
    _logo_fc = (f'<img src="{_logo_uri}" style="width:56px;height:56px;border-radius:50%;'
                f'box-shadow:0 3px 12px rgba(192,57,43,.28);flex-shrink:0">') if _logo_uri else ''
    st.markdown(f"""
<div style="display:flex;align-items:center;gap:16px;margin-bottom:16px">
  {_logo_fc}
  <div>
    <div style="font-size:1.4rem;font-weight:900;color:#1a1209;letter-spacing:2px;line-height:1.2">🃏 Flash Card</div>
    <div style="color:#9a8a70;font-size:.8rem;letter-spacing:1px;margin-top:2px">Luyện thuộc từ vựng bằng thẻ lật 3D</div>
  </div>
</div>""", unsafe_allow_html=True)

    _fc_lessons = sorted(VOCAB_LESSONS.keys())
    _fc_src_opts = [f"Bài {n}  ({len(VOCAB_LESSONS[n])} từ)" for n in _fc_lessons] + ["🎲 Ngẫu nhiên (tất cả bài)"]

    _c1, _c2, _c3, _c4 = st.columns([2.2, 2, 1.2, 1])
    with _c1:
        _fc_src = st.selectbox("Nguồn", _fc_src_opts, key="fc_src", label_visibility="collapsed")
    with _c2:
        _fc_front = st.selectbox("Mặt trước", ["Từ tiếng Nhật → Nghĩa", "Nghĩa → Từ tiếng Nhật"],
                                 key="fc_front", label_visibility="collapsed")
    with _c3:
        _fc_count = st.selectbox("Số thẻ", [10, 20, 30, "Tất cả"], key="fc_count", label_visibility="collapsed")
    with _c4:
        _fc_go = st.button("🃏 Bắt đầu", type="primary", use_container_width=True, key="fc_go")

    if _fc_go or "fc_html" in st.session_state:
        if _fc_go:
            if _fc_src.startswith("Bài "):
                _ln = int(_fc_src.split()[1])
                _words = list(VOCAB_LESSONS.get(_ln, []))
            else:
                _words = []
                for _ws in VOCAB_LESSONS.values():
                    _words.extend(_ws)
            _rand.shuffle(_words)
            _cnt = _fc_count if _fc_count == "Tất cả" else int(_fc_count)
            if _cnt != "Tất cả" and len(_words) > _cnt:
                _words = _words[:_cnt]
            _jp_front = (_fc_front == "Từ tiếng Nhật → Nghĩa")
            _deck = []
            for _w in _words:
                if _jp_front:
                    _deck.append({
                        "front": _w.get("word", ""), "front_sub": _w.get("reading", ""),
                        "back_main": _w.get("meaning", ""), "back_sub": _w.get("hanviet", ""),
                        "back_ex_jp": _w.get("example", ""), "back_ex_vi": _w.get("exampleVi", ""),
                        "jp_word": _w.get("word", ""),
                    })
                else:
                    _hv = _w.get("hanviet", "")
                    _deck.append({
                        "front": _w.get("meaning", ""), "front_sub": "",
                        "back_main": _w.get("word", ""),
                        "back_sub": _w.get("reading", "") + (" · " + _hv if _hv else ""),
                        "back_ex_jp": _w.get("example", ""), "back_ex_vi": _w.get("exampleVi", ""),
                        "jp_word": _w.get("word", ""),
                    })
            st.session_state["fc_html"] = _build_fc_html(_json.dumps(_deck, ensure_ascii=False))

        _components.html(st.session_state["fc_html"], height=590, scrolling=False)

# ── Site Footer (chỉ hiện ở tab Lộ trình và Từ Vựng) ──────────────────────────
if active_tab != TAB_NAMES[0]:
    _logo_footer = f'<img src="{_logo_uri}">' if _logo_uri else ''
    st.markdown(f"""
<div class="site-footer">
  <div class="site-footer-logo">
    {_logo_footer}
    <span>KANJI HUB</span>
  </div>
  <div class="site-footer-links">
    <a href="#">🔍 Tra Kanji</a>
    <a href="#">🗺️ Lộ trình JLPT</a>
    <a href="#">📖 Từ Vựng</a>
  </div>
  <hr class="site-footer-divider">
  <div class="site-footer-copy">
    © 2026 Kanji Hub · Miễn phí · Không quảng cáo · Dành cho người học tiếng Nhật
  </div>
  <div class="site-footer-jp">漢字 · 学習 · 平仮名 · 片仮名</div>
</div>
""", unsafe_allow_html=True)

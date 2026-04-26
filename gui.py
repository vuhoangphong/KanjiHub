"""
Giao diện tra cứu Kanji – nhập kanji → tự động hiện nghĩa + từ vựng.
Chạy: python gui.py
"""
import asyncio
import json
import os
import subprocess
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import customtkinter as ctk
from kanji_lookup import (
    get_kanji_info, search_by_viet, search_by_viet_gemini,
    MANUAL_VI, N4_VI, MNN_N4_EXTRA, MNN_N5, N3_VI, N2_VI, N1_VI,
    get_gemini_key, set_gemini_key, GeminiQuotaError, AIQuotaError,
    lookup_kanji_gemini, lookup_kanji_openrouter,
    lookup_vocab,
)
from pdf_generator import generate_pdf, generate_vocab_table_pdf
from vocab_lessons import VOCAB_LESSONS

import sys
import requests
from PIL import Image, ImageTk
import io

def _get_app_dir() -> str:
    """Trả về thư mục chứa EXE (frozen) hoặc source (dev)."""
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


HISTORY_FILE = os.path.join(_get_app_dir(), "history.json")
PROGRESS_FILE = os.path.join(_get_app_dir(), "progress.json")
EXPORT_DIR   = os.path.join(_get_app_dir(), "exports")
CACHE_DIR    = os.path.join(_get_app_dir(), "cache_stroke")

os.makedirs(EXPORT_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# ─── TTS ─────────────────────────────────────────────────────────────────────
import ctypes

def _play_mp3(path: str):
    """Phát MP3 — cross-platform (Windows: winmm.dll, macOS: afplay, Linux: mpg123)."""
    if sys.platform == "win32":
        winmm = ctypes.WinDLL("winmm")
        alias = "kanji_tts"
        winmm.mciSendStringW(f'open "{path}" type mpegvideo alias {alias}', None, 0, None)
        winmm.mciSendStringW(f"play {alias} wait", None, 0, None)
        winmm.mciSendStringW(f"close {alias}", None, 0, None)
    elif sys.platform == "darwin":
        import subprocess
        subprocess.run(["afplay", path], check=True)
    else:
        import subprocess
        subprocess.run(["mpg123", "-q", path], check=True)


def speak(text: str):
    """Phát âm text tiếng Nhật bằng edge-tts (Microsoft Neural TTS)."""
    if not text.strip():
        return
    def _run():
        try:
            import edge_tts
            async def _gen(path):
                tts = edge_tts.Communicate(text, voice="ja-JP-NanamiNeural")
                await tts.save(path)

            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                path = f.name
            asyncio.run(_gen(path))
            _play_mp3(path)
            try:
                os.unlink(path)
            except Exception:
                pass
        except Exception as e:
            print(f"[TTS] {e}")
    threading.Thread(target=_run, daemon=True).start()

# ─── Theme ────────────────────────────────────────────────────────────────────
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

ALL_DB = {**MANUAL_VI, **N4_VI, **MNN_N4_EXTRA, **MNN_N5, **N3_VI, **N2_VI, **N1_VI}

# Màu trạng thái
CLR_FOUND    = "#2ECC71"   # xanh lá → có trong DB
CLR_FALLBACK = "#F39C12"   # cam → tìm qua Jisho
CLR_GEMINI   = "#A78BFA"   # tím → Gemini AI
CLR_ERROR    = "#E74C3C"   # đỏ → không tìm thấy gì
CLR_CARD_BG  = "#1E1E2E"
CLR_CARD_BD  = "#313244"


# ─── Hàm tiện ích ─────────────────────────────────────────────────────────────
def extract_kanji(text: str) -> list[str]:
    """Tách từng chữ kanji/CJK từ chuỗi nhập, bỏ khoảng trắng & trùng lặp."""
    seen = set()
    result = []
    for ch in text:
        cp = ord(ch)
        is_cjk = (
            0x3000 <= cp <= 0x9FFF
            or 0xF900 <= cp <= 0xFAFF
            or 0x20000 <= cp <= 0x2A6DF
        )
        if is_cjk and ch not in seen:
            seen.add(ch)
            result.append(ch)
    return result


def has_cjk(text: str) -> bool:
    """Trả về True nếu chuỗi có ít nhất 1 ký tự CJK."""
    for ch in text:
        cp = ord(ch)
        if 0x3000 <= cp <= 0x9FFF or 0xF900 <= cp <= 0xFAFF or 0x20000 <= cp <= 0x2A6DF:
            return True
    return False


def status_of(info: dict) -> tuple[str, str]:
    """Trả về (nhãn trạng thái, màu)."""
    kanji = info.get("kanji", "")
    if kanji in ALL_DB:
        return "✓ Có trong cơ sở dữ liệu", CLR_FOUND
    if info.get("source") == "mazii":
        return "🇻🇳 Mazii Dictionary", "#4CAF50" # Màu xanh lá cho Mazii
    if info.get("source") == "gemini":
        return "✨ Gemini AI", CLR_GEMINI
    if info.get("source") == "openrouter":
        return "🚀 OpenRouter AI", "#89B4FA"
    if info.get("meaning_vi") and info["meaning_vi"] != kanji:
        return "⚡ Tìm qua Jisho API", CLR_FALLBACK
    return "✗ Không tìm thấy", CLR_ERROR


# ─── Card từng kanji ──────────────────────────────────────────────────────────
class KanjiCard(ctk.CTkFrame):
    def __init__(self, master, info: dict, **kw):
        super().__init__(master, fg_color=CLR_CARD_BG, border_color=CLR_CARD_BD,
                         border_width=1, corner_radius=10, **kw)
        self._build(info)

    def _build(self, info: dict):
        kanji   = info.get("kanji", "")
        viet    = info.get("viet", "")
        reading = info.get("reading", "")
        meaning = info.get("meaning_vi", "")
        meo     = info.get("meo", "")
        vocab   = info.get("vocab", [])
        status_text, status_color = status_of(info)

        # Cột trái: chữ kanji lớn + nút phát âm
        # Tự động điều chỉnh kích thước theo độ dài từ
        char_count = len(kanji)
        if char_count <= 1:
            kanji_font_size = 52
            left_width = 90
        elif char_count <= 2:
            kanji_font_size = 40
            left_width = 110
        elif char_count <= 3:
            kanji_font_size = 32
            left_width = 130
        elif char_count <= 4:
            kanji_font_size = 26
            left_width = 150
        else:
            kanji_font_size = 20
            left_width = max(160, char_count * 22)

        left = ctk.CTkFrame(self, fg_color="transparent", width=left_width)
        left.pack(side="left", padx=(14, 0), pady=12)
        left.pack_propagate(False)

        lbl_kanji = ctk.CTkLabel(left, text=kanji, font=("Noto Sans JP", kanji_font_size),
                                 text_color="#CDD6F4", wraplength=left_width - 4,
                                 cursor="hand2")
        lbl_kanji.pack()
        lbl_kanji.bind("<Button-1>", lambda e, k=kanji: self._show_stroke_order(k))

        # Giới hạn reading 1 dòng duy nhất, tránh đẩy nút ra ngoài card
        _max_reading_len = 18
        display_reading = reading if len(reading) <= _max_reading_len else reading[:_max_reading_len - 1] + "…"
        lbl_reading = ctk.CTkLabel(left, text=display_reading, font=("Noto Sans JP", 12),
                                   text_color="#89B4FA", wraplength=0)
        lbl_reading.pack()

        # Nút 🔊 phát âm kanji
        speak_text = kanji if kanji else reading
        btn_speak = ctk.CTkButton(
            left, text="🔊", width=52, height=26,
            font=("Arial", 13),
            fg_color="#1E3A5F", hover_color="#2A5080",
            command=lambda t=speak_text: speak(t),
        )
        btn_speak.pack(pady=(4, 0))

        # Nút 🔍 Phân tích AI
        btn_analyze = ctk.CTkButton(
            left, text="🔍 Phân tích", width=70, height=26,
            font=("Arial", 11),
            fg_color="#313244", hover_color="#45475A",
            command=lambda k=kanji: self._analyze_with_ai(k),
        )
        btn_analyze.pack(pady=(4, 0))

        # Cột phải: thông tin
        right = ctk.CTkFrame(self, fg_color="transparent")
        right.pack(side="left", fill="both", expand=True, padx=12, pady=10)

        # Hàng 1: âm HV + trạng thái
        row1 = ctk.CTkFrame(right, fg_color="transparent")
        row1.pack(fill="x")
        ctk.CTkLabel(row1, text=viet, font=("Arial", 15, "bold"),
                     text_color="#CBA6F7").pack(side="left")
        ctk.CTkLabel(row1, text=status_text, font=("Arial", 11),
                     text_color=status_color).pack(side="right")

        # Nghĩa
        ctk.CTkLabel(right, text=f"Nghĩa: {meaning}",
                     font=("Arial", 13), text_color="#CDD6F4",
                     anchor="w").pack(fill="x", pady=(2, 0))

        # Mẹo nhớ
        if meo:
            ctk.CTkLabel(right, text=f"💡 {meo}",
                         font=("Arial", 12, "italic"), text_color="#A6E3A1",
                         anchor="w", wraplength=700).pack(fill="x", pady=(1, 0))

        # Từ vựng ví dụ
        if vocab:
            sep = ctk.CTkFrame(right, height=1, fg_color="#313244")
            sep.pack(fill="x", pady=(5, 3))
            for i, item in enumerate(vocab[:2]):
                w, r, m = item[0], item[1], item[2]
                ctk.CTkLabel(right,
                             text=f"  {i+1}. {w}（{r}）— {m}",
                             font=("Noto Sans JP", 12), text_color="#F5C2E7",
                             anchor="w").pack(fill="x")
        elif info.get("meanings_en"):
            sep = ctk.CTkFrame(right, height=1, fg_color="#313244")
            sep.pack(fill="x", pady=(5, 3))
            ctk.CTkLabel(right,
                         text="  " + " / ".join(info["meanings_en"][:3]),
                         font=("Arial", 12), text_color="#F5C2E7",
                         anchor="w").pack(fill="x")

        # Lỗi nếu không tìm thấy gì
        if status_text.startswith("✗"):
            from kanji_lookup import get_ai_provider, get_gemini_key, get_openrouter_key
            p = get_ai_provider()
            has_key = get_gemini_key() if p == "gemini" else get_openrouter_key()
            p_name = p.upper()
            
            hint = f"⚠ Không tìm thấy. Đã thử {p_name} AI." if has_key else f"⚠ Không tìm thấy. Đặt {p_name} API key (biểu tượng ⚙️) để tra AI."
            ctk.CTkLabel(right,
                         text=hint,
                         font=("Arial", 11), text_color=CLR_ERROR,
                          anchor="w").pack(fill="x", pady=(4, 0))

    def _show_stroke_order(self, kanji: str):
        if not kanji: return
        
        # Tạo popup
        dlg = ctk.CTkToplevel(self)
        dlg.title(f"Cách viết chữ {kanji}")
        dlg.geometry("420x580")
        dlg.resizable(False, False)
        dlg.attributes("-topmost", True)
        dlg.grab_set()
        
        # Center popup
        self.update_idletasks()
        x = self.winfo_toplevel().winfo_x() + (self.winfo_toplevel().winfo_width() // 2) - 210
        y = self.winfo_toplevel().winfo_y() + (self.winfo_toplevel().winfo_height() // 2) - 290
        dlg.geometry(f"+{max(0, x)}+{max(0, y)}")

        # Tab điều khiển
        tab_var = ctk.StringVar(value="Ảnh động")
        seg_button = ctk.CTkSegmentedButton(dlg, values=["Ảnh động", "Sơ đồ"],
                                            variable=tab_var, command=lambda v: switch_tab(v))
        seg_button.pack(pady=15)

        lbl_display = ctk.CTkLabel(dlg, text="⏳ Đang tải dữ liệu...", font=("Arial", 14))
        lbl_display.pack(pady=10)

        # Cache cho ảnh
        data_cache = {"diagram": None, "frames": [], "current_job": None}
        
        def switch_tab(mode):
            # Dừng animation cũ nếu có
            if data_cache["current_job"]:
                dlg.after_cancel(data_cache["current_job"])
                data_cache["current_job"] = None

            if mode == "Sơ đồ":
                if data_cache["diagram"]:
                    lbl_display.configure(text="", image=data_cache["diagram"])
                else:
                    lbl_display.configure(text="⏳ Đang tải sơ đồ...", image="")
                    threading.Thread(target=load_diagram, daemon=True).start()
            else:
                if data_cache["frames"]:
                    play_gif(0)
                else:
                    lbl_display.configure(text="⏳ Đang tải ảnh động...", image="")
                    threading.Thread(target=load_gif, daemon=True).start()

        def play_gif(idx):
            if not data_cache["frames"] or tab_var.get() != "Ảnh động": return
            lbl_display.configure(text="", image=data_cache["frames"][idx])
            next_idx = (idx + 1) % len(data_cache["frames"])
            # Tốc độ animation (khoảng 100ms mỗi khung hình)
            data_cache["current_job"] = dlg.after(100, lambda: play_gif(next_idx))

        def load_gif():
            try:
                h = hex(ord(kanji[0]))[2:].lower()
                cache_path = os.path.join(CACHE_DIR, f"{h}.gif")
                
                # Kiểm tra cache trước
                if os.path.exists(cache_path):
                    with open(cache_path, "rb") as f:
                        resp_content = f.read()
                else:
                    url = f"https://raw.githubusercontent.com/mistval/kanji_images/master/gifs/{h}.gif"
                    resp = requests.get(url, timeout=10)
                    if resp.status_code == 200:
                        resp_content = resp.content
                        with open(cache_path, "wb") as f:
                            f.write(resp_content)
                    else:
                        dlg.after(0, lambda: lbl_display.configure(text="✗ Không tìm thấy ảnh động."))
                        return

                img_data = io.BytesIO(resp_content)
                with Image.open(img_data) as img:
                    frames = []
                    for i in range(getattr(img, "n_frames", 1)):
                        img.seek(i)
                        frame = img.convert("RGBA").resize((320, 320), Image.Resampling.LANCZOS)
                        ctk_f = ctk.CTkImage(light_image=frame, dark_image=frame, size=(320, 320))
                        frames.append(ctk_f)
                    data_cache["frames"] = frames
                if tab_var.get() == "Ảnh động":
                    dlg.after(0, lambda: play_gif(0))
            except Exception as e:
                dlg.after(0, lambda: lbl_display.configure(text=f"✗ Lỗi: {str(e)[:30]}"))

        def load_diagram():
            try:
                h = hex(ord(kanji[0]))[2:].lower()
                cache_path = os.path.join(CACHE_DIR, f"{h}.png")
                
                if os.path.exists(cache_path):
                    with open(cache_path, "rb") as f:
                        resp_content = f.read()
                else:
                    url = f"https://raw.githubusercontent.com/SethClydesdale/kanji-stroke-order-image-search/main/stroke-order/img/{h}.png"
                    resp = requests.get(url, timeout=10)
                    if resp.status_code == 200:
                        resp_content = resp.content
                        with open(cache_path, "wb") as f:
                            f.write(resp_content)
                    else:
                        dlg.after(0, lambda: lbl_display.configure(text="✗ Không tìm thấy sơ đồ."))
                        return

                pil_img = Image.open(io.BytesIO(resp_content)).resize((320, 320), Image.Resampling.LANCZOS)
                data_cache["diagram"] = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(320, 320))
                if tab_var.get() == "Sơ đồ":
                    dlg.after(0, lambda: lbl_display.configure(text="", image=data_cache["diagram"]))
            except:
                dlg.after(0, lambda: lbl_display.configure(text="✗ Lỗi tải sơ đồ."))

        # Tự động tải cả 2 dữ liệu ngay khi mở cửa sổ
        threading.Thread(target=load_gif, daemon=True).start()
        threading.Thread(target=load_diagram, daemon=True).start()

        # Hiện tab Ảnh động làm mặc định
        switch_tab("Ảnh động")

        ctk.CTkLabel(dlg, text=f"Chữ: {kanji}", font=("Arial", 16, "bold"), text_color="#CBA6F7").pack(pady=5)
        ctk.CTkLabel(dlg, text="(Dữ liệu: KanjiVG / Kanji Images)", 
                     font=("Arial", 12, "italic"), text_color="#6C7086").pack(pady=5)

    def _analyze_with_ai(self, kanji: str):
        if not kanji: return
        
        from kanji_lookup import get_ai_provider, get_gemini_key, get_openrouter_key
        p = get_ai_provider()
        has_key = get_gemini_key() if p == "gemini" else get_openrouter_key()
        
        if not has_key:
            from tkinter import messagebox
            messagebox.showwarning("Thiếu API Key", f"Vui lòng cài đặt API Key cho {p.upper()} để dùng tính năng này!")
            return

        # Tạo popup hiển thị kết quả
        dlg = ctk.CTkToplevel(self)
        dlg.title(f"Phân tích chuyên sâu: {kanji}")
        dlg.geometry("550x650")
        dlg.attributes("-topmost", True)
        dlg.grab_set()

        # Center popup
        self.update_idletasks()
        # Lấy tọa độ cửa sổ chính
        root = self.winfo_toplevel()
        x = root.winfo_x() + (root.winfo_width() // 2) - 275
        y = root.winfo_y() + (root.winfo_height() // 2) - 325
        dlg.geometry(f"+{max(0, x)}+{max(0, y)}")

        title = ctk.CTkLabel(dlg, text=f"✨ Phân tích AI ({p.upper()})", font=("Arial", 18, "bold"), text_color="#CBA6F7")
        title.pack(pady=15)

        txt = ctk.CTkTextbox(dlg, font=("Noto Sans JP", 14), corner_radius=10, border_width=1, border_color="#313244")
        txt.pack(fill="both", expand=True, padx=20, pady=(0, 20))
        txt.insert("0.0", f"⏳ Đang hỏi {p.capitalize()} AI về chữ '{kanji}'...\nVui lòng đợi trong giây lát.")
        txt.configure(state="disabled")

        # Chặn cuộn chuột lan ra ngoài nhưng vẫn cho phép cuộn bên trong
        def on_txt_scroll(event):
            # Cuộn nội dung của chính nó
            if event.num == 4 or event.delta > 0:
                txt._textbox.yview_scroll(-1, "units")
            elif event.num == 5 or event.delta < 0:
                txt._textbox.yview_scroll(1, "units")
            return "break"

        txt.bind("<MouseWheel>", on_txt_scroll)
        txt.bind("<Button-4>", on_txt_scroll)
        txt.bind("<Button-5>", on_txt_scroll)

        def perform_analysis():
            from kanji_lookup import analyze_kanji_ai, get_ai_provider, get_openrouter_key
            provider = get_ai_provider()
            
            try:
                content = analyze_kanji_ai(kanji)
                
                # Nếu Gemini lỗi quota, thử tự động sang OpenRouter
                if "quota" in content.lower() and provider == "gemini" and get_openrouter_key():
                    dlg.after(0, lambda: txt.configure(state="normal") or txt.delete("0.0", "end") or txt.insert("0.0", f"⚠️ Gemini hết lượt, đang thử bằng OpenRouter...") or txt.configure(state="disabled"))
                    from kanji_lookup import _ANALYZE_PROMPT, OPENROUTER_MODEL
                    import requests
                    prompt = _ANALYZE_PROMPT.format(kanji=kanji)
                    url = "https://openrouter.ai/api/v1/chat/completions"
                    headers = {"Authorization": f"Bearer {get_openrouter_key()}", "Content-Type": "application/json"}
                    payload = {"model": OPENROUTER_MODEL, "messages": [{"role": "user", "content": prompt}]}
                    resp = requests.post(url, headers=headers, json=payload, timeout=20)
                    content = resp.json()["choices"][0]["message"]["content"].strip()

                def update_ui():
                    txt.configure(state="normal")
                    txt.delete("0.0", "end")
                    txt.insert("0.0", content)
                    txt.configure(state="disabled")
                
                dlg.after(0, update_ui)
            except Exception as e:
                dlg.after(0, lambda e=e: txt.configure(state="normal") or txt.delete("0.0", "end") or txt.insert("0.0", f"✗ Lỗi: {str(e)}"))

        threading.Thread(target=perform_analysis, daemon=True).start()


# ─── Cửa sổ chính ─────────────────────────────────────────────────────────────
class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Kanji Hub")
        self.geometry("1150x800")
        self.minsize(1020, 500)
        self.resizable(True, True)

        # Icon app
        _icon = os.path.join(os.path.dirname(os.path.abspath(__file__)), "icon.ico")
        if os.path.exists(_icon):
            try:
                self.iconbitmap(_icon)
            except Exception:
                pass

        self._lookup_job = None   # debounce timer
        self._cards: list[KanjiCard] = []
        self._infos: list[dict] = []
        self._history: list[dict] = self._load_history()
        self._progress: dict = self._load_progress()

        self._build_ui()

    # ── Giao diện ──────────────────────────────────────────────────────────────
    def _update_settings_badge(self):
        from kanji_lookup import get_gemini_key, get_openrouter_key, get_ai_provider
        p = get_ai_provider()
        has_key = get_gemini_key() if p == "gemini" else get_openrouter_key()
        status = "Đã có Key" if has_key else "Chưa có Key"
        color = "#A6E3A1" if has_key else "#F38BA8"
        # Hiển thị tên AI đang chọn + trạng thái key
        self.btn_settings.configure(text=f"⚙️ {p.upper()}: {status}", text_color=color)
        
        # Cập nhật tên AI trên các nút chọn chế độ tìm kiếm
        p_name = p.capitalize()
        new_values = ["DB", f"DB + {p_name}", f"{p_name} AI"]
        
        # Lưu lại lựa chọn hiện tại để chọn lại sau khi đổi tên
        current_sel = self.opt_search.get()
        self.opt_search.configure(values=new_values)
        
        if "AI" in current_sel or "Gemini" in current_sel or "Openrouter" in current_sel:
            if "+" in current_sel: self.opt_search.set(f"DB + {p_name}")
            else: self.opt_search.set(f"{p_name} AI")
        else:
            self.opt_search.set("DB")

    def _build_ui(self):
        # Tiêu đề + nút cài đặt
        top_bar = ctk.CTkFrame(self, fg_color="transparent")
        top_bar.pack(fill="x", padx=20, pady=(14, 0))

        ctk.CTkLabel(top_bar, text="✍️  Kanji Hub",
                     font=("Arial", 22, "bold"),
                     text_color="#CBA6F7").pack(side="left")

        self.btn_settings = ctk.CTkButton(
            top_bar, text="⚙️  Cài đặt AI",
            font=("Arial", 12), width=120, height=32,
            fg_color="#313244", hover_color="#45475A",
            command=self._open_settings,
        )
        self.btn_settings.pack(side="right")

        # Ô nhập
        input_frame = ctk.CTkFrame(self, fg_color="transparent")
        input_frame.pack(fill="x", padx=20, pady=(10, 0))

        self.entry = ctk.CTkEntry(
            input_frame,
            placeholder_text="Nhập kanji vào đây, VD: 山川田…",
            font=("Noto Sans JP", 18),
            height=46,
        )
        self.entry.pack(side="left", fill="x", expand=True)
        self.entry.bind("<KeyRelease>", self._on_key)
        self.entry.focus()

        # Badge chế độ tìm kiếm
        self.lbl_mode = ctk.CTkLabel(
            self, text="",
            font=("Arial", 11, "bold"), corner_radius=6,
            fg_color="transparent", text_color="#6C7086",
        )
        self.lbl_mode.pack(pady=(4, 0))

        self.btn_clear = ctk.CTkButton(
            input_frame, text="✕", width=46, height=46,
            fg_color="#313244", hover_color="#45475A",
            command=self._clear
        )
        self.btn_clear.pack(side="left", padx=(6, 0))

        # Tùy chọn số hàng luyện viết + chế độ tìm kiếm
        opt_frame = ctk.CTkFrame(self, fg_color="transparent")
        opt_frame.pack(fill="x", padx=20, pady=(6, 0))

        ctk.CTkLabel(opt_frame, text="Tìm kiếm:",
                     font=("Arial", 12), text_color="#CDD6F4").pack(side="left")
        self.opt_search = ctk.CTkSegmentedButton(
            opt_frame,
            values=["DB", "DB + AI", "Chế độ AI"],
            font=("Arial", 12),
            height=30,
        )
        self.opt_search.set("DB + AI")
        self.opt_search.pack(side="left", padx=(6, 0))

        ctk.CTkLabel(opt_frame, text="    Số hàng luyện:",
                     font=("Arial", 12), text_color="#CDD6F4").pack(side="left")

        self._row_map = {
            "1 hàng (6 ô)": 0,
            "2 hàng (18 ô)": 1,
            "3 hàng (30 ô)": 2,
            "4 hàng (42 ô)": 3,
        }
        self.opt_rows = ctk.CTkSegmentedButton(
            opt_frame,
            values=list(self._row_map.keys()),
            font=("Arial", 12),
            height=30,
        )
        self.opt_rows.set("1 hàng (6 ô)")
        self.opt_rows.pack(side="left", padx=(10, 0))

        # Thanh trạng thái
        self.lbl_status = ctk.CTkLabel(self, text="",
                                        font=("Arial", 12), text_color="#6C7086")
        self.lbl_status.pack(pady=(1, 0))

        # Khu vực kết quả (scroll)
        self.scroll = ctk.CTkScrollableFrame(self, fg_color="transparent")
        self.scroll.pack(fill="both", expand=True, padx=20, pady=(8, 4))

        # Nút xuất PDF
        bottom = ctk.CTkFrame(self, fg_color="transparent")
        bottom.pack(fill="x", padx=20, pady=(0, 16))

        self.btn_pdf = ctk.CTkButton(
            bottom, text="📄 Xuất PDF luyện viết",
            font=("Arial", 13, "bold"),
            height=40,
            command=self._export_pdf,
            state="disabled",
        )
        self.btn_pdf.pack(side="right")
        
        self.btn_flashcard = ctk.CTkButton(
            bottom, text="🎴 Flashcard",
            font=("Arial", 13, "bold"),
            height=40, width=120,
            fg_color="#89B4FA", hover_color="#B4BEFE", text_color="#11111B",
            command=self._open_flashcard,
            state="disabled",
        )
        self.btn_flashcard.pack(side="right", padx=(0, 8))

        self.lbl_pdf = ctk.CTkLabel(bottom, text="",
                                     font=("Arial", 11), text_color="#A6E3A1")
        self.lbl_pdf.pack(side="right", padx=(0, 12))

        self.btn_history = ctk.CTkButton(
            bottom, text="📋 Lịch sử",
            font=("Arial", 13), width=110, height=40,
            fg_color="#313244", hover_color="#45475A",
            command=self._open_history,
        )
        self.btn_history.pack(side="left")

        self.btn_json = ctk.CTkButton(
            bottom, text="📥 Nhập JSON",
            font=("Arial", 13), width=120, height=40,
            fg_color="#2D1B4E", hover_color="#3D2B6E",
            command=self._open_json_import,
        )
        self.btn_json.pack(side="left", padx=(8, 0))

        self.btn_path = ctk.CTkButton(
            bottom, text="🗺️ Lộ trình học",
            font=("Arial", 13, "bold"), width=120, height=40,
            fg_color="#D97706", hover_color="#B45309",
            command=self._open_learning_path,
        )
        self.btn_path.pack(side="left", padx=(8, 0))

        self.btn_vocab = ctk.CTkButton(
            bottom, text="📖 Tra Từ",
            font=("Arial", 13, "bold"), width=100, height=40,
            fg_color="#1A5276", hover_color="#2471A3",
            command=self._open_vocab_lookup,
        )
        self.btn_vocab.pack(side="left", padx=(8, 0))
        self._update_settings_badge()

    def _load_all_n4(self):
        self.entry.delete(0, "end")
        n4_kanji = "".join(N4_VI.keys())
        self.entry.insert(0, n4_kanji)
        self._do_lookup()

    def _load_all_n5(self):
        self.entry.delete(0, "end")
        n5_kanji = "".join(MNN_N5.keys())
        self.entry.insert(0, n5_kanji)
        self._do_lookup()

    def _open_settings(self):
        dlg = ctk.CTkToplevel(self)
        dlg.title("Cài đặt AI")
        dlg.geometry("500x420")
        dlg.resizable(False, False)
        dlg.grab_set()
        dlg.after(100, dlg.lift)
        dlg.after(100, dlg.focus_force)

        # ── Provider Selection ──
        ctk.CTkLabel(dlg, text="Chọn AI cung cấp",
                     font=("Arial", 15, "bold"), text_color="#CBA6F7").pack(pady=(20, 10))
        
        from kanji_lookup import get_ai_provider, set_ai_provider, get_gemini_key, set_gemini_key, get_openrouter_key, set_openrouter_key
        
        provider_var = ctk.StringVar(value=get_ai_provider())
        provider_seg = ctk.CTkSegmentedButton(dlg, values=["gemini", "openrouter"],
                                              variable=provider_var)
        provider_seg.pack(pady=10)

        # ── Gemini Key ──
        ctk.CTkLabel(dlg, text="Gemini API Key",
                     font=("Arial", 13, "bold")).pack(pady=(15, 0))
        gemini_entry = ctk.CTkEntry(dlg, placeholder_text="AIzaSy…",
                                     font=("Arial", 13), width=400, height=36, show="•")
        gemini_entry.pack(pady=5)
        if get_gemini_key(): gemini_entry.insert(0, get_gemini_key())

        # ── OpenRouter Key ──
        ctk.CTkLabel(dlg, text="OpenRouter API Key",
                     font=("Arial", 13, "bold")).pack(pady=(15, 0))
        or_entry = ctk.CTkEntry(dlg, placeholder_text="sk-or-…",
                                 font=("Arial", 13), width=400, height=36, show="•")
        or_entry.pack(pady=5)
        if get_openrouter_key(): or_entry.insert(0, get_openrouter_key())

        lbl_result = ctk.CTkLabel(dlg, text="", font=("Arial", 11))
        lbl_result.pack(pady=(10, 0))

        def save():
            selected_provider = provider_seg.get()
            set_ai_provider(selected_provider)
            set_gemini_key(gemini_entry.get().strip())
            set_openrouter_key(or_entry.get().strip())
            lbl_result.configure(text=f"✓ Đã lưu: {selected_provider.upper()}!", text_color="#A6E3A1")
            self._update_settings_badge()
            dlg.after(800, dlg.destroy)

        ctk.CTkButton(dlg, text="💾 Lưu cài đặt", width=160, height=40, command=save).pack(pady=20)

    # ── Sự kiện ────────────────────────────────────────────────────────────────
    def _on_key(self, event=None):
        """Debounce 400ms rồi tra cứu."""
        if self._lookup_job:
            self.after_cancel(self._lookup_job)
        self._lookup_job = self.after(400, self._do_lookup)

    def _do_lookup(self):
        text = self.entry.get().strip()
        search_mode = self.opt_search.get()  # "DB" | "DB + AI" | "AI"

        self._clear_cards()

        if not text:
            self.lbl_mode.configure(text="", fg_color="transparent")
            self.lbl_status.configure(text="")
            self.btn_pdf.configure(state="disabled")
            return

        # ── Chế độ AI trực tiếp ──────────────────────────────────────────
        if "AI" in search_mode and "+" not in search_mode:
            from kanji_lookup import get_ai_provider
            provider = get_ai_provider()
            self.lbl_mode.configure(
                text=f"  ✨ Chế độ: {provider.capitalize()} AI  ",
                fg_color="#2D1B4E", text_color="#CBA6F7")
            self.lbl_status.configure(text=f"⏳ Đang hỏi {provider.capitalize()} AI…", text_color="#6C7086")
            self.update_idletasks()

            if has_cjk(text):
                # Nhập kanji → tra thông tin bằng Gemini
                kanji_list = extract_kanji(text)
                def worker_ai_kanji():
                    from kanji_lookup import get_ai_provider, lookup_kanji_gemini, lookup_kanji_openrouter, AIQuotaError, get_openrouter_key
                    provider = get_ai_provider()
                    results = []
                    for k in kanji_list:
                        try:
                            g = None
                            final_source = provider
                            
                            if provider == "gemini":
                                g = lookup_kanji_gemini(k)
                                if not g and get_openrouter_key():
                                    print(f"⚠️ Gemini lỗi, tự động thử OpenRouter cho '{k}'...")
                                    g = lookup_kanji_openrouter(k)
                                    final_source = "openrouter"
                            else:
                                g = lookup_kanji_openrouter(k)
                                final_source = "openrouter"
                            
                            if g:
                                info = {"kanji": k, "meanings_en": [], "source": final_source, **g}
                            else:
                                info = get_kanji_info(k)
                            results.append(info)
                        except AIQuotaError as ex:
                            # Nếu lỗi quota bắn ra, vẫn thử failover một lần nữa
                            if provider == "gemini" and get_openrouter_key():
                                try:
                                    g = lookup_kanji_openrouter(k)
                                    if g:
                                        results.append({"kanji": k, "meanings_en": [], "source": "openrouter", **g})
                                        continue
                                except: pass
                            
                            self.after(0, lambda msg=str(ex): self._show_quota_error(msg))
                            return
                    self.after(0, lambda: self._show_results(results))
                threading.Thread(target=worker_ai_kanji, daemon=True).start()
            else:
                # Nhập tiếng Việt → tìm kanji bằng Gemini
                def worker_ai_viet():
                    from kanji_lookup import get_ai_provider, search_by_viet_gemini, search_by_viet_openrouter, AIQuotaError
                    provider = get_ai_provider()
                    try:
                        if provider == "gemini":
                            infos = search_by_viet_gemini(text)
                        else:
                            infos = search_by_viet_openrouter(text)
                    except AIQuotaError as ex:
                        self.after(0, lambda msg=str(ex): self._show_quota_error(msg))
                        return
                    self.after(0, lambda: self._show_results(infos, reverse=True))
                threading.Thread(target=worker_ai_viet, daemon=True).start()
            return

        # ── Chế độ DB / DB + AI ─────────────────────────────────────────
        if has_cjk(text):
            # ── Phát hiện chế độ nhập danh sách bằng dấu phẩy ──────────────
            _CSV_SEPS = (',', '、', '，')   # dấu phẩy ASCII, JP, full-width
            is_csv = any(sep in text for sep in _CSV_SEPS)

            if is_csv:
                # Tách theo dấu phẩy, giữ nguyên từ ghép (国王, 女王, ...)
                import re as _re
                raw_parts = _re.split(r'[,、，]', text)
                seen_csv: set[str] = set()
                kanji_list = []
                for part in raw_parts:
                    part = part.strip()
                    # Lấy các ký tự CJK để tạo thành từ (bỏ khoảng trắng, dấu câu lạ)
                    word = ''.join(extract_kanji(part)) if len(part) == 1 else ''.join(
                        ch for ch in part if has_cjk(ch))
                    if word and word not in seen_csv:
                        seen_csv.add(word)
                        kanji_list.append(word)
                self.lbl_mode.configure(
                    text=f"  📋 Chế độ: Danh sách ({len(kanji_list)} từ)  ",
                    fg_color="#1E3A5F", text_color="#89B4FA")
            else:
                kanji_list = extract_kanji(text)
                self.lbl_mode.configure(
                    text="  🔤 Chế độ: Nhập Kanji  ",
                    fg_color="#1E3A5F", text_color="#89B4FA")

            if not kanji_list:
                return
            self.lbl_status.configure(text="⏳ Đang tra cứu…", text_color="#6C7086")
            self.update_idletasks()

            def worker_kanji():
                n = len(kanji_list)
                infos = [None] * n
                done = [0]
                quota_error = [None]
                with ThreadPoolExecutor(max_workers=8) as ex:
                    # Trong chế độ DB + AI, chúng ta vẫn ưu tiên DB trước
                    future_to_idx = {ex.submit(get_kanji_info, k): i
                                     for i, k in enumerate(kanji_list)}
                    for fut in as_completed(future_to_idx):
                        idx = future_to_idx[fut]
                        kanji = kanji_list[idx]
                        try:
                            info = fut.result()
                            # Nếu Jisho fallback mà có AI key, thử hỏi AI thay vì dùng Jisho
                            if info.get("source") == "jisho" and ("AI" in search_mode or "Gemini" in search_mode or "Openrouter" in search_mode):
                                from kanji_lookup import get_ai_provider, lookup_kanji_gemini, lookup_kanji_openrouter, get_openrouter_key
                                p = get_ai_provider()
                                g = None
                                final_source = p
                                if p == "gemini":
                                    g = lookup_kanji_gemini(kanji)
                                    if not g and get_openrouter_key():
                                        g = lookup_kanji_openrouter(kanji)
                                        final_source = "openrouter"
                                else:
                                    g = lookup_kanji_openrouter(kanji)
                                    final_source = "openrouter"
                                
                                if g:
                                    info = {"kanji": kanji, "meanings_en": [], "source": final_source, **g}
                            
                            infos[idx] = info
                        except Exception as e:
                            print(f"[worker_kanji] error: {e}")
                            infos[idx] = {"kanji": kanji, "vocab": []}
                        
                        done[0] += 1
                        d, t = done[0], n
                        self.after(0, lambda d=d, t=t: self.lbl_status.configure(
                            text=f"⏳ Đang tra cứu… {d}/{t}", text_color="#6C7086"))
                
                self.after(0, lambda: self._show_results(infos))

            threading.Thread(target=worker_kanji, daemon=True).start()
        else:
            self.lbl_mode.configure(
                text="  🔍 Chế độ: Tìm theo âm Hán Việt / nghĩa  ",
                fg_color="#2D1B4E", text_color="#CBA6F7")
            self.lbl_status.configure(text="⏳ Đang tìm…", text_color="#6C7086")
            self.update_idletasks()

            def worker_viet():
                infos = search_by_viet(text)
                if not infos and ("AI" in search_mode or "Gemini" in search_mode or "Openrouter" in search_mode):
                    from kanji_lookup import get_ai_provider
                    provider = get_ai_provider()
                    self.after(0, lambda p=provider: self.lbl_status.configure(
                        text=f"⏳ Không tìm thấy trong DB, đang hỏi {p.capitalize()} AI…"))
                    try:
                        from kanji_lookup import search_by_viet_gemini, search_by_viet_openrouter, get_openrouter_key
                        if provider == "gemini":
                            infos = search_by_viet_gemini(text)
                            # Failover sang OpenRouter nếu Gemini không ra hoặc hết lượt
                            if not infos and get_openrouter_key():
                                infos = search_by_viet_openrouter(text)
                        else:
                            infos = search_by_viet_openrouter(text)
                    except AIQuotaError as ex:
                        # Failover cho lỗi quota
                        from kanji_lookup import get_openrouter_key, search_by_viet_openrouter
                        if provider == "gemini" and get_openrouter_key():
                            try:
                                infos = search_by_viet_openrouter(text)
                                if infos:
                                    for i in infos: i["source"] = "openrouter"
                                    self.after(0, lambda: self._show_results(infos, reverse=True))
                                    return
                            except: pass
                        self.after(0, lambda msg=str(ex): self._show_quota_error(msg))
                        return
                if not infos:
                    self.after(0, lambda: self.lbl_status.configure(
                        text="Không tìm thấy. Thử gõ có dấu (Tĩnh, Học…) hoặc dùng chế độ Gemini AI.",
                        text_color="#F39C12"))
                    self.after(0, lambda: self.btn_pdf.configure(state="disabled"))
                    return
                self.after(0, lambda: self._show_results(infos, reverse=True))

            threading.Thread(target=worker_viet, daemon=True).start()

    def _show_quota_error(self, msg: str):
        self.lbl_status.configure(
            text=f"⚠ {msg}",
            text_color="#F38BA8",
        )
        self.btn_pdf.configure(state="disabled")

    def _show_results(self, infos: list[dict], reverse: bool = False):
        # Lọc bỏ None (kanji lookup failed)
        infos = [i for i in infos if i is not None]
        self._infos = infos

        errors    = [i for i in infos if status_of(i)[0].startswith("✗")]
        fallbacks = [i for i in infos if status_of(i)[0].startswith("⚡")]

        # ── Status bar ────────────────────────────────────────────────────────
        if reverse:
            status_text = f"Tìm thấy {len(infos)} kanji khớp" if infos else "Không tìm thấy kanji nào khớp"
        else:
            found = len(infos) - len(errors) - len(fallbacks)
            parts = [f"{len(infos)} kanji"]
            if found:      parts.append(f"✓ {found} trong DB")
            if fallbacks:  parts.append(f"⚡ {len(fallbacks)} qua Jisho")
            if errors:     parts.append(f"✗ {len(errors)} không tìm được")
            status_text = "  •  ".join(parts)

        self.lbl_status.configure(text=status_text, text_color="#CDD6F4")
        
        has_results = any(i.get("viet") for i in infos)
        self.btn_pdf.configure(state="normal" if has_results else "disabled")
        self.btn_flashcard.configure(state="normal" if has_results else "disabled")

        if not infos:
            return

        # Batch render via _pending_infos queue (_clear_cards resets it)
        self._pending_infos = list(infos)
        self._status_text = status_text
        self._schedule_render()


    def _schedule_render(self):
        """Render từng batch 5 card, gọi lại qua after(25) cho đến hết để UI không lag."""
        pending = getattr(self, "_pending_infos", [])
        if not pending:
            return
        
        # Giảm số lượng card render trong 1 batch để UI mượt hơn
        batch_size = 5
        batch = pending[:batch_size]
        self._pending_infos = pending[batch_size:]
        
        for info in batch:
            try:
                card = KanjiCard(self.scroll, info)
                card.pack(fill="x", pady=4)
                self._cards.append(card)
            except Exception as e:
                print(f"[card error] {e}")
                
        remaining = self._pending_infos
        if remaining:
            total = len(self._infos)
            done = total - len(remaining)
            self.lbl_status.configure(text=f"⏳ Đang hiện… {done}/{total}")
            # Tăng thời gian nghỉ giữa các batch để nhường CPU cho GUI update (cuộn, click)
            self.after(25, self._schedule_render)
        else:
            st = getattr(self, "_status_text", "")
            self.lbl_status.configure(text=st, text_color="#CDD6F4")

    def _clear_cards(self):
        self._pending_infos = []   # huỷ batch đang chờ
        for c in self._cards:
            c.destroy()
        self._cards.clear()
        self._infos.clear()


    def _clear(self):
        self.entry.delete(0, "end")
        self._clear_cards()
        self.lbl_mode.configure(text="", fg_color="transparent")
        self.lbl_status.configure(text="")
        self.lbl_pdf.configure(text="")
        self.btn_pdf.configure(state="disabled")

    def _open_json_import(self):
        dlg = ctk.CTkToplevel(self)
        dlg.title("📥 Nhập JSON từ vựng")
        dlg.geometry("700x540")
        dlg.resizable(True, True)
        dlg.grab_set()
        dlg.after(100, dlg.lift)
        dlg.after(100, dlg.focus_force)

        ctk.CTkLabel(dlg, text="Dán JSON từ vựng vào ô bên dưới",
                     font=("Arial", 14, "bold"), text_color="#CBA6F7").pack(pady=(14, 4))
        ctk.CTkLabel(dlg,
                     text='Format: [{"word": "漢字", "reading": "よみ", "hanviet": "ÂM HÁN VIỆT", "meaning": "nghĩa"}]',
                     font=("Arial", 10), text_color="#6C7086").pack()

        txt = ctk.CTkTextbox(dlg, font=("Arial", 12), wrap="none")
        txt.pack(fill="both", expand=True, padx=16, pady=(8, 4))

        lbl_err = ctk.CTkLabel(dlg, text="", font=("Arial", 11), text_color="#F38BA8")
        lbl_err.pack()

        btn_row = ctk.CTkFrame(dlg, fg_color="transparent")
        btn_row.pack(pady=(4, 14))

        def do_import():
            raw = txt.get("1.0", "end").strip()
            if not raw:
                lbl_err.configure(text="⚠ Chưa nhập gì cả!")
                return
            try:
                data = json.loads(raw)
            except json.JSONDecodeError as e:
                lbl_err.configure(text=f"⚠ JSON không hợp lệ: {e}")
                return
            if not isinstance(data, list) or not data:
                lbl_err.configure(text="⚠ JSON phải là mảng [...] không rỗng")
                return

            # Convert sang format nội bộ
            infos = []
            for item in data:
                if not isinstance(item, dict):
                    continue
                word = item.get("word", "").strip()
                if not word:
                    continue
                infos.append({
                    "kanji": word,
                    "viet": item.get("hanviet", "").strip(),
                    "reading": item.get("reading", "").strip(),
                    "meaning_vi": item.get("meaning", "").strip(),
                    "meo": item.get("meo", "").strip(),
                    "vocab": item.get("vocab", []),
                    "source": "json_import",
                })

            if not infos:
                lbl_err.configure(text="⚠ Không đọc được mục nào hợp lệ")
                return

            dlg.destroy()
            # Xóa entry và hiện kết quả từ JSON
            self.entry.delete(0, "end")
            self._clear_cards()
            self.lbl_mode.configure(
                text=f"  📥 Nhập JSON — {len(infos)} mục  ",
                fg_color="#2D1B4E", text_color="#CBA6F7")
            self.lbl_status.configure(text="", text_color="#CDD6F4")
            self._show_results(infos)

        ctk.CTkButton(btn_row, text="✅ Nhập và hiện thị", width=160, height=38,
                      command=do_import).pack(side="left", padx=6)
        ctk.CTkButton(btn_row, text="Hủy", width=80, height=38,
                      fg_color="#313244", hover_color="#45475A",
                      command=dlg.destroy).pack(side="left", padx=6)

    def _load_history(self) -> list[dict]:
        try:
            with open(HISTORY_FILE, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []

    def _save_history(self):
        try:
            with open(HISTORY_FILE, "w", encoding="utf-8") as f:
                json.dump(self._history, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def _add_history(self, kanji_str: str, filename: str):
        from datetime import datetime
        entry = {"kanji": kanji_str, "file": filename,
                 "time": datetime.now().strftime("%d/%m/%Y %H:%M")}
        self._history = [h for h in self._history if h["kanji"] != kanji_str]
        self._history.insert(0, entry)
        self._history = self._history[:50]
        self._save_history()

    def _open_history(self):
        dlg = ctk.CTkToplevel(self)
        dlg.title("Lịch sử xuất PDF")
        dlg.geometry("540x480")
        dlg.resizable(False, True)
        dlg.grab_set()
        dlg.after(100, dlg.lift)
        dlg.after(100, dlg.focus_force)

        ctk.CTkLabel(dlg, text="📋  Lịch sử xuất PDF",
                     font=("Arial", 16, "bold"), text_color="#CBA6F7").pack(pady=(16, 8))

        if not self._history:
            ctk.CTkLabel(dlg, text="Chưa có lần xuất nào.",
                         font=("Arial", 13), text_color="#6C7086").pack(pady=40)
            return

        scroll = ctk.CTkScrollableFrame(dlg, fg_color="transparent")
        scroll.pack(fill="both", expand=True, padx=16, pady=(0, 16))

        def make_row(entry):
            row = ctk.CTkFrame(scroll, fg_color="#1E1E2E", corner_radius=8)
            row.pack(fill="x", pady=3)

            left = ctk.CTkFrame(row, fg_color="transparent")
            left.pack(side="left", fill="both", expand=True, padx=10, pady=8)

            ctk.CTkLabel(left, text=entry["kanji"],
                         font=("Noto Sans JP", 15, "bold"), text_color="#CBA6F7",
                         anchor="w").pack(fill="x")
            ctk.CTkLabel(left, text=entry["time"],
                         font=("Arial", 10), text_color="#6C7086",
                         anchor="w").pack(fill="x")

            def load_this(k=entry["kanji"]):
                self.entry.delete(0, "end")
                self.entry.insert(0, k)
                self._do_lookup()
                dlg.destroy()

            ctk.CTkButton(row, text="Tra lại", width=80, height=32,
                          fg_color="#313244", hover_color="#45475A",
                          command=load_this).pack(side="right", padx=10, pady=8)

        for entry in self._history:
            make_row(entry)

    def _export_pdf(self):
        if not self._infos:
            return
        valid = [i for i in self._infos if i.get("viet") or i.get("meaning_vi")]
        if not valid:
            return

        is_vocab_table = all(i.get("source") == "json_import" for i in valid)
        kanji_str = "".join(i["kanji"] for i in valid)
        
        # Rút ngắn tên file nếu quá dài (max 10 chữ + "...")
        if len(kanji_str) > 10:
            name_part = kanji_str[:10] + f"_{len(kanji_str)}chu"
        else:
            name_part = kanji_str
            
        os.makedirs(EXPORT_DIR, exist_ok=True)
        prefix = "VocabTable_" if is_vocab_table else "Kanji_"
        filename = f"{prefix}{name_part}.pdf"
        out = os.path.join(EXPORT_DIR, filename)
        extra_rows = self._row_map[self.opt_rows.get()]
        self.lbl_pdf.configure(text="⏳ Đang xuất…", text_color="#F38BA8")
        self.update_idletasks()

        def _open_folder(path: str):
            """Mở folder chứa file, highlight file (cross-platform)."""
            try:
                if sys.platform == "win32":
                    subprocess.Popen(["explorer", "/select,", os.path.normpath(path)])
                elif sys.platform == "darwin":
                    subprocess.Popen(["open", "-R", path])
                else:
                    subprocess.Popen(["xdg-open", os.path.dirname(path)])
            except Exception as e:
                print(f"[open_folder] {e}")

        def worker():
            try:
                if is_vocab_table:
                    generate_vocab_table_pdf(valid, out)
                else:
                    generate_pdf(valid, out, extra_rows=extra_rows)
                
                # Sau khi xong, xóa chữ "Đang xuất" và hiện popup
                self.after(0, lambda: self.lbl_pdf.configure(text=""))
                self.after(0, lambda: self._add_history(kanji_str, out))
                self.after(0, lambda: self._show_success_dialog(filename, out))
            except PermissionError:
                self.after(0, lambda: self.lbl_pdf.configure(
                    text="✗ Đóng file PDF trước khi xuất!", text_color="#F38BA8"))
            except Exception as e:
                self.after(0, lambda: self.lbl_pdf.configure(
                    text=f"✗ Lỗi: {str(e)[:30]}", text_color="#F38BA8"))

        threading.Thread(target=worker, daemon=True).start()

    def _show_success_dialog(self, filename, full_path):
        dlg = ctk.CTkToplevel(self)
        dlg.title("Xuất PDF thành công")
        dlg.geometry("420x220")
        dlg.resizable(False, False)
        dlg.attributes("-topmost", True)
        dlg.grab_set()

        # Center dialog
        self.update_idletasks()
        x = self.winfo_x() + (self.winfo_width() // 2) - 210
        y = self.winfo_y() + (self.winfo_height() // 2) - 110
        dlg.geometry(f"+{max(0, x)}+{max(0, y)}")

        ctk.CTkLabel(dlg, text="🎉 Đã tạo file thành công!", 
                     font=("Arial", 18, "bold"), text_color="#A6E3A1").pack(pady=(30, 5))
        
        # Hiển thị tên file (rút gọn nếu quá dài để ko tràn dialog)
        display_name = filename if len(filename) < 50 else filename[:47] + "..."
        ctk.CTkLabel(dlg, text=display_name, font=("Arial", 13), text_color="#CDD6F4").pack(pady=5)
        
        btn_frame = ctk.CTkFrame(dlg, fg_color="transparent")
        btn_frame.pack(pady=25)

        def open_and_close():
            try:
                if sys.platform == "win32":
                    subprocess.Popen(["explorer", "/select,", os.path.normpath(full_path)])
                elif sys.platform == "darwin":
                    subprocess.Popen(["open", "-R", full_path])
                else:
                    subprocess.Popen(["xdg-open", os.path.dirname(full_path)])
            except: pass
            dlg.destroy()

        ctk.CTkButton(btn_frame, text="Mở thư mục", width=120, height=35,
                      fg_color="#313244", hover_color="#45475A",
                      command=open_and_close).pack(side="left", padx=10)
        
        ctk.CTkButton(btn_frame, text="Đóng", width=100, height=35,
                      fg_color="#1E1E2E", border_width=1, border_color="#45475A",
                      command=dlg.destroy).pack(side="left", padx=10)

    def _load_progress(self) -> dict:
        try:
            with open(PROGRESS_FILE, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {"N5": [], "N4": []}

    def _save_progress(self):
        try:
            with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
                json.dump(self._progress, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def _open_learning_path(self):
        dlg = ctk.CTkToplevel(self)
        dlg.title("Lộ trình học Kanji")
        dlg.geometry("640x540")
        dlg.resizable(False, True)
        dlg.grab_set()
        dlg.after(100, dlg.lift)
        dlg.after(100, dlg.focus_force)

        ctk.CTkLabel(dlg, text="🗺️ Lộ trình học Kanji",
                     font=("Arial", 18, "bold"), text_color="#CBA6F7").pack(pady=(16, 8))

        tabview = ctk.CTkTabview(dlg, width=600, height=440)
        tabview.pack(padx=20, pady=10, fill="both", expand=True)
        tabview.add("N5")
        tabview.add("N4")
        tabview.add("N3")
        tabview.add("N2")
        tabview.add("N1")
        tabview.add("Từ Vựng")

        def chunker(seq, size):
            return [seq[pos:pos + size] for pos in range(0, len(seq), size)]

        tab_data = {
            "N5": chunker(list(MNN_N5.keys()), 10),
            "N4": chunker(list(N4_VI.keys()), 10),
            "N3": chunker(list(N3_VI.keys()), 10),
            "N2": chunker(list(N2_VI.keys()), 10),
            "N1": chunker(list(N1_VI.keys()), 10),
        }
        rendered = set()  # track which tabs already built

        def make_tab(tab_name):
            if tab_name in rendered:
                return
            rendered.add(tab_name)
            chunks = tab_data[tab_name]
            parent_tab = tabview.tab(tab_name)

            scroll = ctk.CTkScrollableFrame(parent_tab, fg_color="transparent")
            scroll.pack(fill="both", expand=True, padx=10, pady=10)

            for i, chunk in enumerate(chunks):
                lesson_id = f"{tab_name}_lesson_{i+1}"
                is_done = lesson_id in self._progress.get(tab_name, [])

                row = ctk.CTkFrame(scroll, fg_color="#1E1E2E", corner_radius=8)
                row.pack(fill="x", pady=4)

                left = ctk.CTkFrame(row, fg_color="transparent")
                left.pack(side="left", fill="both", expand=True, padx=10, pady=8)

                title_color = "#A6E3A1" if is_done else "#CDD6F4"
                check_mark = "✅ " if is_done else "📖 "

                ctk.CTkLabel(left, text=f"{check_mark}Bài {i+1} ({len(chunk)} chữ)",
                             font=("Arial", 14, "bold"), text_color=title_color,
                             anchor="w").pack(fill="x")
                ctk.CTkLabel(left, text=" ".join(chunk),
                             font=("Noto Sans JP", 14), text_color="#89B4FA",
                             anchor="w").pack(fill="x")

                def on_learn(c=chunk, lid=lesson_id, t=tab_name):
                    if t not in self._progress:
                        self._progress[t] = []
                    if lid not in self._progress[t]:
                        self._progress[t].append(lid)
                        self._save_progress()
                    self.entry.delete(0, "end")
                    self.entry.insert(0, "".join(c))
                    self._do_lookup()
                    dlg.destroy()

                btn_text = "Ôn lại" if is_done else "Học ngay"
                btn_color = "#313244" if is_done else "#0F52BA"
                btn_hover = "#45475A" if is_done else "#1E65D0"

                ctk.CTkButton(row, text=btn_text, width=80, height=32,
                              fg_color=btn_color, hover_color=btn_hover,
                              command=on_learn).pack(side="right", padx=10)

        def on_tab_change():
            tab = tabview.get()
            if tab == "Từ Vựng":
                make_vocab_tab()
            else:
                make_tab(tab)

        def make_vocab_tab():
            if "Từ Vựng" in rendered:
                return
            rendered.add("Từ Vựng")
            parent_tab = tabview.tab("Từ Vựng")
            scroll = ctk.CTkScrollableFrame(parent_tab, fg_color="transparent")
            scroll.pack(fill="both", expand=True, padx=10, pady=10)

            lesson_nums = sorted(VOCAB_LESSONS.keys())
            if not lesson_nums:
                ctk.CTkLabel(scroll, text="Chưa có bài từ vựng nào.",
                             font=("Arial", 13), text_color="#6C7086").pack(pady=20)
                return

            for lesson_num in lesson_nums:
                words = VOCAB_LESSONS[lesson_num]
                lesson_id = f"vocab_lesson_{lesson_num}"
                is_done = lesson_id in self._progress.get("Từ Vựng", [])

                row = ctk.CTkFrame(scroll, fg_color="#1E1E2E", corner_radius=8)
                row.pack(fill="x", pady=4)

                left = ctk.CTkFrame(row, fg_color="transparent")
                left.pack(side="left", fill="both", expand=True, padx=10, pady=8)

                check_mark = "✅ " if is_done else "📖 "
                title_color = "#A6E3A1" if is_done else "#CDD6F4"
                preview = "  ".join(w["word"] for w in words[:5]) + "…"

                ctk.CTkLabel(left, text=f"{check_mark}Bài {lesson_num}  ({len(words)} từ)",
                             font=("Arial", 14, "bold"), text_color=title_color,
                             anchor="w").pack(fill="x")
                ctk.CTkLabel(left, text=preview,
                             font=("Noto Sans JP", 13), text_color="#89B4FA",
                             anchor="w").pack(fill="x")

                def on_study(ln=lesson_num, lid=lesson_id):
                    if "Từ Vựng" not in self._progress:
                        self._progress["Từ Vựng"] = []
                    if lid not in self._progress["Từ Vựng"]:
                        self._progress["Từ Vựng"].append(lid)
                        self._save_progress()
                    self._open_vocab_lesson(ln)

                btn_text = "Ôn lại" if is_done else "Học ngay"
                btn_color = "#313244" if is_done else "#0F52BA"
                btn_hover = "#45475A" if is_done else "#1E65D0"
                ctk.CTkButton(row, text=btn_text, width=80, height=32,
                              fg_color=btn_color, hover_color=btn_hover,
                              command=on_study).pack(side="right", padx=10)

        tabview.configure(command=on_tab_change)

        # Render tab đầu tiên ngay lập tức
        make_tab("N5")

    def _open_vocab_lesson(self, lesson_num: int):
        """Mở dialog học từ vựng cho một bài cụ thể."""
        words = VOCAB_LESSONS.get(lesson_num, [])
        if not words:
            return

        dlg = ctk.CTkToplevel(self)
        dlg.title(f"📚 Từ Vựng Bài {lesson_num}")
        dlg.geometry("700x600")
        dlg.resizable(True, True)
        dlg.grab_set()
        dlg.after(100, dlg.lift)
        dlg.after(100, dlg.focus_force)

        # Center popup
        self.update_idletasks()
        x = self.winfo_x() + (self.winfo_width() // 2) - 350
        y = self.winfo_y() + (self.winfo_height() // 2) - 300
        dlg.geometry(f"+{max(0, x)}+{max(0, y)}")

        # Header
        hdr = ctk.CTkFrame(dlg, fg_color="#0D1B2A", corner_radius=0)
        hdr.pack(fill="x")
        ctk.CTkLabel(hdr, text=f"📚  Từ Vựng Bài {lesson_num}",
                     font=("Arial", 18, "bold"), text_color="#CBA6F7").pack(side="left", padx=20, pady=12)
        ctk.CTkLabel(hdr, text=f"{len(words)} từ",
                     font=("Arial", 12), text_color="#6C7086").pack(side="right", padx=20)

        # Scrollable word list
        scroll = ctk.CTkScrollableFrame(dlg, fg_color="transparent")
        scroll.pack(fill="both", expand=True, padx=12, pady=10)

        for idx, item in enumerate(words):
            card = ctk.CTkFrame(scroll, fg_color="#1E1E2E", corner_radius=8, border_width=1,
                                border_color="#313244")
            card.pack(fill="x", pady=4, padx=2)

            # Số thứ tự + từ + reading
            top = ctk.CTkFrame(card, fg_color="transparent")
            top.pack(fill="x", padx=12, pady=(8, 2))

            ctk.CTkLabel(top, text=f"{idx + 1}.", font=("Arial", 12),
                         text_color="#6C7086", width=24, anchor="w").pack(side="left")

            ctk.CTkLabel(top, text=item["word"], font=("Noto Sans JP", 20, "bold"),
                         text_color="#CDD6F4", anchor="w").pack(side="left", padx=(4, 0))

            ctk.CTkLabel(top, text=f"（{item['reading']}）", font=("Noto Sans JP", 14),
                         text_color="#89B4FA", anchor="w").pack(side="left", padx=(6, 0))

            if item.get("hanviet"):
                ctk.CTkLabel(top, text=item["hanviet"], font=("Arial", 11, "bold"),
                             text_color="#CBA6F7", anchor="e").pack(side="right")

            # Nghĩa tiếng Việt
            ctk.CTkLabel(card, text=f"  ▸  {item['meaning']}",
                         font=("Arial", 13), text_color="#A6E3A1",
                         anchor="w").pack(fill="x", padx=12, pady=(0, 4))

            # Ví dụ
            if item.get("example"):
                ex_frame = ctk.CTkFrame(card, fg_color="#181825", corner_radius=6)
                ex_frame.pack(fill="x", padx=12, pady=(0, 8))
                ctk.CTkLabel(ex_frame, text=f"  {item['example']}",
                             font=("Noto Sans JP", 12), text_color="#F5C2E7",
                             anchor="w").pack(fill="x", padx=6, pady=(4, 0))
                if item.get("exampleVi"):
                    ctk.CTkLabel(ex_frame, text=f"  → {item['exampleVi']}",
                                 font=("Arial", 11), text_color="#CDD6F4",
                                 anchor="w").pack(fill="x", padx=6, pady=(2, 4))

            # Nút phát âm
            def _speak(t=item["word"]):
                speak(t)
            ctk.CTkButton(card, text="🔊", width=40, height=24,
                          font=("Arial", 12), fg_color="#1E3A5F", hover_color="#2A5080",
                          command=_speak).place(relx=1.0, rely=0.0, x=-50, y=8)

    def _open_vocab_lookup(self):
        """Mở cửa sổ tra cứu từ vựng tiếng Nhật."""
        dlg = ctk.CTkToplevel(self)
        dlg.title("📖 Tra Cứu Từ Vựng")
        dlg.geometry("740x680")
        dlg.resizable(True, True)
        dlg.grab_set()
        dlg.after(100, dlg.lift)
        dlg.after(100, dlg.focus_force)

        # Căn giữa popup
        self.update_idletasks()
        x = self.winfo_x() + (self.winfo_width() // 2) - 370
        y = self.winfo_y() + (self.winfo_height() // 2) - 340
        dlg.geometry(f"+{max(0, x)}+{max(0, y)}")

        # ── Header ──────────────────────────────────────────────────────────
        hdr = ctk.CTkFrame(dlg, fg_color="#0D1B2A", corner_radius=0)
        hdr.pack(fill="x")
        ctk.CTkLabel(hdr, text="📖  Tra Cứu Từ Vựng",
                     font=("Arial", 20, "bold"), text_color="#89B4FA").pack(side="left", padx=20, pady=14)
        ctk.CTkLabel(hdr, text="Mazii · Jisho · AI",
                     font=("Arial", 11), text_color="#45475A").pack(side="right", padx=20)

        # ── Ô tìm kiếm ──────────────────────────────────────────────────────
        search_frame = ctk.CTkFrame(dlg, fg_color="transparent")
        search_frame.pack(fill="x", padx=20, pady=(14, 4))

        ent_search = ctk.CTkEntry(
            search_frame,
            placeholder_text="Nhập từ tiếng Nhật (食べ物, 勉強する…) hoặc tiếng Việt",
            font=("Noto Sans JP", 15), height=46,
        )
        ent_search.pack(side="left", fill="x", expand=True)

        # Điền sẵn từ đang xem (nếu là từ đơn)
        current_text = self.entry.get().strip()
        if current_text and len(current_text) <= 6:
            ent_search.insert(0, current_text)

        lbl_src = ctk.CTkLabel(dlg, text="", font=("Arial", 11), text_color="#6C7086")
        lbl_src.pack(pady=(2, 0))

        # Khu vực kết quả (scroll)
        result_scroll = ctk.CTkScrollableFrame(dlg, fg_color="transparent")
        result_scroll.pack(fill="both", expand=True, padx=20, pady=(4, 4))

        # ── Hàm nội bộ ──────────────────────────────────────────────────────
        def _clear_results():
            for w in result_scroll.winfo_children():
                w.destroy()

        def _show_loading():
            _clear_results()
            lbl_src.configure(text="⏳ Đang tra cứu…", text_color="#6C7086")
            frame_load = ctk.CTkFrame(result_scroll, fg_color="#1E1E2E", corner_radius=12)
            frame_load.pack(fill="x", pady=20)
            ctk.CTkLabel(frame_load, text="⏳", font=("Arial", 40)).pack(pady=(30, 4))
            ctk.CTkLabel(frame_load, text="Đang tra cứu từ điển…",
                         font=("Arial", 14), text_color="#6C7086").pack(pady=(0, 30))

        def _build_result(result: dict, query: str):
            _clear_results()

            if not result or not result.get("meanings_vi"):
                lbl_src.configure(text="✗ Không tìm thấy", text_color="#E74C3C")
                frame_err = ctk.CTkFrame(result_scroll, fg_color="#1E1E2E", corner_radius=12)
                frame_err.pack(fill="x", pady=20)
                ctk.CTkLabel(frame_err, text="😅", font=("Arial", 40)).pack(pady=(30, 4))
                ctk.CTkLabel(
                    frame_err,
                    text=f"Không tìm thấy '{query}'.\nThử từ khác hoặc bật AI trong Cài đặt.",
                    font=("Arial", 13), text_color="#6C7086", justify="center",
                ).pack(pady=(0, 30))
                return

            # Nhãn nguồn
            source = result.get("source", "")
            src_map = {
                "mazii":       ("🇻🇳 Mazii Dictionary", "#4CAF50"),
                "jisho":       ("⚡ Jisho API",          "#F39C12"),
                "gemini":      ("✨ Gemini AI",           "#A78BFA"),
                "openrouter":  ("🚀 OpenRouter AI",      "#89B4FA"),
            }
            src_text, src_color = src_map.get(source, ("🔍 Kết quả", "#6C7086"))
            lbl_src.configure(text=src_text, text_color=src_color)

            word_display = result.get("word", query)
            reading      = result.get("reading", "")
            han_viet     = result.get("han_viet", "")
            meanings     = result.get("meanings_vi", [])
            examples     = result.get("examples", [])
            related      = result.get("related", [])

            # ── Card chính ────────────────────────────────────────────────
            card = ctk.CTkFrame(result_scroll, fg_color="#1E1E2E",
                                corner_radius=14, border_width=1, border_color="#313244")
            card.pack(fill="x", pady=(0, 10))

            # Dòng từ + đọc + HV + TTS
            top_row = ctk.CTkFrame(card, fg_color="transparent")
            top_row.pack(fill="x", padx=16, pady=(16, 0))

            # Cột chữ lớn
            lbl_word = ctk.CTkLabel(
                top_row, text=word_display,
                font=("Noto Sans JP", 48, "bold"), text_color="#CDD6F4",
                cursor="hand2",
            )
            lbl_word.pack(side="left", anchor="w")
            lbl_word.bind("<Button-1>", lambda e, w=word_display: speak(w))

            # Cột info bên phải từ
            info_col = ctk.CTkFrame(top_row, fg_color="transparent")
            info_col.pack(side="left", padx=(12, 0), anchor="sw", pady=(0, 6))
            if reading:
                ctk.CTkLabel(info_col, text=reading,
                             font=("Noto Sans JP", 16), text_color="#89B4FA",
                             anchor="w").pack(anchor="w")
            if han_viet:
                ctk.CTkLabel(info_col, text=han_viet,
                             font=("Arial", 14, "bold"), text_color="#CBA6F7",
                             anchor="w").pack(anchor="w")

            # Nút 🔊 phát âm góc phải
            ctk.CTkButton(
                top_row, text="🔊", width=44, height=36,
                font=("Arial", 14),
                fg_color="#1E3A5F", hover_color="#2A5080",
                command=lambda w=word_display: speak(w),
            ).pack(side="right", anchor="se", pady=(0, 6))

            # Separator
            ctk.CTkFrame(card, height=1, fg_color="#313244").pack(fill="x", padx=16, pady=(10, 6))

            # ── Nghĩa ─────────────────────────────────────────────────────
            if meanings:
                ctk.CTkLabel(card, text="📝 Nghĩa:",
                             font=("Arial", 13, "bold"), text_color="#A6E3A1",
                             anchor="w").pack(fill="x", padx=16)
                for i, m in enumerate(meanings[:4]):
                    ctk.CTkLabel(
                        card,
                        text=f"  {i+1}. {m}",
                        font=("Arial", 13), text_color="#CDD6F4",
                        anchor="w", wraplength=650, justify="left",
                    ).pack(fill="x", padx=16, pady=(1, 0))

            # ── Ví dụ câu ─────────────────────────────────────────────────
            if examples:
                ctk.CTkFrame(card, height=1, fg_color="#313244").pack(fill="x", padx=16, pady=(10, 4))
                ctk.CTkLabel(card, text="💬 Ví dụ câu:",
                             font=("Arial", 13, "bold"), text_color="#F9E2AF",
                             anchor="w").pack(fill="x", padx=16)

                for ex in examples[:3]:
                    ex_frame = ctk.CTkFrame(card, fg_color="#252540", corner_radius=8)
                    ex_frame.pack(fill="x", padx=16, pady=(4, 0))

                    sentence   = ex.get("sentence", "")
                    reading_ex = ex.get("reading", "")
                    meaning_ex = ex.get("meaning", "")

                    if sentence:
                        sen_row = ctk.CTkFrame(ex_frame, fg_color="transparent")
                        sen_row.pack(fill="x", padx=10, pady=(6, 0))
                        ctk.CTkLabel(
                            sen_row, text=sentence,
                            font=("Noto Sans JP", 13), text_color="#CDD6F4",
                            anchor="w", wraplength=570, justify="left",
                        ).pack(side="left", fill="x", expand=True)
                        ctk.CTkButton(
                            sen_row, text="🔊", width=30, height=24,
                            font=("Arial", 11),
                            fg_color="#1E3A5F", hover_color="#2A5080",
                            command=lambda s=sentence: speak(s),
                        ).pack(side="right")

                    if reading_ex:
                        ctk.CTkLabel(ex_frame, text=reading_ex,
                                     font=("Noto Sans JP", 11), text_color="#89B4FA",
                                     anchor="w").pack(fill="x", padx=10)
                    if meaning_ex:
                        ctk.CTkLabel(ex_frame, text=f"→ {meaning_ex}",
                                     font=("Arial", 12), text_color="#A6E3A1",
                                     anchor="w", wraplength=640, justify="left",
                                     ).pack(fill="x", padx=10, pady=(0, 6))

            # ── Từ liên quan ───────────────────────────────────────────────
            if related:
                ctk.CTkFrame(card, height=1, fg_color="#313244").pack(fill="x", padx=16, pady=(10, 4))
                ctk.CTkLabel(card, text="🔗 Từ liên quan:",
                             font=("Arial", 13, "bold"), text_color="#F5C2E7",
                             anchor="w").pack(fill="x", padx=16)
                rel_row = ctk.CTkFrame(card, fg_color="transparent")
                rel_row.pack(fill="x", padx=16, pady=(2, 12))

                for rel in related[:5]:
                    if not isinstance(rel, (list, tuple)) or len(rel) < 1:
                        continue
                    rel_w = rel[0]
                    rel_r = rel[1] if len(rel) > 1 else ""
                    rel_m = rel[2] if len(rel) > 2 else ""
                    btn_lbl = f"{rel_w}（{rel_r}）" if rel_r else rel_w
                    tip_text = rel_m[:30] if rel_m else ""

                    def _on_rel_click(w=rel_w):
                        ent_search.delete(0, "end")
                        ent_search.insert(0, w)
                        do_search()

                    rel_btn = ctk.CTkButton(
                        rel_row, text=btn_lbl,
                        font=("Noto Sans JP", 12),
                        fg_color="#313244", hover_color="#45475A",
                        height=30,
                        command=_on_rel_click,
                    )
                    rel_btn.pack(side="left", padx=(0, 4), pady=2)

            else:
                ctk.CTkFrame(card, height=12, fg_color="transparent").pack()  # bottom padding

        def do_search(event=None):
            word = ent_search.get().strip()
            if not word:
                return
            _show_loading()
            dlg.update_idletasks()

            def worker():
                result = lookup_vocab(word)
                dlg.after(0, lambda r=result: _build_result(r, word))

            threading.Thread(target=worker, daemon=True).start()

        # Nút tìm + phát âm
        btn_search = ctk.CTkButton(
            search_frame, text="🔍 Tra", width=78, height=46,
            font=("Arial", 13, "bold"),
            command=do_search,
        )
        btn_search.pack(side="left", padx=(6, 0))

        ctk.CTkButton(
            search_frame, text="🔊", width=46, height=46,
            font=("Arial", 14),
            fg_color="#1E3A5F", hover_color="#2A5080",
            command=lambda: speak(ent_search.get().strip()),
        ).pack(side="left", padx=(4, 0))

        ent_search.bind("<Return>", do_search)

        # Tự tra nếu ô đã điền sẵn từ
        if current_text and len(current_text) <= 6 and has_cjk(current_text):
            dlg.after(250, do_search)

    def _open_flashcard(self):
        valid_infos = [i for i in self._infos if i.get("viet")]
        if not valid_infos:
            return

        import random

        dlg = ctk.CTkToplevel(self)
        dlg.title("Chế độ Flashcard")
        dlg.geometry("700x560")
        dlg.resizable(False, False)
        dlg.grab_set()
        dlg.after(100, dlg.lift)
        dlg.after(100, dlg.focus_force)

        # Trạng thái flashcard
        state = {
            "index": 0,
            "flipped": False,
            "is_animating": False,
            "cards": valid_infos[:]
        }

        # Khung bọc ngoài giữ nguyên kích thước
        container = ctk.CTkFrame(dlg, fg_color="transparent")
        container.pack(fill="both", expand=True, padx=40, pady=(30, 10))

        # Khung chứa card có thể thay đổi relwidth để tạo hiệu ứng lật
        card_frame = ctk.CTkFrame(container, fg_color="#1E1E2E", corner_radius=20)
        card_frame.place(relx=0.5, rely=0.5, relwidth=1.0, relheight=1.0, anchor="center")

        lbl_kanji = ctk.CTkLabel(card_frame, text="", font=("Noto Sans JP", 140), text_color="#CDD6F4")
        lbl_kanji.place(relx=0.5, rely=0.45, anchor="center")

        lbl_reading = ctk.CTkLabel(card_frame, text="", font=("Noto Sans JP", 20), text_color="#89B4FA")
        lbl_reading.place(relx=0.5, rely=0.15, anchor="center")

        lbl_viet = ctk.CTkLabel(card_frame, text="", font=("Arial", 26, "bold"), text_color="#CBA6F7")
        lbl_viet.place(relx=0.5, rely=0.75, anchor="center")
        
        lbl_meaning = ctk.CTkLabel(card_frame, text="", font=("Arial", 18), text_color="#A6E3A1")
        lbl_meaning.place(relx=0.5, rely=0.85, anchor="center")

        lbl_vocab = ctk.CTkLabel(card_frame, text="", font=("Arial", 14), text_color="#F5C2E7", justify="center")
        lbl_vocab.place(relx=0.5, rely=0.95, anchor="center")

        def update_card():
            card = state["cards"][state["index"]]
            lbl_progress.configure(text=f"{state['index'] + 1} / {len(state['cards'])}")
            
            lbl_kanji.configure(text=card["kanji"])
            lbl_kanji.place(relx=0.5, rely=0.4 if state["flipped"] else 0.5, anchor="center")
            
            if state["flipped"]:
                lbl_reading.configure(text=card.get("reading", ""))
                lbl_viet.configure(text=card.get("viet", ""))
                meaning = card.get("meaning_vi", "") or card.get("meaning", "")
                lbl_meaning.configure(text=meaning)
                
                vocab_lines = []
                if card.get("vocab"):
                    for v in card["vocab"][:2]:
                        if isinstance(v, dict):
                            vocab_lines.append(f"{v.get('word','')} ({v.get('reading','')}): {v.get('meaning','')}")
                        elif isinstance(v, (list, tuple)) and len(v) >= 3:
                            vocab_lines.append(f"{v[0]} ({v[1]}): {v[2]}")
                lbl_vocab.configure(text="\n".join(vocab_lines))
                
                lbl_reading.place(relx=0.5, rely=0.15, anchor="center")
                lbl_viet.place(relx=0.5, rely=0.75, anchor="center")
                lbl_meaning.place(relx=0.5, rely=0.85, anchor="center")
                lbl_vocab.place(relx=0.5, rely=0.95, anchor="center")
            else:
                lbl_reading.place_forget()
                lbl_viet.place_forget()
                lbl_meaning.place_forget()
                lbl_vocab.place_forget()

        def animate_flip(step=0, direction=-1):
            if step == 0 and direction == -1:
                state["is_animating"] = True
                lbl_kanji.place_forget()
                lbl_reading.place_forget()
                lbl_viet.place_forget()
                lbl_meaning.place_forget()
                lbl_vocab.place_forget()
            
            w = max(0.01, 1.0 - (step * 0.15)) if direction == -1 else min(1.0, step * 0.15)
            card_frame.place_configure(relwidth=w)
            
            if direction == -1 and step >= 6:
                state["flipped"] = not state["flipped"]
                dlg.after(15, lambda: animate_flip(1, 1))
            elif direction == 1 and step >= 6:
                card_frame.place_configure(relwidth=1.0)
                update_card()
                state["is_animating"] = False
            else:
                dlg.after(15, lambda: animate_flip(step + 1, direction))

        def flip(event=None):
            if state["is_animating"]: return
            animate_flip()

        def next_card(event=None):
            if state["is_animating"]: return
            if state["index"] < len(state["cards"]) - 1:
                state["index"] += 1
                state["flipped"] = False
                update_card()

        def prev_card(event=None):
            if state["is_animating"]: return
            if state["index"] > 0:
                state["index"] -= 1
                state["flipped"] = False
                update_card()

        def shuffle_cards():
            if state["is_animating"]: return
            random.shuffle(state["cards"])
            state["index"] = 0
            state["flipped"] = False
            update_card()

        # Click thẻ để lật
        card_frame.bind("<Button-1>", flip)
        lbl_kanji.bind("<Button-1>", flip)

        # Phím tắt
        dlg.bind("<space>", flip)
        dlg.bind("<Right>", next_card)
        dlg.bind("<Left>", prev_card)
        dlg.bind("<Up>", flip)
        dlg.bind("<Down>", flip)

        # Điều khiển
        ctrl_frame = ctk.CTkFrame(dlg, fg_color="transparent")
        ctrl_frame.pack(fill="x", padx=40, pady=(0, 20))

        btn_prev = ctk.CTkButton(ctrl_frame, text="◀ Trước", width=100, height=36, command=prev_card)
        btn_prev.pack(side="left")

        lbl_progress = ctk.CTkLabel(ctrl_frame, text="", font=("Arial", 14, "bold"), text_color="#6C7086")
        lbl_progress.pack(side="left", expand=True)

        btn_shuffle = ctk.CTkButton(ctrl_frame, text="🔀 Xáo trộn", width=100, height=36, 
                                    fg_color="#313244", hover_color="#45475A", command=shuffle_cards)
        btn_shuffle.pack(side="left", padx=10)

        btn_next = ctk.CTkButton(ctrl_frame, text="Tiếp ▶", width=100, height=36, command=next_card)
        btn_next.pack(side="right")

        update_card()

# ─── Chạy ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = App()
    app.mainloop()

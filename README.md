# Kanji Hub — Tra cứu & Luyện viết Kanji

Ứng dụng tra cứu kanji tiếng Nhật cho người Việt, tự động tạo file PDF luyện viết.

---

## Tính năng

- Tra cứu kanji theo chữ hoặc âm Hán Việt / nghĩa tiếng Việt
- Cơ sở dữ liệu N4/N5 với hơn 257 kanji có mẹo nhớ
- Tích hợp Gemini AI để tra kanji ngoài DB
- Phát âm chuẩn tiếng Nhật (Microsoft Neural TTS)
- Xuất file PDF luyện viết với ô mẫu + ô luyện mờ dần
- Lịch sử xuất PDF để tra lại nhanh

---

## Chạy trên Windows

### Lần đầu

1. Cài [Python 3.10+](https://python.org/downloads) — tick **Add to PATH**
2. Mở PowerShell trong thư mục project, chạy:
   ```powershell
   python -m venv .venv
   .venv\Scripts\activate
   pip install customtkinter reportlab google-genai edge-tts requests
   ```

### Mở app

Double-click **`KanjiApp.vbs`** — không hiện cửa sổ terminal.

Hoặc chạy thẳng:
```powershell
python gui.py
```

---

## Chạy trên macOS

### Yêu cầu

- macOS 11 (Big Sur) trở lên
- Python 3.10+ — tải tại [python.org/downloads](https://python.org/downloads)

### Lần đầu (chỉ làm 1 lần)

Mở Terminal, chạy 2 lệnh sau (thay đường dẫn cho đúng):
```bash
# 1. Xóa cảnh báo "Apple could not verify" cho toàn bộ thư mục
xattr -rd com.apple.quarantine ~/Desktop/TaoFileVietKanji

# 2. Cấp quyền chạy cho script launcher
chmod +x ~/Desktop/TaoFileVietKanji/KanjiApp_mac.command
```

### Mở app

Double-click **`KanjiApp_mac.command`**

Script tự làm hết: tạo môi trường Python, cài packages, tải font, mở app.

> **Vẫn bị chặn?** Vào **System Settings → Privacy & Security** → kéo xuống → nhấn **"Allow Anyway"**

---

## Cài Gemini API Key (tùy chọn)

Dùng để tra kanji ngoài DB và tìm theo nghĩa tiếng Việt bằng AI.

1. Lấy key miễn phí tại [aistudio.google.com](https://aistudio.google.com)
2. Trong app nhấn **⚙️ Gemini API** góc trên phải → dán key vào → Lưu

---

## Cách dùng

| Nhập | Kết quả |
|------|---------|
| `山川田` | Tra thông tin từng kanji |
| `Học` | Tìm kanji có âm HV là "Học" |
| `núi` | Tìm kanji có nghĩa là "núi" (cần Gemini AI) |

**Xuất PDF luyện viết:**
1. Tra kanji muốn luyện
2. Chọn số hàng ô luyện (1–4 hàng)
3. Nhấn **📄 Xuất PDF luyện viết**
4. File lưu vào thư mục `exports/`

---

## Cấu trúc thư mục

```
TaoFileVietKanji/
├── gui.py                  # Giao diện chính
├── kanji_lookup.py         # Cơ sở dữ liệu + tra cứu
├── pdf_generator.py        # Tạo file PDF
├── config.json             # Lưu Gemini API key
├── history.json            # Lịch sử xuất PDF
├── exports/                # Thư mục chứa file PDF xuất ra
├── KanjiApp.vbs            # Launcher Windows
├── KanjiApp_mac.command    # Launcher macOS
└── NotoSansJP-VF.ttf       # Font (tự tải về khi chạy trên Mac)
```

---

## Tác giả

**Kanji Hub by Phong Vu**

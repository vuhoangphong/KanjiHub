#!/bin/bash
# ────────────────────────────────────────────────────────────────────────────
# KanjiApp_mac.command — Double-click để chạy Kanji Hub trên macOS
# Lần đầu chạy: tự tạo venv + cài packages + tải font tự động
# ────────────────────────────────────────────────────────────────────────────

# Chuyển vào thư mục chứa script này
cd "$(dirname "$0")"

# ── Tìm Python 3 ─────────────────────────────────────────────────────────────
PYTHON=""
for p in /opt/homebrew/bin/python3 /usr/local/bin/python3 python3; do
    if command -v "$p" &>/dev/null; then
        PYTHON="$p"
        break
    fi
done

if [ -z "$PYTHON" ]; then
    osascript -e 'display alert "Chưa cài Python 3" message "Vào python.org tải Python 3.10+ rồi chạy lại."'
    exit 1
fi

# ── Kiểm tra Python 3.10+ ────────────────────────────────────────────────────
PY_VER=$("$PYTHON" -c "import sys; print(sys.version_info >= (3,10))" 2>/dev/null)
if [ "$PY_VER" != "True" ]; then
    osascript -e 'display alert "Python quá cũ" message "App cần Python 3.10 trở lên. Vào python.org tải bản mới nhất rồi chạy lại."'
    exit 1
fi

# ── Tạo venv nếu chưa có ─────────────────────────────────────────────────────
if [ ! -f ".venv/bin/python" ]; then
    echo "⏳ Tạo môi trường Python (chỉ lần đầu)..."
    "$PYTHON" -m venv .venv
fi

PY=".venv/bin/python"
PIP=".venv/bin/pip"

# ── Cài packages nếu thiếu ───────────────────────────────────────────────────
MISSING=0
"$PY" -c "import customtkinter" 2>/dev/null || MISSING=1
"$PY" -c "import reportlab" 2>/dev/null || MISSING=1
"$PY" -c "import google.genai" 2>/dev/null || MISSING=1
"$PY" -c "import edge_tts" 2>/dev/null || MISSING=1
"$PY" -c "import requests" 2>/dev/null || MISSING=1

if [ "$MISSING" -eq 1 ]; then
    echo "⏳ Cài packages (chỉ lần đầu)..."
    "$PIP" install -q --upgrade pip
    "$PIP" install -q customtkinter reportlab google-genai edge-tts requests
fi

# ── Tải font Noto Sans JP nếu chưa có ────────────────────────────────────────
FONT_FILE="NotoSansJP-VF.ttf"
if [ ! -f "$FONT_FILE" ]; then
    echo "⏳ Tải font Noto Sans JP (chỉ lần đầu)..."
    FONT_URL="https://github.com/google/fonts/raw/main/ofl/notosansjp/NotoSansJP%5Bwght%5D.ttf"
    curl -sL "$FONT_URL" -o "$FONT_FILE" || {
        # Fallback URL
        curl -sL "https://fonts.gstatic.com/s/notosansjp/v52/-F6jfjtqLzI2JPCgQBnw7HFyzSD-AsregP8VFBEi75vY0rw-oME.ttf" -o "$FONT_FILE" || true
    }
    if [ -f "$FONT_FILE" ] && [ -s "$FONT_FILE" ]; then
        echo "✓ Font đã tải xong"
    else
        echo "⚠ Không tải được font, kanji trong PDF có thể không hiện — app vẫn chạy bình thường"
        rm -f "$FONT_FILE"
    fi
fi

# ── Chạy app ─────────────────────────────────────────────────────────────────
echo "🚀 Khởi động Kanji Hub..."
"$PY" gui.py

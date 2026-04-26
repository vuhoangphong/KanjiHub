#!/usr/bin/env python3
"""
Script thêm bài từ vựng mới vào vocab_lessons.py.

Cách dùng:
  1. Paste JSON vào file input.json (hoặc truyền qua stdin)
  2. Chạy: python add_vocab_lesson.py <số_bài> [input.json]

Ví dụ:
  python add_vocab_lesson.py 28 bai28.json
  python add_vocab_lesson.py 28          (nhập JSON thủ công rồi Ctrl+Z/Ctrl+D)
"""
import json
import sys
import os
import re

VOCAB_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vocab_lessons.py")


def load_json(lesson_num: int, json_file: str | None) -> list[dict]:
    if json_file:
        with open(json_file, encoding="utf-8") as f:
            data = json.load(f)
    else:
        print(f"Nhập JSON cho bài {lesson_num} (kết thúc bằng Ctrl+Z / Ctrl+D):")
        raw = sys.stdin.read()
        data = json.load(raw if hasattr(raw, "read") else __import__("io").StringIO(raw))

    # Normalize fields
    result = []
    for item in data:
        result.append({
            "word": item.get("word", ""),
            "reading": item.get("reading", ""),
            "hanviet": item.get("hanviet", ""),
            "meaning": item.get("meaning", ""),
            "example": item.get("example", ""),
            "exampleVi": item.get("exampleVi", ""),
        })
    return result


def format_lesson(lesson_num: int, words: list[dict]) -> str:
    lines = [f"    {lesson_num}: ["]
    for w in words:
        def esc(s): return s.replace("\\", "\\\\").replace('"', '\\"')
        lines.append(
            f'        {{"word": "{esc(w["word"])}", "reading": "{esc(w["reading"])}", '
            f'"hanviet": "{esc(w["hanviet"])}", "meaning": "{esc(w["meaning"])}", '
            f'"example": "{esc(w["example"])}", "exampleVi": "{esc(w["exampleVi"])}"}},')
    lines.append("    ],")
    return "\n".join(lines)


def inject(lesson_num: int, words: list[dict]):
    with open(VOCAB_FILE, encoding="utf-8") as f:
        content = f.read()

    # Kiểm tra bài đã tồn tại chưa
    pattern = re.compile(rf"^\s+{lesson_num}\s*:", re.MULTILINE)
    if pattern.search(content):
        print(f"⚠  Bài {lesson_num} đã tồn tại trong vocab_lessons.py!")
        ans = input("Ghi đè? (y/N): ").strip().lower()
        if ans != "y":
            print("Hủy.")
            return

        # Xóa block cũ: từ "    {lesson_num}: [" đến "    ],"
        block_pat = re.compile(
            rf"(    {lesson_num}: \[.*?    \],\n)", re.DOTALL
        )
        content = block_pat.sub("", content)

    # Chèn trước dòng cuối "}"
    new_block = format_lesson(lesson_num, words) + "\n"
    content = content.rstrip()
    # Tìm dấu "}" cuối cùng (đóng dict VOCAB_LESSONS)
    last_brace = content.rfind("}")
    content = content[:last_brace] + new_block + "}\n"

    with open(VOCAB_FILE, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"✅ Đã thêm bài {lesson_num} ({len(words)} từ) vào vocab_lessons.py")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    lesson_num = int(sys.argv[1])
    json_file = sys.argv[2] if len(sys.argv) > 2 else None

    words = load_json(lesson_num, json_file)
    print(f"Đọc được {len(words)} từ cho bài {lesson_num}.")
    inject(lesson_num, words)


if __name__ == "__main__":
    main()

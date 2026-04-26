"""
Chương trình chính: Nhập kanji → Xuất PDF luyện viết.
Cách dùng:
    python main.py
    > Nhập kanji cách nhau bằng dấu cách hoặc dấu phẩy, ví dụ: 毛 刀 力 丸
    > Hoặc nhập file danh sách (mỗi dòng 1 kanji)
"""
import sys
import os
from kanji_lookup import get_kanji_info
from pdf_generator import generate_pdf


def parse_input(raw: str) -> list[str]:
    """Tách kanji từ chuỗi nhập."""
    # Thay phẩy, xuống dòng, tab bằng dấu cách rồi split
    for ch in [",", "\n", "\t", "、"]:
        raw = raw.replace(ch, " ")
    return [k.strip() for k in raw.split() if k.strip()]


def main():
    print("=" * 55)
    print("   XÓA MÙ KANJI - Tạo file luyện viết PDF")
    print("=" * 55)

    # ── Chọn chế độ nhập ─────────────────────────────────────────────────────
    print("\nChọn cách nhập:")
    print("  1. Nhập danh sách kanji trực tiếp")
    print("  2. Đọc từ file văn bản (.txt)")
    choice = input("\nLựa chọn (1/2, mặc định 1): ").strip() or "1"

    kanji_list_raw: list[str] = []

    if choice == "2":
        path = input("Đường dẫn file .txt: ").strip().strip('"')
        if not os.path.exists(path):
            print(f"[!] Không tìm thấy file: {path}")
            sys.exit(1)
        with open(path, encoding="utf-8") as f:
            content = f.read()
        kanji_list_raw = parse_input(content)
    else:
        raw = input("\nNhập kanji (cách nhau bằng dấu cách hoặc phẩy):\n> ").strip()
        kanji_list_raw = parse_input(raw)

    if not kanji_list_raw:
        print("[!] Không có kanji nào được nhập.")
        sys.exit(1)

    print(f"\n[i] Tìm thấy {len(kanji_list_raw)} kanji: {' '.join(kanji_list_raw)}")

    # ── Tra thông tin từng kanji ──────────────────────────────────────────────
    print("\n[i] Đang tra cứu thông tin kanji...")
    kanji_data = []
    for k in kanji_list_raw:
        info = get_kanji_info(k)
        # Nếu không có nghĩa tiếng Việt, hỏi người dùng
        if not info.get("viet") and not info.get("meaning_vi"):
            print(f"\n[?] Không tìm thấy nghĩa cho '{k}'.")
            viet = input(f"    Âm Hán Việt của '{k}' (có thể để trống): ").strip()
            meaning = input(f"    Nghĩa tiếng Việt của '{k}': ").strip()
            meo = input(f"    Mẹo nhớ (có thể để trống): ").strip()
            info["viet"] = viet
            info["meaning_vi"] = meaning or k
            if meo:
                info["meo"] = meo
        elif not info.get("viet"):
            print(f"    '{k}' → {info.get('meaning_vi', '')}")
        else:
            print(f"    '{k}' → {info['viet']} ({info.get('meaning_vi', '')})")
        kanji_data.append(info)

    # ── Tên file đầu ra ───────────────────────────────────────────────────────
    default_out = "kanji_luyen_viet.pdf"
    out_input = input(f"\nTên file PDF đầu ra (mặc định: {default_out}): ").strip()
    out_path = out_input if out_input else default_out

    # Đảm bảo có đuôi .pdf
    if not out_path.lower().endswith(".pdf"):
        out_path += ".pdf"

    # Nếu không có đường dẫn tuyệt đối, lưu cùng thư mục script
    if not os.path.isabs(out_path):
        out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), out_path)

    # ── Tạo PDF ───────────────────────────────────────────────────────────────
    print("\n[i] Đang tạo PDF...")
    generate_pdf(kanji_data, out_path)
    print(f"\n✓ Hoàn thành! File đã lưu tại:\n  {out_path}")


if __name__ == "__main__":
    main()

"""
Tạo PDF luyện viết Kanji theo phong cách XÓA MÙ KANJI - CHIẾT TỰ.
Layout: Tiêu đề kanji, ô lớn mẫu + 11 ô luyện viết (5 đặc + 6 mờ dần).
"""
import os
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase.cidfonts import UnicodeCIDFont

# ─── Cấu hình ────────────────────────────────────────────────────────────────
PAGE_W, PAGE_H = A4                      # 595 x 842 pt
MARGIN_X = 15 * mm
MARGIN_TOP = 20 * mm
MARGIN_BOT = 10 * mm

HEADER_H = 20 * mm                      # chiều cao vùng tiêu đề mỗi kanji
ROW_H = 22 * mm                         # chiều cao hàng ô luyện viết
CELL_W = ROW_H                          # ô vuông
GRID_COLS = 12                          # 1 mẫu + 5 đặc + 6 mờ

USABLE_W = PAGE_W - 2 * MARGIN_X       # độ rộng thực dùng

# Màu sắc
CLR_BG_CELL     = colors.white
CLR_BORDER      = colors.HexColor("#888888")
CLR_GUIDE       = colors.HexColor("#BBBBBB")   # đường kẻ ô luyện mờ
CLR_SAMPLE_BG   = colors.HexColor("#FFFFFF")
CLR_HEADER_LINE = colors.HexColor("#444444")
CLR_KANJI_FADED = [                            # 6 mức độ mờ dần
    colors.HexColor("#888888"),
    colors.HexColor("#999999"),
    colors.HexColor("#AAAAAA"),
    colors.HexColor("#BBBBBB"),
    colors.HexColor("#CCCCCC"),
    colors.HexColor("#DDDDDD"),
]

# ─── Font ──────────────────────────────────────────────────────────────────────
# Tìm font hệ thống hỗ trợ CJK
def find_cjk_font() -> str | None:
    import sys
    # Ưu tiên font đặt ngay trong thư mục project
    _here = os.path.dirname(os.path.abspath(__file__))
    local_candidates = [
        os.path.join(_here, "NotoSansJP-VF.ttf"),
        os.path.join(_here, "fonts", "NotoSansJP-VF.ttf"),
    ]
    for p in local_candidates:
        if os.path.exists(p):
            return p
    candidates = []
    if sys.platform == "win32":
        candidates = [
            r"C:\Windows\Fonts\NotoSansJP-VF.ttf",
            r"C:\Windows\Fonts\yumin.ttf",
            r"C:\Windows\Fonts\NotoSerifJP-VF.ttf",
            r"C:\Windows\Fonts\BIZ-UDMinchoM.ttc",
            r"C:\Windows\Fonts\msmincho.ttc",
            r"C:\Windows\Fonts\YuGothM.ttc",
            r"C:\Windows\Fonts\msgothic.ttc",
            r"C:\Windows\Fonts\meiryo.ttc",
        ]
    elif sys.platform == "darwin":
        home = os.path.expanduser("~")
        candidates = [
            os.path.join(home, "Library", "Fonts", "NotoSansJP-VF.ttf"),
            "/Library/Fonts/NotoSansJP-VF.ttf",
            os.path.join(home, "Library", "Fonts", "NotoSansJP-Regular.otf"),
            "/Library/Fonts/NotoSansJP-Regular.otf",
            "/System/Library/Fonts/Hiragino Sans.ttc",
            "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc",
            "/System/Library/Fonts/Hiragino Sans GB.ttc",
            "/Library/Fonts/Osaka.ttf",
        ]
    else:  # Linux
        home = os.path.expanduser("~")
        candidates = [
            os.path.join(home, ".local", "share", "fonts", "NotoSansJP-VF.ttf"),
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJKjp-Regular.otf",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-VF.ttc",
            "/usr/share/fonts/truetype/noto/NotoSansJP-Regular.ttf",
            "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def register_fonts():
    """Đăng ký font CJK và font Latin cho ReportLab."""
    _here = os.path.dirname(os.path.abspath(__file__))
    # Font cho tiếng Việt + Latin
    vi_font = None
    import sys
    local_vi = os.path.join(_here, "arial.ttf")
    if os.path.exists(local_vi):
        vi_candidates = [local_vi]
    elif sys.platform == "win32":
        vi_candidates = [
            r"C:\Windows\Fonts\arial.ttf",
            r"C:\Windows\Fonts\calibri.ttf",
            r"C:\Windows\Fonts\times.ttf",
        ]
    elif sys.platform == "darwin":
        vi_candidates = [
            "/Library/Fonts/Arial.ttf",
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/Library/Fonts/Times New Roman.ttf",
        ]
    else:
        vi_candidates = [
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]
    for p in vi_candidates:
        if os.path.exists(p):
            vi_font = p
            break

    cjk_font = find_cjk_font()

    registered = {}
    if vi_font:
        try:
            pdfmetrics.registerFont(TTFont("ViFont", vi_font))
            registered["vi"] = "ViFont"
        except Exception:
            pass

    if cjk_font:
        try:
            pdfmetrics.registerFont(TTFont("CJKFont", cjk_font, subfontIndex=0))
            registered["cjk"] = "CJKFont"
        except Exception:
            try:
                pdfmetrics.registerFont(TTFont("CJKFont", cjk_font))
                registered["cjk"] = "CJKFont"
            except Exception:
                pass

    # Fallback: dùng CID font có sẵn trong ReportLab (không cần file ngoài)
    if "cjk" not in registered:
        try:
            pdfmetrics.registerFont(UnicodeCIDFont("HeiseiMin-W3"))
            registered["cjk"] = "HeiseiMin-W3"
        except Exception:
            try:
                pdfmetrics.registerFont(UnicodeCIDFont("HeiseiKakuGo-W5"))
                registered["cjk"] = "HeiseiKakuGo-W5"
            except Exception:
                pass

    return registered


# ─── Helpers ──────────────────────────────────────────────────────────────────
def is_cjk(ch: str) -> bool:
    """Kiểm tra ký tự có phải CJK (kanji/kana) không."""
    cp = ord(ch)
    return (
        0x3000 <= cp <= 0x9FFF   # CJK + Kana
        or 0xF900 <= cp <= 0xFAFF
        or 0x20000 <= cp <= 0x2A6DF
    )


def draw_mixed_string(c: canvas.Canvas, x: float, y: float, text: str,
                      font_vi: str, font_cjk: str, font_size: float) -> float:
    """
    Vẽ chuỗi hỗn hợp tiếng Việt + kanji.
    Tự động chuyển font giữa từng đoạn.
    Trả về tổng chiều rộng đã vẽ.
    """
    if not text:
        return 0.0

    # Chia thành các segment (font, đoạn_text)
    segments: list[tuple[str, str]] = []
    cur_font = font_cjk if is_cjk(text[0]) else font_vi
    cur_buf = ""
    for ch in text:
        ch_font = font_cjk if is_cjk(ch) else font_vi
        if ch_font == cur_font:
            cur_buf += ch
        else:
            if cur_buf:
                segments.append((cur_font, cur_buf))
            cur_font = ch_font
            cur_buf = ch
    if cur_buf:
        segments.append((cur_font, cur_buf))

    cursor_x = x
    for seg_font, seg_text in segments:
        c.setFont(seg_font, font_size)
        c.drawString(cursor_x, y, seg_text)
        cursor_x += c.stringWidth(seg_text, seg_font, font_size)

    return cursor_x - x


def mixed_string_width(text: str, font_vi: str, font_cjk: str,
                       font_size: float, canvas_obj: canvas.Canvas) -> float:
    """Tính tổng chiều rộng của chuỗi hỗn hợp."""
    total = 0.0
    cur_font = font_cjk if (text and is_cjk(text[0])) else font_vi
    cur_buf = ""
    for ch in text:
        ch_font = font_cjk if is_cjk(ch) else font_vi
        if ch_font == cur_font:
            cur_buf += ch
        else:
            total += canvas_obj.stringWidth(cur_buf, cur_font, font_size)
            cur_font = ch_font
            cur_buf = ch
    if cur_buf:
        total += canvas_obj.stringWidth(cur_buf, cur_font, font_size)
    return total


def wrap_mixed_text(text: str, font_vi: str, font_cjk: str,
                    font_size: float, max_w: float,
                    canvas_obj: canvas.Canvas) -> list[str]:
    """
    Word-wrap chuỗi hỗn hợp Vi+CJK thành các dòng không vượt max_w.
    Tách theo khoảng trắng; nếu 1 từ đơn vẫn rộng hơn max_w thì vẫn giữ nguyên.
    """
    if not text:
        return []
    words = text.split(' ')
    lines: list[str] = []
    current = ''
    for word in words:
        test = (current + ' ' + word).strip() if current else word
        if mixed_string_width(test, font_vi, font_cjk, font_size, canvas_obj) <= max_w:
            current = test
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines if lines else [text]


def draw_practice_cell(c: canvas.Canvas, x: float, y: float, size: float,
                       kanji: str = "", alpha_color=None, is_sample=False):
    """
    Vẽ 1 ô luyện viết (hình vuông + đường kẻ chéo + kanji).
    x, y = góc dưới-trái của ô.
    """
    # Nền trắng
    c.setFillColor(CLR_BG_CELL)
    c.rect(x, y, size, size, fill=1, stroke=0)

    # Đường kẻ gạch ngang giữa và dọc giữa (màu nhạt)
    c.setStrokeColor(CLR_GUIDE)
    c.setLineWidth(0.4)
    c.setDash(3, 3)
    c.line(x, y + size / 2, x + size, y + size / 2)
    c.line(x + size / 2, y, x + size / 2, y + size)
    c.setDash()

    # Đường chéo góc (rất nhạt)
    c.setStrokeColor(CLR_GUIDE)
    c.setLineWidth(0.4)
    c.setDash(2, 4)
    c.line(x, y + size, x + size, y)
    c.line(x, y, x + size, y + size)
    c.setDash()

    # Viền ô
    border_color = CLR_BORDER if is_sample else colors.HexColor("#666666")
    c.setStrokeColor(border_color)
    c.setLineWidth(0.8 if is_sample else 0.9)
    c.rect(x, y, size, size, fill=0, stroke=1)

    # Kanji
    if kanji:
        font_name = "CJKFont" if "CJKFont" in pdfmetrics.getRegisteredFontNames() else "Helvetica"
        c.setFillColor(alpha_color if alpha_color else colors.black)
        font_size = size * 0.72 if is_sample else size * 0.65
        c.setFont(font_name, font_size)
        # Căn giữa ô
        text_w = c.stringWidth(kanji, font_name, font_size)
        tx = x + (size - text_w) / 2
        ty = y + size * 0.14
        c.drawString(tx, ty, kanji)


def draw_kanji_block(c: canvas.Canvas, info: dict, y_top: float, fonts: dict,
                     extra_rows: int = 0) -> float:
    """
    Vẽ 1 block kanji (tiêu đề + hàng ô luyện).
    extra_rows: số hàng ô trắng bổ sung sau hàng mẫu (0 = mặc định, 1-3 = luyện thêm).
    Trả về y_bottom của block.
    """
    kanji = info["kanji"]
    viet = info.get("viet", "")
    meaning_vi = info.get("meaning_vi", "")
    meo = info.get("meo", "")

    font_vi = fonts.get("vi", "Helvetica")
    font_cjk = fonts.get("cjk", "Helvetica")

    x0 = MARGIN_X
    cell_size = (USABLE_W / GRID_COLS)
    total_row_w = cell_size * GRID_COLS

    # ── Tiêu đề ──────────────────────────────────────────────────────────────
    reading = info.get("reading", "")
    vocab   = info.get("vocab", [])          # list of (chữ, hiragana, nghĩa)

    # ── Mẹo nhớ: word-wrap full width thành nhiều dòng ─────────────────────
    MEO_FONT  = 6.5
    MEO_LINE_H = 5.5 * mm   # khoảng cách giữa các dòng mẹo
    meo_lines: list[str] = []
    if meo:
        meo_lines = wrap_mixed_text(meo, font_vi, font_cjk, MEO_FONT, total_row_w, c)

    # ── Tính width còn lại cho meaning_vi sau viet+(reading) ────────────────
    MEANING_FONT = 9
    MEANING_LINE_H = 5.5 * mm
    viet_w = (c.stringWidth(viet, font_vi, 11) if viet else 0)
    reading_w = (c.stringWidth(f" ({reading})", font_cjk, 9) if reading else 0)
    gap_w = c.stringWidth("  ", font_vi, MEANING_FONT) if (viet or reading) else 0
    remaining_w = total_row_w - viet_w - reading_w - gap_w

    # Wrap meaning_vi: dòng đầu dùng remaining_w, dòng sau dùng full width
    meaning_lines: list[str] = []
    if meaning_vi:
        # Thử fit dòng đầu
        first_line_words: list[str] = []
        rest_words: list[str] = []
        words = meaning_vi.split(" ")
        cur = ""
        overflow_idx = None
        for wi, w in enumerate(words):
            test = (cur + " " + w).strip() if cur else w
            if mixed_string_width(test, font_vi, font_cjk, MEANING_FONT, c) <= remaining_w:
                cur = test
            else:
                overflow_idx = wi
                break
        if overflow_idx is not None:
            meaning_lines.append(cur)
            rest = " ".join(words[overflow_idx:])
            meaning_lines += wrap_mixed_text(rest, font_vi, font_cjk, MEANING_FONT, total_row_w, c)
        else:
            meaning_lines.append(cur)

    extra_meaning_lines = max(0, len(meaning_lines) - 1)  # dòng 2+ của meaning

    # ── Tính dynamic header height ────────────────────────────────────────────
    # Dòng 1 (viet+reading+meaning): chiếm 7mm
    # Mỗi dòng meaning overflow: 5.5mm
    # Mỗi dòng mẹo: 5.5mm
    # Vocab: 6mm
    # Padding trước separator: 3mm
    LINE1_OFF  = 7 * mm
    MEO_START  = 13 * mm + extra_meaning_lines * MEANING_LINE_H
    VOCAB_H    = 6 * mm
    SEP_PAD    = 3 * mm

    n_meo = len(meo_lines)
    meo_block_h = n_meo * MEO_LINE_H if n_meo else 0
    dyn_header_h = MEO_START + meo_block_h + (VOCAB_H if vocab else 0) + SEP_PAD
    dyn_header_h = max(dyn_header_h, HEADER_H)   # không nhỏ hơn mặc định

    # ── Dòng 1: VIET (hiragana)  Nghĩa tiếng Việt (full width, không clip) ───
    y_line1 = y_top - LINE1_OFF
    x_cur = x0

    # Âm Hán Việt
    if viet:
        c.setFont(font_vi, 11)
        c.setFillColor(colors.black)
        c.drawString(x_cur, y_line1, viet)
        x_cur += c.stringWidth(viet, font_vi, 11)

    # (hiragana) kế ngay bên
    if reading:
        kana_str = f" ({reading})"
        c.setFont(font_cjk, 9)
        c.setFillColor(colors.HexColor("#444444"))
        c.drawString(x_cur, y_line1, kana_str)
        x_cur += c.stringWidth(kana_str, font_cjk, 9)

    # Nghĩa tiếng Việt — dòng đầu kế sau viet+reading, dòng dư wrap xuống
    if meaning_lines:
        gap_str = "  " if viet or reading else ""
        c.setFillColor(colors.HexColor("#333333"))
        draw_mixed_string(c, x_cur, y_line1, f"{gap_str}{meaning_lines[0]}", font_vi, font_cjk, MEANING_FONT)
        for li, extra_line in enumerate(meaning_lines[1:], 1):
            y_extra = y_line1 - li * MEANING_LINE_H
            draw_mixed_string(c, x0, y_extra, extra_line, font_vi, font_cjk, MEANING_FONT)

    # ── Mẹo nhớ: từng dòng đã wrap ───────────────────────────────────────────
    if meo_lines:
        c.setFillColor(colors.HexColor("#555555"))
        for ln_idx, ln_text in enumerate(meo_lines):
            y_meo = y_top - MEO_START - ln_idx * MEO_LINE_H
            draw_mixed_string(c, x0, y_meo, ln_text, font_vi, font_cjk, MEO_FONT)

    # ── Vocab: bên dưới mẹo nhớ ──────────────────────────────────────────────
    if vocab:
        y_line2 = y_top - MEO_START - meo_block_h - (MEO_LINE_H * 0.2 if meo_lines else 0)
        x_cursor = x0
        for i, item in enumerate(vocab[:2]):
            word, reading_v, meaning_v = item
            # Từ kanji (màu xanh đậm)
            c.setFillColor(colors.HexColor("#0D2D47"))
            seg_w = draw_mixed_string(c, x_cursor, y_line2,
                                      word, font_vi, font_cjk, 8)
            x_cursor += seg_w
            # (hiragana)
            c.setFont(font_cjk, 7.5)
            c.setFillColor(colors.HexColor("#222222"))
            kana_txt = f"({reading_v})"
            c.drawString(x_cursor, y_line2, kana_txt)
            x_cursor += c.stringWidth(kana_txt, font_cjk, 7.5)
            # nghĩa
            c.setFont(font_vi, 7.5)
            c.setFillColor(colors.HexColor("#111111"))
            meaning_txt = f" {meaning_v}"
            c.drawString(x_cursor, y_line2, meaning_txt)
            x_cursor += c.stringWidth(meaning_txt, font_vi, 7.5)
            # dấu phân cách
            if i == 0 and len(vocab) > 1:
                c.setFont(font_vi, 7.5)
                c.setFillColor(colors.HexColor("#777777"))
                sep = "     |     "
                c.drawString(x_cursor, y_line2, sep)
                x_cursor += c.stringWidth(sep, font_vi, 7.5)

    # ── Đường kẻ dưới header (dùng dynamic height) ──────────────────────────
    y_header = y_top - dyn_header_h
    c.setStrokeColor(CLR_HEADER_LINE)
    c.setLineWidth(0.5)
    c.line(x0, y_header + 2, x0 + total_row_w, y_header + 2)

    # ── Hàng ô luyện ─────────────────────────────────────────────────────────
    y_row = y_header - ROW_H

    for col in range(GRID_COLS):
        cx = x0 + col * cell_size
        cy = y_row

        if col == 0:
            draw_practice_cell(c, cx, cy, cell_size, kanji,
                                alpha_color=colors.black, is_sample=True)
        elif col <= 5:
            fade = CLR_KANJI_FADED[col - 1]
            draw_practice_cell(c, cx, cy, cell_size, kanji,
                                alpha_color=fade, is_sample=False)
        else:
            draw_practice_cell(c, cx, cy, cell_size, "",
                                alpha_color=None, is_sample=False)

    # ── Hàng ô trắng bổ sung ─────────────────────────────────────────────────
    for r in range(extra_rows):
        y_extra = y_row - ROW_H * (r + 1)
        for col in range(GRID_COLS):
            cx = x0 + col * cell_size
            draw_practice_cell(c, cx, y_extra, cell_size, "",
                                alpha_color=None, is_sample=False)

    y_bottom = y_row - ROW_H * extra_rows
    return y_bottom  # trả về y dưới cùng của block


# ─── Main ─────────────────────────────────────────────────────────────────────
def generate_pdf(kanji_list: list[dict], output_path: str = "output.pdf",
                 extra_rows: int = 0):
    """
    kanji_list: list of dict từ kanji_lookup.get_kanji_info()
    extra_rows : số hàng ô trắng thêm vào sau hàng mẫu mỗi kanji (0-3).
    """
    fonts = register_fonts()

    c = canvas.Canvas(output_path, pagesize=A4)
    c.setTitle("Kanji")

    font_vi = fonts.get("vi", "Helvetica")
    font_cjk = fonts.get("cjk", "Helvetica")

    cell_size = USABLE_W / GRID_COLS
    # Ước tính chiều cao tối đa mỗi block (mẹo nhớ có thể dài thêm ~4 dòng = 22mm)
    block_height = HEADER_H + 22 * mm + ROW_H * (1 + extra_rows) + 4 * mm

    # ── Header / Footer trang ─────────────────────────────────────────────────
    def draw_page_header():
        pass

    def draw_page_footer():
        footer = "Kanji Hub by Phong Vu"
        c.setFont(font_vi, 8)
        c.setFillColor(colors.HexColor("#888888"))
        fw = c.stringWidth(footer, font_vi, 8)
        c.drawString((PAGE_W - fw) / 2, 6 * mm, footer)

    draw_page_header()
    draw_page_footer()
    y_cursor = PAGE_H - MARGIN_TOP - 5 * mm

    for info in kanji_list:
        # Kiểm tra còn đủ chỗ không
        if y_cursor - block_height < MARGIN_BOT:
            c.showPage()
            draw_page_header()
            draw_page_footer()
            y_cursor = PAGE_H - MARGIN_TOP - 5 * mm

        y_bottom = draw_kanji_block(c, info, y_cursor, fonts, extra_rows=extra_rows)
        y_cursor = y_bottom - 4 * mm   # khoảng cách giữa các block

    c.save()
    print(f"[✓] Đã xuất: {output_path}")


# ─── Bảng từ vựng ───────────────────────────────────────────────────────────────────
def generate_vocab_table_pdf(vocab_list: list[dict], output_path: str = "output.pdf"):
    """
    Tạo PDF dạng bảng từ vựng:
    # | Từ vựng | Hiragana | Nghĩa | Kanji
    Cột Từ vựng điền sẵn, 3 cột còn lại trống để học sinh lực nhớ.
    """
    from datetime import datetime
    fonts = register_fonts()
    font_vi  = fonts.get("vi", "Helvetica")
    font_cjk = fonts.get("cjk", "Helvetica")

    c = canvas.Canvas(output_path, pagesize=A4)
    c.setTitle("Bảng từ vựng Tiếng Nhật")

    now_str = datetime.now().strftime("%H:%M  %d/%m/%y")

    # ── Layout ───────────────────────────────────────────────────────────────────
    MX      = 10 * mm
    USABLE  = PAGE_W - 2 * MX
    C_STT   = 10 * mm
    C_WORD  = 35 * mm
    C_HIRA  = 45 * mm
    C_NGHIA = USABLE - C_STT - C_WORD - C_HIRA - 38 * mm
    C_KANJI = 38 * mm

    col_widths = [C_STT, C_WORD, C_HIRA, C_NGHIA, C_KANJI]
    col_xs = []
    _x = MX
    for _w in col_widths:
        col_xs.append(_x)
        _x += _w

    ROW_H   = 6.5 * mm
    THEAD_H = 7.5 * mm
    MY_TOP  = 8 * mm
    MY_BOT  = 10 * mm
    PHEAD_H = 7 * mm

    def _table_start_y():
        return PAGE_H - MY_TOP - PHEAD_H - 2 * mm

    def _draw_page_chrome():
        # Timestamp góc trên trái
        c.setFont(font_vi, 7.5)
        c.setFillColor(colors.HexColor("#888888"))
        c.drawString(MX, PAGE_H - MY_TOP, now_str)
        # Footer
        footer = "Kanji Hub by Phong Vu"
        c.setFont(font_vi, 7.5)
        c.setFillColor(colors.HexColor("#AAAAAA"))
        fw = c.stringWidth(footer, font_vi, 7.5)
        c.drawString((PAGE_W - fw) / 2, MY_BOT - 4, footer)

    def _draw_col_header(y_top):
        """Vẽ hàng tiêu đề cột, trả về y_bottom."""
        y_bot = y_top - THEAD_H
        c.setFillColor(colors.HexColor("#DFF0D8"))
        c.rect(MX, y_bot, USABLE, THEAD_H, fill=1, stroke=0)
        c.setStrokeColor(colors.HexColor("#999999"))
        c.setLineWidth(0.6)
        c.rect(MX, y_bot, USABLE, THEAD_H, fill=0, stroke=1)
        c.setLineWidth(0.4)
        for _cx in col_xs[1:]:
            c.line(_cx, y_bot, _cx, y_top)
        labels = ["#", "Từ vựng", "Hiragana", "Nghĩa", "Kanji"]
        c.setFillColor(colors.HexColor("#2D4A2D"))
        for label, _cx, _cw in zip(labels, col_xs, col_widths):
            c.setFont(font_vi, 8.5)
            lw = c.stringWidth(label, font_vi, 8.5)
            c.drawString(_cx + (_cw - lw) / 2, y_bot + 2.2 * mm, label)
        return y_bot

    def _draw_row(y_top, idx, info, shade):
        """Vẽ 1 dòng dữ liệu, trả về y_bottom."""
        word = info.get("kanji", "")
        y_bot = y_top - ROW_H
        # Nền zebra
        c.setFillColor(colors.HexColor("#F6F6F6") if shade else colors.white)
        c.rect(MX, y_bot, USABLE, ROW_H, fill=1, stroke=0)
        # Viền ngang
        c.setStrokeColor(colors.HexColor("#DDDDDD"))
        c.setLineWidth(0.3)
        c.rect(MX, y_bot, USABLE, ROW_H, fill=0, stroke=1)
        # Viền dọc
        for _cx in col_xs[1:]:
            c.line(_cx, y_bot, _cx, y_top)

        ty = y_bot + ROW_H * 0.27

        # STT
        c.setFont(font_vi, 7.5)
        c.setFillColor(colors.HexColor("#999999"))
        s = str(idx)
        sw = c.stringWidth(s, font_vi, 7.5)
        c.drawString(col_xs[0] + (C_STT - sw) / 2, ty, s)

        # Từ vựng (căn giữa, đen)
        if word:
            fs = 10.5
            c.setFont(font_cjk, fs)
            c.setFillColor(colors.black)
            ww = c.stringWidth(word, font_cjk, fs)
            max_w = C_WORD - 2 * mm
            if ww > max_w:
                fs = fs * max_w / ww
                c.setFont(font_cjk, fs)
                ww = c.stringWidth(word, font_cjk, fs)
            c.drawString(col_xs[1] + (C_WORD - ww) / 2, ty, word)

        return y_bot

    # ── Vẽ ───────────────────────────────────────────────────────────────────────
    _draw_page_chrome()
    y = _table_start_y()
    y = _draw_col_header(y)

    for i, info in enumerate(vocab_list, 1):
        if y - ROW_H < MY_BOT + 5 * mm:
            c.showPage()
            _draw_page_chrome()
            y = _table_start_y()
            y = _draw_col_header(y)
        y = _draw_row(y, i, info, shade=(i % 2 == 0))

    c.save()
    print(f"[✓] Đã xuất bảng từ vựng: {output_path}")

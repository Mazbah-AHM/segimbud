from __future__ import annotations

from io import BytesIO
from textwrap import wrap

from PIL import Image, ImageDraw, ImageFont

PAGE_SIZE = (1654, 2339)
PAGE_BACKGROUND = "#f6f1e6"
CARD_BACKGROUND = "#ffffff"
CARD_BORDER = "#d8d2c5"
TEXT_PRIMARY = "#16324b"
TEXT_MUTED = "#66768a"
ACCENT = "#156b52"


def _load_font(size: int, bold: bool = False):
    candidates = [
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _fit_image(image: Image.Image, max_width: int, max_height: int) -> Image.Image:
    fitted = image.convert("RGB").copy()
    fitted.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
    return fitted


def _draw_wrapped_text(draw: ImageDraw.ImageDraw, text: str, box: tuple[int, int, int, int], font, fill):
    x0, y0, x1, y1 = box
    max_chars = max(int((x1 - x0) / max(font.size * 0.55, 8)), 16)
    lines = []
    for paragraph in text.splitlines():
        if not paragraph.strip():
            lines.append("")
            continue
        lines.extend(wrap(paragraph, width=max_chars))

    y = y0
    for line in lines:
        draw.text((x0, y), line, font=font, fill=fill)
        y += int(font.size * 1.45)
        if y > y1:
            break


def build_pdf_report(
    report_title: str,
    mode_label: str,
    summary_metrics: list[tuple[str, str]],
    image_panels: list[tuple[str, Image.Image]],
    insights: list[str],
    stats_rows: list[dict],
    extra_notes: list[str] | None = None,
) -> bytes:
    title_font = _load_font(54, bold=True)
    subtitle_font = _load_font(26, bold=False)
    section_font = _load_font(28, bold=True)
    body_font = _load_font(22, bold=False)
    small_font = _load_font(18, bold=False)

    page_1 = Image.new("RGB", PAGE_SIZE, PAGE_BACKGROUND)
    draw_1 = ImageDraw.Draw(page_1)

    draw_1.rounded_rectangle((70, 70, PAGE_SIZE[0] - 70, PAGE_SIZE[1] - 70), radius=38, fill=PAGE_BACKGROUND)
    draw_1.text((110, 110), report_title, font=title_font, fill=TEXT_PRIMARY)
    draw_1.text((112, 182), mode_label, font=subtitle_font, fill=ACCENT)
    draw_1.text((112, 225), "SegImBud analytical export", font=small_font, fill=TEXT_MUTED)

    metric_y = 290
    metric_x = 110
    metric_width = 338
    metric_gap = 18
    for label, value in summary_metrics[:4]:
        draw_1.rounded_rectangle(
            (metric_x, metric_y, metric_x + metric_width, metric_y + 120),
            radius=24,
            fill=CARD_BACKGROUND,
            outline=CARD_BORDER,
            width=2,
        )
        draw_1.text((metric_x + 24, metric_y + 20), label.upper(), font=small_font, fill=TEXT_MUTED)
        draw_1.text((metric_x + 24, metric_y + 58), value, font=section_font, fill=TEXT_PRIMARY)
        metric_x += metric_width + metric_gap

    panel_positions = [
        (110, 460, 760, 1120),
        (840, 460, 1490, 1120),
        (110, 1170, 1490, 1890),
    ]
    for (panel_title, panel_image), box in zip(image_panels[:3], panel_positions):
        x0, y0, x1, y1 = box
        draw_1.rounded_rectangle(box, radius=28, fill=CARD_BACKGROUND, outline=CARD_BORDER, width=2)
        draw_1.text((x0 + 24, y0 + 18), panel_title, font=section_font, fill=TEXT_PRIMARY)
        fitted = _fit_image(panel_image, x1 - x0 - 48, y1 - y0 - 90)
        paste_x = x0 + ((x1 - x0) - fitted.width) // 2
        paste_y = y0 + 70 + ((y1 - y0 - 92) - fitted.height) // 2
        page_1.paste(fitted, (paste_x, paste_y))

    draw_1.rounded_rectangle((110, 1930, PAGE_SIZE[0] - 110, 2240), radius=28, fill=CARD_BACKGROUND, outline=CARD_BORDER, width=2)
    draw_1.text((134, 1955), "Operational notes", font=section_font, fill=TEXT_PRIMARY)
    note_lines = insights[:4] + (extra_notes or [])[:2]
    _draw_wrapped_text(
        draw_1,
        "\n".join(f"- {line}" for line in note_lines),
        (136, 2002, PAGE_SIZE[0] - 140, 2220),
        body_font,
        TEXT_MUTED,
    )

    page_2 = Image.new("RGB", PAGE_SIZE, PAGE_BACKGROUND)
    draw_2 = ImageDraw.Draw(page_2)
    draw_2.text((110, 110), "Detailed class statistics", font=title_font, fill=TEXT_PRIMARY)
    draw_2.text((112, 182), report_title, font=subtitle_font, fill=ACCENT)

    table_box = (110, 270, PAGE_SIZE[0] - 110, 1450)
    draw_2.rounded_rectangle(table_box, radius=30, fill=CARD_BACKGROUND, outline=CARD_BORDER, width=2)
    headers = ["Class", "Pixels", "Percent", "Area km2"]
    header_positions = [140, 640, 960, 1220]
    for header, x in zip(headers, header_positions):
        draw_2.text((x, 310), header, font=section_font, fill=TEXT_PRIMARY)

    y = 380
    for row in stats_rows[:18]:
        draw_2.line((130, y - 16, PAGE_SIZE[0] - 130, y - 16), fill="#ebe4d8", width=2)
        draw_2.text((140, y), str(row.get("class", "")), font=body_font, fill=TEXT_PRIMARY)
        draw_2.text((640, y), str(row.get("pixels", "")), font=body_font, fill=TEXT_PRIMARY)
        draw_2.text((960, y), str(row.get("percentage", "")), font=body_font, fill=TEXT_PRIMARY)
        draw_2.text((1220, y), str(row.get("area_sq_km", "")), font=body_font, fill=TEXT_PRIMARY)
        y += 58

    draw_2.rounded_rectangle((110, 1510, PAGE_SIZE[0] - 110, 2230), radius=30, fill=CARD_BACKGROUND, outline=CARD_BORDER, width=2)
    draw_2.text((134, 1540), "Interpretation guide", font=section_font, fill=TEXT_PRIMARY)
    guide_lines = insights[:8]
    if extra_notes:
        guide_lines.extend(extra_notes[:6])
    _draw_wrapped_text(
        draw_2,
        "\n".join(f"- {line}" for line in guide_lines),
        (136, 1590, PAGE_SIZE[0] - 140, 2200),
        body_font,
        TEXT_MUTED,
    )

    buffer = BytesIO()
    page_1.save(buffer, format="PDF", save_all=True, append_images=[page_2])
    return buffer.getvalue()

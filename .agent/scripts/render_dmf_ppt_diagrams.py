from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "Tomtat-paper" / "acc-loss" / "ppt_ready"
WIDTH, HEIGHT = 1920, 1080
FONT_DIR = Path("C:/Windows/Fonts")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


@dataclass(frozen=True)
class Theme:
    name: str
    edge: str
    fill: str
    final: str
    accent_fill: str


@dataclass(frozen=True)
class DiagramSpec:
    filename: str
    title: str
    models: tuple[str, ...]
    theme: Theme


THEMES = {
    "blue": Theme("blue", "#1f66cc", "#edf4ff", "#2266ad", "#e8f1ff"),
    "green": Theme("green", "#24814a", "#eef9f2", "#2f8e55", "#eaf7ee"),
    "purple": Theme("purple", "#8a49d8", "#f6efff", "#7740b2", "#f2eaff"),
    "gold": Theme("gold", "#f2a100", "#fff8df", "#f4c33f", "#fff0b5"),
}

SPECS = (
    DiagramSpec(
        filename="dmf_accuracy_lstm_tlstm_ppt.png",
        title="Sơ đồ kết hợp DMF-LT",
        models=("LSTM", "T-LSTM\n(Transformer-LSTM)"),
        theme=THEMES["blue"],
    ),
    DiagramSpec(
        filename="dmf_accuracy_lstm_graphsage_ppt.png",
        title="Sơ đồ kết hợp DMF-LG",
        models=("LSTM", "GraphSAGE"),
        theme=THEMES["green"],
    ),
    DiagramSpec(
        filename="dmf_accuracy_lstm_tlstm_graphsage_ppt.png",
        title="Sơ đồ kết hợp DMF-LTG",
        models=("LSTM", "T-LSTM\n(Transformer-LSTM)", "GraphSAGE"),
        theme=THEMES["purple"],
    ),
    DiagramSpec(
        filename="dmf_accuracy_tlstm_graphsage_ppt.png",
        title="Sơ đồ kết hợp DMF-TG",
        models=("T-LSTM", "GraphSAGE"),
        theme=THEMES["gold"],
    ),
)


def font(name: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(FONT_DIR / name), size=size)


TITLE_FONT = font("calibrib.ttf", 70)
BOX_FONT = font("calibrib.ttf", 38)
BOX_SMALL_FONT = font("calibrib.ttf", 33)
FINAL_FONT = font("calibrib.ttf", 42)


def text_size(draw: ImageDraw.ImageDraw, text: str, fnt: ImageFont.ImageFont) -> tuple[int, int]:
    bbox = draw.multiline_textbbox((0, 0), text, font=fnt, spacing=7, align="center")
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def center_text(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    text: str,
    fnt: ImageFont.ImageFont,
    fill: str = "#222222",
    spacing: int = 7,
) -> None:
    x1, y1, x2, y2 = box
    tw, th = text_size(draw, text, fnt)
    x = x1 + (x2 - x1 - tw) / 2
    y = y1 + (y2 - y1 - th) / 2 - 2
    draw.multiline_text((x, y), text, font=fnt, fill=fill, spacing=spacing, align="center")


def rounded_box(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    *,
    fill: str,
    outline: str,
    width: int = 4,
    radius: int = 16,
) -> None:
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[int, int],
    end: tuple[int, int],
    *,
    color: str = "#222222",
    width: int = 4,
    head: int = 16,
) -> None:
    sx, sy = start
    ex, ey = end
    draw.line((sx, sy, ex, ey), fill=color, width=width)
    dx, dy = ex - sx, ey - sy
    length = max((dx * dx + dy * dy) ** 0.5, 1)
    ux, uy = dx / length, dy / length
    px, py = -uy, ux
    p1 = (ex, ey)
    p2 = (ex - ux * head + px * head * 0.45, ey - uy * head + py * head * 0.45)
    p3 = (ex - ux * head - px * head * 0.45, ey - uy * head - py * head * 0.45)
    draw.polygon((p1, p2, p3), fill=color)


def polyline(
    draw: ImageDraw.ImageDraw,
    points: Iterable[tuple[int, int]],
    *,
    color: str = "#222222",
    width: int = 4,
) -> None:
    pts = list(points)
    for start, end in zip(pts, pts[1:]):
        draw.line((*start, *end), fill=color, width=width)


def draw_diagram(spec: DiagramSpec) -> Image.Image:
    img = Image.new("RGB", (WIDTH, HEIGHT), "#ffffff")
    draw = ImageDraw.Draw(img)
    theme = spec.theme

    title_box = (0, 46, WIDTH, 130)
    center_text(draw, title_box, spec.title, TITLE_FONT)

    top = (520, 165, 1400, 260)
    compare = (620, 505, 1300, 600)
    left_decision = (170, 710, 710, 855)
    right_decision = (1020, 710, 1750, 855)
    final = (690, 930, 1230, 1018)

    if len(spec.models) == 3:
        model_boxes = (
            (120, 340, 530, 430),
            (755, 340, 1165, 430),
            (1390, 340, 1800, 430),
        )
    else:
        model_boxes = (
            (220, 340, 660, 430),
            (1260, 340, 1700, 430),
        )

    for box in (top, compare, left_decision, right_decision):
        rounded_box(draw, box, fill=theme.fill, outline=theme.edge)
    for box in model_boxes:
        rounded_box(draw, box, fill=theme.fill, outline=theme.edge)
    rounded_box(draw, final, fill=theme.final, outline=theme.edge)

    center_text(draw, top, "Dữ liệu tài chính doanh nghiệp", BOX_FONT)
    for box, label in zip(model_boxes, spec.models):
        center_text(draw, box, label, BOX_FONT)
    center_text(draw, compare, "So sánh nhãn dự đoán", BOX_FONT)
    center_text(draw, left_decision, "Cùng nhãn\n→ Chọn nhãn chung", BOX_SMALL_FONT)
    center_text(
        draw,
        right_decision,
        "Khác nhãn\n→ Dynamic Classifier Selection (DCS)\nChọn mô hình có độ tin cậy cao hơn",
        BOX_SMALL_FONT,
        spacing=5,
    )
    final_text_color = "#222222" if theme.name == "gold" else "#ffffff"
    center_text(draw, final, "Kết quả cuối cùng", FINAL_FONT, fill=final_text_color)

    top_center = ((top[0] + top[2]) // 2, top[3])
    branch_y = 305
    model_centers = [((b[0] + b[2]) // 2, b[1]) for b in model_boxes]
    polyline(draw, (top_center, (top_center[0], branch_y), (model_centers[0][0], branch_y)))
    polyline(draw, ((top_center[0], branch_y), (model_centers[-1][0], branch_y)))
    for cx, cy in model_centers:
        arrow(draw, (cx, branch_y), (cx, cy))

    compare_top = ((compare[0] + compare[2]) // 2, compare[1])
    for box in model_boxes:
        source = ((box[0] + box[2]) // 2, box[3])
        arrow(draw, source, compare_top)

    compare_bottom = ((compare[0] + compare[2]) // 2, compare[3])
    split_y = 665
    left_center = ((left_decision[0] + left_decision[2]) // 2, left_decision[1])
    right_center = ((right_decision[0] + right_decision[2]) // 2, right_decision[1])
    polyline(draw, (compare_bottom, (compare_bottom[0], split_y), (left_center[0], split_y)))
    polyline(draw, ((compare_bottom[0], split_y), (right_center[0], split_y)))
    arrow(draw, (left_center[0], split_y), left_center)
    arrow(draw, (right_center[0], split_y), right_center)

    join_y = 890
    final_top = ((final[0] + final[2]) // 2, final[1])
    left_bottom = ((left_decision[0] + left_decision[2]) // 2, left_decision[3])
    right_bottom = ((right_decision[0] + right_decision[2]) // 2, right_decision[3])
    polyline(draw, (left_bottom, (left_bottom[0], join_y), (final_top[0], join_y)))
    polyline(draw, (right_bottom, (right_bottom[0], join_y), (final_top[0], join_y)))
    arrow(draw, (final_top[0], join_y), final_top)

    return img


def make_contact_sheet(paths: list[Path]) -> None:
    thumbs: list[Image.Image] = []
    for path in paths:
        with Image.open(path) as im:
            thumb = im.copy()
        thumb.thumbnail((720, 405), Image.Resampling.LANCZOS)
        thumbs.append(thumb)
    sheet = Image.new("RGB", (1500, 900), "#ffffff")
    positions = ((20, 20), (760, 20), (20, 465), (760, 465))
    for thumb, pos in zip(thumbs, positions):
        sheet.paste(thumb, pos)
    sheet.save(OUT_DIR / "dmf_accuracy_ppt_preview.png", dpi=(150, 150))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    output_paths: list[Path] = []
    for spec in SPECS:
        path = OUT_DIR / spec.filename
        image = draw_diagram(spec)
        image.save(path, dpi=(150, 150), optimize=True)
        output_paths.append(path)
        print(f"Đã xuất: {path}")
    make_contact_sheet(output_paths)
    print(f"Đã xuất: {OUT_DIR / 'dmf_accuracy_ppt_preview.png'}")


if __name__ == "__main__":
    main()

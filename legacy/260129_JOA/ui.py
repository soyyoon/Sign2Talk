import cv2
import numpy as np
from PIL import ImageFont, ImageDraw, Image

from .config import FONT_PATH, FONT_SIZE_MAIN, FONT_SIZE_SENT

_FONT_MAIN = ImageFont.truetype(FONT_PATH, FONT_SIZE_MAIN)
_FONT_SENT = ImageFont.truetype(FONT_PATH, FONT_SIZE_SENT)

def draw_panel(img, x, y, w, h, color=(15, 15, 15), alpha=0.75):
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
    return cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

def put_korean_text(
    img_bgr,
    text,
    org,
    color=(255, 255, 255),
    font=_FONT_MAIN,
    stroke_width=4,
    stroke_fill=(0, 0, 0),
):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)

    fill_rgb = (color[2], color[1], color[0])
    stroke_rgb = (stroke_fill[2], stroke_fill[1], stroke_fill[0])

    try:
        draw.text(org, text, font=font, fill=fill_rgb,
                  stroke_width=stroke_width, stroke_fill=stroke_rgb)
    except TypeError:
        x, y = org
        for dx in range(-stroke_width, stroke_width + 1):
            for dy in range(-stroke_width, stroke_width + 1):
                if dx == 0 and dy == 0:
                    continue
                draw.text((x + dx, y + dy), text, font=font, fill=stroke_rgb)
        draw.text(org, text, font=font, fill=fill_rgb)

    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

def wrap_text_by_chars(text, max_chars):
    if len(text) <= max_chars:
        return [text]
    words = text.split(" ")
    lines = []
    cur = ""
    for w in words:
        if len(cur) == 0:
            cur = w
        elif len(cur) + 1 + len(w) <= max_chars:
            cur += " " + w
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines

def get_fonts():
    return _FONT_MAIN, _FONT_SENT

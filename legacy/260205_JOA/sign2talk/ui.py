import cv2
import numpy as np
from PIL import ImageFont, ImageDraw, Image

from .config import FONT_PATH, FONT_SIZE_MAIN, FONT_SIZE_SENT

_FONT_MAIN = ImageFont.truetype(FONT_PATH, FONT_SIZE_MAIN)
_FONT_SENT = ImageFont.truetype(FONT_PATH, FONT_SIZE_SENT)


# 기본 컬러 팔레트 (다크 테마)
COLOR_BG = (18, 18, 26)
COLOR_CARD = (26, 26, 36)
COLOR_CARD_BORDER = (70, 70, 100)
COLOR_ACCENT = (0, 215, 255)
COLOR_SUCCESS = (0, 220, 130)
COLOR_TEXT_MAIN = (245, 245, 245)
COLOR_TEXT_SUB = (180, 180, 190)


def _draw_rounded_rect(img, x, y, w, h, color, radius=18, thickness=-1):
    """
    OpenCV에는 기본 라운드 사각형이 없어서
    사각형 + 4개의 원을 조합해서 그린다.
    """
    if radius <= 0:
        cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness)
        return img

    if thickness < 0:
        # 내부 채우기
        cv2.rectangle(img, (x + radius, y), (x + w - radius, y + h), color, -1)
        cv2.rectangle(img, (x, y + radius), (x + w, y + h - radius), color, -1)
        cv2.circle(img, (x + radius, y + radius), radius, color, -1)
        cv2.circle(img, (x + w - radius, y + radius), radius, color, -1)
        cv2.circle(img, (x + radius, y + h - radius), radius, color, -1)
        cv2.circle(img, (x + w - radius, y + h - radius), radius, color, -1)
    else:
        # 테두리만
        cv2.ellipse(img, (x + radius, y + radius), (radius, radius), 180, 0, 90, color, thickness)
        cv2.ellipse(img, (x + w - radius, y + radius), (radius, radius), 270, 0, 90, color, thickness)
        cv2.ellipse(img, (x + radius, y + h - radius), (radius, radius), 90, 0, 90, color, thickness)
        cv2.ellipse(img, (x + w - radius, y + h - radius), (radius, radius), 0, 0, 90, color, thickness)
        cv2.line(img, (x + radius, y), (x + w - radius, y), color, thickness)
        cv2.line(img, (x + radius, y + h), (x + w - radius, y + h), color, thickness)
        cv2.line(img, (x, y + radius), (x, y + h - radius), color, thickness)
        cv2.line(img, (x + w, y + radius), (x + w, y + h - radius), color, thickness)
    return img


def draw_panel(img, x, y, w, h, color=(15, 15, 15), alpha=0.75):
    """
    기존 패널 함수 (레거시). 내부적으로는 그냥 직사각형 박스만 쓴다.
    """
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
    return cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)


def draw_card(
    img,
    x,
    y,
    w,
    h,
    alpha=0.9,
    bg_color=COLOR_CARD,
    border_color=COLOR_CARD_BORDER,
    radius=18,
    border_thickness=2,
):
    """
    조금 더 현대적인 카드형 UI를 위한 패널.
    - 라운드 코너
    - 옅은 테두리
    - 반투명 오버레이
    """
    overlay = img.copy()
    _draw_rounded_rect(overlay, x, y, w, h, bg_color, radius=radius, thickness=-1)
    _draw_rounded_rect(overlay, x, y, w, h, border_color, radius=radius, thickness=border_thickness)
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
        # 구버전 Pillow 호환용 수동 stroke
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

"""ДЗ №8 (НМ та ГН 2026): детекція об'єктів YOLOv8 у Streamlit з українськими назвами класів."""
import io
from collections import Counter

import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError

from classes_uk import COCO_UK

MODELS = {
    "YOLOv8n (найшвидша)": "yolov8n.pt",
    "YOLOv8s (точніша)": "yolov8s.pt",
}
MAX_SIDE = 4000
PALETTE = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#46f0f0",
           "#f032e6", "#bcf60c", "#fabebe", "#008080", "#e6beff", "#9a6324"]

st.set_page_config(page_title="Детектор об'єктів — YOLOv8", page_icon="🔎", layout="wide")


@st.cache_resource(show_spinner="Завантаження моделі YOLO…")
def load_model(weights: str):
    from ultralytics import YOLO

    return YOLO(weights)


@st.cache_resource
def load_font(size: int):
    for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                 "DejaVuSans-Bold.ttf", "C:/Windows/Fonts/arialbd.ttf"):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_boxes(img: Image.Image, dets: list[dict]) -> Image.Image:
    out = img.copy()
    draw = ImageDraw.Draw(out)
    font = load_font(max(14, img.width // 45))
    width = max(2, img.width // 300)
    for d in dets:
        color = PALETTE[d["class_id"] % len(PALETTE)]
        x1, y1, x2, y2 = d["box"]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
        label = f"{d['name']} {d['conf']:.0%}"
        tb = draw.textbbox((x1, y1), label, font=font)
        th = tb[3] - tb[1] + 6
        ty = y1 - th if y1 - th > 0 else y1
        draw.rectangle([x1, ty, x1 + tb[2] - tb[0] + 8, ty + th], fill=color)
        draw.text((x1 + 4, ty + 2), label, fill="white", font=font)
    return out


# --- Бічна панель ------------------------------------------------------------
st.sidebar.header("⚙️ Налаштування")
model_label = st.sidebar.selectbox("Модель", list(MODELS))
conf_thr = st.sidebar.slider("Поріг упевненості", 0.05, 0.95, 0.35, 0.05)
iou_thr = st.sidebar.slider("Поріг IoU (NMS)", 0.1, 0.9, 0.5, 0.05)
class_filter = st.sidebar.multiselect(
    "Показувати лише класи", COCO_UK, placeholder="усі 80 класів"
)
show_table = st.sidebar.checkbox("Таблиця знайдених об'єктів", value=True)

# --- Основна частина ---------------------------------------------------------
st.title("🔎 Детектор об'єктів на фото")
st.caption(
    f"Модель: **{model_label}** (Ultralytics, навчена на COCO) · "
    "80 класів з українськими назвами"
)

uploaded = st.file_uploader("Завантажте фото (JPG, PNG, WEBP, BMP)",
                            type=["jpg", "jpeg", "png", "webp", "bmp"])
if uploaded is None:
    st.info("Завантажте фото — модель знайде на ньому людей, тварин, транспорт, предмети тощо.")
    st.stop()

try:
    image = Image.open(io.BytesIO(uploaded.getvalue()))
    image.load()
    image = image.convert("RGB")
except (UnidentifiedImageError, OSError, ValueError):
    st.error("❌ Не вдалося відкрити файл як зображення. Можливо, файл пошкоджений "
             "або має інше розширення. Спробуйте інше фото.")
    st.stop()

if min(image.size) < 32:
    st.error("❌ Зображення занадто маленьке (менше 32 пікселів). Завантажте більше фото.")
    st.stop()
if max(image.size) > MAX_SIDE:
    image.thumbnail((MAX_SIDE, MAX_SIDE))
    st.warning(f"Фото дуже велике, його зменшено до {image.size[0]}×{image.size[1]}.")

model = load_model(MODELS[model_label])
with st.spinner("Шукаю об'єкти…"):
    res = model.predict(image, conf=conf_thr, iou=iou_thr, verbose=False)[0]

dets = []
for box, cls, conf in zip(res.boxes.xyxy.tolist(), res.boxes.cls.tolist(), res.boxes.conf.tolist()):
    cid = int(cls)
    name = COCO_UK[cid]
    if class_filter and name not in class_filter:
        continue
    dets.append({"class_id": cid, "name": name, "conf": float(conf),
                 "box": [round(v) for v in box]})
dets.sort(key=lambda d: d["conf"], reverse=True)

col1, col2 = st.columns([3, 2])
with col1:
    annotated = draw_boxes(image, dets)
    st.image(annotated, caption=f"{uploaded.name} · {image.size[0]}×{image.size[1]}",
             use_container_width=True)
    buf = io.BytesIO()
    annotated.save(buf, format="PNG")
    st.download_button("⬇️ Завантажити розмічене фото", buf.getvalue(),
                       file_name="detections.png", mime="image/png")

with col2:
    if not dets:
        st.warning("Об'єктів не знайдено. Спробуйте зменшити поріг упевненості "
                   "або прибрати фільтр класів.")
    else:
        st.success(f"Знайдено об'єктів: **{len(dets)}**")
        counts = Counter(d["name"] for d in dets)
        st.subheader("Підсумок")
        for name, n in counts.most_common():
            st.write(f"• **{name}** — {n}")
        st.subheader("Упевненість")
        for d in dets[:10]:
            st.write(f"{d['name']} — {d['conf']:.1%}")
            st.progress(d["conf"])
        if show_table:
            df = pd.DataFrame(
                [{"Об'єкт": d["name"], "Упевненість": f"{d['conf']:.1%}",
                  "Рамка (x1, y1, x2, y2)": str(tuple(d["box"]))} for d in dets]
            )
            st.dataframe(df, hide_index=True, use_container_width=True)

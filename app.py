"""Streamlit-застосунок для класифікації зображень (ДЗ №8, НМ та ГН 2026).

Модель: MobileNetV2 з Keras Applications, попередньо навчена на ImageNet (1000 класів).
"""
import io

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, UnidentifiedImageError

MODEL_NAME = "MobileNetV2 (Keras Applications, ImageNet)"
INPUT_SIZE = (224, 224)

st.set_page_config(page_title="Класифікатор зображень — MobileNetV2", page_icon="🖼️")


@st.cache_resource(show_spinner="Завантаження моделі MobileNetV2…")
def load_model():
    from tensorflow.keras.applications import MobileNetV2

    return MobileNetV2(weights="imagenet")


def preprocess(img: Image.Image) -> np.ndarray:
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

    img = img.convert("RGB").resize(INPUT_SIZE)
    arr = np.asarray(img, dtype=np.float32)[None, ...]
    return preprocess_input(arr)


def predict(model, img: Image.Image, top_k: int):
    from tensorflow.keras.applications.mobilenet_v2 import decode_predictions

    preds = model.predict(preprocess(img), verbose=0)
    decoded = decode_predictions(preds, top=top_k)[0]
    return [(label.replace("_", " "), float(p)) for _, label, p in decoded]


# --- Бічна панель: налаштування ---------------------------------------------
st.sidebar.header("Налаштування")
st.sidebar.markdown(f"**Модель:** {MODEL_NAME}")
top_k = st.sidebar.slider("Кількість варіантів (top-k)", 1, 10, 5)
threshold = st.sidebar.slider("Поріг упевненості, %", 0, 100, 20, step=5) / 100
show_table = st.sidebar.checkbox("Показати таблицю ймовірностей", value=True)

# --- Основна частина ---------------------------------------------------------
st.title("🖼️ Класифікатор зображень")
st.caption(f"Модель: **{MODEL_NAME}** · 1000 класів ImageNet")

uploaded = st.file_uploader(
    "Завантажте зображення (JPG, PNG, WEBP)", type=["jpg", "jpeg", "png", "webp"]
)

if uploaded is None:
    st.info("Завантажте зображення, щоб отримати прогноз.")
    st.stop()

try:
    image = Image.open(io.BytesIO(uploaded.getvalue()))
    image.load()
except (UnidentifiedImageError, OSError):
    st.error("Не вдалося відкрити файл як зображення. Спробуйте інший файл.")
    st.stop()

if min(image.size) < 32:
    st.warning("Зображення дуже маленьке — прогноз може бути неточним.")

st.image(image, caption=f"{uploaded.name} · {image.size[0]}×{image.size[1]}", use_container_width=True)

model = load_model()
with st.spinner("Розпізнаю…"):
    results = predict(model, image, top_k)

best_label, best_p = results[0]
if best_p >= threshold:
    st.success(f"**{best_label}** — упевненість {best_p:.1%}")
else:
    st.warning(
        f"Модель не впевнена: найкращий варіант **{best_label}** має лише {best_p:.1%}, "
        f"що нижче порогу {threshold:.0%}."
    )

st.subheader(f"Top-{top_k} прогнозів")
for label, p in results:
    mark = "✅" if p >= threshold else "▫️"
    st.write(f"{mark} {label} — {p:.1%}")
    st.progress(min(max(p, 0.0), 1.0))

if show_table:
    df = pd.DataFrame(results, columns=["Клас", "Ймовірність"])
    df["Ймовірність"] = df["Ймовірність"].map(lambda v: f"{v:.2%}")
    st.dataframe(df, hide_index=True, use_container_width=True)

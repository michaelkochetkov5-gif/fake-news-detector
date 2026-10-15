import streamlit as st
import os
import joblib
from main_code import (
    search_duckduckgo,
    search_wikipedia_api,
    get_wikipedia_full_text,
    split_into_claims
)

st.title("🔧 Отладка фактчекера")
from main_code import FACTCHECKER_VERSION

st.info(f"Активная версия фактчекера: {FACTCHECKER_VERSION}")
st.subheader("1. Проверка ML-модели")

model_paths = [
    "russian_fake_news_model_lemma_best.pkl",
    "fake-news-detector/russian_fake_news_model_lemma_best.pkl",
    "pages/russian_fake_news_model_lemma_best.pkl"
]

vectorizer_paths = [
    "russian_fake_news_vectorizer_lemma.pkl",
    "fake-news-detector/russian_fake_news_vectorizer_lemma.pkl",
    "pages/russian_fake_news_vectorizer_lemma.pkl"
]

st.write("Проверяю пути к модели:")

for path in model_paths:
    exists = os.path.exists(path)
    st.write(f"`{path}` — {'✅ найден' if exists else '❌ нет'}")

for path in vectorizer_paths:
    exists = os.path.exists(path)
    st.write(f"`{path}` — {'✅ найден' if exists else '❌ нет'}")

model_path = None
vectorizer_path = None

for path in model_paths:
    if os.path.exists(path):
        model_path = path
        break

for path in vectorizer_paths:
    if os.path.exists(path):
        vectorizer_path = path
        break

if model_path and vectorizer_path:
    try:
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)

        st.success("ML-модель успешно загружается")
        st.write("Классы модели:")
        st.write(model.classes_)

    except Exception as e:
        st.error(f"Ошибка загрузки модели: {e}")

else:
    st.error(
        "ML-модель не найдена. "
        "Проверь, где лежат .pkl файлы."
    )

st.divider()

st.subheader("2. Разбиение на смыслы")

test_text = st.text_input(
    "Текст для проверки",
    value="Земля имеет форму шара."
)

if st.button("Разбить на утверждения"):
    claims = split_into_claims(test_text)

    st.write("Смысловые части:")

    for claim in claims:
        st.write(f"- {claim}")

st.divider()

st.subheader("3. Проверка Wikipedia")

if st.button("Проверить Wikipedia"):
    results = search_wikipedia_api(test_text)

    st.write(f"Wikipedia вернул результатов: {len(results)}")

    for result in results:
        st.write(f"**{result['title']}**")
        st.write(result["url"])
        st.write(result["body"][:300])
        st.divider()

    if results:
        full_text = get_wikipedia_full_text(
            results[0]["title"]
        )

        st.write(
            f"Длина полного текста первой статьи: {len(full_text)}"
        )
        st.write(full_text[:500])

st.divider()

st.subheader("4. Проверка DuckDuckGo")

if st.button("Проверить DuckDuckGo"):
    results = search_duckduckgo(test_text)

    st.write(f"DuckDuckGo вернул результатов: {len(results)}")

    for result in results:
        st.write(f"**{result['title']}**")
        st.write(result["url"])
        st.write(result["body"][:300])
        st.divider()

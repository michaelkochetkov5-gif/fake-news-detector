import streamlit as st
import joblib
import re
import csv
import os
from pymorphy3 import MorphAnalyzer
from scipy.special import expit
from main_code import FactChecker

CSV_FILE = "test_results.csv"

TEST_CASES = [
    {
        "text": "Земля имеет форму шара.",
        "expected": "ПРАВДА"
    },
    {
        "text": "Земля плоская и стоит на трёх китах.",
        "expected": "ФЕЙК"
    },
    {
        "text": "Вода кипит при температуре 100 градусов Цельсия при нормальном атмосферном давлении.",
        "expected": "ПРАВДА"
    },
    {
        "text": "Вода замерзает при температуре 50 градусов Цельсия.",
        "expected": "ФЕЙК"
    },
    {
        "text": "Вакцины помогают организму выработать защиту от некоторых инфекционных заболеваний.",
        "expected": "ПРАВДА"
    },
    {
        "text": "Вакцины вызывают у всех людей мгновенную смерть.",
        "expected": "ФЕЙК"
    },
    {
        "text": "МВФ прогнозирует рост мировой экономики в 2026 году на 3,1%.",
        "expected": "ПРАВДА"
    },
    {
        "text": "МВФ прогнозирует, что мировая экономика в 2026 году сократится на 3,1% из-за глобальной рецессии.",
        "expected": "ФЕЙК"
    },
    {
        "text": "Чемпионат мира по футболу 2026 года пройдёт в США, Канаде и Мексике.",
        "expected": "ПРАВДА"
    },
    {
        "text": "Чемпионат мира по футболу 2026 года прошёл в Аргентине, Уругвае и Парагвае.",
        "expected": "ФЕЙК"
    },
    {
        "text": "Python — это язык программирования, который часто используется в анализе данных и машинном обучении.",
        "expected": "ПРАВДА"
    },
    {
        "text": "Python был создан специально для взлома банковских систем.",
        "expected": "ФЕЙК"
    }
]


@st.cache_resource
def load_fact_checker():
    return FactChecker()


@st.cache_resource
def load_style_model():
    try:
        model = joblib.load(
            "fake-news-detector/russian_fake_news_model_lemma_best.pkl"
        )

        vectorizer = joblib.load(
            "fake-news-detector/russian_fake_news_vectorizer_lemma.pkl"
        )

        return model, vectorizer

    except Exception as e:
        st.error(f"Ошибка загрузки ML-модели: {e}")

        return None, None

fact_checker = load_fact_checker()
model, vectorizer = load_style_model()
morph = MorphAnalyzer()


def clean_text(text):
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"\S+@\S+", "", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.lower().strip()


def lemmatize_text(text):
    words = text.split()
    lemmas = []

    for word in words:
        if word.isalpha() and len(word) > 1:
            try:
                lemmas.append(
                    morph.parse(word)[0].normal_form
                )
            except Exception:
                lemmas.append(word)

    return " ".join(lemmas)


def analyze_style(text):
    if model is None or vectorizer is None:
        return None, 0.0

    cleaned = clean_text(text)
    lemmatized = lemmatize_text(cleaned)

    vectorized = vectorizer.transform([lemmatized])

    prediction = model.predict(vectorized)[0]

    try:
        probabilities = model.predict_proba(vectorized)[0]
        confidence = max(probabilities) * 100
    except Exception:
        decision = model.decision_function(vectorized)[0]
        confidence = expit(decision) * 100

    # ВАЖНО:
    # Если после запуска теста метки окажутся перевёрнутыми,
    # поменяй 0 и 1 местами.
    if prediction == 0:
        return "ПРАВДА", confidence

    return "ФЕЙК", confidence


st.title("🧪 Тест фактчекера и ML-модели")

st.write(
    "Тест подаёт проверенные и ложные утверждения по очереди, "
    "разбивает их на смысловые части и проверяет каждую отдельно."
)

if st.button("▶️ Запустить тест"):
    results = []

    fact_correct = 0
    style_correct = 0
    style_checked = 0

    progress = st.progress(0)
    status = st.empty()

    for i, case in enumerate(TEST_CASES):
        text = case["text"]
        expected = case["expected"]

        status.write(
            f"Проверяю {i + 1} из {len(TEST_CASES)}: {text}"
        )

        fact_result = fact_checker.verify(text)

        fact_verdict = fact_result["verdict"]
        fact_is_correct = fact_verdict == expected

        if fact_is_correct:
            fact_correct += 1

        style_verdict, style_confidence = analyze_style(text)

        if style_verdict is not None:
            style_checked += 1

            style_is_correct = style_verdict == expected

            if style_is_correct:
                style_correct += 1

        else:
            style_is_correct = None

        sources = []

        for detail in fact_result.get("details", []):
            for link in detail.get("links", []):
                if link:
                    sources.append(link)

        results.append({
            "Текст": text,
            "Ожидалось": expected,
            "Фактчекер": fact_verdict,
            "Фактчекер верно": (
                "Да" if fact_is_correct else "Нет"
            ),
            "Уверенность фактчекера": round(
                fact_result["confidence"] * 100, 1
            ),
            "Причина фактчекера": fact_result["reason"],
            "ML-модель": style_verdict or "Не применялась",
            "ML верно": (
                "Да"
                if style_is_correct is True
                else "Нет"
                if style_is_correct is False
                else "—"
            ),
            "Уверенность ML": round(style_confidence, 1),
            "Количество источников": len(sources),
            "Первая ссылка": sources[0] if sources else ""
        })

        progress.progress((i + 1) / len(TEST_CASES))

    status.success("Тест завершён")

    total = len(TEST_CASES)

    fact_accuracy = fact_correct / total * 100

    if style_checked > 0:
        style_accuracy = style_correct / style_checked * 100
    else:
        style_accuracy = 0

    st.subheader("📊 Итоги теста")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Точность фактчекера",
            f"{fact_accuracy:.1f}%"
        )

    with col2:
        st.metric(
            "Точность ML-модели",
            f"{style_accuracy:.1f}%"
        )

    with col3:
        st.metric(
            "Проверено примеров",
            total
        )

    st.subheader("📋 Подробные результаты")

    for item in results:
        fact_icon = (
            "✅"
            if item["Фактчекер верно"] == "Да"
            else "❌"
        )

        style_icon = (
            "✅"
            if item["ML верно"] == "Да"
            else "❌"
            if item["ML верно"] == "Нет"
            else "➖"
        )

        st.markdown(
            f"""
            **{item['Текст']}**

            - Ожидалось: **{item['Ожидалось']}**
            - {fact_icon} Фактчекер: **{item['Фактчекер']}** 
              ({item['Уверенность фактчекера']}%)
            - {style_icon} ML-модель: **{item['ML-модель']}** 
              ({item['Уверенность ML']}%)
            - Источников найдено: **{item['Количество источников']}**
            - Причина: {item['Причина фактчекера']}
            """
        )

        if item["Первая ссылка"]:
            st.markdown(
                f"[Открыть первый источник]({item['Первая ссылка']})"
            )

        st.divider()

    file_exists = os.path.exists(CSV_FILE)

    with open(
        CSV_FILE,
        "a",
        newline="",
        encoding="utf-8-sig"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(results[0].keys())
        )

        if not file_exists:
            writer.writeheader()

        for item in results:
            writer.writerow(item)

    st.success(
        f"Результаты сохранены в файл {CSV_FILE}"
    )

import streamlit as st
import pandas as pd
import os
import joblib

from main_code import FactChecker

# БАГ 1: единая константа для метки фейка (по семантике site.py: 0 = ФЕЙК, 1 = ПРАВДА)
FAKE_LABEL = 0

# Пути к модели — относительно папки этого файла, чтобы работало
# и локально, и на Streamlit Cloud независимо от рабочей директории
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "russian_fake_news_model_lemma_best.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "russian_fake_news_vectorizer_lemma.pkl")


def prepare_text(text):
    """
    БАГ 2: убираем лемматизацию - модель обучена на сырых текстах.
    Оставляем только lowercase для соответствия TfidfVectorizer(lowercase=True).
    """
    return text.lower()


@st.cache_resource
def load_model():
    """
    Загружает ML-модель и векторизатор.
    Если файлов нет — возвращает (None, None), чтобы приложение не падало.
    """
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
        return None, None

    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer


model, vectorizer = load_model()

if model is None or vectorizer is None:
    st.warning(
        "⚠️ ML-модель не найдена в репозитории "
        f"({os.path.basename(MODEL_PATH)}, {os.path.basename(VECTORIZER_PATH)}). "
        "Тест будет работать только с чистым фактчекером. "
        "Добавьте .pkl-файлы в Git и запушьте их, чтобы включить ML-модель."
    )


def analyze_style(text):
    """
    Независимый вердикт ML-модели.
    Возвращает (вердикт, уверенность в процентах) или (None, 0.0),
    если модель не загружена.
    """
    if model is None or vectorizer is None:
        return None, 0.0

    prepared = prepare_text(text)
    features = vectorizer.transform([prepared])

    proba = model.predict_proba(features)[0]

    # БАГ 1: находим индекс FAKE_LABEL в model.classes_ вместо жёсткого индекса [1]
    try:
        fake_idx = list(model.classes_).index(FAKE_LABEL)
        fake_prob = proba[fake_idx]
    except (ValueError, AttributeError):
        fake_prob = proba[0]

    if fake_prob >= 0.60:
        return "ФЕЙК", round(fake_prob * 100, 1)
    elif fake_prob <= 0.40:
        return "ПРАВДА", round((1 - fake_prob) * 100, 1)
    else:
        return "НЕИЗВЕСТНО", round(max(fake_prob, 1 - fake_prob) * 100, 1)


fact_checker = FactChecker()

TEST_CASES = [
    {
        "text": (
            "Правительство России утвердило новый национальный проект "
            "по развитию цифровой экономики. Программа предусматривает "
            "расширение доступа к высокоскоростному интернету в малых "
            "населённых пунктах, поддержку отечественных IT-компаний "
            "и развитие цифровых государственных услуг. На реализацию "
            "проекта планируется направить значительные бюджетные средства "
            "в течение нескольких лет."
        ),
        "expected": "ПРАВДА",
    },
    {
        "text": (
            "Центральный банк России объявил о полном запрете наличных "
            "денег с 1 января следующего года. Согласно сообщению регулятора, "
            "все граждане обязаны в течение 24 часов перевести сбережения "
            "в цифровую валюту, иначе их банковские счета будут заблокированы. "
            "Решение уже вступило в силу без обсуждения в Государственной думе."
        ),
        "expected": "ФЕЙК",
    },
    {
        "text": (
            "Всемирная организация здравоохранения рекомендует регулярную "
            "физическую активность, сбалансированное питание и достаточный сон "
            "как важные факторы поддержания здоровья. По данным организации, "
            "малоподвижный образ жизни повышает риск сердечно-сосудистых "
            "заболеваний, диабета и ожирения. Взрослым рекомендуется уделять "
            "физической активности не менее 150 минут в неделю."
        ),
        "expected": "ПРАВДА",
    },
    {
        "text": (
            "Учёные подтвердили, что употребление обычной питьевой воды "
            "в больших количествах полностью заменяет необходимость сна. "
            "Согласно исследованию, человек может не спать неделями без вреда "
            "для здоровья, если выпивает не менее четырёх литров воды в день. "
            "Результаты работы были опубликованы в ведущем медицинском журнале."
        ),
        "expected": "ФЕЙК",
    },
    {
        "text": (
            "Московский метрополитен ежедневно перевозит миллионы пассажиров "
            "и является одной из крупнейших систем метро в мире. В городе "
            "продолжается строительство новых линий и станций, а также обновление "
            "подвижного состава. Метро остаётся основным видом общественного "
            "транспорта для жителей столицы."
        ),
        "expected": "ПРАВДА",
    },
    {
        "text": (
            "В Москве с завтрашнего дня вводится комендантский час для всех "
            "жителей старше 18 лет. Выходить из дома будет разрешено только "
            "по специальным электронным пропускам, выданным через банки. "
            "Нарушителям грозит штраф и блокировка банковских счетов."
        ),
        "expected": "ФЕЙК",
    },
    {
        "text": (
            "Python — популярный язык программирования, который широко "
            "применяется в анализе данных, машинном обучении, автоматизации "
            "и разработке веб-приложений. Язык отличается простым синтаксисом "
            "и большим количеством библиотек. Python используется как крупными "
            "компаниями, так и независимыми разработчиками по всему миру."
        ),
        "expected": "ПРАВДА",
    },
    {
        "text": (
            "Python был создан специально для взлома банковских систем "
            "и запрещён в большинстве стран мира. Использовать язык можно "
            "только с разрешения специальных государственных органов. "
            "За незаконное применение Python предусмотрена уголовная ответственность."
        ),
        "expected": "ФЕЙК",
    },
    {
        "text": (
            "Чемпионат мира по футболу 2026 года пройдёт в США, Канаде и Мексике. "
            "Турнир станет первым чемпионатом мира, который примут три страны. "
            "Финальный матч запланирован на стадионе MetLife в Нью-Джерси. "
            "Ожидается рекордное количество участников — 48 сборных."
        ),
        "expected": "ПРАВДА",
    },
    {
        "text": (
            "Чемпионат мира по футболу 2026 года пройдёт в Аргентине, Уругвае "
            "и Парагвае. Все матчи будут проводиться на стадионах, построенных "
            "специально к турниру 1930 года. Организаторы отказались от использования "
            "современных арен в честь столетия первого чемпионата мира."
        ),
        "expected": "ФЕЙК",
    },
    {
        "text": (
            "Вакцинация помогает организму сформировать защиту от ряда "
            "инфекционных заболеваний. Перед массовым применением вакцины "
            "проходят клинические испытания и оценку безопасности. "
            "Прививки рекомендованы Всемирной организацией здравоохранения "
            "как эффективный способ предотвращения эпидемий."
        ),
        "expected": "ПРАВДА",
    },
    {
        "text": (
            "Новая универсальная вакцина вызывает мгновенную смерть у всех "
            "привитых людей. Правительства скрывают эту информацию, а все "
            "отчёты о клинических испытаниях подделаны. Официальная статистика "
            "о побочных эффектах намеренно занижена в тысячи раз."
        ),
        "expected": "ФЕЙК",
    },
]

st.title("🧪 Тест фактчекера и ML-модели")
st.write(
    "Тест подаёт проверенные и ложные утверждения по очереди. "
    "Фактчекер и ML-модель проверяют текст независимо, "
    "затем формируется общий вердикт."
)

if st.button("Запустить тест"):
    results = []

    fact_correct = 0
    pure_fact_correct = 0
    style_correct = 0
    style_checked = 0

    progress = st.progress(0)
    status = st.empty()

    for i, case in enumerate(TEST_CASES):
        text = case["text"]
        expected = case["expected"]

        status.write(f"Проверяю {i + 1} из {len(TEST_CASES)}: {text[:80]}...")

        # --- Независимый вердикт фактчекера ---
        fact_result = fact_checker.verify(text)
        pure_fact_verdict = fact_result["verdict"]
        pure_fact_confidence = fact_result["confidence"]

        sources = []
        for detail in fact_result.get("details", []):
            for link in detail.get("links", []):
                if link:
                    sources.append(link)
        sources_count = len(sources)

        # --- Независимый вердикт ML-модели ---
        style_verdict, style_confidence = analyze_style(text)

        # --- Общий вердикт ---
        if (
            style_verdict is not None
            and pure_fact_verdict == style_verdict
            and pure_fact_verdict in ("ПРАВДА", "ФЕЙК")
        ):
            final_verdict = pure_fact_verdict
            final_confidence = max(
                pure_fact_confidence,
                style_confidence / 100
            )
            final_reason = (
                "Фактчекер и ML-модель согласны: вердикт подтверждён "
                "обоими методами."
            )

        elif pure_fact_verdict in ("ПРАВДА", "ФЕЙК"):
            final_verdict = pure_fact_verdict
            final_confidence = pure_fact_confidence
            final_reason = fact_result["reason"]

        elif style_verdict in ("ПРАВДА", "ФЕЙК"):
            final_verdict = style_verdict
            final_confidence = style_confidence / 100
            final_reason = (
                "Фактчекер не определил вердикт; "
                "ответ дан ML-моделью по стилю текста."
            )

        else:
            final_verdict = "НЕИЗВЕСТНО"
            final_confidence = pure_fact_confidence
            final_reason = fact_result["reason"]

        # --- Подсчёт точности ---
        fact_is_correct = final_verdict == expected
        if fact_is_correct:
            fact_correct += 1

        pure_fact_is_correct = pure_fact_verdict == expected
        if pure_fact_is_correct:
            pure_fact_correct += 1

        if style_verdict is not None:
            style_checked += 1
            style_is_correct = style_verdict == expected
            if style_is_correct:
                style_correct += 1
        else:
            style_is_correct = None

        results.append({
            "Текст": text,
            "Ожидалось": expected,
            "Чистый фактчекер": pure_fact_verdict,
            "Уверенность фактчекера": round(
                pure_fact_confidence * 100, 1
            ),
            "ML-модель": style_verdict or "Не применялась",
            "Уверенность ML": round(style_confidence, 1),
            "Общий вердикт": final_verdict,
            "Уверенность общего": round(final_confidence * 100, 1),
            "Причина": final_reason,
            "Общий вердикт верен": "Да" if fact_is_correct else "Нет",
            "Количество источников": sources_count,
            "Первая ссылка": sources[0] if sources else ""
        })

        progress.progress((i + 1) / len(TEST_CASES))

    status.success("Тест завершён")

    total = len(TEST_CASES)

    fact_accuracy = fact_correct / total * 100
    pure_fact_accuracy = pure_fact_correct / total * 100

    if style_checked > 0:
        style_accuracy = style_correct / style_checked * 100
    else:
        style_accuracy = 0

    st.subheader("📊 Итоги теста")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Общий вердикт",
            f"{fact_accuracy:.1f}%"
        )

    with col2:
        st.metric(
            "Чистый фактчекер",
            f"{pure_fact_accuracy:.1f}%"
        )

    with col3:
        st.metric(
            "ML-модель",
            f"{style_accuracy:.1f}%"
        )

    with col4:
        st.metric(
            "Проверено примеров",
            total
        )

    st.subheader("📋 Подробные результаты")

    df = pd.DataFrame(results)

    for _, row in df.iterrows():
        expected = row["Ожидалось"]
        final = row["Общий вердикт"]
        pure = row["Чистый фактчекер"]
        ml = row["ML-модель"]

        if final == expected:
            icon = "✅"
        else:
            icon = "❌"

        st.markdown(f"### {icon} {row['Текст'][:120]}...")
        st.write(f"**Ожидалось:** {expected}")
        st.write(
            f"**Чистый фактчекер:** {pure} "
            f"({row['Уверенность фактчекера']}%)"
        )
        st.write(
            f"**ML-модель:** {ml} "
            f"({row['Уверенность ML']}%)"
        )
        st.write(
            f"**Общий вердикт:** {final} "
            f"({row['Уверенность общего']}%)"
        )
        st.write(f"**Причина:** {row['Причина']}")
        st.write(f"**Источников найдено:** {row['Количество источников']}")

        if row["Первая ссылка"]:
            st.markdown(
                f"[Открыть первый источник]({row['Первая ссылка']})"
            )

        st.divider()

else:
    st.info("Нажми «Запустить тест», чтобы начать проверку.")

import streamlit as st
import joblib
import re
import json
from pymorphy3 import MorphAnalyzer
from main_code import FactChecker
from scipy.special import expit
from datetime import datetime

# ===== ФАКТЧЕКЕР =====
@st.cache_resource
def load_fact_checker():
    return FactChecker()

fact_checker = load_fact_checker()

# ===== КЭШ ПРОВЕРОК =====
CACHE_FILE = "cache.json"

def load_cache():
    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as file:
            return json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

def save_cache(cache):
    with open(CACHE_FILE, "w", encoding="utf-8") as file:
        json.dump(cache, file, ensure_ascii=False, indent=2, default=str)

# ===== СТИЛЕВОЙ ОКРАС =====
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
    except (FileNotFoundError, OSError):
        return None, None

model, vectorizer = load_style_model()
morph = MorphAnalyzer()

# ===== ФУНКЦИИ =====
def clean_text(text):
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"\S+@\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"[^\w\s.,!?;:()\-\"]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()

def lemmatize_text(text):
    words = text.split()
    lemmas = []

    for word in words:
        if word.isalpha() and len(word) > 1:
            try:
                lemmas.append(morph.parse(word)[0].normal_form)
            except Exception:
                lemmas.append(word)

    return " ".join(lemmas)

def analyze_style(text):
    if model is None or vectorizer is None:
        return None, 0.0, ""

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

    return prediction, confidence, lemmatized

# ===== SESSION STATE =====
if "history" not in st.session_state:
    st.session_state.history = []

if "stats" not in st.session_state:
    st.session_state.stats = {
        "total": 0,
        "fake_style": 0,
        "truth_style": 0,
        "fake_fact": 0,
        "truth_fact": 0,
        "themes": {}
    }

# ===== СТРАНИЦА =====
st.set_page_config(
    page_title="🔍 Детектор Фейков",
    page_icon="🕵️",
    layout="wide"
)

st.title("🔍 Детектор Фейковых Новостей + Фактчекинг")
st.markdown("---")

# ===== БОКОВАЯ ПАНЕЛЬ =====
with st.sidebar:
    st.header("📈 Статистика")
    stats = st.session_state.stats

    st.metric("Всего проверок", stats["total"])
    st.metric("Фейков (стилевой окрас)", stats["fake_style"])
    st.metric("Правд (стилевой окрас)", stats["truth_style"])
    st.metric("Фейков (факты)", stats["fake_fact"])
    st.metric("Правд (факты)", stats["truth_fact"])

    st.header("📜 История")

    if st.session_state.history:
        for i, item in enumerate(reversed(st.session_state.history[-10:]), 1):
            theme_emoji = {
                "еда": "🍕", "учёба": "📚", "ии": "🤖", "наука": "🔬",
                "it": "💻", "игры": "🎮", "кино": "🎬", "музыка": "🎵",
                "спорт": "⚽", "путешествия": "✈️", "факты": "🧠",
                "общее": "📰"
            }

            theme = item.get("theme", "общее")
            emoji = theme_emoji.get(theme, "📰")
            verdict_icon = "❌" if item["fact_verdict"] == "ФЕЙК" else "✅"
            short_text = (
                item["text"][:30] + "..."
                if len(item["text"]) > 30
                else item["text"]
            )

            if st.button(
                f"{emoji} {verdict_icon} {short_text}",
                key=f"history_{i}"
            ):
                st.session_state["load_history"] = item

        st.session_state["load_history"] = None
    else:
        st.write("Пока нет проверок")

    st.header("📝 Примеры")
    st.markdown("""
    **🍕 Еда:**  
    - Сахар вызывает зависимость

    **📚 Учёба:**  
    - ЕГЭ отменят в 2026

    **🤖 ИИ:**  
    - ChatGPT заменит программистов

    **🔬 Наука:**  
    - Учёные создали материал

    **🎮 Игры:**  
    - Minecraft рекордсмен
    """)

# ===== ОСНОВНОЙ ИНТЕРФЕЙС =====
st.write("**Введите текст для проверки:**")

user_input = st.text_area(
    "Текст",
    height=200,
    placeholder="Введите текст новости, факта или утверждения..."
)

theme_choice = st.selectbox(
    "Тема новости (необязательно — если не уверены, оставьте «авто»):",
    [
        "авто", "общее", "еда", "учёба", "ии", "it", "наука",
        "игры", "кино", "музыка", "спорт", "путешествия", "факты"
    ]
)

col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    check_button = st.button(
        "🔍 Проверить",
        type="primary",
        use_container_width=True
    )

with col2:
    clear_button = st.button("🗑️ Очистить", use_container_width=True)

with col3:
    examples_button = st.button("📚 Примеры", use_container_width=True)

if clear_button:
    for key in [
        "example", "fact_result", "prediction",
        "confidence", "lemmatized"
    ]:
        if key in st.session_state:
            del st.session_state[key]

    st.rerun()

if examples_button:
    st.session_state["example"] = (
        "Сахар вызывает зависимость сильнее кокаина!"
    )

if st.session_state.get("load_history"):
    item = st.session_state["load_history"]

    st.session_state["example"] = item["text"]
    st.session_state["fact_result"] = item["fact_result"]
    st.session_state["prediction"] = item["prediction"]
    st.session_state["confidence"] = item["confidence"]
    st.session_state["lemmatized"] = item["lemmatized"]
    st.session_state["load_history"] = None

    st.rerun()

if check_button or "example" in st.session_state:
    text_to_check = (
        st.session_state.get("example", user_input)
        if "example" in st.session_state
        else user_input
    )

    if text_to_check.strip() == "":
        st.warning("⚠️ Пожалуйста, введите текст!")
    else:
        cache = load_cache()
        cache_key = f"{text_to_check}||{theme_choice}"

        if cache_key in cache:
            st.info("ℹ️ Результат загружен из кэша")
            prediction, confidence, lemmatized, fact_result = cache[cache_key]
        else:
            word_count = len(text_to_check.split())

            if word_count < 8:
                prediction = None
                confidence = 0.0
                lemmatized = lemmatize_text(clean_text(text_to_check))
            else:
                with st.spinner("🔄 Анализ стилевого окраса..."):
                    prediction, confidence, lemmatized = analyze_style(
                        text_to_check
                    )

            with st.spinner("🌐 Проверка фактов и RSS-источников..."):
                fact_result = fact_checker.verify(
                    text_to_check,
                    theme=theme_choice
                )

            cache[cache_key] = (
                prediction,
                confidence,
                lemmatized,
                fact_result
            )
            save_cache(cache)

        st.session_state["fact_result"] = fact_result
        st.session_state["prediction"] = prediction
        st.session_state["confidence"] = confidence
        st.session_state["lemmatized"] = lemmatized

        st.session_state.stats["total"] += 1

        if prediction == 0:
            st.session_state.stats["fake_style"] += 1
        elif prediction == 1:
            st.session_state.stats["truth_style"] += 1

        if fact_result["verdict"] == "ФЕЙК":
            st.session_state.stats["fake_fact"] += 1
        elif fact_result["verdict"] == "ПРАВДА":
            st.session_state.stats["truth_fact"] += 1

        theme = fact_result.get("theme", "общее")

        if theme not in st.session_state.stats["themes"]:
            st.session_state.stats["themes"][theme] = 0

        st.session_state.stats["themes"][theme] += 1

        st.session_state.history.append({
            "text": text_to_check,
            "prediction": prediction,
            "confidence": confidence,
            "fact_verdict": fact_result["verdict"],
            "theme": theme,
            "time": datetime.now().strftime("%H:%M"),
            "fact_result": fact_result,
            "lemmatized": lemmatized
        })

        if len(st.session_state.history) > 20:
            st.session_state.history = st.session_state.history[-20:]

        # ===== РЕЗУЛЬТАТЫ =====
        st.markdown("---")
        st.subheader("📊 Результат анализа")

        theme_emoji = {
            "еда": "🍕", "учёба": "📚", "ии": "🤖", "наука": "🔬",
            "it": "💻", "игры": "🎮", "кино": "🎬", "музыка": "🎵",
            "спорт": "⚽", "путешествия": "✈️", "факты": "🧠",
            "общее": "📰"
        }

        st.info(f"**Тема:** {theme_emoji.get(theme, '📰')} {theme}")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Длина текста", f"{len(text_to_check)} симв.")

        with col2:
            st.metric(
                "Слов после лемматизации",
                f"{len(lemmatized.split())}"
            )

        with col3:
            if prediction is None:
                st.metric("Уверенность стилевого окраса", "—")
            else:
                st.metric(
                    "Уверенность стилевого окраса",
                    f"{confidence:.1f}%"
                )

        st.markdown("---")

        col_fact, col_style = st.columns(2)

        with col_fact:
            st.markdown("### 🔍 Фактчекинг")

            if fact_result["verdict"] == "ПРАВДА":
                st.success(
                    f"✅ {fact_result['verdict']} "
                    f"({fact_result['confidence']:.0%})"
                )
            elif fact_result["verdict"] == "ФЕЙК":
                st.error(
                    f"❌ {fact_result['verdict']} "
                    f"({fact_result['confidence']:.0%})"
                )
            else:
                st.warning(
                    f"⚠️ {fact_result['verdict']} "
                    f"({fact_result['confidence']:.0%})"
                )

            st.info(f"📖 {fact_result['reason']}")

        with col_style:
            st.markdown("### 🎨 Стилевой окрас")

            if prediction is None:
                st.info(
                    "➖ Не применялся: короткое утверждение, "
                    "решение принимается по фактам."
                )
            elif prediction == 0:
                st.error(f"❌ ФЕЙК ({confidence:.1f}%)")
            else:
                st.success(f"✅ ПРАВДА ({confidence:.1f}%)")

        # ===== ИТОГ =====
        st.markdown("---")
        st.subheader("🎯 Итоговый вердикт")

        if fact_result["verdict"] == "ФЕЙК":
            st.error(
                "❌ **ФЕЙКОВАЯ НОВОСТЬ** "
                "(не соответствует фактам)"
            )
            st.write(f"**Причина:** {fact_result['reason']}")

        elif fact_result["verdict"] == "ПРАВДА":
            st.success(
                "✅ **ПРАВДИВАЯ НОВОСТЬ** "
                "(подтверждено фактами)"
            )
            st.write(f"**Причина:** {fact_result['reason']}")

        elif prediction is None:
            st.warning(
                "⚠️ **НЕ УДАЛОСЬ НАДЁЖНО ПРОВЕРИТЬ** "
                "(короткое утверждение)"
            )
            st.write(f"**Причина:** {fact_result['reason']}")

        elif (
            fact_result["verdict"] == "НЕИЗВЕСТНО"
            and confidence < 70
        ):
            st.warning(
                "⚠️ **НЕДОСТАТОЧНО ДАННЫХ "
                "ДЛЯ НАДЁЖНОГО ВЕРДИКТА**"
            )
            st.write(f"**Причина:** {fact_result['reason']}")
            st.write(
                f"**Стилевой окрас:** "
                f"{'ФЕЙК' if prediction == 0 else 'ПРАВДА'} "
                f"({confidence:.1f}%)"
            )

        elif (
            fact_result["verdict"] == "НЕИЗВЕСТНО"
            and prediction == 0
        ):
            st.error(
                "❌ **Вероятно ФЕЙК** (стилевой окрас)"
            )
            st.write(f"**Уверенность:** {confidence:.1f}%")

        else:
            st.success(
                "✅ **Вероятно ПРАВДА** (стилевой окрас)"
            )
            st.write(f"**Уверенность:** {confidence:.1f}%")

        # ===== ИСТОЧНИКИ =====
        if fact_result.get("details"):
            all_links = []

            for detail in fact_result["details"]:
                links = detail.get("links", [])

                if links:
                    all_links.extend(links)

            if all_links:
                st.markdown("---")
                st.subheader("🔗 Источники информации")

                unique_links = list(dict.fromkeys(all_links))[:10]

                for i, link in enumerate(unique_links, 1):
                    matching_detail = next(
                        (
                            detail for detail in fact_result["details"]
                            if link in detail.get("links", [])
                        ),
                        None
                    )

                    if matching_detail:
                        st.markdown(
                            f"{i}. [{link}]({link}) "
                            f"*( {matching_detail['reason']} )*"
                        )
                    else:
                        st.markdown(f"{i}. [{link}]({link})")

                st.download_button(
                    label="📥 Скачать список ссылок",
                    data="\n".join(unique_links),
                    file_name="sources.txt",
                    mime="text/plain"
                )

        if fact_result.get("details"):
            with st.expander("🔍 Детали проверки фактов"):
                for detail in fact_result["details"]:
                    icon = (
                        "✅" if detail["result"] is True
                        else "❌" if detail["result"] is False
                        else "⚠️"
                    )

                    st.write(f"{icon} **{detail['fact']}**")
                    st.write(f"   *Источник:* {detail['source']}")
                    st.write(f"   *{detail['reason']}*")

                    links = detail.get("links", [])

                    if links:
                        st.write("   **Ссылки:**")

                        for link in links:
                            st.markdown(
                                f"   - [{link}]({link}) "
                                f"*( {detail['reason']} )*"
                            )

                    st.divider()

        with st.expander("🔧 Предобработанный текст"):
            st.code(lemmatized)

        if st.button("🔄 Проверить ещё один текст"):
            for key in [
                "example", "fact_result", "prediction",
                "confidence", "lemmatized"
            ]:
                if key in st.session_state:
                    del st.session_state[key]

            st.rerun()

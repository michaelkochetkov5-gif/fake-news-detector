import streamlit as st
import joblib
import re
import pandas as pd
from pymorphy3 import MorphAnalyzer
from main_code import FactChecker

try:
    from main_code import QwenFactChecker
    QWEN_AVAILABLE = True
except Exception:
    QwenFactChecker = None
    QWEN_AVAILABLE = False

from scipy.special import expit
from datetime import datetime
import json
from pathlib import Path

# ===== ФАКТЧЕКЕР =====
@st.cache_resource
def load_fact_checker():
    return FactChecker()

fact_checker = load_fact_checker()

# ===== QWEN =====
@st.cache_resource
def load_qwen_checker():
    if QwenFactChecker is None:
        return None
    return QwenFactChecker()

qwen_checker = load_qwen_checker()

# ===== КЭШ =====
CACHE_FILE = "cache.json"

def load_cache():
    try:
        with open(CACHE_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

def save_cache(cache):
    with open(CACHE_FILE, 'w', encoding='utf-8') as f:
        json.dump(cache, f, ensure_ascii=False, indent=2, default=str)

# ===== СТИЛЕВАЯ МОДЕЛЬ =====
@st.cache_resource
def load_style_model():
    try:
        model = joblib.load('fake-news-detector/russian_fake_news_model_lemma_best.pkl')
        vectorizer = joblib.load('fake-news-detector/russian_fake_news_vectorizer_lemma.pkl')
        return model, vectorizer
    except (FileNotFoundError, OSError):
        return None, None

model, vectorizer = load_style_model()
morph = MorphAnalyzer()

# ===== ФУНКЦИИ =====
def clean_text(text):
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#', '', text)
    text = re.sub(r'[^\w\s.,!?;:()\-\"]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
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

    return ' '.join(lemmas)

def predict_fake(text):
    if model is None or vectorizer is None:
        return None, 0.0, ""

    cleaned = clean_text(text)
    lemmatized = lemmatize_text(cleaned)
    vectorized = vectorizer.transform([lemmatized])
    prediction = model.predict(vectorized)[0]

    try:
        proba = model.predict_proba(vectorized)[0]
        confidence = max(proba) * 100
    except Exception:
        decision = model.decision_function(vectorized)[0]
        confidence = expit(decision) * 100

    return prediction, confidence, lemmatized

# ===== SESSION STATE =====
if 'history' not in st.session_state:
    st.session_state.history = []

if 'stats' not in st.session_state:
    st.session_state.stats = {
        'total': 0,
        'fake_style': 0,
        'truth_style': 0,
        'fake_fact': 0,
        'truth_fact': 0,
        'themes': {}
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

    st.metric("Всего проверок", stats['total'])
    st.metric("Фейков (стиль)", stats['fake_style'])
    st.metric("Правды (стиль)", stats['truth_style'])
    st.metric("Фейков (факты)", stats['fake_fact'])
    st.metric("Правды (факты)", stats['truth_fact'])

    st.header("📜 История")

    if st.session_state.history:
        for i, item in enumerate(reversed(st.session_state.history[-10:]), 1):
            theme_emoji = {
                'еда': '🍕', 'учёба': '📚', 'ии': '🤖', 'наука': '🔬',
                'it': '💻', 'игры': '🎮', 'кино': '🎬', 'музыка': '🎵',
                'спорт': '⚽', 'путешествия': '✈️', 'факты': '🧠', 'общее': '📰'
            }

            theme = item.get('theme', 'общее')
            emoji = theme_emoji.get(theme, '📰')
            verdict_icon = "❌" if item['fact_verdict'] == 'ФЕЙК' else "✅"
            short_text = item['text'][:30] + "..." if len(item['text']) > 30 else item['text']

            if st.button(f"{emoji} {verdict_icon} {short_text}", key=f"hist_{i}"):
                st.session_state['load_history'] = item

        st.session_state['load_history'] = None
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
    ['авто', 'общее', 'еда', 'учёба', 'ии', 'it', 'наука',
     'игры', 'кино', 'музыка', 'спорт', 'путешествия', 'факты']
)

col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    check_button = st.button("🔍 Проверить", type="primary", use_container_width=True)

with col2:
    clear_button = st.button("🗑️ Очистить", use_container_width=True)

with col3:
    examples_button = st.button("📚 Примеры", use_container_width=True)

if clear_button:
    for key in ['example', 'fact_result', 'prediction', 'confidence', 'lemmatized']:
        if key in st.session_state:
            del st.session_state[key]
    st.rerun()

if examples_button:
    st.session_state['example'] = "Сахар вызывает зависимость сильнее кокаина!"

if st.session_state.get('load_history'):
    item = st.session_state['load_history']
    st.session_state['example'] = item['text']
    st.session_state['fact_result'] = item['fact_result']
    st.session_state['prediction'] = item['prediction']
    st.session_state['confidence'] = item['confidence']
    st.session_state['lemmatized'] = item['lemmatized']
    st.session_state['load_history'] = None
    st.rerun()

if check_button or 'example' in st.session_state:
    text_to_check = (
        st.session_state.get('example', user_input)
        if 'example' in st.session_state
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
                with st.spinner("🔄 Анализ текста..."):
                    prediction, confidence, lemmatized = predict_fake(text_to_check)

            with st.spinner("🌐 Проверка фактов..."):
                fact_result = fact_checker.verify(text_to_check, theme=theme_choice)

            cache[cache_key] = (prediction, confidence, lemmatized, fact_result)
            save_cache(cache)

        st.session_state['fact_result'] = fact_result
        st.session_state['prediction'] = prediction
        st.session_state['confidence'] = confidence
        st.session_state['lemmatized'] = lemmatized

        st.session_state.stats['total'] += 1

        if prediction == 0:
            st.session_state.stats['fake_style'] += 1
        elif prediction == 1:
            st.session_state.stats['truth_style'] += 1

        if fact_result['verdict'] == 'ФЕЙК':
            st.session_state.stats['fake_fact'] += 1
        elif fact_result['verdict'] == 'ПРАВДА':
            st.session_state.stats['truth_fact'] += 1

        theme = fact_result.get('theme', 'общее')

        if theme not in st.session_state.stats['themes']:
            st.session_state.stats['themes'][theme] = 0

        st.session_state.stats['themes'][theme] += 1

        st.session_state.history.append({
            'text': text_to_check,
            'prediction': prediction,
            'confidence': confidence,
            'fact_verdict': fact_result['verdict'],
            'theme': theme,
            'time': datetime.now().strftime("%H:%M"),
            'fact_result': fact_result,
            'lemmatized': lemmatized
        })

        if len(st.session_state.history) > 20:
            st.session_state.history = st.session_state.history[-20:]

        # ===== РЕЗУЛЬТАТЫ =====
        st.markdown("---")
        st.subheader("📊 Результат анализа")

        theme_emoji = {
            'еда': '🍕', 'учёба': '📚', 'ии': '🤖', 'наука': '🔬',
            'it': '💻', 'игры': '🎮', 'кино': '🎬', 'музыка': '🎵',
            'спорт': '⚽', 'путешествия': '✈️', 'факты': '🧠', 'общее': '📰'
        }

        st.info(f"**Тема:** {theme_emoji.get(theme, '📰')} {theme}")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Длина текста", f"{len(text_to_check)} симв.")

        with col2:
            st.metric("Слов после лемматизации", f"{len(lemmatized.split())}")

        with col3:
            if prediction is None:
                st.metric("Уверенность модели", "—")
            else:
                st.metric("Уверенность модели", f"{confidence:.1f}%")

        st.markdown("---")

        col_fact, col_style = st.columns(2)

        with col_fact:
            st.markdown("### 🔍 Фактчекинг")

            if fact_result['verdict'] == 'ПРАВДА':
                st.success(f"✅ {fact_result['verdict']} ({fact_result['confidence']:.0%})")
            elif fact_result['verdict'] == 'ФЕЙК':
                st.error(f"❌ {fact_result['verdict']} ({fact_result['confidence']:.0%})")
            else:
                st.warning(f"⚠️ {fact_result['verdict']} ({fact_result['confidence']:.0%})")

            st.info(f"📖 {fact_result['reason']}")

        with col_style:
            st.markdown("### 📈 Стилевая модель")

            if prediction is None:
                st.info("➖ Не применялась (короткое утверждение — решение по фактам)")
            elif prediction == 0:
                st.error(f"❌ ФЕЙК ({confidence:.1f}%)")
            else:
                st.success(f"✅ ПРАВДА ({confidence:.1f}%)")

        # ===== QWEN =====
        st.markdown("---")
        st.subheader("🤖 ИИ-анализ Qwen")

st.markdown("---")
st.subheader("🤖 ИИ-анализ Qwen")

import os

IS_STREAMLIT_CLOUD = os.environ.get("STREAMLIT_SHARING_MODE") == "1" or os.path.exists("/mount/src")

if IS_STREAMLIT_CLOUD:
    st.info(
        "🤖 ИИ-анализ Qwen доступен только в локальной версии сайта.\n\n"
        "Причина: Qwen2.5-7B требует GPU и больше памяти, чем доступно в Streamlit Cloud."
    )
elif not QWEN_AVAILABLE or qwen_checker is None:
    st.warning(
        "🤖 Qwen не подключён. Проверьте main_code.py и установленные зависимости."
    )
else:
    if st.button("🧠 Запустить ИИ-анализ", use_container_width=True):
        try:
            with st.spinner("Qwen анализирует утверждение и источники..."):
                sources = []

                for detail in fact_result.get('details', []):
                    source = detail.get('source', '')
                    if source:
                        sources.append(source)

                if not sources:
                    sources = ["Надёжные источники не найдены."]

                if prediction is None:
                    ml_hint = "нет данных"
                else:
                    label = "fake" if prediction == 0 else "real"
                    ml_hint = f"style={confidence / 100:.2f} {label}"

                qwen_result = qwen_checker.analyze(
                    text_to_check,
                    sources,
                    ml_hint
                )

            if qwen_result['verdict'] == 'ПРАВДА':
                st.success(
                    f"✅ {qwen_result['verdict']} "
                    f"({qwen_result['confidence']:.0%})"
                )
            elif qwen_result['verdict'] == 'ФЕЙК':
                st.error(
                    f"❌ {qwen_result['verdict']} "
                    f"({qwen_result['confidence']:.0%})"
                )
            else:
                st.warning(
                    f"⚠️ {qwen_result['verdict']} "
                    f"({qwen_result['confidence']:.0%})"
                )

            st.info(f"🧠 {qwen_result['reason']}")

        except Exception as e:
            st.warning(
                "🤖 Qwen недоступен в облачной версии сайта. "
                "Запустите приложение локально с GPU."
            )
            st.caption(str(e))
            try:
                with st.spinner("Qwen анализирует утверждение и источники..."):
                    sources = []

                    for detail in fact_result.get('details', []):
                        source = detail.get('source', '')
                        if source:
                            sources.append(source)

                    if not sources:
                        sources = ["Надёжные источники не найдены."]

                    if prediction is None:
                        ml_hint = "нет данных"
                    else:
                        label = "fake" if prediction == 0 else "real"
                        ml_hint = f"style={confidence / 100:.2f} {label}"

                    qwen_result = qwen_checker.analyze(
                        text_to_check,
                        sources,
                        ml_hint
                    )

                if qwen_result['verdict'] == 'ПРАВДА':
                    st.success(
                        f"✅ {qwen_result['verdict']} "
                        f"({qwen_result['confidence']:.0%})"
                    )
                elif qwen_result['verdict'] == 'ФЕЙК':
                    st.error(
                        f"❌ {qwen_result['verdict']} "
                        f"({qwen_result['confidence']:.0%})"
                    )
                else:
                    st.warning(
                        f"⚠️ {qwen_result['verdict']} "
                        f"({qwen_result['confidence']:.0%})"
                    )

                st.info(f"🧠 {qwen_result['reason']}")

                used = qwen_result.get('sources_used', [])
                if used:
                    st.caption(
                        "Использованные источники: "
                        + ", ".join(map(str, used))
                    )

            except Exception as e:
                st.warning(
                    "🤖 Qwen недоступен в облачной версии сайта. "
                    "Запустите приложение локально с GPU для ИИ-анализа."
                )
                st.caption(str(e))

        # ===== ИТОГ =====
        st.markdown("---")
        st.subheader("🎯 Итоговый вердикт")

        if fact_result['verdict'] == 'ФЕЙК':
            st.error("❌ **ФЕЙКОВАЯ НОВОСТЬ** (не соответствует фактам)")
            st.write(f"**Причина:** {fact_result['reason']}")
        elif fact_result['verdict'] == 'ПРАВДА':
            st.success("✅ **ПРАВДИВАЯ НОВОСТЬ** (подтверждено фактами)")
            st.write(f"**Причина:** {fact_result['reason']}")
        elif prediction is None:
            st.warning("⚠️ **НЕ УДАЛОСЬ НАДЁЖНО ПРОВЕРИТЬ** (короткое утверждение)")
            st.write(f"**Причина:** {fact_result['reason']}")
        elif fact_result['verdict'] == 'НЕИЗВЕСТНО' and confidence < 70:
            st.warning("⚠️ **НЕДОСТАТОЧНО ДАННЫХ ДЛЯ НАДЁЖНОГО ВЕРДИКТА**")
            st.write(f"**Причина:** {fact_result['reason']}")
            st.write(
                f"**Стилевая модель:** "
                f"{'ФЕЙК' if prediction == 0 else 'ПРАВДА'} ({confidence:.1f}%)"
            )
        elif fact_result['verdict'] == 'НЕИЗВЕСТНО' and prediction == 0:
            st.error("❌ **Вероятно ФЕЙК** (стилевая модель)")
            st.write(f"**Уверенность:** {confidence:.1f}%")
        else:
            st.success("✅ **Вероятно ПРАВДА** (стилевая модель)")
            st.write(f"**Уверенность:** {confidence:.1f}%")

        # ===== ИСТОЧНИКИ =====
        if fact_result.get('details'):
            all_links = []

            for detail in fact_result['details']:
                links = detail.get('links', [])
                if links:
                    all_links.extend(links)

            if all_links:
                st.markdown("---")
                st.subheader("🔗 Источники информации")

                unique_links = list(dict.fromkeys(all_links))[:10]

                for i, link in enumerate(unique_links, 1):
                    st.markdown(f"{i}. [{link}]({link})")

                st.download_button(
                    label="📥 Скачать список ссылок",
                    data="\n".join(unique_links),
                    file_name="sources.txt",
                    mime="text/plain"
                )

        if fact_result.get('details'):
            with st.expander("🔍 Детали проверки фактов"):
                for detail in fact_result['details']:
                    icon = (
                        "✅" if detail['result'] is True
                        else "❌" if detail['result'] is False
                        else "⚠️"
                    )

                    st.write(f"{icon} **{detail['fact']}**")
                    st.write(f"   *Источник:* {detail['source']}")
                    st.write(f"   *{detail['reason']}*")

                    links = detail.get('links', [])
                    if links:
                        st.write("   **Ссылки:**")
                        for link in links[:3]:
                            st.write(f"   - [{link}]({link})")

                    st.divider()

        with st.expander("🔧 Предобработанный текст"):
            st.code(lemmatized)

        if st.button("🔄 Проверить ещё один текст"):
            for key in ['example', 'fact_result', 'prediction', 'confidence', 'lemmatized']:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()

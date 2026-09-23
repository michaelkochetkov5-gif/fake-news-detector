import streamlit as st
import joblib
import re
import pandas as pd
from pymorphy3 import MorphAnalyzer
from main_code import FactChecker
from scipy.special import expit
from datetime import datetime
import json
from pathlib import Path

# ===== ИНИЦИАЛИЗАЦИЯ ФАКТЧЕКЕРА =====
fact_checker = FactChecker()

# ===== КЭШИРОВАНИЕ =====
CACHE_FILE = "cache.json"

def load_cache():
    if Path(CACHE_FILE).exists():
        with open(CACHE_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def save_cache(cache):
    with open(CACHE_FILE, 'w', encoding='utf-8') as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)

# ===== ЗАГРУЗКА МОДЕЛИ =====
@st.cache_resource
def load_model():
    model = joblib.load('D:/russian_fake_news_model_lemma_best.pkl')
    vectorizer = joblib.load('D:/russian_fake_news_vectorizer_lemma.pkl')
    return model, vectorizer

model, vectorizer = load_model()
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
    text = text.lower()
    return text

def lemmatize_text(text):
    words = text.split()
    lemmas = []
    for word in words:
        if word.isalpha() and len(word) > 1:
            try:
                normal_form = morph.parse(word)[0].normal_form
                lemmas.append(normal_form)
            except:
                lemmas.append(word)
    return ' '.join(lemmas)

def predict_fake(text):
    cleaned = clean_text(text)
    lemmatized = lemmatize_text(cleaned)
    vectorized = vectorizer.transform([lemmatized])
    prediction = model.predict(vectorized)[0]
    
    try:
        proba = model.predict_proba(vectorized)[0]
        confidence = max(proba) * 100
    except:
        decision = model.decision_function(vectorized)[0]
        confidence = expit(decision) * 100
    
    return prediction, confidence, lemmatized

# ===== ИНИЦИАЛИЗАЦИЯ SESSION STATE =====
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

# ===== НАСТРОЙКА СТРАНИЦЫ =====
st.set_page_config(
    page_title="🔍 Детектор Фейков",
    page_icon="🕵️",
    layout="wide"
)

# ===== ИНТЕРФЕЙС =====
st.title("🔍 Детектор Фейковых Новостей + Фактчекинг")
st.markdown("---")

# Боковая панель
with st.sidebar:
    st.header("📊 О системе")
    st.info("""
    - **Модель**: Linear SVM (95.48%)
    - **Фактчекинг**: Wikipedia + DuckDuckGo + RSS (20+ сайтов)
    - **Темы**: Еда, Учёба, ИИ, Наука, IT, Игры, Кино, Музыка, Спорт, Путешествия, Факты
    - **Язык**: Русский
    """)
    
    # Статистика
    st.header("📈 Статистика")
    stats = st.session_state.stats
    st.metric("Всего проверок", stats['total'])
    st.metric("Фейков (стиль)", stats['fake_style'])
    st.metric("Правды (стиль)", stats['truth_style'])
    st.metric("Фейков (факты)", stats['fake_fact'])
    st.metric("Правды (факты)", stats['truth_fact'])
    
    # История
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
    
    # Примеры
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

# Основной интерфейс
st.write("**Введите текст для проверки:**")

user_input = st.text_area("Текст", height=200, placeholder="Введите текст новости, факта или утверждения...")

# Кнопки
col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    check_button = st.button("🔍 Проверить", type="primary", use_container_width=True)
with col2:
    clear_button = st.button("🗑️ Очистить", use_container_width=True)
with col3:
    examples_button = st.button("📚 Примеры", use_container_width=True)

# Обработка
if clear_button:
    if 'example' in st.session_state:
        del st.session_state['example']
    if 'fact_result' in st.session_state:
        del st.session_state['fact_result']
    st.rerun()

if examples_button:
    st.session_state['example'] = "Сахар вызывает зависимость сильнее кокаина!"

# Загрузка из истории
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
    text_to_check = st.session_state.get('example', user_input) if 'example' in st.session_state else user_input
    
    if text_to_check.strip() == "":
        st.warning("⚠️ Пожалуйста, введите текст!")
    else:
        # Проверяем кэш
        cache = load_cache()
        if text_to_check in cache:
            st.info("ℹ️ Результат загружен из кэша")
            prediction, confidence, lemmatized, fact_result = cache[text_to_check]
        else:
            # 1. Стилевая проверка
            with st.spinner("🔄 Анализ текста..."):
                prediction, confidence, lemmatized = predict_fake(text_to_check)
            
            # 2. Фактчекинг
            with st.spinner("🌐 Проверка фактов..."):
                fact_result = fact_checker.verify(text_to_check)
            
            # Сохраняем в кэш
            cache[text_to_check] = (prediction, confidence, lemmatized, fact_result)
            save_cache(cache)
        
        # Сохраняем в session state
        st.session_state['fact_result'] = fact_result
        st.session_state['prediction'] = prediction
        st.session_state['confidence'] = confidence
        st.session_state['lemmatized'] = lemmatized
        
        # Обновляем статистику
        st.session_state.stats['total'] += 1
        if prediction == 0:
            st.session_state.stats['fake_style'] += 1
        else:
            st.session_state.stats['truth_style'] += 1
        
        if fact_result['verdict'] == 'ФЕЙК':
            st.session_state.stats['fake_fact'] += 1
        elif fact_result['verdict'] == 'ПРАВДА':
            st.session_state.stats['truth_fact'] += 1
        
        theme = fact_result.get('theme', 'общее')
        if theme not in st.session_state.stats['themes']:
            st.session_state.stats['themes'][theme] = 0
        st.session_state.stats['themes'][theme] += 1
        
        # Сохраняем в историю
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
        
        # Ограничиваем историю 20 записями
        if len(st.session_state.history) > 20:
            st.session_state.history = st.session_state.history[-20:]
        
        # 3. Результаты
        st.markdown("---")
        st.subheader("📊 Результат анализа")
        
        # Показываем тему
        theme_emoji = {
            'еда': '🍕', 'учёба': '📚', 'ии': '🤖', 'наука': '🔬',
            'it': '💻', 'игры': '🎮', 'кино': '🎬', 'музыка': '🎵',
            'спорт': '⚽', 'путешествия': '✈️', 'факты': '🧠', 'общее': '📰'
        }
        st.info(f"**Тема:** {theme_emoji.get(theme, '📰')} {theme}")
        
        # Метрики
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Длина текста", f"{len(text_to_check)} симв.")
        with col2:
            st.metric("Слов после лемматизации", f"{len(lemmatized.split())}")
        with col3:
            st.metric("Уверенность модели", f"{confidence:.1f}%")
        
        # Вердикт
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
            if prediction == 0:
                st.error(f"❌ ФЕЙК ({confidence:.1f}%)")
            else:
                st.success(f"✅ ПРАВДА ({confidence:.1f}%)")
        
        # Итоговый вердикт
        st.markdown("---")
        st.subheader("🎯 Итоговый вердикт")
        
        if fact_result['verdict'] == 'ФЕЙК':
            st.error("❌ **ФЕЙКОВАЯ НОВОСТЬ** (не соответствует фактам)")
            st.write(f"**Причина:** {fact_result['reason']}")
        elif fact_result['verdict'] == 'ПРАВДА' and prediction == 1:
            st.success("✅ **ПРАВДИВАЯ НОВОСТЬ** (подтверждено фактами и стилем)")
            st.write(f"**Причина:** {fact_result['reason']}")
        elif fact_result['verdict'] == 'ПРАВДА' and prediction == 0:
            st.warning("⚠️ **ВОЗМОЖНО ФЕЙК** (факты верны, но стиль подозрительный)")
            st.write("**Признаки фейка:**")
            st.write("- Эмоциональные слова (шок, сенсация)")
            st.write("- Отсутствие конкретных фактов")
            st.write("- Призывы к эмоциям")
        elif fact_result['verdict'] == 'НЕИЗВЕСТНО' and prediction == 0:
            st.error("❌ **Вероятно ФЕЙК** (стилевая модель)")
            st.write(f"**Уверенность:** {confidence:.1f}%")
        else:
            st.success("✅ **Вероятно ПРАВДА** (стилевая модель)")
            st.write(f"**Уверенность:** {confidence:.1f}%")
        
        # 🔗 Источники
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
                
                if len(unique_links) > 0:
                    links_text = "\n".join(unique_links)
                    st.download_button(
                        label="📥 Скачать список ссылок",
                        data=links_text,
                        file_name="sources.txt",
                        mime="text/plain"
                    )
        
        # Детали
        if fact_result.get('details'):
            with st.expander("🔍 Детали проверки фактов"):
                for detail in fact_result['details']:
                    icon = "✅" if detail['result'] == True else "❌" if detail['result'] == False else "⚠️"
                    st.write(f"{icon} **{detail['fact']}**")
                    st.write(f"   *Источник:* {detail['source']}")
                    st.write(f"   *{detail['reason']}*")
                    
                    links = detail.get('links', [])
                    if links:
                        st.write("   **Ссылки:**")
                        for link in links[:3]:
                            st.write(f"   - [{link}]({link})")
                    
                    st.divider()
        
        # Лемматизация
        with st.expander("🔧 Предобработанный текст"):
            st.code(lemmatized)
        
        # Кнопка "Ещё раз"
        if st.button("🔄 Проверить ещё один текст"):
            if 'example' in st.session_state:
                del st.session_state['example']
            if 'fact_result' in st.session_state:
                del st.session_state['fact_result']
            st.rerun()

# Футер
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <small>
    🔍 Детектор Фейковых Новостей + Фактчекинг | Модель: Linear SVM (95.48% accuracy)<br>
    Фактчекинг: Wikipedia + DuckDuckGo + RSS (20+ сайтов)
    </small>
</div>
""", unsafe_allow_html=True)

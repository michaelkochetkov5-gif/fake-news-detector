import streamlit as st
import re
import json
import hashlib
from pymorphy3 import MorphAnalyzer
from main_code import FactChecker
from lum_checker import LUMChecker, combine, is_confident, SOURCE_MESSAGES
from nli_checker import NLIChecker
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

def get_cache_key(text):
    # v3: добавлен NLI + numeric checker.
    # Версия в ключе гарантирует, что старые записи не будут прочитаны.
    return hashlib.md5(("nli-numeric-v1|" + text).encode("utf-8")).hexdigest()

# ===== NLI + NUMERIC CHECKER =====
@st.cache_resource
def load_nli_checker():
    return NLIChecker()

nli_checker = load_nli_checker()

# ===== LUM v2 (GPU fallback) =====
@st.cache_resource
def load_lum_checker():
    return LUMChecker()

lum_checker = load_lum_checker()

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
    """
    Лемматизация для отображения в UI.
    НЕ используется для ML-инференса.
    """
    words = text.split()
    lemmas = []

    for word in words:
        if word.isalpha() and len(word) > 1:
            try:
                lemmas.append(morph.parse(word)[0].normal_form)
            except Exception:
                lemmas.append(word)

    return " ".join(lemmas)

# ===== SESSION STATE =====
if "history" not in st.session_state:
    st.session_state.history = []

if "stats" not in st.session_state:
    st.session_state.stats = {
        "total": 0,
        "fake_fact": 0,
        "truth_fact": 0
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
    st.metric("Фейков (факты)", stats["fake_fact"])
    st.metric("Правд (факты)", stats["truth_fact"])

    st.header("📜 История")

    if st.session_state.history:
        for i, item in enumerate(reversed(st.session_state.history[-10:]), 1):
            verdict_icon = (
                "❌"
                if item["fact_verdict"] == "ФЕЙК"
                else "✅"
            )

            short_text = (
                item["text"][:30] + "..."
                if len(item["text"]) > 30
                else item["text"]
            )

            if st.button(
                f"{verdict_icon} {short_text}",
                key=f"history_{i}"
            ):
                st.session_state["load_history"] = item

        st.session_state["load_history"] = None
    else:
        st.write("Пока нет проверок")

    st.header("📝 Примеры")
    st.markdown("""
    **🔬 Наука:**  
    - Земля круглая

    **🍕 Еда:**  
    - Сахар вызывает зависимость

    **🤖 ИИ:**  
    - ChatGPT заменит программистов

    **🎮 Игры:**  
    - Minecraft закроют навсегда
    """)

# ===== ОСНОВНОЙ ИНТЕРФЕЙС =====
st.write("**Введите текст для проверки:**")

user_input = st.text_area(
    "Текст",
    height=200,
    placeholder="Введите текст новости, факта или утверждения..."
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
        "example", "fact_result", "original_fact_result",
        "nli_result", "lum_result", "lemmatized"
    ]:
        if key in st.session_state:
            del st.session_state[key]

    st.rerun()

if examples_button:
    st.session_state["example"] = "Земля круглая"

if st.session_state.get("load_history"):
    item = st.session_state["load_history"]

    st.session_state["example"] = item["text"]
    st.session_state["fact_result"] = item["fact_result"]
    st.session_state["original_fact_result"] = item.get(
        "original_fact_result"
    )
    st.session_state["nli_result"] = item.get("nli_result")
    st.session_state["lum_result"] = item.get("lum_result")
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
        cache_key = get_cache_key(text_to_check)

        if cache_key in cache:
            st.info("ℹ️ Результат загружен из кэша")

            (
                fact_result,
                original_fact_result,
                nli_result,
                lum_result,
            ) = cache[cache_key]

            lemmatized = lemmatize_text(clean_text(text_to_check))
        else:
            lemmatized = lemmatize_text(clean_text(text_to_check))

            with st.spinner("🌐 Проверка фактов, Wikipedia и RSS..."):
                original_fact_result = fact_checker.verify(text_to_check)

            nli_result = None
            lum_result = None

            # NLI + numeric checker подключается, только если
            # FactChecker не дал уверенного вердикта.
            if not is_confident(original_fact_result):
                with st.spinner(
                    "🔎 NLI сравнивает утверждение с источниками..."
                ):
                    nli_result = nli_checker.verify(
                        text_to_check,
                        original_fact_result,
                    )

            # Если NLI дала уверенный вердикт — используем её.
            if nli_result and nli_result["verdict"] in (
                "ПРАВДА",
                "ФЕЙК",
            ):
                fact_result = combine(
                    original_fact_result,
                    nli_result,
                )
            else:
                # LUM остаётся последним fallback-слоем.
                if not is_confident(original_fact_result):
                    with st.spinner(
                        "🧠 Уточнение семантической моделью LUM v2..."
                    ):
                        lum_result = lum_checker.verify(text_to_check)

                fact_result = combine(
                    original_fact_result,
                    lum_result,
                )

            cache[cache_key] = (
                fact_result,
                original_fact_result,
                nli_result,
                lum_result,
            )
            save_cache(cache)

        st.session_state["fact_result"] = fact_result
        st.session_state["original_fact_result"] = original_fact_result
        st.session_state["nli_result"] = nli_result
        st.session_state["lum_result"] = lum_result
        st.session_state["lemmatized"] = lemmatized

        st.session_state.stats["total"] += 1

        if fact_result["verdict"] == "ФЕЙК":
            st.session_state.stats["fake_fact"] += 1
        elif fact_result["verdict"] == "ПРАВДА":
            st.session_state.stats["truth_fact"] += 1

        st.session_state.history.append({
            "text": text_to_check,
            "fact_verdict": fact_result["verdict"],
            "time": datetime.now().strftime("%H:%M"),
            "fact_result": fact_result,
            "original_fact_result": original_fact_result,
            "nli_result": nli_result,
            "lum_result": lum_result,
            "lemmatized": lemmatized
        })

        if len(st.session_state.history) > 20:
            st.session_state.history = st.session_state.history[-20:]

        # ===== РЕЗУЛЬТАТЫ =====
        st.markdown("---")
        st.subheader("📊 Результат анализа")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Длина текста", f"{len(text_to_check)} симв.")

        with col2:
            st.metric(
                "Слов после лемматизации",
                f"{len(lemmatized.split())}"
            )

        with col3:
            st.metric(
                "Уверенность вердикта",
                f"{fact_result['confidence']:.0%}"
            )

        st.markdown("---")

        # ===== ИТОГ =====
        st.subheader("🎯 Итоговый вердикт")

        verdict = fact_result["verdict"]

        if verdict == "ФЕЙК":
            st.error(
                "❌ **ФЕЙКОВАЯ НОВОСТЬ** "
                "(не соответствует фактам)"
            )
        elif verdict == "ПРАВДА":
            st.success(
                "✅ **ПРАВДИВАЯ НОВОСТЬ** "
                "(подтверждено фактами)"
            )
        else:
            st.warning(
                "⚠️ **НЕДОСТАТОЧНО ДАННЫХ "
                "ДЛЯ НАДЁЖНОГО ВЕРДИКТА**"
            )

        st.write(f"**Причина:** {fact_result['reason']}")
        st.caption(
            SOURCE_MESSAGES.get(fact_result.get("decision_source"), "")
        )

        # ===== NLI =====
        if nli_result:
            with st.expander("🔎 Результат NLI + numeric checker"):
                st.write(
                    f"**Вердикт NLI:** {nli_result['verdict']}"
                )
                st.write(
                    f"**Уверенность NLI:** "
                    f"{round(nli_result['confidence'] * 100, 1)}%"
                )
                st.write(
                    f"**Причина NLI:** {nli_result['reason']}"
                )
                st.write(
                    f"**decision_source NLI:** "
                    f"`{nli_result.get('decision_source')}`"
                )

                if nli_result.get("raw_response"):
                    st.code(nli_result["raw_response"])

        # ===== LUM =====
        if lum_result:
            with st.expander("🧠 Результат LUM v2"):
                st.write(
                    f"**Вердикт LUM:** {lum_result['verdict']}"
                )
                st.write(
                    f"**Уверенность LUM:** "
                    f"{round(lum_result['confidence'] * 100, 1)}%"
                )
                st.write(
                    f"**Причина LUM:** {lum_result['reason']}"
                )

        if lum_checker.load_error:
            with st.expander("🔧 Диагностика LUM v2"):
                st.code(lum_checker.load_error)

        if nli_checker.load_error:
            with st.expander("🔧 Диагностика NLI"):
                st.code(nli_checker.load_error)

        # ===== ИСТОЧНИКИ =====
        if original_fact_result.get("details"):
            st.markdown("---")
            st.subheader("🔗 Источники информации")

            unique_details = []
            seen_links = set()

            for detail in original_fact_result["details"]:
                links = detail.get("links", [])

                for link in links:
                    if link and link not in seen_links:
                        seen_links.add(link)

                        unique_details.append({
                            "link": link,
                            "reason": detail.get("reason", "")
                        })

            for i, item in enumerate(unique_details[:10], 1):
                st.markdown(
                    f"{i}. [{item['link']}]({item['link']}) "
                    f"*( {item['reason']} )*"
                )

            st.download_button(
                label="📥 Скачать список ссылок",
                data="\n".join(seen_links),
                file_name="sources.txt",
                mime="text/plain"
            )

        if original_fact_result.get("details"):
            with st.expander("🔍 Детали проверки фактов"):
                for detail in original_fact_result["details"]:
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
                "example", "fact_result", "original_fact_result",
                "nli_result", "lum_result", "lemmatized"
            ]:
                if key in st.session_state:
                    del st.session_state[key]

            st.rerun()

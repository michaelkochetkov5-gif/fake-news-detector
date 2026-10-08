import streamlit as st
from main_code import FactChecker

st.set_page_config(
    page_title="Детектор фейковых новостей",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Детектор фейковых новостей + фактчекинг")

st.write(
    "Введите утверждение или новость — сервис найдёт источники, "
    "проверит их релевантность и покажет ключевые цитаты."
)

checker = FactChecker()

user_text = st.text_area(
    "Введите текст для проверки:",
    height=120,
    placeholder="Например: Земля имеет форму шара"
)

check_button = st.button("Проверить факт")

if check_button:
    if not user_text.strip():
        st.warning("Введите текст для проверки.")
        st.stop()

    with st.spinner("Ищу источники и проверяю утверждение..."):
        result = checker.check_fact(user_text.strip())

    label = result["label"]
    score = result["score"]
    details = result["details"]
    explanation = result["explanation"]

    confidence_percent = int(round(score * 100))

    st.subheader("📊 Результат анализа")

    if label == "ПРАВДА":
        st.success(f"✅ {label} ({confidence_percent}%)")
    elif label == "ФЕЙК":
        st.error(f"❌ {label} ({confidence_percent}%)")
    else:
        st.warning(f"⚠️ {label} ({confidence_percent}%)")

    st.info(explanation)

    st.subheader("🔗 Источники информации")

    if not details:
        st.warning(
            "Не найдено релевантных источников для проверки утверждения."
        )

    else:
        for i, item in enumerate(details, start=1):
            fact = item.get("fact", "")
            source = item.get("source", "")
            reason = item.get("reason", "")
            links = item.get("links", [])
            item_result = item.get("result")

            if item_result is True:
                status_icon = "✅"
            elif item_result is False:
                status_icon = "❌"
            else:
                status_icon = "➖"

            st.markdown(
                f"""
                **{i}. {status_icon} {fact}**  
                *Источник: {source}*

                {reason}
                """
            )

            if links:
                for link in links:
                    if link:
                        st.markdown(
                            f"[Открыть источник]({link})"
                        )

            st.divider()

else:
    st.info(
        "Введите утверждение и нажмите «Проверить факт», "
        "чтобы начать проверку."
    )

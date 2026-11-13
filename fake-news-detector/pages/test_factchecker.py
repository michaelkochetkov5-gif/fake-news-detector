import streamlit as st
import pandas as pd

from main_code import FactChecker
from nli_checker import (
    NLIChecker,
    combine,
    is_confident,
    SOURCE_MESSAGES,
)

@st.cache_resource
def load_lum_checker():
    return NLIChecker()

lum_checker = load_lum_checker()

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

st.title("🧪 Тест фактчекера и LUM")
st.write(
    "Тест подаёт проверенные и ложные утверждения по очереди. "
    "Сначала работает FactChecker; если он не дал уверенного "
    "вердикта, подключается LUM на основе NLI и numeric checker."
)

if st.button("Запустить тест"):
    results = []

    final_correct = 0
    pure_fact_correct = 0
    lum_correct = 0
    lum_checked = 0

    progress = st.progress(0)
    status = st.empty()

    for i, case in enumerate(TEST_CASES):
        text = case["text"]
        expected = case["expected"]

        status.write(
            f"Проверяю {i + 1} из {len(TEST_CASES)}: "
            f"{text[:80]}..."
        )

        fact_result = fact_checker.verify(text)

        pure_fact_verdict = fact_result[
            "verdict"
        ]
        pure_fact_confidence = fact_result[
            "confidence"
        ]

        sources = []

        for detail in fact_result.get(
            "details",
            [],
        ):
            for link in detail.get(
                "links",
                [],
            ):
                if link:
                    sources.append(link)

        sources_count = len(sources)

        lum_result = None

        if not is_confident(fact_result):
            lum_result = lum_checker.verify(
                text,
                fact_result,
            )

        final = combine(
            fact_result,
            lum_result,
        )

        final_verdict = final["verdict"]
        final_confidence = final[
            "confidence"
        ]
        final_reason = final["reason"]
        decision_source = final[
            "decision_source"
        ]

        lum_verdict = "Не вызывалась"
        lum_confidence = None
        lum_raw = ""
        lum_reason = ""

        if lum_result is not None:
            lum_verdict = lum_result[
                "verdict"
            ]
            lum_confidence = round(
                lum_result["confidence"] * 100,
                1,
            )
            lum_raw = lum_result.get(
                "raw_response",
                "",
            )
            lum_reason = lum_result.get(
                "reason",
                "",
            )

        final_is_correct = (
            final_verdict == expected
        )

        if final_is_correct:
            final_correct += 1

        pure_fact_is_correct = (
            pure_fact_verdict == expected
        )

        if pure_fact_is_correct:
            pure_fact_correct += 1

        if lum_verdict in (
            "ПРАВДА",
            "ФЕЙК",
        ):
            lum_checked += 1

            if lum_verdict == expected:
                lum_correct += 1

        results.append({
            "Текст": text,
            "Ожидалось": expected,
            "Чистый фактчекер": (
                pure_fact_verdict
            ),
            "Уверенность фактчекера": round(
                pure_fact_confidence * 100,
                1,
            ),
            "LUM": lum_verdict,
            "Уверенность LUM": (
                lum_confidence
                if lum_confidence is not None
                else "—"
            ),
            "LUM причина": lum_reason,
            "decision_source": decision_source,
            "raw_response": lum_raw,
            "Общий вердикт": final_verdict,
            "Уверенность общего": round(
                final_confidence * 100,
                1,
            ),
            "Причина": final_reason,
            "Общий вердикт верен": (
                "Да"
                if final_is_correct
                else "Нет"
            ),
            "Количество источников": (
                sources_count
            ),
            "Первая ссылка": (
                sources[0]
                if sources
                else ""
            ),
        })

        progress.progress(
            (i + 1) / len(TEST_CASES)
        )

    status.success("Тест завершён")

    total = len(TEST_CASES)

    final_accuracy = (
        final_correct / total * 100
    )
    pure_fact_accuracy = (
        pure_fact_correct / total * 100
    )

    if lum_checked > 0:
        lum_accuracy = (
            lum_correct / lum_checked * 100
        )
    else:
        lum_accuracy = 0

    st.subheader("📊 Итоги теста")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Общий вердикт",
            f"{final_accuracy:.1f}%",
        )

    with col2:
        st.metric(
            "Чистый фактчекер",
            f"{pure_fact_accuracy:.1f}%",
        )

    with col3:
        st.metric(
            "LUM",
            f"{lum_accuracy:.1f}% "
            f"({lum_checked})",
        )

    with col4:
        st.metric(
            "Проверено примеров",
            total,
        )

    st.subheader(
        "📋 Подробные результаты"
    )

    df = pd.DataFrame(results)

    for _, row in df.iterrows():
        expected = row["Ожидалось"]
        final_verdict = row[
            "Общий вердикт"
        ]
        pure = row[
            "Чистый фактчекер"
        ]
        lum = row["LUM"]

        if final_verdict == expected:
            icon = "✅"
        elif final_verdict == (
            "БОЛЬШИЕ СОМНЕНИЯ"
        ):
            icon = "⚠️"
        else:
            icon = "❌"

        st.markdown(
            f"### {icon} "
            f"{row['Текст'][:120]}..."
        )

        st.write(
            f"**Ожидалось:** {expected}"
        )

        st.write(
            f"**Чистый фактчекер:** {pure} "
            f"({row['Уверенность фактчекера']}%)"
        )

        st.write(
            f"**LUM:** {lum} "
            f"({row['Уверенность LUM']}%)"
        )

        st.write(
            f"**Общий вердикт:** "
            f"{final_verdict} "
            f"({row['Уверенность общего']}%)"
        )

        st.write(
            f"**decision_source:** "
            f"`{row['decision_source']}`"
        )

        st.write(
            f"**Причина:** {row['Причина']}"
        )

        if row.get("LUM причина"):
            st.write(
                f"**Причина LUM:** "
                f"{row['LUM причина']}"
            )

        st.write(
            f"**Источников найдено:** "
            f"{row['Количество источников']}"
        )

        if row["Первая ссылка"]:
            st.markdown(
                f"[Открыть первый источник]"
                f"({row['Первая ссылка']})"
            )

        if row.get("raw_response"):
            with st.expander(
                "🔧 raw_response LUM"
            ):
                st.code(
                    row["raw_response"]
                )

        st.divider()

else:
    st.info(
        "Нажми «Запустить тест», "
        "чтобы начать проверку."
    )

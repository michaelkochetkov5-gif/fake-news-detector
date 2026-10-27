import streamlit as st
import requests
import feedparser
import re
import json
import os
import time
import copy
import hashlib
from functools import lru_cache
from bs4 import BeautifulSoup

try:
    from ddgs import DDGS
except ImportError:
    from duckduckgo_search import DDGS

try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    import pymorphy3
    _MORPH = pymorphy3.MorphAnalyzer()
    MORPH_AVAILABLE = True
except Exception:
    _MORPH = None
    MORPH_AVAILABLE = False

# Правка №8: отдельный файл. Раньше site.py (cache.json, md5-ключи) и
# main_code (search_cache/news_cache) писали ОДИН файл, и save_cache() из
# main_code затирал записи site.py целиком.
CACHE_FILE = os.path.join(
    os.path.dirname(__file__),
    "factchecker_cache.json"
)

# Сколько живёт запись поиска. Тела статей по 12 000 символов иначе
# копятся в файле бесконечно и маскируют изменения выдачи.
CACHE_TTL = 24 * 60 * 60

# Версия включается в ключи кэша, чтобы старые записи не маскировали
# изменения в логике проверки.
CACHE_VERSION = "v8-fix"
FACTCHECKER_VERSION = "v8-fix"

if os.path.exists(CACHE_FILE):
    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as file:
            cache = json.load(file)
    except Exception:
        cache = {}
else:
    cache = {}

search_cache = cache.get("search_cache", {})
news_cache = cache.get("news_cache", {})


def save_cache():
    try:
        with open(CACHE_FILE, "w", encoding="utf-8") as file:
            json.dump(
                {
                    "search_cache": search_cache,
                    "news_cache": news_cache
                },
                file,
                ensure_ascii=False,
                indent=2
            )
    except Exception:
        pass


def get_cache_key(text):
    """Полный SHA-1 нормализованного текста.

    Правка №8: раньше ключ обрезался до 180 символов, поэтому два разных
    утверждения с общим длинным префиксом давали один ключ (коллизия).
    """
    normalized = re.sub(r"[^a-zа-яё0-9]+", " ", text.lower()).strip()

    return CACHE_VERSION + "_" + hashlib.sha1(
        normalized.encode("utf-8")
    ).hexdigest()


def _cache_get(store, key):
    """Возвращает копию записи кэша или None, если её нет / истёк TTL."""
    if key not in store:
        return None

    entry = store[key]

    # Старый формат: просто список. TTL к нему не применяем.
    if not isinstance(entry, dict) or "results" not in entry:
        return copy.deepcopy(entry)

    if time.time() - entry.get("time", 0) > CACHE_TTL:
        store.pop(key, None)
        return None

    return copy.deepcopy(entry["results"])


def _cache_set(store, key, results):
    store[key] = {"time": time.time(), "results": copy.deepcopy(results)}


STOP_WORDS = {
    "это", "этот", "эта", "эти", "того", "такой", "такая", "такие",
    "есть", "был", "была", "были", "будет", "будут", "очень",
    "более", "менее", "самый", "самая", "самые", "который",
    "которая", "которые", "чтобы", "потому", "поэтому", "также",
    "тоже", "вот", "там", "тут", "или", "и", "а", "но", "не", "ни",
    "да", "нет", "как", "что", "чем", "при", "для", "от", "до",
    "из", "по", "за", "на", "в", "во", "со", "с", "у", "о", "об",
    "обо", "над", "под", "про", "через", "между", "имеет", "является",
    "года", "году", "прошёл", "прошел", "прошла", "прошли",
    "состоялся", "состоялась", "состоялось",
    "свой", "своя", "своё", "свое", "свои", "своего", "своей",
    "своих", "своим", "своими", "которого", "которой", "которых",
    "которому", "которыми", "весь", "вся", "всё", "все", "всего",
    "мочь", "может", "можно", "нужно", "надо", "если", "когда",
    "где", "кто", "почему", "зачем", "лишь", "только", "уже",
    "ещё", "еще", "теперь", "сейчас", "себя", "себе", "собой",
    "один", "одна", "одно", "два", "две", "три", "ряд", "часть",
    "стать", "становиться", "являться", "представлять", "считать",
    "говорить", "сказать", "отмечать", "отметить", "сообщать",
    "сообщить", "заявлять", "заявить", "указывать", "указать",
    "данный", "данные", "данных", "данным", "некоторый", "любой",
    "каждый", "другой", "иной", "сам", "сама", "само", "сами",
    "кроме", "кроме того", "при этом", "однако", "хотя", "однако"
}

SYNONYMS = {
    "круглый": [
        "круглый", "шарообразный", "сферический",
        "геоид", "эллипсоид"
    ],
    "шар": [
        "шарообразный", "круглый", "сферический",
        "геоид", "эллипсоид"
    ],
    "вредный": [
        "вредный", "опасный"
    ],
    "опасный": [
        "вредный", "опасный"
    ],
    "лечить": [
        "помогать", "эффективный"
    ],
    "запрещать": [
        "запрет", "запрещённый", "запретить"
    ],
    "отменять": [
        "отмена", "отменённый", "отменить"
    ],
    "вводить": [
        "введение", "ввести", "вводиться"
    ]
}

TRUSTED_DOMAINS = {
    "wikipedia.org": 1.0,
    "nasa.gov": 1.0,
    "science.nasa.gov": 1.0,
    "who.int": 1.0,
    "un.org": 0.9,
    "gov.ru": 0.9,
    "edu.ru": 0.9,
    "edu": 0.8,
    "mit.edu": 0.95,
    "khanacademy.org": 0.9,
    "stanford.edu": 0.95,
    "ourworldindata.org": 0.95,
    "medlineplus.gov": 1.0,
    "cdc.gov": 1.0,
    "nih.gov": 1.0,
    "imf.org": 0.95,
    "worldbank.org": 0.95,
    "nature.com": 0.95,
    "reuters.com": 0.9,
    "apnews.com": 0.9,
    "bbc.com": 0.8,
    "interfax.ru": 0.7,
    "tass.ru": 0.7,
    "ria.ru": 0.7,
    "lenta.ru": 0.7
}

CONTRADICTION_WORDS = [
    "не является", "не имеет", "не соответствует",
    "опровергает", "опровергнуто", "ложно", "ложная",
    "фейк", "миф", "неверно", "неправда", "дезинформация",
    "не пройдёт", "не пройдет",
    "не состоится", "не состоятся"
]

GROWTH_WORDS = [
    "вырастет", "вырастут", "вырасти", "рост", "роста",
    "увеличится", "увеличатся", "увеличение", "увеличения",
    "повысится", "повысятся", "повышение", "повышения",
    "поднимется", "поднимутся", "прирост", "прироста"
]

DECLINE_WORDS = [
    "сократится", "сократятся", "сокращение", "сокращения",
    "снизится", "снизятся", "снижение", "снижения",
    "уменьшится", "уменьшатся", "уменьшение", "уменьшения",
    "упадёт", "упадут", "падение", "падения", "спад", "спада"
]

# Сенсационные маркеры. Наличие любого из них ЗАПРЕЩАЕТ выдачу ПРАВДА.
SENSATIONAL_MARKERS = [
    "запрет", "запретят", "запретили", "запрещён", "запрещено",
    "запрещать", "обязан", "обязаны", "обяжут", "заблокир",
    "скрывают", "скрывать", "скрыва", "подделан", "подделка",
    "подделк", "мгновенн", "навсегда", "полностью заменяет",
    "все люди", "все граждане", "все жители", "никто не",
    "смертельно", "мгновенную смерть", "уголовную ответственность",
    "комендантский час", "смерть у всех", "убивает всех",
    "в тысячи раз", "занижена"
]

# Страны и регионы в леммах pymorphy3 — для проверки географических противоречий.
COUNTRY_LEMMAS = {
    "россия", "москва", "петербург", "ссср", "сша", "америка",
    "канада", "мексика", "аргентина", "уругвай", "парагвай",
    "бразилия", "европа", "азия", "африка", "украина", "китай",
    "германия", "франция", "британия", "англия", "япония",
    "италия", "испания", "португалия", "нидерланды", "польша",
    "турция", "египет", "катар", "саудовская", "индия",
    "австралия", "новая", "корея", "казахстан", "беларусь"
}

# Опорные сущности: если лемма из этого набора совпала — совпадение весомее.
ANCHOR_HINTS = {
    "python", "javascript", "java", "c++", "metlife",
    "вакцина", "вакцинация", "прививка", "метрополитен", "метро",
    "чемпионат", "футбол", "олимпиада", "стадион", "сборная",
    "правительство", "министерство", "президент", "госдума",
    "центробанк", "банк", "экономика", "интернет", "здравоохранение"
}

@lru_cache(maxsize=50000)
def _lemma(word):
    """Лемма слова через pymorphy3. Кэшируется — вызовов очень много."""
    if not word:
        return ""

    if not MORPH_AVAILABLE:
        return word.lower()

    try:
        return _MORPH.parse(word)[0].normal_form
    except Exception:
        return word.lower()


TOKEN_RE = re.compile(r"[а-яёa-z]+|\d+(?:[.,]\d+)?", re.IGNORECASE)
NUMBER_RE = re.compile(r"^\d+(?:[.,]\d+)?$")


def lemma_tokens(text):
    """Список лемм (числа остаются числами) в порядке появления."""
    if not text:
        return []

    result = []

    for raw in TOKEN_RE.findall(text.lower()):
        raw = raw.strip()

        if not raw:
            continue

        if NUMBER_RE.match(raw):
            result.append(raw.replace(",", "."))
        else:
            result.append(_lemma(raw))

    return result


STOP_LEMMAS = {_lemma(w) for w in STOP_WORDS}
STOP_LEMMAS.update({
    "свой", "весь", "который", "этот", "мочь", "быть", "это",
    "при", "для", "от", "до", "из", "по", "за", "на", "в",
    "о", "об", "над", "под", "про", "через", "между", "у",
    "и", "а", "но", "не", "ни", "да", "нет", "как", "что",
    "чем", "или", "либо", "же", "ли", "бы", "ведь", "вот"
})


def clean_text(text):
    if not text:
        return ""

    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def normalize_word(word):
    """
    Возвращает лемму слова (а не первые 5 символов, как раньше).
    Сигнатура сохранена для обратной совместимости.
    """
    if not word:
        return ""

    word = word.lower().strip()

    if NUMBER_RE.match(word):
        return word.replace(",", ".")

    return _lemma(word)


def get_base_words(text):
    """Значимые леммы текста без стоп-слов, в порядке появления."""
    lemmas = lemma_tokens(text)

    result = []

    for lem in lemmas:
        if NUMBER_RE.match(lem):
            result.append(lem)
            continue

        if len(lem) < 3:
            continue

        if lem in STOP_LEMMAS:
            continue

        result.append(lem)

    return list(dict.fromkeys(result))


def _term_weight(lemma):
    """Вес термина: числа и латиница — сильнее всего, затем длина."""
    if NUMBER_RE.match(lemma):
        return 3.0

    if lemma in COUNTRY_LEMMAS:
        return 3.0

    if re.fullmatch(r"[a-z]+", lemma):
        return 3.0

    if lemma in ANCHOR_HINTS:
        return 3.0

    if len(lemma) >= 8:
        return 2.0

    if len(lemma) >= 6:
        return 1.6

    return 1.0


def claim_key_terms(text):
    """
    Ключевые термины утверждения с весами.
    Возвращает список (lemma, weight), отсортированный по весу убыв.
    """
    lemmas = get_base_words(text)

    weights = {}

    for lem in lemmas:
        w = _term_weight(lem)
        if lem in weights:
            weights[lem] = max(weights[lem], min(w + 0.2, 3.0))
        else:
            weights[lem] = w

    return sorted(weights.items(), key=lambda kv: (-kv[1], -len(kv[0])))


def get_claim_word_groups(text):
    """
    Группы лемм с синонимами. Сигнатура сохранена для обратной совместимости.
    """
    lemmas = get_base_words(text)

    groups = []

    for lem in lemmas:
        group = [lem]

        if lem in SYNONYMS:
            for synonym in SYNONYMS[lem]:
                group.append(_lemma(synonym))

        groups.append(list(dict.fromkeys(group)))

    return groups


def has_entity_match(claim, matched_words=None):
    """
    Есть ли в утверждении опорная сущность: число, страна, латиница
    или термин из ANCHOR_HINTS. Сигнатура сохранена.
    """
    lemmas = get_base_words(claim)

    for lem in lemmas:
        if NUMBER_RE.match(lem):
            print(f"      [entity] число '{lem}' → True")
            return True

        if lem in COUNTRY_LEMMAS:
            print(f"      [entity] страна '{lem}' → True")
            return True

        if re.fullmatch(r"[a-z]+", lem):
            print(f"      [entity] латиница '{lem}' → True")
            return True

        if lem in ANCHOR_HINTS:
            print(f"      [entity] опорный термин '{lem}' → True")
            return True

    print("      [entity] опорных сущностей нет → False")
    return False


def has_sensational_markers(text):
    text_lower = text.lower()
    return any(marker in text_lower for marker in SENSATIONAL_MARKERS)


def get_domain_trust(url):
    try:
        domain = url.lower()
        domain = domain.replace("https://", "").replace("http://", "")
        domain = domain.split("/")[0]

        for trusted_domain, trust in TRUSTED_DOMAINS.items():
            if trusted_domain in domain:
                return trust

    except Exception:
        pass

    return 0.5


def split_into_claims(text):
    """
    Разбивает текст на отдельные проверяемые утверждения.
    Сохраняет исходную формулировку (нужна для поиска), но режет
    по границам предложений и по связкам ("однако", "но", "также").
    """
    text = clean_text(text)

    sentences = re.split(r"(?<=[.!?;])\s+", text)

    split_words = [
        "а", "но", "однако", "хотя", "при этом",
        "кроме того", "также", "поэтому",
        "потому что", "так как",
        "утверждается, что", "сообщается, что"
    ]

    claims = []

    for sentence in sentences:
        sentence = sentence.strip()

        if not sentence:
            continue

        parts = [sentence]

        for word in split_words:
            new_parts = []

            for part in parts:
                pattern = rf",?\s+{re.escape(word)}\s+"
                pieces = re.split(pattern, part, flags=re.IGNORECASE)

                for piece in pieces:
                    piece = piece.strip()

                    if piece:
                        new_parts.append(piece)

            parts = new_parts

        for part in parts:
            part = re.sub(
                r"^(сообщается|заявляется|утверждается|по данным|по словам)"
                r"\s+(что|о том, что)?\s*,?\s*",
                "",
                part,
                flags=re.IGNORECASE
            )

            part = part.strip(" .,;:!?")

            if len(part.split()) >= 3:
                claims.append(part)

    unique_claims = []

    for claim in claims:
        if claim.lower() not in [c.lower() for c in unique_claims]:
            unique_claims.append(claim)

    return unique_claims


def classify_claim_importance(claim):
    """
    Важное утверждение — то, что несёт проверяемый факт:
    числа, запреты, обязательства, причинно-следственные связи.
    Контекстное — фон, который не должен определять вердикт.
    """
    text = claim.lower()

    important_markers = [
        "запрет", "запретят", "запретили", "введут", "ввели",
        "отменят", "отменили", "разрешат", "разрешили",
        "убийств", "убивает", "смерт", "вызывает", "вызовут",
        "докажут", "доказали", "подтвердили", "опроверг",
        "заблокир", "обязан", "обязаны", "скрывают", "подделан",
        "полностью", "все люди", "все жители", "никто", "никогда",
        "заменяет", "заменит", "произойдёт", "произойдет",
        "пройдёт", "пройдет", "сократится", "вырастет", "составит"
    ]

    has_numbers = bool(re.search(r"\d", text))
    has_marker = any(marker in text for marker in important_markers)

    if has_numbers or has_marker:
        return "important"

    return "context"




# Канонизация стран: прилагательные сводим к стране, чтобы
# "московский" и "москва" не считались разными сущностями.
COUNTRY_CANON = {
    "россия": "россия", "российский": "россия", "рф": "россия",
    "москва": "москва", "московский": "москва",
    "петербург": "петербург", "петербургский": "петербург",
    "ссср": "ссср", "советский": "ссср",
    "сша": "сша", "американский": "сша", "америка": "сша",
    "канада": "канада", "канадский": "канада",
    "мексика": "мексика", "мексиканский": "мексика",
    "аргентина": "аргентина", "аргентинский": "аргентина",
    "уругвай": "уругвай", "уругвайский": "уругвай",
    "парагвай": "парагвай", "парагвайский": "парагвай",
    "бразилия": "бразилия", "бразильский": "бразилия",
    "европа": "европа", "европейский": "европа",
    "азия": "азия", "азиатский": "азия",
    "африка": "африка", "африканский": "африка",
    "украина": "украина", "украинский": "украина",
    "китай": "китай", "китайский": "китай",
    "германия": "германия", "германский": "германия",
    "франция": "франция", "французский": "франция",
    "британия": "британия", "британский": "британия",
    "англия": "англия", "английский": "англия",
    "япония": "япония", "японский": "япония",
    "италия": "италия", "итальянский": "италия",
    "испания": "испания", "испанский": "испания",
    "португалия": "португалия", "португальский": "португалия",
    "польша": "польша", "польский": "польша",
    "турция": "турция", "турецкий": "турция",
    "египет": "египет", "египетский": "египет",
    "катар": "катар", "индия": "индия", "индийский": "индия",
    "австралия": "австралия", "австралийский": "австралия",
    "корея": "корея", "корейский": "корея",
    "казахстан": "казахстан", "беларусь": "беларусь",
    "белоруссия": "беларусь", "белорусский": "беларусь"
}


def extract_countries(text):
    """Множество канонических стран, упомянутых в тексте."""
    found = set()

    for lem in lemma_tokens(text):
        canon = COUNTRY_CANON.get(lem)
        if canon:
            found.add(canon)

    return found


def extract_years(text):
    """Годы (4 цифры) из текста — отдельно от прочих чисел."""
    return {
        t for t in lemma_tokens(text)
        if NUMBER_RE.match(t) and len(t.split(".")[0]) == 4
    }


def extract_numbers(text):
    return {t for t in lemma_tokens(text) if NUMBER_RE.match(t)}


# ─────────────────────────────────────────────────────────────────────────────
# ПОИСК
# ─────────────────────────────────────────────────────────────────────────────

def extract_key_terms(claim, max_terms=6):
    """
    Запрос из 3-6 ключевых слов в исходной форме (нижний регистр).
    Порядок: годы -> латиница/страны/опорные термины -> длинные слова.

    ИСПРАВЛЕНИЕ v8: раньше regex был [а-яёa-zA-Z]+ — в нём нет заглавной
    кириллицы А-ЯЁ. Поэтому "Чемпионат" -> "емпионат", "Канаде" -> "анаде",
    а "США" выпадало целиком. Теперь класс включает А-ЯЁ, а страны и
    аббревиатуры короче 4 букв (США, ВОЗ) не отбрасываются.
    """
    raw_words = re.findall(r"[а-яёА-ЯЁa-zA-Z]+|\d{4}", claim)

    numbers = []
    priority = []
    long_words = []
    other_words = []

    seen = set()

    for raw in raw_words:
        low = raw.lower()

        if low in seen:
            continue

        if low.isdigit():
            if len(low) == 4:
                numbers.append(low)
                seen.add(low)
            continue

        lem = _lemma(low)

        is_country = lem in COUNTRY_CANON or lem in COUNTRY_LEMMAS
        is_latin = bool(re.fullmatch(r"[a-z]+", lem))
        is_anchor = lem in ANCHOR_HINTS

        if not (is_country or is_latin or is_anchor):
            if len(low) < 4 or lem in STOP_LEMMAS:
                continue

        seen.add(low)

        if is_country or is_latin or is_anchor:
            priority.append(low)
        elif len(lem) >= 8:
            long_words.append(low)
        else:
            other_words.append(low)

    ordered = numbers + priority + long_words + other_words

    if not ordered:
        return claim[:60].lower()

    query = " ".join(ordered[:max_terms])
    print(f"      [key_terms] {query}")

    return query


def make_search_queries(claim):
    """
    Два запроса:
      1) ключевые термины (числа + сущности + значимые слова);
      2) значимые леммы в исходном порядке, 5 штук.
    Полное предложение больше не отправляется.
    """
    query1 = extract_key_terms(claim, max_terms=6)

    lemmas = get_base_words(claim)
    query2 = " ".join(lemmas[:5])

    queries = []

    if query1 and len(query1.split()) >= 2:
        queries.append(query1)

    if query2 and query2 != query1:
        queries.append(query2)

    if not queries:
        queries = [claim[:60]]

    print(f"   [queries] {queries}")

    return list(dict.fromkeys(queries))


def get_page_text(url, max_length=12000):
    if not url:
        return ""

    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0 Safari/537.36"
            )
        }

        response = requests.get(url, headers=headers, timeout=10)

        if response.status_code != 200:
            return ""

        soup = BeautifulSoup(response.text, "html.parser")

        for tag in soup(["script", "style", "nav", "header",
                         "footer", "aside", "form", "button"]):
            tag.decompose()

        text = soup.get_text(separator=" ")
        text = clean_text(text)

        return text[:max_length]

    except Exception:
        return ""


def search_duckduckgo(claim, max_results=6):
    cache_key = "ddg_" + get_cache_key(claim)

    cached = _cache_get(search_cache, cache_key)

    if cached is not None:
        print(f"   [DDG] кэш: {len(cached)}")
        return cached

    results = []
    seen_urls = set()
    queries = make_search_queries(claim)

    try:
        with DDGS() as ddgs:
            for query in queries:
                try:
                    web_results = ddgs.text(
                        query,
                        region="ru-ru",
                        safesearch="moderate",
                        max_results=4
                    )

                    before = len(results)

                    for result in web_results:
                        url = result.get("href", "")

                        if url and url not in seen_urls:
                            seen_urls.add(url)
                            results.append({
                                "title": result.get("title", ""),
                                "summary": clean_text(result.get("body", "")),
                                "body": clean_text(result.get("body", "")),
                                "url": url,
                                "source": "DuckDuckGo"
                            })

                    print(f"   [DDG] '{query[:50]}': +{len(results) - before}")
                    time.sleep(0.4)

                except Exception as e:
                    print(f"   [DDG] ошибка '{query[:40]}': {e}")
                    continue

    except Exception as e:
        print(f"   [DDG] недоступен: {e}")

    print(f"   [DDG] итого {len(results)}")

    if results:
        _cache_set(search_cache, cache_key, results[:max_results])
        save_cache()

    return results[:max_results]


# Википедия требует User-Agent с контактом (политика Wikimedia API).
# Без него сервер отвечает 403/429, а код раньше молча считал это
# пустым поиском. Замените адрес на свой.
WIKI_CONTACT = "your-email@example.com"
WIKI_USER_AGENT = (
    "FakeNewsDetector/1.0 (educational project; contact: "
    + WIKI_CONTACT + ")"
)


def _wiki_get(params, timeout=15, max_retries=1):
    """
    GET к ru.wikipedia API с логированием статуса.
    При 429/503 повторяет запрос, уважая Retry-After (не дольше 5 с).
    Возвращает (response, None) при успехе или (None, причина).
    """
    last_status = None

    for attempt in range(max_retries + 1):
        try:
            response = requests.get(
                "https://ru.wikipedia.org/w/api.php",
                params=params,
                timeout=timeout,
                headers={"User-Agent": WIKI_USER_AGENT}
            )
        except Exception as e:
            print(f"   [Wiki] сетевая ошибка: {type(e).__name__}: {e}")
            return None, f"exception {type(e).__name__}"

        last_status = response.status_code

        if response.status_code == 200:
            return response, None

        retry_after = response.headers.get("retry-after", "")
        print(
            f"   [Wiki] HTTP {response.status_code} "
            f"(попытка {attempt + 1}/{max_retries + 1}), "
            f"Retry-After={retry_after or '-'}"
        )

        if response.status_code not in (429, 503) or attempt == max_retries:
            break

        try:
            # Retry-After от Википедии часто слишком мал (0.1 с) — повтор
            # сразу снова падает в 429. Ждём не меньше 2 с, не больше 5 с.
            wait = min(max(float(retry_after), 2.0), 5.0)
        except ValueError:
            wait = 2.0

        time.sleep(wait)

    return None, f"HTTP {last_status}"


def get_wikipedia_full_text(title):
    try:
        api_url = "https://ru.wikipedia.org/w/api.php"

        params = {
            "action": "query",
            "format": "json",
            "formatversion": "2",
            "prop": "extracts",
            "explaintext": "1",
            "redirects": "1",
            "titles": title
        }

        response, failure = _wiki_get(params)

        if response is None:
            print(f"   [Wiki] полный текст '{title[:50]}' не получен: {failure}")
            return ""

        data = response.json()
        pages = data.get("query", {}).get("pages", [])

        if not pages:
            return ""

        extract = pages[0].get("extract", "")

        return clean_text(extract[:12000])

    except Exception:
        return ""


def search_wikipedia_api(claim, max_results=6):
    cache_key = "wiki_" + get_cache_key(claim)

    cached = _cache_get(search_cache, cache_key)

    if cached is not None:
        print(f"   [Wiki] кэш: {len(cached)}")
        return cached

    results = []
    seen_titles = set()

    query1 = extract_key_terms(claim, max_terms=5)
    lemmas = get_base_words(claim)
    query2 = " ".join(lemmas[:4])

    queries = list(dict.fromkeys([q for q in (query1, query2) if q]))

    if not queries:
        queries = [claim[:60]]

    print(f"   [Wiki] {queries}")

    for query in queries:
        try:
            api_url = "https://ru.wikipedia.org/w/api.php"

            params = {
                "action": "query",
                "format": "json",
                "list": "search",
                "srsearch": query,
                "srlimit": 4,
                "srprop": "snippet",
                "utf8": 1
            }

            print(f"   [Wiki query] '{query}'")

            response, failure = _wiki_get(params)

            if response is None:
                print(f"   [Wiki] запрос '{query[:50]}' не выполнен: {failure}")
                continue

            data = response.json()
            pages = data.get("query", {}).get("search", [])
            before = len(results)

            for page in pages:
                title = page.get("title", "")
                snippet = page.get("snippet", "")

                if not title or title in seen_titles:
                    continue

                seen_titles.add(title)
                snippet = re.sub(r"<[^>]+>", "", snippet)

                page_url = (
                    "https://ru.wikipedia.org/wiki/"
                    + requests.utils.quote(title)
                )

                full_text = get_wikipedia_full_text(title)
                body = full_text if full_text else snippet

                results.append({
                    "title": title,
                    "summary": snippet,
                    "body": body,
                    "url": page_url,
                    "source": "Wikipedia"
                })

            print(f"   [Wiki] '{query[:50]}': +{len(results) - before}")
            time.sleep(0.3)

        except Exception as e:
            print(f"   [Wiki] ошибка: {e}")
            continue

    unique = []
    seen_urls = set()

    for result in results:
        if result["url"] not in seen_urls:
            seen_urls.add(result["url"])
            unique.append(result)

    print(f"   [Wiki] итого {len(unique)}")

    if unique:
        _cache_set(search_cache, cache_key, unique[:max_results])
        save_cache()

    return unique[:max_results]


RSS_FEEDS = [
    {"name": "РИА Новости", "url": "https://ria.ru/export/rss2/archive/index.xml"},
    {"name": "ТАСС",        "url": "https://tass.ru/rss/v2.xml"},
    {"name": "Интерфакс",   "url": "https://www.interfax.ru/rss.asp"},
    {"name": "Lenta.ru",    "url": "https://lenta.ru/rss/news"},
    {"name": "BBC Russian", "url": "https://feeds.bbci.co.uk/russian/rss.xml"}
]


def get_relevant_articles(claim, max_articles=3):
    cache_key = "news_" + get_cache_key(claim)
    current_time = time.time()

    if cache_key in news_cache:
        cached = news_cache[cache_key]
        if current_time - cached.get("time", 0) < 3600:
            articles = cached.get("articles", [])
            print(f"   [RSS] кэш: {len(articles)}")
            return copy.deepcopy(articles)

    key_terms = dict(claim_key_terms(claim)[:10])
    total_weight = sum(key_terms.values()) or 1.0

    articles = []

    for feed in RSS_FEEDS:
        try:
            parsed_feed = feedparser.parse(feed["url"])

            for entry in parsed_feed.entries[:25]:
                title = clean_text(entry.get("title", ""))
                summary = clean_text(entry.get("summary", ""))

                text = title + " " + summary
                lemmas = set(lemma_tokens(text))

                matched = sum(
                    w for lem, w in key_terms.items() if lem in lemmas
                )
                coverage = matched / total_weight

                if coverage >= 0.30 and sum(
                    1 for lem in key_terms if lem in lemmas
                ) >= 2:
                    articles.append({
                        "title": title,
                        "summary": summary,
                        "link": entry.get("link", ""),
                        "source": feed["name"],
                        "published": entry.get("published", ""),
                        "matches": sum(1 for lem in key_terms if lem in lemmas),
                        "coverage": coverage
                    })

        except Exception:
            continue

    articles.sort(key=lambda x: (x["coverage"], x["matches"]), reverse=True)
    articles = articles[:max_articles]

    print(f"   [RSS] итого {len(articles)}")

    if articles:
        news_cache[cache_key] = {"time": current_time, "articles": articles}
        save_cache()

    return articles




# ─────────────────────────────────────────────────────────────────────────────
# ЧИСЛА, НАПРАВЛЕНИЯ, NLI
# ─────────────────────────────────────────────────────────────────────────────

def normalize_number(text):
    text = text.lower().replace(",", ".")
    return re.findall(r"\d+(?:\.\d+)?", text)


def get_direction(text):
    text_lower = text.lower()

    has_growth = any(w in text_lower for w in GROWTH_WORDS)
    has_decline = any(w in text_lower for w in DECLINE_WORDS)

    if has_growth and not has_decline:
        return "рост"

    if has_decline and not has_growth:
        return "снижение"

    return None


def check_numeric_relation(claim, sentence):
    """
    Противоречие по числам: общее число есть, но направление
    (рост/снижение) противоположное.
    """
    claim_numbers = normalize_number(claim)
    sentence_numbers = normalize_number(sentence)

    if not claim_numbers or not sentence_numbers:
        return None

    if not (set(claim_numbers) & set(sentence_numbers)):
        return None

    claim_direction = get_direction(claim)
    sentence_direction = get_direction(sentence)

    if not claim_direction or not sentence_direction:
        return None

    if claim_direction != sentence_direction:
        return "противоречит"

    return "подтверждает"


def check_year_conflict(claim, sentence):
    """
    Противоречие по годам: в утверждении и предложении есть годы,
    но пересечения нет. Например, claim «ЧМ-2026 в Аргентине»,
    а источник говорит только о 1930 годе.
    """
    claim_years = extract_years(claim)
    sentence_years = extract_years(sentence)

    if not claim_years or not sentence_years:
        return None

    if claim_years & sentence_years:
        return None

    return "противоречит"


# Группы совместимых географий: внутри группы противоречия быть не может.
# "Московский метрополитен" и "метро СССР" — не конфликт.
COUNTRY_GROUP = {
    "россия": "ru", "москва": "ru", "петербург": "ru", "ссср": "ru",
    "сша": "us", "америка": "us",
    "британия": "uk", "англия": "uk"
}

# Континенты не сравниваем со странами: "США" и "Америка/Европа" — не конфликт.
CONTINENTS = {"европа", "азия", "африка"}


def _country_groups(countries):
    return {
        COUNTRY_GROUP.get(c, c)
        for c in countries
        if c not in CONTINENTS
    }


# P3: хозяйский предикат. Предложение говорит о ПРОВЕДЕНИИ события в стране
# ("пройдёт в", "примут", "проводится в", "страны-хозяйки"), а не об участии
# в нём ("сборная X сыграет", "принимает участие").
_HOST_RE = re.compile(
    r"(?<![а-яё])("
    r"пройд[её]т|пройдут|прошл[аи]|прош[её]л|"
    r"проходит|проходят|проходил\w*|"
    r"проводит\w*|проводят\w*|проводил\w*|провед[её]н\w*|"
    r"принима\w*|примут|примет|принял\w*|"
    r"хозя\w+|страны-организатор\w*|места? проведения"
    r")(?![а-яё])"
)

# "принимает участие", "примут участие" — это участие, а не хозяйство.
_PARTICIPATION_RE = re.compile(
    r"(?<![а-яё])(принима\w*|примут|примет|принял\w*|принять)\s+участи\w*"
)


# HOST_BINDING: хозяйский предикат привязывается к стране по синтаксической
# роли, а не по расстоянию в токенах.
#
# Расстояние не работает: в тесте "Финалиссима 2026 — ... сборными Аргентины
# и Испании, который прошёл в Пасадине" между "Аргентины" и "прошёл" всего
# ~5 токенов, и любое окно их "свяжет". Но "прошёл" управляет не "Аргентины",
# а подлежащим "который" (-> "матч").
#
# Синтаксическая привязка = страна стоит объектом предлога "в/на", который
# управляется хозяйским предикатом: "пройдёт В Аргентине", "прошёл В США".
# "между сборными Аргентины" — предлог "между", не "в", привязки нет.
_HOST_VERB_RE = re.compile(
    r"(?<![а-яё])("
    r"пройд[её]т|пройдут|прошл[аи]|прош[её]л|проходит|проходят|проходил\w*|"
    r"проводит\w*|проводят\w*|проводил\w*|провед[её]н\w*|"
    r"принима\w*|примут|примет|принял\w*"
    r")(?![а-яё])"
)

_HOST_NOUN_RE = re.compile(
    r"(?<![а-яё])(хозя\w+|организатор\w*|страна-хозяйка|страны-хозяйки)(?![а-яё])"
)

_HOST_PREP_RE = re.compile(r"(?<![а-яё])(в|во|на)(?![а-яё])")

# Сколько токенов после предиката просматривать в поисках "в <страна>".
HOST_WINDOW = 14

# Насколько далеко может стоять страна от слова "хозяйка/организатор".
HOST_NOUN_WINDOW = 5

# Правка №1: «организатор/хозяйка» привязывает страну к хозяйству только если
# рядом есть турнирное слово. Иначе «организатор товарищеского матча» рядом с
# «Аргентиной» давал привязку и ложный ФЕЙК на правдивом утверждении.
_HOST_TOURNAMENT_RE = re.compile(
    r"(?<![а-яё])("
    r"турнир\w*|чемпионат\w*|мундиаль\w*|первенств\w*|кубок|кубка|кубке"
    r")(?![а-яё])"
)

# Насколько близко к «организатор/хозяйка» должно стоять турнирное слово.
HOST_NOUN_CONTEXT = 2

# Правка №4: эти глаголы управляют прямым объектом («Матчи примут США»),
# поэтому привязка по предлогу «в/на» для них не единственный вариант.
_HOST_DIRECT_VERB_RE = re.compile(
    r"(?<![а-яё])(примут|примет|приму|примешь|"
    r"принимают|принимает|принимал\w*|принял\w*|принять)(?![а-яё])"
)

_LIST_JOINERS = {",", "и", "а", "также", "или"}


def extract_hosted_countries(text):
    """
    Страны, к которым привязан хозяйский предикат.

    Привязка считается, если:
      1) страна стоит объектом предлога "в/на", управляемого хозяйским
         глаголом ("пройдёт в Аргентине", "прошёл в США");
      2) или страна стоит рядом со словом "хозяйка/организатор турнира".

    Возвращает множество канонических стран (см. COUNTRY_CANON).
    Пустое множество = "хозяйство ни к какой стране не привязано".
    """
    if not text:
        return set()

    low = _PARTICIPATION_RE.sub(" ", text.lower())
    tokens = [m.group(0) for m in TOKEN_RE.finditer(low)]

    if not tokens:
        return set()

    country_at = {}

    for i, tok in enumerate(tokens):
        canon = COUNTRY_CANON.get(_lemma(tok))
        if canon:
            country_at[i] = canon

    if not country_at:
        return set()

    hosted = set()

    for i, tok in enumerate(tokens):
        if not _HOST_VERB_RE.fullmatch(tok):
            continue

        limit = min(len(tokens), i + 1 + HOST_WINDOW)

        for j in range(i + 1, limit):
            nxt = tokens[j]

            if _HOST_VERB_RE.fullmatch(nxt):
                break

            if not _HOST_PREP_RE.fullmatch(nxt):
                continue

            # Собираем список стран после предлога: "в США, Канаде и Мексике"
            k = j + 1
            found = False

            while k < len(tokens):
                tk = tokens[k]

                if k in country_at:
                    hosted.add(country_at[k])
                    found = True
                    k += 1
                elif tk in _LIST_JOINERS:
                    k += 1
                else:
                    break

            if found:
                # Не break: в предложении может быть несколько списков
                # ("ЧМ-2002 в Японии..., ЧМ-2026 в США, Канаде и Мексике").
                # Раньше сбор останавливался на первом и терял нужный.
                continue

    def collect_country_list(start):
        """Собирает список стран начиная с индекса start. Возвращает (страны, найдено)."""
        found_countries = set()
        k = start

        while k < len(tokens):
            if k in country_at:
                found_countries.add(country_at[k])
                k += 1
            elif tokens[k] in _LIST_JOINERS:
                k += 1
            else:
                break

        return found_countries

    # Правка №4: прямой объект при глаголах «примут/примет/принимают».
    # «примут участие» уже вырезано _PARTICIPATION_RE выше, поэтому здесь
    # остаются только хозяйственные конструкции.
    for i, tok in enumerate(tokens):
        if not _HOST_DIRECT_VERB_RE.fullmatch(tok):
            continue

        if i + 1 >= len(tokens) or (i + 1) not in country_at:
            continue

        hosted.update(collect_country_list(i + 1))

    # Правка №1: «организатор/хозяйка» + турнирное слово рядом.
    for i, tok in enumerate(tokens):
        if not _HOST_NOUN_RE.fullmatch(tok):
            continue

        context_start = max(0, i - HOST_NOUN_CONTEXT)
        context_end = min(len(tokens), i + HOST_NOUN_CONTEXT + 1)

        has_tournament = any(
            _HOST_TOURNAMENT_RE.fullmatch(tokens[c])
            for c in range(context_start, context_end)
        )

        if not has_tournament:
            continue

        for k, canon in country_at.items():
            if abs(k - i) <= HOST_NOUN_WINDOW:
                hosted.add(canon)

    return hosted


def has_host_predicate(text):
    """True, если в тексте есть хозяйский предикат (а не просто участие)."""
    if not text:
        return False

    low = _PARTICIPATION_RE.sub(" ", text.lower())

    return bool(_HOST_VERB_RE.search(low) or _HOST_NOUN_RE.search(low))


def check_country_conflict(claim, sentence, matched_count, coverage):
    """
    Противоречие по странам — только для взаимоисключающих стран
    (аргентина vs сша), не для совместимых (москва vs ссср).

    P3 + HOST_BINDING: противоречие возможно, только если в предложении есть
    хозяйский предикат, ПРИВЯЗАННЫЙ к стране (страна — объект предлога "в/на"
    при этом предикате). "Сборная Аргентины сыграет на ЧМ" и "матч ...
    Аргентины, который прошёл в Пасадине" не дают ни подтверждения,
    ни противоречия по странам.

    Условия:
      - предложение релевантно: matched_count >= 3 и coverage >= 0.30;
      - хозяйский предикат привязан к стране и в утверждении, и в предложении;
      - если в утверждении есть год, предложение содержит тот же год;
      - группы привязанных стран не пересекаются.
    """
    if matched_count < 3 or coverage < 0.30:
        return None

    claim_hosted = _country_groups(extract_hosted_countries(claim))

    if not claim_hosted:
        return None

    sentence_hosted = _country_groups(extract_hosted_countries(sentence))

    if not sentence_hosted:
        return None

    claim_years = extract_years(claim)
    sentence_years = extract_years(sentence)

    if claim_years and not (claim_years & sentence_years):
        return None

    # Предложение, где перечислены ДРУГИЕ турниры ("ЧМ-2002 в Японии,
    # ЧМ-2026 в США, ЧМ-2030 в Испании") — это обзор, а не утверждение
    # о нашем событии. Страновой конфликт по нему не считаем.
    if claim_years and (sentence_years - claim_years):
        print(
            f"      [country_conflict] пропуск: в предложении другие годы "
            f"{sorted(sentence_years - claim_years)}"
        )
        return None

    if claim_hosted & sentence_hosted:
        return None

    print(
        f"      [country_conflict] предикат привязан к стране; "
        f"claim={sorted(claim_hosted)} "
        f"vs sentence={sorted(sentence_hosted)} -> противоречит"
    )

    return "противоречит"


def slots_satisfied(claim, sentence):
    """
    P1: обязательные слоты. Если в утверждении есть страны, числа или годы,
    подтверждение ("подтверждает"/"косвенно") возможно только когда в
    предложении источника совпал соответствующий слот.

    Строже буквального правила "хотя бы один слот": год или число НЕ заменяют
    страну. Иначе заголовочные предложения статьи ("Чемпионат мира по
    футболу 2026") подтверждали бы любое утверждение про ЧМ-2026,
    независимо от того, в какой стране оно пройдёт.

    HOST_BINDING: если утверждение о проведении события ("пройдёт в X"),
    в предложении нужна та же привязанная страна: страна — объект предлога
    "в/на" при хозяйском предикате, а не просто соседнее упоминание.

    Порядок:
      1) в утверждении есть привязанная страна -> в предложении должна быть
         та же привязанная страна;
      2) страны есть, но хозяйство не привязано -> нужна общая страна;
      3) стран нет, но есть числа/годы -> нужно общее число/год;
      4) слотов нет -> ограничений нет.
    Возвращает (ok, причина).
    """
    claim_groups = _country_groups(extract_countries(claim))

    if claim_groups:
        claim_hosted = _country_groups(extract_hosted_countries(claim))

        if claim_hosted:
            sentence_hosted = _country_groups(extract_hosted_countries(sentence))

            if not (claim_hosted & sentence_hosted):
                return False, "хозяйство не привязано к стране утверждения"

            return True, "страна+предикат"

        sentence_groups = _country_groups(extract_countries(sentence))

        if not (claim_groups & sentence_groups):
            return False, "нет общей страны"

        return True, "страна"

    claim_numbers = extract_numbers(claim)

    if claim_numbers:
        if not (claim_numbers & extract_numbers(sentence)):
            return False, "нет общего числа/года"

        return True, "число/год"

    return True, "слотов нет"


def _has_contradiction_phrase(sentence_lower):
    """Маркер противоречия с границами слов, не по подстроке."""
    for phrase in CONTRADICTION_WORDS:
        pattern = r"(?<![а-яёa-z])" + re.escape(phrase) + r"(?![а-яёa-z])"
        if re.search(pattern, sentence_lower):
            return phrase
    return None


def highlight_matched_words(sentence, claim_groups):
    highlighted = sentence

    group_lemmas = set()

    for group in claim_groups:
        for item in group:
            group_lemmas.add(item)

    for match in re.finditer(r"[а-яёa-z0-9.,%]+", sentence, flags=re.IGNORECASE):
        raw = match.group()
        word_clean = raw.strip(".,;:!?()\"'")

        if not word_clean:
            continue

        if NUMBER_RE.match(word_clean):
            lem = word_clean.replace(",", ".")
        else:
            lem = _lemma(word_clean)

        if lem in group_lemmas:
            pattern = re.compile(
                r"(?<![\w])" + re.escape(raw) + r"(?![\w])",
                flags=re.IGNORECASE
            )
            highlighted = pattern.sub(f"**{raw}**", highlighted, count=1)

    return highlighted


# ─────────────────────────────────────────────────────────────────────────────
# ОЦЕНКА ПОДТВЕРЖДЕНИЯ
# ─────────────────────────────────────────────────────────────────────────────

CONFIRM_THRESHOLD = 0.60
PARTIAL_THRESHOLD = 0.30


def split_source_clauses(text):
    """
    Единый сплиттер текста источника: по заголовкам Википедии (==, ===),
    по границам предложений (. ! ? ; :) и по тире с пробелами.

    Используется и в find_best_sentences, и в сборке details для UI,
    чтобы пользователь видел ровно ту клаузу, которую оценивала система.

    Почему режем по ":" и по тире: иначе обломки вида
    "Финалиссима 2026 — ... прошёл в Пасадине" склеиваются, и страна с
    предикатом из разных фраз дают ложное подтверждение.
    """
    if not text:
        return []

    clauses = []

    for chunk in re.split(r"=+", text):
        for part in re.split(r"(?<=[.!?;:])\s+|\s[—–]\s", chunk):
            part = part.strip()

            if part:
                clauses.append(part)

    return clauses


def find_best_sentences(claim, text, max_sentences=3):
    """
    Ищет предложения источника, релевантные утверждению.

    Метрика — взвешенное покрытие ключевых терминов claim:
      coverage = сумма весов найденных лемм / сумма весов всех ключевых лемм

    Веса: числа, страны, латиница, опорные термины — 3.0;
    слова длиной 8+ — 2.0; 6-7 — 1.6; остальные — 1.0.

    Статусы:
      подтверждает          — coverage >= 0.60 и >= 3 совпавших термина
      косвенно подтверждает — coverage >= 0.35 и >= 2 совпавших термина
      противоречит          — числовое/годовое/страновое противоречие
                              либо маркер противоречия при >= 2 совпадениях
      связан с темой        — остальное
    """
    key_terms = claim_key_terms(claim)

    if not key_terms:
        return []

    weights = dict(key_terms[:10])
    total_weight = sum(weights.values()) or 1.0

    # CLAUSE_SPLIT_FIX: режем текст источника по заголовкам Википедии
    # (===, ==) и по клаузам (. ! ? ; : и тире с пробелами) ДО разбивки на
    # предложения. Хозяйский предикат и страна должны совпасть внутри одной
    # клаузы, иначе обломок вида "Финал 2026 — ... США, прошёл в ..." даёт
    # ложное подтверждение, потому что страны и предикат попали в одну
    # строку, но из разных фраз.
    sentences = split_source_clauses(text)

    # Годы утверждения и годы заголовка источника (первый фрагмент текста —
    # это title, так его собирает estimate_support).
    claim_years = extract_years(claim)
    title_years = extract_years(text.split(". ", 1)[0]) if text else set()

    matches = []

    for sentence in sentences:
        sentence_clean = sentence.strip()

        if len(sentence_clean) < 20:
            continue

        sentence_lower = sentence_clean.lower()
        sentence_lemmas = set(lemma_tokens(sentence_clean))

        matched = [lem for lem in weights if lem in sentence_lemmas]
        matched_count = len(matched)
        matched_weight = sum(weights[lem] for lem in matched)
        coverage = matched_weight / total_weight

        if matched_count < 1:
            continue

        numeric_check = check_numeric_relation(claim, sentence_clean)
        # v8: если в предложении нет своего года, берём год из заголовка
        # источника ("Чемпионат мира по футболу 2030"). Иначе фраза
        # "три матча пройдут в Аргентине" из статьи о ЧМ-2030
        # "подтверждала" бы фейк про ЧМ-2026 в Аргентине.
        year_check = check_year_conflict(claim, sentence_clean)
        if (
            year_check is None
            and claim_years
            and title_years
            and not extract_years(sentence_clean)
            and not (claim_years & title_years)
        ):
            year_check = "противоречит"

        # v8: маркер противоречия засчитываем только в действительно
        # релевантном предложении (>= 3 терминов и coverage >= 0.30).
        # Иначе фраза "миф" из статьи "Мифы о вакцинации" рушит правду.
        contradiction_phrase = None
        if matched_count >= 3 and coverage >= 0.30:
            contradiction_phrase = _has_contradiction_phrase(sentence_lower)

        country_check = check_country_conflict(
            claim, sentence_clean, matched_count, coverage
        )

        # v8: несовпадение годов или стран БЛОКИРУЕТ подтверждение,
        # но само по себе не является противоречием (в Википедии про ЧМ
        # полно предложений о 1930 или 2022 годе — это другие события).
        claim_geo = _country_groups(extract_countries(claim))
        sentence_geo = _country_groups(extract_countries(sentence_clean))
        geo_mismatch = bool(
            claim_geo and sentence_geo and not (claim_geo & sentence_geo)
        )
        year_mismatch = year_check == "противоречит"
        # P1: обязательные слоты (страны/числа/годы утверждения).
        slots_ok, slot_reason = slots_satisfied(claim, sentence_clean)
        block_confirm = geo_mismatch or year_mismatch or not slots_ok

        if numeric_check == "противоречит":
            status = "противоречит"
        elif country_check == "противоречит":
            status = "противоречит"
        elif contradiction_phrase:
            status = "противоречит"
        elif block_confirm:
            status = "связан с темой утверждения"
        elif numeric_check == "подтверждает":
            status = "подтверждает"
        elif coverage >= CONFIRM_THRESHOLD and matched_count >= 3:
            status = "подтверждает"
        elif coverage >= PARTIAL_THRESHOLD and matched_count >= 2:
            status = "косвенно подтверждает"
        else:
            status = "связан с темой утверждения"

        print(
            f"      [sentence] matched={matched_count} "
            f"coverage={coverage:.2f} year_mismatch={year_mismatch} "
            f"geo_mismatch={geo_mismatch} slots_ok={slots_ok} "
            f"({slot_reason}) status={status}"
        )

        matches.append({
            "sentence": sentence_clean,
            "highlighted_sentence": highlight_matched_words(
                sentence_clean, get_claim_word_groups(claim)
            ),
            "status": status,
            "coverage": coverage,
            "matched_groups": matched_count,
            "matched_words": matched,
            "numeric_check": numeric_check,
            "year_check": year_check,
            "country_check": country_check
        })

    matches.sort(
        key=lambda item: (item["matched_groups"], item["coverage"]),
        reverse=True
    )

    return matches[:max_sentences]


def estimate_support(claim, sources):
    """
    Возвращает (support_score, evidence).
    support_score — доля подтверждающего веса от общего числа источников,
    используется только для уверенности, не для вердикта.
    """
    key_terms = claim_key_terms(claim)

    if not key_terms:
        return 0.0, []

    evidence = []
    positive = 0.0
    negative = 0.0

    for source in sources:
        text = (
            source.get("title", "") + ". "
            + source.get("summary", "") + ". "
            + source.get("body", "")
        )

        sentence_matches = find_best_sentences(claim, text, max_sentences=2)

        if not sentence_matches:
            continue

        best = sentence_matches[0]
        trust = get_domain_trust(source.get("url", ""))
        score = best["coverage"] * 0.7 + trust * 0.3

        evidence.append({
            "title": source.get("title", ""),
            "url": source.get("url", ""),
            "source": source.get("source", ""),
            "quotes": sentence_matches,
            "coverage": best["coverage"],
            "matched_groups": best["matched_groups"],
            "matched_words": best["matched_words"],
            "trust": trust,
            "score": score
        })

        for match in sentence_matches:
            if match["status"] == "подтверждает":
                positive += 1.0
            elif match["status"] == "косвенно подтверждает":
                positive += 0.5
            elif match["status"] == "противоречит":
                negative += 1.0

    denom = max(len(key_terms[:10]), 1)
    support_score = max(0.0, min((positive - negative) / denom, 1.0))

    evidence.sort(key=lambda item: item["score"], reverse=True)

    return support_score, evidence




# ─────────────────────────────────────────────────────────────────────────────
# ФАКТЧЕКЕР
# ─────────────────────────────────────────────────────────────────────────────

class FactChecker:
    def check_fact(self, text):
        result = self.verify(text)

        return {
            "label": result["verdict"],
            "score": result["confidence"],
            "details": result["details"],
            "explanation": result["reason"]
        }

    def verify(self, text):
        claims = split_into_claims(text)

        if not claims:
            claims = [clean_text(text)]

        claims = claims[:4]

        print(f"\n[verify] claims: {len(claims)}")

        results = []
        important_results = []

        for index, claim in enumerate(claims, 1):
            print(f"\n[claim {index}] {claim[:90]}")

            result = self._verify_single_claim(claim)

            if result is None:
                result = {
                    "verdict": "НЕИЗВЕСТНО",
                    "confidence": 0.5,
                    "reason": "Не удалось проверить часть утверждения.",
                    "details": [],
                    "importance": "context"
                }

            importance = classify_claim_importance(claim)
            result["importance"] = importance
            results.append(result)

            if importance == "important":
                important_results.append(result)

            print(
                f"   → {result['verdict']} ({result['confidence']:.2f}) "
                f"[{importance}]"
            )

        if important_results:
            combined = self._combine_results(important_results)
        else:
            combined = self._combine_results(results)

        # v8: глобальный guardrail. Сенсационные маркеры проверяются по
        # ВСЕМУ тексту, а не только по claim: даже если отдельная нейтральная
        # фраза фейка подтвердилась, весь текст не может стать ПРАВДОЙ.
        if combined["verdict"] == "ПРАВДА" and has_sensational_markers(text):
            print("   [guardrail] сенсационные маркеры в тексте -> ПРАВДА запрещена")
            combined["verdict"] = "НЕИЗВЕСТНО"
            combined["confidence"] = 0.45
            combined["reason"] = (
                "Текст содержит сенсационные формулировки (запрет, "
                "обязательства, сокрытие, подделка, мгновенный эффект); "
                "подтверждены лишь второстепенные детали, главное "
                "утверждение источниками не подтверждено."
            )

        combined["claims_checked"] = len(important_results)
        combined["claims_total"] = len(results)

        print(
            f"\n[verify] ИТОГ: {combined['verdict']} "
            f"({combined['confidence']:.2f})"
        )

        return combined

    def _verify_single_claim(self, text):
        print(f"   [claim] {text[:90]}")

        key_terms = claim_key_terms(text)
        print(f"   [lemms] {[t for t, _ in key_terms[:10]]}")

        duckduckgo_results = search_duckduckgo(text)
        wikipedia_results = search_wikipedia_api(text)
        rss_results = get_relevant_articles(text)

        print(
            f"   [sources] DDG={len(duckduckgo_results)}, "
            f"Wiki={len(wikipedia_results)}, RSS={len(rss_results)}"
        )

        all_sources = []

        for result in duckduckgo_results:
            all_sources.append(result)

        for result in wikipedia_results:
            all_sources.append(result)

        for result in rss_results:
            all_sources.append({
                "title": result["title"],
                "summary": result["summary"],
                "body": result["summary"],
                "url": result["link"],
                "source": result["source"]
            })

        # Углубляем первый источник страницы (если DDG дал мало тела)
        if all_sources:
            first_url = all_sources[0].get("url", "") if all_sources[0].get("body", "") and len(all_sources[0].get("body", "")) < 500 else ""

            if first_url:
                page_text = get_page_text(first_url)
                if page_text:
                    all_sources[0]["body"] = (
                        all_sources[0].get("body", "") + " " + page_text
                    )

        support, evidence = estimate_support(text, all_sources)

        print(f"   [support] {support:.3f}, evidence={len(evidence)}")

        # ── details для UI ────────────────────────────────────────────────
        details = []

        for item in evidence[:8]:
            quotes = item["quotes"]
            quote_texts = []

            for quote in quotes[:2]:
                sentence_for_output = quote.get(
                    "highlighted_sentence", quote["sentence"]
                )
                quote_texts.append(
                    f"«{sentence_for_output}» "
                    f"— источник {quote['status']} утверждению."
                )

            has_confirmation = any(
                q["status"] == "подтверждает" for q in quotes
            )
            has_contradiction = any(
                q["status"] == "противоречит" for q in quotes
            )

            if has_contradiction and not has_confirmation:
                result_flag = False
            elif has_confirmation:
                result_flag = True
            else:
                result_flag = None

            details.append({
                "fact": item["title"],
                "source": item["source"],
                "reason": " ".join(quote_texts),
                "links": [item["url"]] if item["url"] else [],
                "result": result_flag
            })

        # ── релевантные источники ─────────────────────────────────────────
        relevant_sources = []

        for source in all_sources:
            source_text = (
                source.get("title", "") + " "
                + source.get("summary", "") + " "
                + source.get("body", "")
            )

            source_lemmas = set(lemma_tokens(source_text))
            matched = [lem for lem, _ in key_terms[:10] if lem in source_lemmas]
            coverage = (
                sum(w for lem, w in key_terms[:10] if lem in source_lemmas)
                / (sum(w for _, w in key_terms[:10]) or 1.0)
            )

            if len(matched) >= 2 and coverage >= 0.20:
                relevant_sources.append({
                    "source": source,
                    "coverage": coverage,
                    "matched_groups": len(matched)
                })

        relevant_sources.sort(
            key=lambda x: (x["matched_groups"], x["coverage"]),
            reverse=True
        )

        print(f"   [relevant] {len(relevant_sources)}")

        if not details:
            for item in relevant_sources[:6]:
                source = item["source"]

                sentences = split_source_clauses(
                    (source.get("summary", "") + " " + source.get("body", "")).strip()
                )

                sentence = ""

                for candidate in sentences:
                    if len(candidate.strip()) > 20:
                        sentence = candidate.strip()
                        break

                if not sentence:
                    sentence = source.get("title", "")

                details.append({
                    "fact": source.get("title", ""),
                    "source": source.get("source", ""),
                    "reason": f"Источник связан с темой утверждения: «{sentence[:300]}»",
                    "links": [source.get("url", "")] if source.get("url") else [],
                    "result": None
                })

        # ── классификация evidence ────────────────────────────────────────
        confirming_sources = [
            item for item in evidence
            if any(q["status"] == "подтверждает" for q in item["quotes"])
        ]
        partially_confirming_sources = [
            item for item in evidence
            if any(q["status"] == "косвенно подтверждает" for q in item["quotes"])
        ]
        contradicting_sources = [
            item for item in evidence
            if any(q["status"] == "противоречит" for q in item["quotes"])
        ]
        numeric_contradictions = [
            item for item in evidence
            if any(q.get("numeric_check") == "противоречит" for q in item["quotes"])
        ]
        strong_contradictions = [
            item for item in contradicting_sources
            if item.get("trust", 0.5) >= 0.8
        ]

        print(
            f"   [evidence] подтв={len(confirming_sources)}, "
            f"косв={len(partially_confirming_sources)}, "
            f"против={len(contradicting_sources)}, "
            f"числ_против={len(numeric_contradictions)}, "
            f"сильн_против={len(strong_contradictions)}"
        )

        sensational = has_sensational_markers(text)
        print(f"   [sensational] {sensational}")

        # ── ЦЕПОЧКА ВЕРДИКТОВ ─────────────────────────────────────────────
        if not all_sources:
            verdict = "НЕИЗВЕСТНО"
            confidence = 0.5
            reason = (
                "Поиск не вернул ни одного источника: DuckDuckGo, Wikipedia "
                "и RSS-ленты не дали результатов."
            )

        elif numeric_contradictions:
            verdict = "ФЕЙК"
            confidence = min(0.78 + len(numeric_contradictions) * 0.04, 0.93)
            reason = (
                "Найден источник с той же числовой информацией, но "
                "противоположным направлением изменения — утверждение "
                "противоречит данным источника."
            )

        elif strong_contradictions:
            verdict = "ФЕЙК"
            confidence = min(0.72 + len(strong_contradictions) * 0.04, 0.9)
            reason = (
                "Надёжный источник прямо противоречит утверждению."
            )

        # P6: если есть и подтверждение, и противоречие (слабое — доверие
        # источника < 0.8, иначе сработала бы ветка strong_contradictions
        # выше), это спорный случай: ПРАВДА не выдаём.
        elif (
            (confirming_sources or partially_confirming_sources)
            and contradicting_sources
        ):
            verdict = "НЕИЗВЕСТНО"
            confidence = 0.5
            reason = (
                "Источники расходятся: одни подтверждают утверждение, "
                "другие ему противоречат. Надёжно определить, правда это "
                "или фейк, нельзя."
            )

        elif confirming_sources and not sensational:
            verdict = "ПРАВДА"

            if len(confirming_sources) >= 2:
                confidence = max(0.68, min(0.68 + support * 0.2, 0.9))
            else:
                confidence = 0.58

            reason = (
                "Найден источник, который подтверждает утверждение: "
                "совпадают ключевые сущности и смысл."
            )

        elif partially_confirming_sources and not sensational:
            if len(partially_confirming_sources) >= 2:
                verdict = "ПРАВДА"
                confidence = 0.55
                reason = (
                    "Два и более релевантных источника согласуются с "
                    "утверждением по ключевым сущностям; противоречий нет."
                )
            elif relevant_sources:
                verdict = "ПРАВДА"
                confidence = 0.52
                reason = (
                    "Релевантный источник согласуется с утверждением по "
                    "ключевым сущностям; противоречий нет."
                )
            else:
                verdict = "НЕИЗВЕСТНО"
                confidence = 0.48
                reason = (
                    "Источники лишь отчасти касаются темы утверждения, "
                    "уверенного вывода сделать нельзя."
                )

        elif sensational:
            verdict = "НЕИЗВЕСТНО"
            confidence = 0.45
            reason = (
                "Утверждение содержит сенсационные формулировки "
                "(запрет, обязательства, сокрытие, подделка) и не "
                "подтверждено надёжными источниками."
            )

        elif contradicting_sources:
            verdict = "НЕИЗВЕСТНО"
            confidence = 0.5
            reason = (
                "Найдены возможные расхождения с источниками, но "
                "их недостаточно для вывода о фейке."
            )

        elif relevant_sources:
            verdict = "НЕИЗВЕСТНО"
            confidence = 0.5
            reason = (
                "Найдены источники по теме, но ни один не подтверждает "
                "утверждение и не противоречит ему."
            )

        else:
            verdict = "НЕИЗВЕСТНО"
            confidence = 0.5
            reason = (
                "Источники по теме утверждения не найдены: результаты "
                "поиска нерелевантны."
            )

        # ── корректировка для контекстных claims ─────────────────────────
        importance = classify_claim_importance(text)

        if importance == "context":
            if verdict == "ФЕЙК":
                final_verdict = "ФЕЙК"
                final_confidence = confidence
                final_reason = reason
            elif verdict == "ПРАВДА":
                final_verdict = "ПРАВДА"
                final_confidence = 0.55
                final_reason = (
                    "Контекстная часть утверждения согласуется с "
                    "источниками; противоречий не найдено."
                )
            else:
                final_verdict = "НЕИЗВЕСТНО"
                final_confidence = 0.5
                final_reason = (
                    "Контекстная часть утверждения не определяет "
                    "итоговый вердикт."
                )
        else:
            final_verdict = verdict
            final_confidence = confidence
            final_reason = reason

        print(f"   [итог claim] {final_verdict} ({final_confidence:.2f})")

        return {
            "verdict": final_verdict,
            "confidence": final_confidence,
            "reason": final_reason,
            "details": details,
            "importance": importance
        }

    def _combine_results(self, results):
        """
        Агрегация по claims:
          - любой ФЕЙК -> ФЕЙК;
          - ПРАВДА при строгом большинстве ПРАВДА, либо при ровно
            половине, если подтверждено первое (главное) утверждение;
          - иначе НЕИЗВЕСТНО.
        """
        if not results:
            return {
                "verdict": "НЕИЗВЕСТНО",
                "confidence": 0.5,
                "reason": "Не удалось проверить ни одного утверждения.",
                "details": []
            }

        verdicts = [r["verdict"] for r in results]
        total = len(verdicts)

        truth_count = verdicts.count("ПРАВДА")
        fake_count = verdicts.count("ФЕЙК")

        if fake_count > 0:
            final_verdict = "ФЕЙК"
            reason = (
                "Хотя бы одна часть утверждения противоречит источникам."
            )
        elif truth_count > total / 2 or (
            verdicts[0] == "ПРАВДА" and truth_count >= total / 2
        ):
            # v8: ПРАВДА — строгое большинство, либо ровно половина при
            # условии, что подтверждено ПЕРВОЕ (главное) утверждение.
            # Так фейк "ЧМ-2026 в Аргентине" не станет правдой из-за того,
            # что подтвердилась второстепенная фраза про стадионы 1930 года.
            final_verdict = "ПРАВДА"
            reason = (
                "Большинство проверенных частей утверждения подтверждено "
                "источниками; противоречий не найдено."
            )
        else:
            final_verdict = "НЕИЗВЕСТНО"
            reason = (
                "Подтвердить удалось лишь меньшую часть утверждения."
            )

        all_details = []

        for result in results:
            all_details.extend(result.get("details", []))

        confidence = sum(
            r.get("confidence", 0.5) for r in results
        ) / len(results)

        return {
            "verdict": final_verdict,
            "confidence": confidence,
            "reason": reason,
            "details": all_details
        }


# КОНЕЦ ФАЙЛА main_code.py

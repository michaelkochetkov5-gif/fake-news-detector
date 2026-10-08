import streamlit as st
import requests
import feedparser
import re
import json
import os
import time
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS

CACHE_FILE = "cache.json"

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
    return re.sub(r"[^a-zа-яё0-9]+", "_", text.lower()).strip("_")

STOP_WORDS = {
    "это", "этот", "эта", "эти", "того", "такой", "такая", "такие",
    "есть", "был", "была", "были", "будет", "будут", "очень",
    "более", "менее", "самый", "самая", "самые", "который",
    "которая", "которые", "чтобы", "потому", "поэтому", "также",
    "тоже", "вот", "там", "тут", "или", "и", "а", "но", "не", "ни",
    "да", "нет", "как", "что", "чем", "при", "для", "от", "до",
    "из", "по", "за", "на", "в", "во", "со", "с", "у", "о", "об",
    "обо", "над", "под", "про", "через", "между", "имеет", "является"
}

SYNONYMS = {
    "круглая": [
        "круглая", "шарообразная", "форме шара", "сферическая",
        "геоид", "эллипсоид", "форма земли"
    ],
    "шара": [
        "шара", "шарообразная", "круглая", "сферическая",
        "геоид", "эллипсоид", "форма земли"
    ],
    "форма": [
        "форма", "форме", "формы", "имеет форму", "форма земли"
    ],
    "земля": [
        "земля", "планета земля", "наша планета", "форма земли"
    ],
    "вреден": [
        "вреден", "вредна", "вредно", "опасен", "опасна", "опасно",
        "негативно влияет", "негативное влияние"
    ],
    "опасен": [
        "вреден", "вредна", "вредно", "опасен", "опасна", "опасно",
        "негативно влияет", "негативное влияние"
    ],
    "вызывает": [
        "вызывает", "приводит", "приводит к", "связан",
        "связана", "влияет", "влияние"
    ],
    "лечит": [
        "лечит", "помогает", "эффективен", "эффективна",
        "эффективность", "терапия"
    ],
    "запретят": [
        "запретят", "запрет", "запрещён", "запрещено", "запрещать"
    ],
    "отменят": [
        "отменят", "отмена", "отменён", "отменено", "отменять"
    ]
}

def expand_keywords(words):
    expanded = []

    for word in words:
        expanded.append(word)

        if word in SYNONYMS:
            expanded.extend(SYNONYMS[word])

    return list(dict.fromkeys(expanded))

def clean_text(text):
    if not text:
        return ""

    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()

def normalize_word(word):
    word = word.lower().strip()

    if len(word) <= 5:
        return word

    return word[:5]

def get_claim_words(text):
    words = re.findall(r"[а-яёa-z]{4,}", text.lower())

    words = [
        word for word in words
        if word not in STOP_WORDS
    ]

    words = expand_keywords(list(dict.fromkeys(words)))

    normalized = [
        normalize_word(word)
        for word in words
    ]

    return list(dict.fromkeys(normalized))

def make_search_queries(claim):
    queries = [claim]

    words = re.findall(r"[а-яёa-z]{4,}", claim.lower())

    words = [
        word for word in words
        if word not in STOP_WORDS
    ]

    if len(words) >= 2:
        queries.append(" ".join(words))

    claim_lower = claim.lower()

    if "земля" in claim_lower:
        queries.append("Земля")

    if "круглая" in claim_lower or "шара" in claim_lower:
        queries.append("Форма Земли")
        queries.append("Земля сферическая")

    if "вакцина" in claim_lower:
        queries.append("вакцина безопасность исследования")

    if "вред" in claim_lower or "вреден" in claim_lower:
        queries.append("научные исследования влияние")

    return list(dict.fromkeys(queries))

def search_duckduckgo(claim, max_results=8):
    cache_key = "ddg_" + get_cache_key(claim)

    if cache_key in search_cache:
        return search_cache[cache_key]

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
                        max_results=5
                    )

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

                except Exception:
                    continue

            try:
                news_results = ddgs.news(
                    claim,
                    region="ru-ru",
                    safesearch="moderate",
                    max_results=5
                )

                for result in news_results:
                    url = result.get("href", "") or result.get("url", "")

                    if url and url not in seen_urls:
                        seen_urls.add(url)

                        results.append({
                            "title": result.get("title", ""),
                            "summary": clean_text(result.get("body", "")),
                            "body": clean_text(result.get("body", "")),
                            "url": url,
                            "source": result.get("source", "DuckDuckGo News")
                        })

            except Exception:
                pass

    except Exception:
        pass

    search_cache[cache_key] = results[:max_results]
    save_cache()

    return results[:max_results]

def search_wikipedia_api(claim, max_results=5):
    cache_key = "wiki_" + get_cache_key(claim)

    if cache_key in search_cache:
        return search_cache[cache_key]

    results = []
    seen_titles = set()

    queries = make_search_queries(claim)

    for query in queries:
        try:
            search_url = "https://ru.wikipedia.org/w/api.php"

            search_params = {
                "action": "query",
                "list": "search",
                "srsearch": query,
                "srlimit": 3,
                "format": "json",
                "utf8": 1
            }

            response = requests.get(
                search_url,
                params=search_params,
                timeout=10
            )

            data = response.json()

            pages = data.get("query", {}).get("search", [])

            for page in pages:
                title = page.get("title", "")

                if title in seen_titles:
                    continue

                seen_titles.add(title)

                extract_params = {
                    "action": "query",
                    "prop": "extracts|info",
                    "explaintext": 1,
                    "exintro": 1,
                    "inprop": "url",
                    "titles": title,
                    "format": "json",
                    "utf8": 1
                }

                extract_response = requests.get(
                    search_url,
                    params=extract_params,
                    timeout=10
                )

                extract_data = extract_response.json()

                pages_data = extract_data.get(
                    "query", {}
                ).get("pages", {})

                for page_id, page_data in pages_data.items():
                    extract = page_data.get("extract", "")
                    url = page_data.get("fullurl", "")

                    if extract and url:
                        results.append({
                            "title": title,
                            "summary": clean_text(extract),
                            "body": clean_text(extract),
                            "url": url,
                            "source": "Wikipedia"
                        })

        except Exception:
            continue

    unique_results = []
    seen_urls = set()

    for result in results:
        if result["url"] not in seen_urls:
            seen_urls.add(result["url"])
            unique_results.append(result)

    search_cache[cache_key] = unique_results[:max_results]
    save_cache()

    return unique_results[:max_results]

RSS_FEEDS = [
    {
        "name": "РИА Новости",
        "url": "https://ria.ru/export/rss2/archive/index.xml"
    },
    {
        "name": "ТАСС",
        "url": "https://tass.ru/rss/v2.xml"
    },
    {
        "name": "Интерфакс",
        "url": "https://www.interfax.ru/rss.asp"
    },
    {
        "name": "Lenta.ru",
        "url": "https://lenta.ru/rss/news"
    },
    {
        "name": "BBC Russian",
        "url": "https://feeds.bbci.co.uk/russian/rss.xml"
    }
]

def get_relevant_articles(claim, max_articles=5):
    cache_key = "news_" + get_cache_key(claim)

    current_time = time.time()

    if cache_key in news_cache:
        cached = news_cache[cache_key]

        if current_time - cached.get("time", 0) < 3600:
            return cached.get("articles", [])

    claim_words = get_claim_words(claim)

    articles = []

    for feed in RSS_FEEDS:
        try:
            parsed_feed = feedparser.parse(feed["url"])

            for entry in parsed_feed.entries[:50]:
                title = clean_text(entry.get("title", ""))
                summary = clean_text(entry.get("summary", ""))

                text = (title + " " + summary).lower()

                text_words = re.findall(
                    r"[а-яёa-z]{4,}",
                    text
                )

                text_stems = {
                    normalize_word(word)
                    for word in text_words
                }

                matched_words = []

                for word in claim_words:
                    if word in text_stems:
                        matched_words.append(word)

                unique_matches = list(dict.fromkeys(matched_words))

                coverage = len(unique_matches) / max(len(claim_words), 1)

                if len(unique_matches) >= 2 and coverage >= 0.4:
                    articles.append({
                        "title": title,
                        "summary": summary,
                        "link": entry.get("link", ""),
                        "source": feed["name"],
                        "published": entry.get("published", ""),
                        "matches": len(unique_matches),
                        "coverage": coverage
                    })

        except Exception:
            continue

    articles.sort(
        key=lambda item: (
            item["coverage"],
            item["matches"]
        ),
        reverse=True
    )

    articles = articles[:max_articles]

    news_cache[cache_key] = {
        "time": current_time,
        "articles": articles
    }

    save_cache()

    return articles

def find_best_sentence(claim, text):
    sentences = re.split(r"(?<=[.!?])\s+", text)

    claim_words = get_claim_words(claim)

    best_match = None

    for sentence in sentences:
        sentence_clean = sentence.strip()

        if len(sentence_clean) < 30:
            continue

        sentence_words = re.findall(
            r"[а-яёa-z]{4,}",
            sentence_clean.lower()
        )

        sentence_stems = {
            normalize_word(word)
            for word in sentence_words
        }

        matched_words = []

        for word in claim_words:
            if word in sentence_stems:
                matched_words.append(word)

        unique_matches = list(dict.fromkeys(matched_words))

        coverage = len(unique_matches) / max(len(claim_words), 1)

        if coverage >= 0.45:
            status = "подтверждает"
        elif coverage >= 0.25:
            status = "косвенно подтверждает"
        elif coverage >= 0.1:
            status = "связан с темой утверждения"
        else:
            continue

        if best_match is None or coverage > best_match["coverage"]:
            best_match = {
                "sentence": sentence_clean,
                "status": status,
                "coverage": coverage
            }

    return best_match

def estimate_support(claim, sources):
    claim_words = get_claim_words(claim)

    if not claim_words:
        return 0.0, []

    evidence = []
    support_score = 0.0

    for source in sources:
        text = (
            source.get("title", "") + ". " +
            source.get("summary", "") + ". " +
            source.get("body", "")
        )

        best_match = find_best_sentence(claim, text)

        if best_match:
            evidence.append({
                "title": source.get("title", ""),
                "url": source.get("url", ""),
                "source": source.get("source", ""),
                "sentence": best_match["sentence"],
                "status": best_match["status"],
                "coverage": best_match["coverage"]
            })

            if best_match["status"] == "подтверждает":
                support_score += 1.0
            elif best_match["status"] == "косвенно подтверждает":
                support_score += 0.6
            else:
                support_score += 0.25

    support_score = min(
        support_score / max(len(claim_words), 1),
        1.0
    )

    evidence.sort(
        key=lambda item: item["coverage"],
        reverse=True
    )

    return support_score, evidence

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
        duckduckgo_results = search_duckduckgo(text)
        wikipedia_results = search_wikipedia_api(text)
        rss_results = get_relevant_articles(text)

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

        support, evidence = estimate_support(text, all_sources)

        details = []

        for item in evidence[:8]:
            details.append({
                "fact": item["title"],
                "source": item["source"],
                "reason": (
                    f"Источник {item['status']} утверждение: "
                    f"«{item['sentence']}»"
                ),
                "links": [item["url"]] if item["url"] else [],
                "result": (
                    True
                    if item["status"] == "подтверждает"
                    else None
                )
            })

        relevant_sources = []

        claim_words = get_claim_words(text)

        for source in all_sources:
            source_text = (
                source.get("title", "") + " " +
                source.get("summary", "") + " " +
                source.get("body", "")
            )

            source_words = re.findall(
                r"[а-яёa-z]{4,}",
                source_text.lower()
            )

            source_stems = {
                normalize_word(word)
                for word in source_words
            }

            matched_words = []

            for word in claim_words:
                if word in source_stems:
                    matched_words.append(word)

            unique_matches = list(dict.fromkeys(matched_words))

            coverage = len(unique_matches) / max(len(claim_words), 1)

            if len(unique_matches) >= 2 or coverage >= 0.4:
                relevant_sources.append({
                    "source": source,
                    "coverage": coverage
                })

        relevant_sources.sort(
            key=lambda item: item["coverage"],
            reverse=True
        )

        if not details:
            for item in relevant_sources[:8]:
                source = item["source"]

                sentences = re.split(
                    r"(?<=[.!?])\s+",
                    (
                        source.get("summary", "") + " " +
                        source.get("body", "")
                    ).strip()
                )

                sentence = ""

                for sentence_candidate in sentences:
                    if len(sentence_candidate.strip()) > 30:
                        sentence = sentence_candidate.strip()
                        break

                if not sentence:
                    sentence = source.get("title", "")

                details.append({
                    "fact": source.get("title", ""),
                    "source": source.get("source", ""),
                    "reason": (
                        f"Источник связан с темой утверждения: "
                        f"«{sentence}»"
                    ),
                    "links": [source.get("url", "")] if source.get("url") else [],
                    "result": None
                })

        strong_evidence = [
            item for item in evidence
            if item["status"] in [
                "подтверждает",
                "косвенно подтверждает"
            ]
        ]

        if len(strong_evidence) >= 2:
            verdict = "ПРАВДА"
            confidence = min(0.65 + support * 0.25, 0.9)
            reason = (
                "Найдено несколько релевантных источников, которые "
                "подтверждают или косвенно подтверждают утверждение."
            )

        elif len(strong_evidence) == 1:
            verdict = "ПРАВДА"
            confidence = 0.62
            reason = (
                "Найден релевантный источник, который подтверждает "
                "или косвенно подтверждает утверждение. Для более "
                "надёжного вывода желательно больше независимых "
                "источников."
            )

        elif relevant_sources:
            verdict = "НЕИЗВЕСТНО"
            confidence = 0.5
            reason = (
                "Найдены источники по теме, но в них нет достаточно "
                "точного подтверждения или опровержения утверждения."
            )

        else:
            verdict = "НЕИЗВЕСТНО"
            confidence = 0.5
            reason = (
                "Поиск не нашёл достаточного количества релевантных "
                "источников для проверки утверждения."
            )

        return {
            "verdict": verdict,
            "confidence": confidence,
            "reason": reason,
            "details": details
        }

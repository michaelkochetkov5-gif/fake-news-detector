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

TRUSTED_DOMAINS = {
    "wikipedia.org": 1.0,
    "nasa.gov": 1.0,
    "who.int": 1.0,
    "un.org": 0.9,
    "gov.ru": 0.9,
    "edu.ru": 0.9,
    "edu": 0.8,
    "ria.ru": 0.7,
    "tass.ru": 0.7,
    "interfax.ru": 0.7,
    "bbc.com": 0.8,
    "reuters.com": 0.9,
    "apnews.com": 0.9
}

CONTRADICTION_WORDS = [
    "не является", "не имеет", "не соответствует",
    "опровергает", "опровергнуто", "ложно", "ложная",
    "фейк", "миф", "неверно", "неправда", "дезинформация"
]

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

def get_domain_trust(url):
    try:
        domain = url.lower()
        domain = domain.replace("https://", "")
        domain = domain.replace("http://", "")
        domain = domain.split("/")[0]

        for trusted_domain, trust in TRUSTED_DOMAINS.items():
            if trusted_domain in domain:
                return trust

    except Exception:
        pass

    return 0.5

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

    if "форма" in claim_lower:
        queries.append("Форма Земли")

    if "шара" in claim_lower or "круглая" in claim_lower:
        queries.append("Земля сферическая")

    if "вакцина" in claim_lower:
        queries.append("вакцина безопасность исследования")

    if "вред" in claim_lower or "вреден" in claim_lower:
        queries.append("научные исследования влияние")

    return list(dict.fromkeys(queries))

def get_page_text(url, max_length=12000):
    if not url:
        return ""

    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (compatible; FactChecker/1.0)"
            )
        }

        response = requests.get(
            url,
            headers=headers,
            timeout=10
        )

        if response.status_code != 200:
            return ""

        soup = BeautifulSoup(
            response.text,
            "html.parser"
        )

        for tag in soup([
            "script",
            "style",
            "nav",
            "header",
            "footer",
            "aside",
            "form",
            "button"
        ]):
            tag.decompose()

        text = soup.get_text(separator=" ")

        text = clean_text(text)

        return text[:max_length]

    except Exception:
        return ""

def search_duckduckgo(claim, max_results=10):
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

                    time.sleep(1)

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

                time.sleep(1)

            except Exception:
                pass

    except Exception:
        pass

    search_cache[cache_key] = results[:max_results]
    save_cache()

    return results[:max_results]

def search_wikipedia_api(claim, max_results=6):
    cache_key = "wiki_" + get_cache_key(claim)

    if cache_key in search_cache:
        return search_cache[cache_key]

    results = []
    seen_titles = set()

    queries = make_search_queries(claim)

    for query in queries:
        try:
            api_url = "https://ru.wikipedia.org/w/api.php"

            search_params = {
                "action": "query",
                "list": "search",
                "srsearch": query,
                "srlimit": 3,
                "format": "json",
                "utf8": 1
            }

            response = requests.get(
                api_url,
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
                    api_url,
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

            time.sleep(0.5)

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

                if len(unique_matches) >= 2 or coverage >= 0.4:
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

def find_best_sentences(claim, text, max_sentences=3):
    sentences = re.split(r"(?<=[.!?])\s+", text)

    claim_words = get_claim_words(claim)

    matches = []

    for sentence in sentences:
        sentence_clean = sentence.strip()

        if len(sentence_clean) < 30:
            continue

        sentence_lower = sentence_clean.lower()

        sentence_words = re.findall(
            r"[а-яёa-z]{4,}",
            sentence_lower
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

        has_contradiction = any(
            word in sentence_lower
            for word in CONTRADICTION_WORDS
        )

        if has_contradiction and coverage >= 0.25:
            status = "противоречит"
        elif coverage >= 0.45:
            status = "подтверждает"
        elif coverage >= 0.25:
            status = "косвенно подтверждает"
        elif coverage >= 0.1:
            status = "связан с темой утверждения"
        else:
            continue

        matches.append({
            "sentence": sentence_clean,
            "status": status,
            "coverage": coverage
        })

    matches.sort(
        key=lambda item: item["coverage"],
        reverse=True
    )

    return matches[:max_sentences]

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

        sentence_matches = find_best_sentences(
            claim,
            text,
            max_sentences=3
        )

        if sentence_matches:
            best_coverage = sentence_matches[0]["coverage"]
            trust = get_domain_trust(source.get("url", ""))

            final_score = (
                best_coverage * 0.7 +
                trust * 0.3
            )

            evidence.append({
                "title": source.get("title", ""),
                "url": source.get("url", ""),
                "source": source.get("source", ""),
                "quotes": sentence_matches,
                "coverage": best_coverage,
                "trust": trust,
                "score": final_score
            })

            for match in sentence_matches:
                if match["status"] == "подтверждает":
                    support_score += 1.0
                elif match["status"] == "косвенно подтверждает":
                    support_score += 0.6
                elif match["status"] == "противоречит":
                    support_score -= 1.0

    support_score = min(
        max(support_score / max(len(claim_words), 1), 0.0),
        1.0
    )

    evidence.sort(
        key=lambda item: item["score"],
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

        # Открываем до 5 наиболее перспективных страниц
        pages_to_open = []

        for source in all_sources:
            url = source.get("url", "")

            if url and url not in pages_to_open:
                pages_to_open.append(url)

        pages_to_open = pages_to_open[:5]

        for url in pages_to_open:
            page_text = get_page_text(url)

            if page_text:
                for source in all_sources:
                    if source.get("url") == url:
                        source["body"] = (
                            source.get("body", "") + " " +
                            page_text
                        )

                        break

        support, evidence = estimate_support(text, all_sources)

        details = []

        for item in evidence[:8]:
            quotes = item["quotes"]

            quote_texts = []

            for quote in quotes[:2]:
                quote_texts.append(
                    f"«{quote['sentence']}» "
                    f"— источник {quote['status']} утверждение."
                )

            has_confirmation = any(
                quote["status"] in [
                    "подтверждает",
                    "косвенно подтверждает"
                ]
                for quote in quotes
            )

            has_contradiction = any(
                quote["status"] == "противоречит"
                for quote in quotes
            )

            if has_contradiction and not has_confirmation:
                result = False
            elif has_confirmation:
                result = True
            else:
                result = None

            details.append({
                "fact": item["title"],
                "source": item["source"],
                "reason": " ".join(quote_texts),
                "links": [item["url"]] if item["url"] else [],
                "result": result
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

        confirming_sources = [
            item for item in evidence
            if any(
                quote["status"] in [
                    "подтверждает",
                    "косвенно подтверждает"
                ]
                for quote in item["quotes"]
            )
        ]

        contradicting_sources = [
            item for item in evidence
            if any(
                quote["status"] == "противоречит"
                for quote in item["quotes"]
            )
        ]

        if contradicting_sources and not confirming_sources:
            verdict = "ФЕЙК"
            confidence = min(
                0.6 + len(contradicting_sources) * 0.1,
                0.88
            )
            reason = (
                "Найдены релевантные источники, которые противоречат "
                "утверждению."
            )

        elif confirming_sources:
            verdict = "ПРАВДА"

            if len(confirming_sources) >= 2:
                confidence = max(
                    0.70,
                    min(
                        0.70 + support * 0.20,
                        0.92
                    )
                )
            else:
                confidence = 0.65

            reason = (
                "Найдены релевантные источники, которые подтверждают "
                "или косвенно подтверждают утверждение."
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

import streamlit as st
import requests
import feedparser
import re
import json
import os
import time
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS

try:
    import wikipedia
except ImportError:
    wikipedia = None

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
    "это", "этот", "эта", "эти", "того", "того", "такой", "такая",
    "такие", "есть", "был", "была", "были", "будет", "будут",
    "очень", "более", "менее", "самый", "самая", "самые",
    "который", "которая", "которые", "чтобы", "потому", "поэтому",
    "также", "тоже", "вот", "там", "тут", "или", "и", "а", "но",
    "не", "ни", "да", "нет", "как", "что", "чем", "при", "для",
    "от", "до", "из", "по", "за", "на", "в", "во", "со", "с", "у",
    "о", "об", "обо", "над", "под", "про", "через", "между"
}

SYNONYMS = {
    "круглая": [
        "круглая", "шарообразная", "форме шара",
        "сферическая", "геоид", "эллипсоид"
    ],
    "шара": [
        "шара", "шарообразная", "круглая",
        "сферическая", "геоид", "эллипсоид"
    ],
    "форма": [
        "форма", "форме", "формы", "имеет форму"
    ],
    "вреден": [
        "вреден", "вредна", "вредно", "опасен", "опасна", "опасно"
    ],
    "опасен": [
        "вреден", "вредна", "вредно", "опасен", "опасна", "опасно"
    ],
    "вызывает": [
        "вызывает", "приводит", "связан", "связана", "влияет"
    ],
    "лечит": [
        "лечит", "помогает", "эффективен", "эффективна"
    ],
    "запретят": [
        "запретят", "запрет", "запрещён", "запрещено"
    ],
    "отменят": [
        "отменят", "отмена", "отменён", "отменено"
    ],
    "земля": [
        "земля", "планета земля", "наша планета"
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

def search_duckduckgo(claim, max_results=5):
    cache_key = "ddg_" + get_cache_key(claim)

    if cache_key in search_cache:
        return search_cache[cache_key]

    results = []

    try:
        with DDGS() as ddgs:
            search_results = ddgs.text(
                claim,
                region="ru-ru",
                safesearch="moderate",
                max_results=max_results
            )

            for result in search_results:
                results.append({
                    "title": result.get("title", ""),
                    "summary": clean_text(result.get("body", "")),
                    "body": clean_text(result.get("body", "")),
                    "url": result.get("href", ""),
                    "source": "DuckDuckGo"
                })

    except Exception:
        pass

    search_cache[cache_key] = results
    save_cache()

    return results

def search_wikipedia(claim, max_results=3):
    cache_key = "wiki_" + get_cache_key(claim)

    if cache_key in search_cache:
        return search_cache[cache_key]

    results = []

    queries = [claim]

    claim_lower = claim.lower()

    if "земля" in claim_lower:
        queries.append("Земля")

    if "круглая" in claim_lower or "шара" in claim_lower:
        queries.append("Форма Земли")

    if wikipedia:
        try:
            wikipedia.set_lang("ru")

            for query in queries:
                try:
                    search_results = wikipedia.search(
                        query,
                        results=max_results
                    )

                    for title in search_results:
                        try:
                            page = wikipedia.page(
                                title,
                                auto_suggest=False
                            )

                            results.append({
                                "title": page.title,
                                "summary": page.summary,
                                "body": page.summary,
                                "url": page.url,
                                "source": "Wikipedia"
                            })

                        except Exception:
                            continue

                except Exception:
                    continue

        except Exception:
            pass

    unique_results = []
    seen_urls = set()

    for result in results:
        if result["url"] not in seen_urls:
            seen_urls.add(result["url"])
            unique_results.append(result)

    search_cache[cache_key] = unique_results
    save_cache()

    return unique_results

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

    claim_words = [
        word for word in re.findall(r"[а-яёa-z]{4,}", claim.lower())
        if word not in STOP_WORDS
    ]

    claim_words = expand_keywords(claim_words)

    articles = []

    for feed in RSS_FEEDS:
        try:
            parsed_feed = feedparser.parse(feed["url"])

            for entry in parsed_feed.entries[:30]:
                title = clean_text(entry.get("title", ""))
                summary = clean_text(entry.get("summary", ""))

                text = (title + " " + summary).lower()

                matches = [
                    word for word in claim_words
                    if word in text
                ]

                if matches:
                    articles.append({
                        "title": title,
                        "summary": summary,
                        "link": entry.get("link", ""),
                        "source": feed["name"],
                        "published": entry.get("published", ""),
                        "matches": len(matches)
                    })

        except Exception:
            continue

    articles.sort(key=lambda item: item["matches"], reverse=True)

    articles = articles[:max_articles]

    news_cache[cache_key] = {
        "time": current_time,
        "articles": articles
    }

    save_cache()

    return articles

def extract_matching_sentences(claim, text):
    sentences = re.split(r"(?<=[.!?])\s+", text)

    claim_words = [
        word for word in re.findall(r"[а-яёa-z]{4,}", claim.lower())
        if word not in STOP_WORDS
    ]

    claim_words = expand_keywords(claim_words)

    matches = []

    for sentence in sentences:
        sentence_clean = sentence.strip()

        if len(sentence_clean) < 20:
            continue

        sentence_lower = sentence_clean.lower()

        matched_words = []

        for word in claim_words:
            if word in sentence_lower:
                matched_words.append(word)

        coverage = len(matched_words) / max(len(claim_words), 1)

        if coverage >= 0.4:
            status = "подтверждает"
        elif coverage >= 0.15:
            status = "косвенно подтверждает"
        elif coverage > 0:
            status = "связан с темой утверждения"
        else:
            continue

        matches.append({
            "sentence": sentence_clean,
            "status": status,
            "coverage": coverage
        })

    matches.sort(key=lambda item: item["coverage"], reverse=True)

    return matches[:2]

def estimate_support(claim, sources):
    claim_words = [
        word for word in re.findall(r"[а-яёa-z]{4,}", claim.lower())
        if word not in STOP_WORDS
    ]

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

        matches = extract_matching_sentences(claim, text)

        if matches:
            best = matches[0]

            evidence.append({
                "title": source.get("title", ""),
                "url": source.get("url", ""),
                "source": source.get("source", ""),
                "sentence": best["sentence"],
                "status": best["status"],
                "coverage": best["coverage"]
            })

            if best["status"] == "подтверждает":
                support_score += 1.0
            elif best["status"] == "косвенно подтверждает":
                support_score += 0.5
            else:
                support_score += 0.2

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
        wikipedia_results = search_wikipedia(text)
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

        if not details:
            for source in all_sources[:8]:
                sentences = re.split(
                    r"(?<=[.!?])\s+",
                    (
                        source.get("summary", "") + " " +
                        source.get("body", "")
                    ).strip()
                )

                sentence = ""

                for sentence_candidate in sentences:
                    if len(sentence_candidate.strip()) > 40:
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

        if support >= 0.7 and len(evidence) >= 2:
            verdict = "ПРАВДА"
            confidence = min(0.55 + support * 0.35, 0.92)
            reason = (
                "Найдены источники, подтверждающие или косвенно "
                "подтверждающие утверждение."
            )

        elif support <= 0.2 and len(all_sources) >= 3:
            verdict = "ФЕЙК"
            confidence = min(0.55 + (1 - support) * 0.3, 0.88)
            reason = (
                "Релевантные источники не подтверждают утверждение "
                "или противоречат его основной мысли."
            )

        else:
            verdict = "НЕИЗВЕСТНО"
            confidence = 0.5
            reason = (
                "Найдены источники по теме, но их недостаточно "
                "для надёжного подтверждения или опровержения."
            )

        return {
            "verdict": verdict,
            "confidence": confidence,
            "reason": reason,
            "details": details
        }

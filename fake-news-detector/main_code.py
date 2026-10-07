import re
import json
import time
import hashlib
import requests
import feedparser
from datetime import datetime
from duckduckgo_search import DDGS

# ===== НАСТРОЙКИ =====
RSS_CACHE_TIME = 600  # секунд, 10 минут
MAX_FEED_ARTICLES = 8
FEED_TIMEOUT = 5

STOP_WORDS = {
    "это", "того", "которые", "который", "чтобы", "очень", "такой",
    "такие", "будет", "будут", "есть", "быть", "можно", "нужно",
    "говорят", "заявляют", "утверждают", "сообщают", "новости",
    "случилось", "произошло", "сообщается", "сообщил", "заявил"
}

RSS_FEEDS = {
    "общее": [
        "https://lenta.ru/rss/news",
        "https://ria.ru/export/rss2/archive/index.xml",
        "https://tass.ru/rss/v2.xml",
        "https://www.interfax.ru/rss.asp",
        "https://www.kommersant.ru/RSS/news.xml",
    ],
    "наука": [
        "https://nplus1.ru/rss",
        "https://naked-science.ru/rss",
        "https://www.popmech.ru/rss/all.xml",
        "https://phys.org/rss-feed/",
        "https://www.sciencedaily.com/rss/all.xml",
    ],
    "ии": [
        "https://venturebeat.com/category/ai/feed/",
        "https://techcrunch.com/category/artificial-intelligence/feed/",
        "https://www.theverge.com/rss/ai-artificial-intelligence/index.xml",
        "https://habr.com/ru/rss/hubs/artificial_intelligence/?fl=ru",
    ],
    "it": [
        "https://habr.com/ru/rss/hubs/programming/?fl=ru",
        "https://habr.com/ru/rss/hubs/it/?fl=ru",
        "https://techcrunch.com/feed/",
        "https://www.theverge.com/rss/index.xml",
        "https://arstechnica.com/feed/",
    ],
    "игры": [
        "https://www.igromania.ru/rss/all.xml",
        "https://stopgame.ru/rss/news.rss",
        "https://www.ign.com/rss.xml",
        "https://www.pcgamer.com/rss/",
        "https://www.gamespot.com/feeds/news/",
    ],
    "кино": [
        "https://www.kinopoisk.ru/rss/news.xml",
        "https://www.film.ru/rss/news",
        "https://variety.com/feed/",
        "https://deadline.com/feed/",
    ],
    "музыка": [
        "https://www.billboard.com/feed/",
        "https://www.rollingstone.com/music/rss/",
        "https://pitchfork.com/rss/news/",
        "https://www.theguardian.com/music/rss",
    ],
    "спорт": [
        "https://www.sports.ru/rss/all/news.xml",
        "https://www.championat.com/rss/news.xml",
        "https://www.eurosport.com/rss.xml",
        "https://www.bbc.com/sport/rss.xml",
    ],
    "путешествия": [
        "https://www.tourister.ru/rss/news",
        "https://www.tourprom.ru/rss/news/",
        "https://www.lonelyplanet.com/rss",
        "https://www.theguardian.com/travel/rss",
    ],
    "еда": [
        "https://www.gastronom.ru/text/rss",
        "https://www.edimdoma.ru/news/rss",
        "https://www.bbcgoodfood.com/rss.xml",
        "https://www.healthline.com/nutrition/rss",
    ],
    "факты": [
        "https://nplus1.ru/rss",
        "https://naked-science.ru/rss",
        "https://www.popmech.ru/rss/all.xml",
        "https://www.bbc.com/russian/rss",
    ],
    "учёба": [
        "https://www.gazeta.ru/education/rss.xml",
        "https://tass.ru/rss/v2.xml",
        "https://www.theguardian.com/education/rss",
    ],
}

THEME_KEYWORDS = {
    "еда": ["еда", "питание", "сахар", "кофе", "еда", "диета", "витамин",
             "вакцина", "здоровье", "лекарство", "болезнь", "врач", "медицина"],
    "учёба": ["школа", "школы", "егэ", "университет", "студент", "экзамен",
               "учёба", "образование", "учитель", "домашнее задание"],
    "ии": ["ии", "искусственный интеллект", "chatgpt", "нейросеть",
            "нейросети", "робот", "роботы", "gpt", "gemini"],
    "it": ["компьютер", "программирование", "python", "windows", "google",
            "apple", "интернет", "сайт", "приложение", "технологии", "хакер"],
    "игры": ["игра", "игры", "minecraft", "gta", "fortnite", "roblox",
              "counter-strike", "киберспорт", "консоль", "steam"],
    "кино": ["фильм", "фильмы", "кино", "сериал", "режиссёр", "актёр",
              "оскар", "киностудия", "премьера"],
    "музыка": ["музыка", "песня", "альбом", "концерт", "spotify",
                "артист", "певец", "гитара", "фестиваль"],
    "спорт": ["футбол", "спорт", "олимпиада", "матч", "чемпионат",
               "спортсмен", "хоккей", "баскетбол", "тренер", "рекорд"],
    "путешествия": ["путешествие", "туризм", "виза", "самолёт", "аэропорт",
                     "отель", "паспорт", "граница", "поездка", "отпуск"],
    "факты": ["наука", "учёные", "исследование", "земля", "космос",
               "луна", "солнце", "днк", "физика", "химия", "биология"],
}

# ===== КЭШ RSS =====
rss_cache = {}


def get_cache_key(text):
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def get_cached_rss(theme):
    now = time.time()
    cached = rss_cache.get(theme)

    if cached and now - cached["time"] < RSS_CACHE_TIME:
        return cached["articles"]

    return None


def save_rss_cache(theme, articles):
    rss_cache[theme] = {
        "time": time.time(),
        "articles": articles
    }


# ===== ОПРЕДЕЛЕНИЕ ТЕМЫ =====
def detect_theme(text, forced_theme="авто"):
    if forced_theme != "авто":
        return forced_theme

    text_lower = text.lower()
    scores = {}

    for theme, keywords in THEME_KEYWORDS.items():
        score = 0

        for keyword in keywords:
            if keyword in text_lower:
                score += 1

        if score > 0:
            scores[theme] = score

    if not scores:
        return "общее"

    return max(scores, key=scores.get)


# ===== ПОЛУЧЕНИЕ СТАТЕЙ ИЗ RSS =====
def fetch_feed(feed_url):
    try:
        response = requests_get(feed_url)
        feed = feedparser.parse(response.content)

        articles = []

        for entry in feed.entries[:30]:
            articles.append({
                "title": entry.get("title", ""),
                "summary": entry.get("summary", ""),
                "link": entry.get("link", ""),
                "source": feed.feed.get("title", feed_url)
            })

        return articles

    except Exception:
        return []


def requests_get(url):
    import requests

    return requests.get(url, timeout=FEED_TIMEOUT)


def get_rss_articles(theme):
    cached = get_cached_rss(theme)

    if cached is not None:
        return cached

    feeds = RSS_FEEDS.get(theme, RSS_FEEDS["общее"])[:5]
    articles = []

    for feed_url in feeds:
        articles.extend(fetch_feed(feed_url))

    save_rss_cache(theme, articles)

    return articles


def get_relevant_articles(claim, theme, max_articles=MAX_FEED_ARTICLES):
    articles = get_rss_articles(theme)

    keywords = [
        word for word in re.findall(r"[а-яёa-z]{4,}", claim.lower())
        if word not in STOP_WORDS
    ][:6]

    relevant = []

    for article in articles:
        text = (article["title"] + " " + article["summary"]).lower()
        score = sum(1 for keyword in keywords if keyword in text)

        if score > 0:
            article["score"] = score
            relevant.append(article)

    relevant.sort(key=lambda item: item["score"], reverse=True)

    return relevant[:max_articles]


# ===== ПОИСК DUCKDUCKGO =====
def search_duckduckgo(claim, max_results=5):
    results = []

    try:
        with DDGS() as ddgs:
            search_results = ddgs.text(
                claim,
                max_results=max_results
            )

            for result in search_results:
                results.append({
                    "title": result.get("title", ""),
                    "body": result.get("body", ""),
                    "href": result.get("href", "")
                })

    except Exception:
        pass

    return results


# ===== ПОИСК WIKIPEDIA =====
def search_wikipedia(claim, max_results=3):
    results = []

    try:
        import wikipedia

        search_results = wikipedia.search(claim, results=max_results)

        for title in search_results:
            try:
                page = wikipedia.page(title, auto_suggest=False)

                results.append({
                    "title": page.title,
                    "summary": page.summary[:700],
                    "url": page.url
                })

            except Exception:
                continue

    except Exception:
        pass

    return results


# ===== ОЦЕНКА ПОДТВЕРЖДЕНИЯ =====
def estimate_support(claim, sources):
    claim_words = [
        word for word in re.findall(r"[а-яёa-z]{4,}", claim.lower())
        if word not in STOP_WORDS
    ]

    if not claim_words:
        return 0.0

    supported = 0

    for source in sources:
        text = (
            source.get("title", "") + " " +
            source.get("summary", "") + " " +
            source.get("body", "")
        ).lower()

        matches = sum(1 for word in claim_words if word in text)
        supported += matches

    return min(supported / max(len(claim_words), 1), 1.0)


# ===== ГЛАВНЫЙ КЛАСС =====
class FactChecker:
    def verify(self, text, theme="авто"):
        theme = detect_theme(text, theme)

        duckduckgo_results = search_duckduckgo(text)
        wikipedia_results = search_wikipedia(text)
        rss_results = get_relevant_articles(text, theme)

        all_sources = []

        for result in duckduckgo_results:
            all_sources.append({
                "title": result["title"],
                "summary": result["body"],
                "url": result["href"],
                "source": "DuckDuckGo"
            })

        for result in wikipedia_results:
            all_sources.append({
                "title": result["title"],
                "summary": result["summary"],
                "url": result["url"],
                "source": "Wikipedia"
            })

        for result in rss_results:
            all_sources.append({
                "title": result["title"],
                "summary": result["summary"],
                "url": result["link"],
                "source": result["source"]
            })

        support = estimate_support(text, all_sources)

        details = []

        for source in all_sources[:8]:
            details.append({
                "fact": source["title"],
                "source": source["source"],
                "reason": "Найден потенциально релевантный источник.",
                "links": [source["url"]] if source["url"] else [],
                "result": None
            })

        if support >= 0.7 and len(all_sources) >= 3:
            verdict = "ПРАВДА"
            confidence = min(0.55 + support * 0.35, 0.92)
            reason = (
                "Найдено несколько релевантных источников, "
                "подтверждающих основные слова утверждения."
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
                "Найденных источников недостаточно или они лишь косвенно "
                "связаны с утверждением."
            )

        return {
            "verdict": verdict,
            "confidence": confidence,
            "reason": reason,
            "theme": theme,
            "details": details
        }

import streamlit as st
import requests
import feedparser
import re
import json
import os
import time
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

CACHE_FILE = os.path.join(
    os.path.dirname(__file__),
    "cache.json"
)
FACTCHECKER_VERSION = "v4-nli-0.60"

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
    return re.sub(
        r"[^a-zа-яё0-9]+",
        "_",
        text.lower()
    ).strip("_")


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
    "состоялся", "состоялась", "состоялось"
}

SYNONYMS = {
    "круглая": [
        "круглая", "шарообразная", "сферическая",
        "геоид", "эллипсоид"
    ],
    "шара": [
        "шарообразная", "круглая", "сферическая",
        "геоид", "эллипсоид"
    ],
    "вреден": [
        "вредна", "вредно", "опасен", "опасна", "опасно"
    ],
    "опасен": [
        "вреден", "вредна", "вредно", "опасна", "опасно"
    ],
    "вызывает": [
        "приводит", "связан", "связана", "влияет"
    ],
    "лечит": [
        "помогает", "эффективен", "эффективна"
    ],
    "запретят": [
        "запрет", "запрещён", "запрещено"
    ],
    "отменят": [
        "отмена", "отменён", "отменено"
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
    "ria.ru": 0.7
}

CONTRADICTION_WORDS = [
    "не является", "не имеет", "не соответствует",
    "опровергает", "опровергнуто", "ложно", "ложная",
    "фейк", "миф", "неверно", "неправда", "дезинформация"
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


NLI_LOAD_ERROR = None


@st.cache_resource
def load_nli_model():
    global NLI_LOAD_ERROR

    if not TRANSFORMERS_AVAILABLE:
        NLI_LOAD_ERROR = "transformers не установлен"
        return None

    try:
        model = pipeline(
            "text-classification",
            model="MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
            truncation=True
        )

        NLI_LOAD_ERROR = None
        return model

    except Exception as e:
        NLI_LOAD_ERROR = str(e)
        print("NLI LOAD ERROR:", e)
        return None


nli_model = load_nli_model()


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


def get_base_words(text):
    words = re.findall(r"[а-яёa-z]{4,}", text.lower())

    words = [
        word for word in words
        if word not in STOP_WORDS
    ]

    return list(dict.fromkeys(words))


def get_claim_word_groups(text):
    words = get_base_words(text)

    groups = []

    for word in words:
        group = [normalize_word(word)]

        if word in SYNONYMS:
            for synonym in SYNONYMS[word]:
                group.append(normalize_word(synonym))

        groups.append(list(dict.fromkeys(group)))

    return groups


def get_claim_words(text):
    words = get_base_words(text)

    expanded = []

    for word in words:
        expanded.append(word)

        if word in SYNONYMS:
            expanded.extend(SYNONYMS[word])

    normalized = [
        normalize_word(word)
        for word in expanded
    ]

    return list(dict.fromkeys(normalized))


def split_into_claims(text):
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
                pieces = re.split(
                    pattern,
                    part,
                    flags=re.IGNORECASE
                )

                for piece in pieces:
                    piece = piece.strip()

                    if piece:
                        new_parts.append(piece)

            parts = new_parts

        for part in parts:
            part = re.sub(
                r"^(сообщается|заявляется|утверждается|по данным|по словам)\s+(что|о том, что)?\s*,?\s*",
                "",
                part,
                flags=re.IGNORECASE
            )

            part = part.strip(" .,;:!?")

            if len(part.split()) >= 3:
                claims.append(part)

    unique_claims = []

    for claim in claims:
        if claim.lower() not in [
            item.lower() for item in unique_claims
        ]:
            unique_claims.append(claim)

    return unique_claims


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

    words = get_base_words(claim)

    if len(words) >= 2:
        queries.append(" ".join(words))

    if len(words) >= 3:
        queries.append(" ".join(words[:4]))

    factcheck_queries = [
        claim + " миф",
        claim + " опровергнут",
        claim + " фейк",
        claim + " дезинформация"
    ]

    queries.extend(factcheck_queries)

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
            "script", "style", "nav", "header",
            "footer", "aside", "form", "button"
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
                        max_results=3
                    )

                    for result in web_results:
                        url = result.get("href", "")

                        if url and url not in seen_urls:
                            seen_urls.add(url)

                            results.append({
                                "title": result.get("title", ""),
                                "summary": clean_text(
                                    result.get("body", "")
                                ),
                                "body": clean_text(
                                    result.get("body", "")
                                ),
                                "url": url,
                                "source": "DuckDuckGo"
                            })

                    time.sleep(0.5)

                except Exception:
                    continue

    except Exception:
        pass

    if results:
        search_cache[cache_key] = results[:max_results]
        save_cache()

    return results[:max_results]


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

        response = requests.get(
            api_url,
            params=params,
            timeout=15,
            headers={
                "User-Agent": "FakeNewsDetector/1.0"
            }
        )

        if response.status_code != 200:
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

    if cache_key in search_cache:
        return search_cache[cache_key]

    results = []
    seen_titles = set()

    queries = [claim]

    words = get_base_words(claim)

    if words:
        queries.append(" ".join(words[:5]))

    queries = list(dict.fromkeys(queries))

    for query in queries:
        try:
            api_url = "https://ru.wikipedia.org/w/api.php"

            params = {
                "action": "query",
                "format": "json",
                "list": "search",
                "srsearch": query,
                "srlimit": 5,
                "srprop": "snippet",
                "utf8": 1
            }

            response = requests.get(
                api_url,
                params=params,
                timeout=15,
                headers={
                    "User-Agent": "FakeNewsDetector/1.0"
                }
            )

            if response.status_code != 200:
                continue

            data = response.json()

            pages = data.get("query", {}).get("search", [])

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

            time.sleep(0.3)

        except Exception:
            continue

    unique_results = []
    seen_urls = set()

    for result in results:
        if result["url"] not in seen_urls:
            seen_urls.add(result["url"])
            unique_results.append(result)

    if unique_results:
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

    claim_groups = get_claim_word_groups(claim)

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

                matched_groups = 0

                for group in claim_groups:
                    if any(
                        word in text_stems
                        for word in group
                    ):
                        matched_groups += 1

                coverage = matched_groups / max(
                    len(claim_groups), 1
                )

                if matched_groups >= 2 and coverage >= 0.30:
                    articles.append({
                        "title": title,
                        "summary": summary,
                        "link": entry.get("link", ""),
                        "source": feed["name"],
                        "published": entry.get("published", ""),
                        "matches": matched_groups,
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

    if articles:
        news_cache[cache_key] = {
            "time": current_time,
            "articles": articles
        }

        save_cache()

    return articles


def normalize_number(text):
    text = text.lower().replace(",", ".")

    return re.findall(r"\d+(?:\.\d+)?", text)


def get_direction(text):
    text_lower = text.lower()

    has_growth = any(
        word in text_lower
        for word in GROWTH_WORDS
    )

    has_decline = any(
        word in text_lower
        for word in DECLINE_WORDS
    )

    if has_growth and not has_decline:
        return "рост"

    if has_decline and not has_growth:
        return "снижение"

    return None


def check_numeric_relation(claim, sentence):
    claim_numbers = normalize_number(claim)
    sentence_numbers = normalize_number(sentence)

    if not claim_numbers or not sentence_numbers:
        return None

    common_numbers = set(claim_numbers) & set(sentence_numbers)

    if not common_numbers:
        return None

    claim_direction = get_direction(claim)
    sentence_direction = get_direction(sentence)

    if not claim_direction or not sentence_direction:
        return None

    if claim_direction != sentence_direction:
        return "противоречит"

    return "подтверждает"


def check_sentence_with_nli(claim, sentence):
    if nli_model is None:
        return None, 0.0

    try:
        result = nli_model(
            {
                "text": sentence,
                "text_pair": claim
            }
        )

        if isinstance(result, list):
            result = result[0]

        label = result.get("label", "").lower()
        score = float(result.get("score", 0))

        print(
            "NLI DEBUG:",
            "claim =", claim,
            "| sentence =", sentence[:100],
            "| label =", label,
            "| score =", score
        )

        if label == "entailment" and score >= 0.50:
            return "подтверждает", score

        if label == "contradiction" and score >= 0.70:
            return "противоречит", score

        return "нейтрально", score

    except Exception as e:
        print("NLI ERROR:", e)

        return None, 0.0


def highlight_matched_words(sentence, claim_groups):
    sentence_words = re.findall(
        r"[а-яёa-z0-9.,%]+",
        sentence.lower()
    )

    highlighted = sentence

    for group in claim_groups:
        group_stems = set(group)

        for word in sentence_words:
            word_clean = word.strip(".,;:!?()\"'")

            if not word_clean:
                continue

            stem = normalize_word(word_clean)

            if stem in group_stems:
                pattern = re.compile(
                    r"(?<![\w])"
                    + re.escape(word_clean)
                    + r"(?![\w])",
                    re.IGNORECASE
                )

                highlighted = pattern.sub(
                    f"**{word_clean}**",
                    highlighted,
                    count=1
                )

    return highlighted


def find_best_sentences(claim, text, max_sentences=3):
    sentences = re.split(r"(?<=[.!?])\s+", text)

    claim_groups = get_claim_word_groups(claim)

    matches = []

    for sentence in sentences:
        sentence_clean = sentence.strip()

        if len(sentence_clean) < 20:
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

        matched_groups = 0
        matched_words = []

        for group in claim_groups:
            found = any(
                word in sentence_stems
                for word in group
            )

            if found:
                matched_groups += 1
                matched_words.append(group[0])

        coverage = matched_groups / max(
            len(claim_groups), 1
        )

        if matched_groups < 1:
            continue

        numeric_check = check_numeric_relation(
            claim,
            sentence_clean
        )

        nli_status, nli_score = check_sentence_with_nli(
            claim,
            sentence_clean
        )

        has_contradiction = any(
            word in sentence_lower
            for word in CONTRADICTION_WORDS
        )

        if numeric_check == "противоречит":
            status = "противоречит"

        elif nli_status == "противоречит" and nli_score >= 0.65:
            status = "противоречит"

        elif numeric_check == "подтверждает":
            status = "подтверждает"

        elif nli_status == "подтверждает" and nli_score >= 0.60:
            status = "подтверждает"

        elif has_contradiction:
            status = "противоречит"

        elif matched_groups >= 2 and coverage >= 0.35:
            status = "косвенно подтверждает"

        elif matched_groups >= 1 and coverage >= 0.15:
            status = "косвенно подтверждает"

        else:
            status = "связан с темой утверждения"

        highlighted_sentence = highlight_matched_words(
            sentence_clean,
            claim_groups
        )

        matches.append({
            "sentence": sentence_clean,
            "highlighted_sentence": highlighted_sentence,
            "status": status,
            "coverage": coverage,
            "matched_groups": matched_groups,
            "matched_words": matched_words,
            "numeric_check": numeric_check,
            "nli_status": nli_status,
            "nli_score": nli_score
        })

    matches.sort(
        key=lambda item: (
            item["matched_groups"],
            item["coverage"]
        ),
        reverse=True
    )

    return matches[:max_sentences]


def estimate_support(claim, sources):
    claim_groups = get_claim_word_groups(claim)

    if not claim_groups:
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
            best_match = sentence_matches[0]

            trust = get_domain_trust(
                source.get("url", "")
            )

            final_score = (
                best_match["coverage"] * 0.7 +
                trust * 0.3
            )

            evidence.append({
                "title": source.get("title", ""),
                "url": source.get("url", ""),
                "source": source.get("source", ""),
                "quotes": sentence_matches,
                "coverage": best_match["coverage"],
                "matched_groups": best_match["matched_groups"],
                "matched_words": best_match["matched_words"],
                "trust": trust,
                "score": final_score
            })

            for match in sentence_matches:
                if match["status"] == "подтверждает":
                    support_score += 1.0
                elif match["status"] == "косвенно подтверждает":
                    support_score += 0.5
                elif match["status"] == "противоречит":
                    support_score -= 1.0

    support_score = min(
        max(
            support_score / max(len(claim_groups), 1),
            0.0
        ),
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
        claims = split_into_claims(text)

        if not claims:
            claims = [clean_text(text)]

        claims = claims[:3]

        results = []

        for claim in claims:
            results.append(
                self._verify_single_claim(claim)
            )

        return self._combine_results(results)

    def _verify_single_claim(self, text):
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

        support, evidence = estimate_support(
            text,
            all_sources
        )

        details = []

        for item in evidence[:8]:
            quotes = item["quotes"]

            quote_texts = []

            for quote in quotes[:2]:
                sentence_for_output = quote.get(
                    "highlighted_sentence",
                    quote["sentence"]
                )

                quote_texts.append(
                    f"«{sentence_for_output}» "
                    f"— источник {quote['status']} утверждению."
                )

            has_confirmation = any(
                quote["status"] == "подтверждает"
                for quote in quotes
            )

            has_partial_confirmation = any(
                quote["status"] == "косвенно подтверждает"
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
            elif has_partial_confirmation:
                result = None
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

        claim_groups = get_claim_word_groups(text)

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

            matched_groups = 0

            for group in claim_groups:
                if any(
                    word in source_stems
                    for word in group
                ):
                    matched_groups += 1

            coverage = matched_groups / max(
                len(claim_groups), 1
            )

            if matched_groups >= 1 and coverage >= 0.15:
                relevant_sources.append({
                    "source": source,
                    "coverage": coverage,
                    "matched_groups": matched_groups
                })

        relevant_sources.sort(
            key=lambda item: (
                item["matched_groups"],
                item["coverage"]
            ),
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
                    if len(sentence_candidate.strip()) > 20:
                        sentence = sentence_candidate.strip()
                        break

                if not sentence:
                    sentence = source.get("title", "")

                highlighted_sentence = highlight_matched_words(
                    sentence,
                    claim_groups
                )

                details.append({
                    "fact": source.get("title", ""),
                    "source": source.get("source", ""),
                    "reason": (
                        f"Источник связан с темой утверждения: "
                        f"«{highlighted_sentence}»"
                    ),
                    "links": (
                        [source.get("url", "")]
                        if source.get("url") else []
                    ),
                    "result": None
                })

        confirming_sources = [
            item for item in evidence
            if any(
                quote["status"] == "подтверждает"
                for quote in item["quotes"]
            )
        ]

        partially_confirming_sources = [
            item for item in evidence
            if any(
                quote["status"] == "косвенно подтверждает"
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

        numeric_contradictions = [
            item for item in evidence
            if any(
                quote.get("numeric_check") == "противоречит"
                for quote in item["quotes"]
            )
        ]

        strong_contradictions = [
            item for item in contradicting_sources
            if item.get("trust", 0.5) >= 0.8
        ]

        if numeric_contradictions:
            verdict = "ФЕЙК"
            confidence = min(
                0.78 + len(numeric_contradictions) * 0.05,
                0.93
            )
            reason = (
                "Найден релевантный источник с той же числовой "
                "информацией, но противоположным направлением."
            )

        elif strong_contradictions:
            verdict = "ФЕЙК"
            confidence = min(
                0.72 + len(strong_contradictions) * 0.05,
                0.9
            )
            reason = (
                "Найден надёжный источник, который противоречит "
                "утверждению."
            )

        elif contradicting_sources:
            verdict = "НЕИЗВЕСТНО"
            confidence = 0.55
            reason = (
                "Найдены возможные противоречия, но их недостаточно "
                "для надёжного вывода о фейке."
            )

        elif confirming_sources:
            verdict = "ПРАВДА"

            if len(confirming_sources) >= 2:
                confidence = max(
                    0.72,
                    min(
                        0.72 + support * 0.2,
                        0.92
                    )
                )
            else:
                confidence = 0.68

            reason = (
                "Найден релевантный источник, который подтверждает "
                "утверждение по смыслу."
            )

        elif partially_confirming_sources:
            verdict = "НЕИЗВЕСТНО"
            confidence = 0.55
            reason = (
                "Найдены источники по теме, но они не подтверждают "
                "утверждение по смыслу."
            )

        elif relevant_sources:
            verdict = "НЕИЗВЕСТНО"
            confidence = 0.5
            reason = (
                "Найдены источники по теме, но они не подтверждают "
                "и не опровергают утверждение."
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

    def _combine_results(self, results):
        verdicts = [
            result["verdict"] for result in results
        ]

        if "ФЕЙК" in verdicts:
            final_verdict = "ФЕЙК"
            reason = (
                "В утверждении найдена часть, которая противоречит "
                "источникам."
            )

        elif "ПРАВДА" in verdicts and "НЕИЗВЕСТНО" in verdicts:
            final_verdict = "НЕИЗВЕСТНО"
            reason = (
                "Часть утверждений подтверждена, но не все удалось "
                "надёжно проверить."
            )

        elif verdicts and all(
            verdict == "ПРАВДА" for verdict in verdicts
        ):
            final_verdict = "ПРАВДА"
            reason = (
                "Все смысловые части утверждения подтверждены "
                "источниками."
            )

        else:
            final_verdict = "НЕИЗВЕСТНО"
            reason = (
                "Не удалось найти достаточно надёжных источников "
                "для проверки."
            )

        all_details = []

        for result in results:
            all_details.extend(
                result.get("details", [])
            )

        confidence = 0.0

        if results:
            confidence = sum(
                result.get("confidence", 0)
                for result in results
            ) / len(results)

        return {
            "verdict": final_verdict,
            "confidence": confidence,
            "reason": reason,
            "details": all_details
        }

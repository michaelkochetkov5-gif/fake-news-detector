from ddgs import DDGS
import wikipedia
import re
import feedparser
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ===== RSS ЛЕНТЫ =====
RSS_FEEDS = {
    'lenta': 'https://lenta.ru/rss/news',
    'ria': 'https://ria.ru/export/rss2/news/index.xml',
    'tass': 'https://tass.ru/rss/v2.xml',
    'meduza': 'https://meduza.io/rss/all',
    'kommersant': 'https://www.kommersant.ru/RSS/news.xml',
    'rbc': 'https://www.rbc.ru/rss/news.rss',
    'vedomosti': 'https://www.vedomosti.ru/rss/news',
    'interfax': 'https://www.interfax.ru/rss.asp',
    'gazeta': 'https://www.gazeta.ru/export/rss/lenta.xml',
    'izvestia': 'https://iz.ru/rss',
    'rg': 'https://rg.ru/rss.xml',
    'aif': 'https://aif.ru/rss/all',
    'mk': 'https://www.mk.ru/rss/news/index.xml',
    'kp': 'https://www.kp.ru/rss/all.xml',

    'eda_ru': 'https://eda.ru/rss',
    'povarenok': 'https://www.povarenok.ru/rss/',
    'gastronom': 'https://www.gastronom.ru/xml/rss.xml',

    'postupi_online': 'https://postupi.online/news/rss/',
    'hse': 'https://www.hse.ru/news/rss/',
    'mel': 'https://mel.fm/rss',

    'habr': 'https://habr.com/ru/rss/',
    'vc_ru': 'https://vc.ru/rss',
    'tproger': 'https://tproger.ru/feed/',
    'xakep': 'https://xakep.ru/feed/',
    'opennet': 'https://www.opennet.ru/opennews/opennews_all.rss',

    'nplus1': 'https://nplus1.ru/rss',
    'elementy': 'https://elementy.ru/rss',
    'indicator': 'https://indicator.ru/rss.xml',
    'scientificrussia': 'https://scientificrussia.ru/feed',
    'naked-science': 'https://naked-science.ru/rss',
    'popmech': 'https://www.popmech.ru/rss/all.xml',

    'dtf': 'https://dtf.ru/rss',
    'kanobu': 'https://kanobu.ru/rss/',
    'igromania': 'https://www.igromania.ru/rss/',
    'stopgame': 'https://stopgame.ru/rss/data.rss',
    'cybersport': 'https://www.cybersport.ru/rss/news.rss',

    'sports': 'https://www.sports.ru/rss/',
    'matchtv': 'https://matchtv.ru/rss',
    'championat': 'https://www.championat.com/rss/news.xml',
    'sovsport': 'https://www.sovsport.ru/rss/all.xml',

    'vokrugsveta': 'https://www.vokrugsveta.ru/rss/',
    'natgeo': 'https://www.national-geographic.ru/rss/',
    'tonkosti': 'https://tonkosti.ru/rss',
    'tourister': 'https://www.tourister.ru/rss/news',

    'kinopoisk': 'https://www.kinopoisk.ru/rss/news/',
    'film_ru': 'https://www.film.ru/rss/',
    'afisha': 'https://www.afisha.ru/rss/news/',

    'factroom': 'https://factroom.ru/feed',
    'fishki': 'https://fishki.net/rss',
    'adme': 'https://www.adme.ru/rss/',
    'lifehacker': 'https://lifehacker.ru/feed/',
}

# ===== КЛЮЧЕВЫЕ СЛОВА ДЛЯ ТЕМ =====
THEME_KEYWORDS = {
    'еда': [
        'рецепт', 'еда', 'продукт', 'питание', 'диета', 'кулинар', 'вкус', 'блюдо',
        'калор', 'белок', 'жир', 'углевод', 'витами', 'минерал', 'полезн', 'вредн',
        'сахар', 'соль', 'мясо', 'рыба', 'овощ', 'фрукт', 'молок', 'хлеб', 'вода',
        'ресторан', 'кафе', 'доставк', 'меню', 'завтрак', 'обед', 'ужин'
    ],
    'учёба': [
        'егэ', 'экзамен', 'университет', 'курс', 'образование', 'школа', 'студент',
        'учеб', 'лекци', 'семинар', 'диплом', 'диссертац', 'наука', 'исследован',
        'знани', 'умени', 'навык', 'тренинг', 'сертификат', 'степень',
        'бакалавр', 'магистр', 'аспирант', 'доцент', 'профессор', 'преподават',
        'олимпиада', 'конкурс', 'грант', 'стипендия', 'бюджет', 'платн'
    ],
    'ии': [
        'нейросеть', 'chatgpt', 'искусственный интеллект', 'ai', 'алгоритм',
        'машинное обучение', 'deep learning', 'neural network', 'трансформер',
        'генеративн', 'gpt', 'языковая модель', 'llm', 'бот', 'чат',
        'автоматизац', 'робот', 'компьютерное зрение', 'nlp', 'обработка текста',
        'данные', 'big data', 'аналитика', 'предсказан', 'классификац',
        'технолог', 'инноваци', 'стартап', 'цифров', 'виртуальн', 'дополненн'
    ],
    'наука': [
        'наука', 'исследован', 'открыти', 'учён', 'лаборатор', 'эксперимент',
        'физика', 'химия', 'биология', 'астроном', 'космос', 'планета', 'звезда',
        'ген', 'днк', 'клетка', 'вирус', 'бактерия', 'эволюция', 'вид', 'организм',
        'энергия', 'атом', 'молекула', 'частица', 'волна', 'излучение', 'поле'
    ],
    'it': [
        'программ', 'код', 'разработк', 'язык программирован', 'python', 'java',
        'сайт', 'приложение', 'софт', 'браузер', 'операционная система', 'windows',
        'сервер', 'база данных', 'api', 'фреймворк', 'библиотека', 'github',
        'кибербезопасност', 'вирус', 'взлом', 'пароль', 'шифрование'
    ],
    'игры': [
        'игра', 'гейм', 'игрок', 'прохождени', 'уровень', 'босс', 'квест',
        'playstation', 'xbox', 'nintendo', 'pc', 'steam', 'epic games',
        'киберспорт', 'турнир', 'команда', 'чемпионат', 'призовые',
        'minecraft', 'roblox', 'fortnite', 'gta', 'dota', 'cs'
    ],
    'кино': [
        'фильм', 'кино', 'сериал', 'актёр', 'режиссёр', 'премьера', 'кинопоиск',
        'оскар', 'премия', 'блокбастер', 'триллер', 'комедия', 'драма', 'ужасы',
        'марвел', 'dc', 'киновселенная', 'экранизация', 'адаптац'
    ],
    'музыка': [
        'музыка', 'альбом', 'трек', 'песня', 'исполнитель', 'группа', 'концерт',
        'клип', 'сингл', 'чарт', 'хит', 'премьера', 'фестиваль', 'тур',
        'рэп', 'хип-хоп', 'поп', 'рок', 'электронная', 'классическая', 'джаз'
    ],
    'спорт': [
        'спорт', 'футбол', 'хоккей', 'баскетбол', 'теннис', 'бокс', 'mma',
        'чемпионат', 'турнир', 'лига', 'кубок', 'медаль', 'золото', 'рекорд',
        'команда', 'клуб', 'тренер', 'матч', 'игра', 'счёт', 'гол', 'победа'
    ],
    'путешествия': [
        'путешеств', 'туризм', 'отдых', 'отель', 'билет', 'авиа', 'поезд',
        'страна', 'город', 'курорт', 'пляж', 'море', 'горы', 'экскурсия',
        'виза', 'паспорт', 'таможня', 'маршрут', 'путеводитель', 'достопримечательност'
    ],
    'факты': [
        'интересн', 'удивительн', 'необычн', 'поража', 'шокирующ',
        'факт', 'правда', 'оказывается', 'знаете ли вы', 'мало кто знает',
        'топ', 'лучш', 'сам', 'рекорд', 'перв', 'последн'
    ],
}

# ===== БАЗА ФАКТОВ =====
QUICK_FACTS = {
    'сахар вызывает зависимость сильнее кокаина': False,
    'яблоки полезны для сердца': True,
    'в макдоналдсе используют мясо червей': False,
    'вода помогает похудеть': True,
    'глютен вреден всем': False,
    'егэ отменят в 2026': False,
    'мгу входит в топ 100 университетов': True,
    'ночью информация усваивается лучше': False,
    'chatgpt заменит всех программистов': False,
    'нейросети уже сознательные': False,
    'ии не может чувствовать эмоции': True,
    'gpt-4 понимает русский язык': True,
    'minecraft самая продаваемая игра': True,
    'киберспорт на олимпиаде': False,
    'земля круглая': True,
    'земля плоская': False,
}

STOP_WORDS = {'и', 'в', 'на', 'с', 'по', 'за', 'из', 'у', 'к', 'о', 'а', 'но',
              'или', 'это', 'как', 'что', 'не', 'был', 'была', 'быть', 'есть'}


class FactChecker:
    def __init__(self):
        wikipedia.set_lang("ru")

        # ===== ЛОКАЛЬНАЯ ML-МОДЕЛЬ =====
        self.model_path = "lastikfff/fake-news-detector"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
        self.model.eval()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def detect_theme(self, text):
        """Определяет тему по ключевым словам."""
        text_lower = text.lower()
        scores = {}

        for theme, keywords in THEME_KEYWORDS.items():
            score = 0

            for kw in keywords:
                if len(kw) < 4:
                    pattern = rf'\b{re.escape(kw)}\b'
                else:
                    pattern = rf'\b{re.escape(kw)}'

                if re.search(pattern, text_lower):
                    score += 1

            scores[theme] = score

        best_theme = max(scores, key=scores.get)

        if scores[best_theme] > 0:
            return best_theme

        return 'общее'

    def check_quick_facts(self, text):
        text_lower = text.lower()

        for fact, is_true in QUICK_FACTS.items():
            if fact in text_lower:
                if is_true:
                    return True, f"✓ Подтверждено: {fact}"
                else:
                    return False, f"✗ ФЕЙК: {fact}"

        return None, "Нет в базе быстрых фактов"

    def check_model(self, text):
        """Проверка текста локальной моделью.

        Метки модели:
        0 — фейк
        1 — правда
        """
        try:
            inputs = self.tokenizer(
                text,
                truncation=True,
                max_length=256,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)[0]

            fake_probability = float(probs[0])
            real_probability = float(probs[1])

            if fake_probability > real_probability:
                prediction = "ФЕЙК"
            else:
                prediction = "ПРАВДА"

            return {
                "fake_probability": fake_probability,
                "real_probability": real_probability,
                "prediction": prediction,
                "confidence": max(fake_probability, real_probability)
            }

        except Exception as e:
            return {
                "fake_probability": 0.5,
                "real_probability": 0.5,
                "prediction": "НЕИЗВЕСТНО",
                "confidence": 0.5,
                "error": str(e)
            }

    def extract_entities(self, text):
        entities = {
            'persons': [], 'positions': [], 'countries': [],
            'events': [], 'organizations': []
        }

        text_norm = text.strip()
        if text_norm:
            text_norm = text_norm[0].upper() + text_norm[1:]

        pattern1 = r'([А-Яа-яЁё]+)\s+(президент|министр|глава|канцлер|директор)\s+([А-Яа-яЁё]+)'
        matches1 = re.findall(pattern1, text_norm, re.IGNORECASE)
        for match in matches1:
            entities['persons'].append(match[0])
            entities['positions'].append(match[1])
            entities['countries'].append(match[2])

        pattern2 = r'(взрыв|стрельба|атака|произошло|случилось)\s+(в|на)\s+([А-Я][а-я]+)'
        matches2 = re.findall(pattern2, text, re.IGNORECASE)
        for match in matches2:
            entities['events'].append(match[0])
            entities['countries'].append(match[2])

        pattern3 = r'(компания|университет|школа|институт|организация)\s+([А-Я][а-я]+)'
        matches3 = re.findall(pattern3, text, re.IGNORECASE)
        for match in matches3:
            entities['organizations'].append(match[1])

        return entities

    def get_rss_for_theme(self, theme):
        if theme == 'еда':
            return {k: v for k, v in RSS_FEEDS.items() if k in ['eda_ru', 'povarenok', 'gastronom']}
        elif theme == 'учёба':
            return {k: v for k, v in RSS_FEEDS.items() if k in ['postupi_online', 'hse', 'mel']}
        elif theme == 'ии' or theme == 'it':
            return {k: v for k, v in RSS_FEEDS.items() if k in ['habr', 'vc_ru', 'tproger', 'xakep', 'opennet']}
        elif theme == 'наука':
            return {k: v for k, v in RSS_FEEDS.items() if k in ['nplus1', 'elementy', 'indicator', 'scientificrussia', 'naked-science', 'popmech']}
        elif theme == 'игры':
            return {k: v for k, v in RSS_FEEDS.items() if k in ['dtf', 'kanobu', 'igromania', 'stopgame', 'cybersport']}
        elif theme == 'спорт':
            return {k: v for k, v in RSS_FEEDS.items() if k in ['sports', 'matchtv', 'championat', 'sovsport']}
        elif theme == 'путешествия':
            return {k: v for k, v in RSS_FEEDS.items() if k in ['vokrugsveta', 'natgeo', 'tonkosti', 'tourister']}
        elif theme == 'кино':
            return {k: v for k, v in RSS_FEEDS.items() if k in ['kinopoisk', 'film_ru', 'afisha']}
        elif theme == 'факты':
            return {k: v for k, v in RSS_FEEDS.items() if k in ['factroom', 'fishki', 'adme', 'lifehacker']}
        else:
            return RSS_FEEDS

    def search_rss_news(self, query, theme='общее'):
        try:
            feeds = self.get_rss_for_theme(theme)
            all_matches = []

            words = [
                w for w in re.findall(r"[а-яёa-z0-9]+", str(query).lower())
                if len(w) >= 5
            ]
            words = list(dict.fromkeys(words))[:8]

            if not words:
                return None, "Не удалось выделить ключевые слова для поиска", []

            for source, url in feeds.items():
                try:
                    feed = feedparser.parse(url, request_headers={
                        "User-Agent": "Mozilla/5.0"
                    })

                    for entry in feed.entries[:30]:
                        title = str(entry.get("title", "")).lower()
                        summary = str(entry.get("summary", "")).lower()
                        combined = title + " " + summary

                        matches = sum(1 for word in words if word in combined)

                        if matches >= 2:
                            all_matches.append({
                                "source": source,
                                "title": entry.get("title", ""),
                                "link": entry.get("link", ""),
                                "published": entry.get("published", "N/A"),
                                "matches": matches,
                            })
                except Exception:
                    continue

            all_matches.sort(key=lambda x: x["matches"], reverse=True)

            if len(all_matches) >= 2:
                return True, f"✓ Найдено {len(all_matches)} релевантных материалов", all_matches[:5]
            if len(all_matches) == 1:
                return None, "? Найден 1 релевантный материал", all_matches
            return False, "✗ Релевантных материалов не найдено", []

        except Exception as e:
            return None, f"Ошибка RSS: {e}", []

    def search_duckduckgo(self, query):
        """Ищет упоминания через DuckDuckGo с сокращением запроса и повторами."""
        query = " ".join(str(query).split())

        search_query = query
        if len(search_query) > 220:
            search_query = search_query[:220].rsplit(" ", 1)[0]

        attempts = [
            search_query,
            " ".join(search_query.split()[:18]),
            " ".join(search_query.split()[:10]),
        ]

        last_error = None

        for attempt_query in attempts:
            for backend in ("auto", "html", "lite"):
                try:
                    with DDGS() as ddgs:
                        results = list(
                            ddgs.text(
                                attempt_query,
                                max_results=10,
                                backend=backend,
                            )
                        )

                    if not results:
                        continue

                    query_words = attempt_query.lower().split()
                    matches = 0
                    links = []

                    for r in results:
                        combined = (
                            str(r.get("title", "")) + " " +
                            str(r.get("body", ""))
                        ).lower()

                        if all(word in combined for word in query_words):
                            matches += 1
                            links.append(r.get("href", ""))

                    links = [link for link in links if link]

                    if matches >= 3:
                        return (
                            None,
                            f"? Найдено {matches} упоминаний, требуется проверка смысла",
                            links[:5],
                        )
                    if matches >= 1:
                        return (
                            None,
                            f"? Найдено {matches} упоминание, требуется проверка смысла",
                            links[:5],
                        )

                except Exception as e:
                    last_error = e
                    continue

        if last_error:
            return None, f"Ошибка поиска: {last_error}", []

        return False, "✗ Не найдено даже упоминаний темы", []

    def check_refutations(self, query):
        """Ищет статьи, опровергающие утверждение."""
        refute_queries = [
            f"{query} опровержение",
            f"{query} фейк разоблачение",
            f"{query} это ложь",
        ]

        negation_words = ['не является', 'не был', 'опроверг', 'фейк', 'ложь',
                          'неправда', 'миф', 'дезинформация', 'не соответствует']

        refutations_found = 0
        links = []

        try:
            with DDGS() as ddgs:
                for rq in refute_queries:
                    results = ddgs.text(rq, max_results=5)
                    query_words = query.lower().split()

                    for r in results:
                        combined = (r['title'] + " " + r['body']).lower()

                        topic_match = sum(1 for w in query_words if w in combined)
                        has_negation = any(nw in combined for nw in negation_words)

                        if topic_match >= 2 and has_negation:
                            refutations_found += 1
                            links.append(r['href'])

            if refutations_found >= 2:
                return True, f"Найдено {refutations_found} возможных опровержения — требуется проверка смысла", links[:5]
            elif refutations_found == 1:
                return True, "Найдено 1 возможное опровержение — требуется проверка смысла", links[:5]
            return False, "Опровержений не найдено", []

        except Exception as e:
            return False, f"Ошибка поиска опровержений: {e}", []

    def check_wikipedia_claim(self, text):
        """Проверка короткого утверждения по Википедии."""
        try:
            words = [
                w for w in re.findall(r'[а-яё]+', text.lower())
                if w not in STOP_WORDS and len(w) > 2
            ]

            if not words:
                return None, "Нет значимых слов для проверки"

            search_results = wikipedia.search(text, results=3)

            if not search_results:
                return None, "Статья не найдена"

            page = wikipedia.page(search_results[0], auto_suggest=False)
            content = (page.summary + " " + page.content[:5000]).lower()

            found = sum(1 for w in words if w[:5] in content)
            ratio = found / len(words)

            if ratio >= 0.99:
                return True, f"✓ Утверждение согласуется со статьёй «{page.title}» (Википедия)"
            elif ratio >= 0.5:
                return None, f"? Тема найдена («{page.title}»), но утверждение требует проверки смысла"
            else:
                return None, f"? Тема найдена («{page.title}»), но утверждение не подтверждено напрямую"

        except Exception as e:
            return None, f"Ошибка Wikipedia: {e}"

    def check_wikipedia(self, person=None, position=None, country=None, organization=None, query=None):
        try:
            search_query = person or organization or query

            if not search_query:
                return None, "Нет запроса для поиска"

            search_results = wikipedia.search(search_query, results=3)

            if not search_results:
                return None, f"Не найдено статью о {search_query}"

            page = wikipedia.page(search_results[0], auto_suggest=False)
            content = page.content.lower()
            summary = page.summary.lower()

            if person and position and country:
                position_lower = position.lower()
                country_lower = country.lower()

                if "президент" in position_lower:
                    if country_lower == "россии" and "путин" in person.lower():
                        if "президент россии" in content or "президент российской" in content:
                            return True, "✓ Путин — президент России (Википедия)"
                        elif "президент сша" in content:
                            return False, "✗ Путин НЕ президент США (Википедия)"
                    elif country_lower == "сша" and "путин" in person.lower():
                        return False, "✗ Путин НЕ президент США (Википедия)"
                    elif country_lower == "сша" and "трамп" in person.lower():
                        if "президент сша" in content or "45-й президент" in content:
                            return True, "✓ Трамп — президент США (Википедия)"

            if position and country:
                position_lower = position.lower()
                country_lower = country.lower()

                position_found = position_lower in content or position_lower in summary
                country_found = country_lower in content or country_lower in summary

                if position_found and country_found:
                    return True, "✓ Подтверждено в Википедии"
                elif position_found:
                    return None, f"? {person} — {position} (страна неясна)"
                else:
                    return False, "✗ Не найдено в Википедии"

            if organization:
                if organization.lower() in content or organization.lower() in summary:
                    return True, "✓ Найдено в Википедии"
                else:
                    return False, "✗ Не найдено в Википедии"

            return True, "✓ Статья найдена в Википедии"

        except Exception as e:
            return None, f"Ошибка Wikipedia: {e}"

    def verify(self, text, theme=None):
        if not theme or theme == 'авто':
            theme = self.detect_theme(text)

        quick_result = self.check_quick_facts(text)

        if quick_result[0] is not None:
            model_result = self.check_model(text)

            return {
                'verdict': 'ФЕЙК' if not quick_result[0] else 'ПРАВДА',
                'reason': quick_result[1],
                'confidence': 1.0,
                'details': [{
                    'fact': text,
                    'source': 'База фактов',
                    'result': quick_result[0],
                    'reason': quick_result[1],
                    'links': []
                }],
                'theme': theme,
                'model': model_result
            }

        model_result = self.check_model(text)
        entities = self.extract_entities(text)

        # ===== ВЕТКА: НЕТ СУЩНОСТЕЙ =====
        if not entities['persons'] and not entities['events'] and not entities['organizations']:
            wiki_claim = self.check_wikipedia_claim(text)

            if wiki_claim[0] is True:
                return {
                    'verdict': 'ПРАВДА',
                    'reason': wiki_claim[1],
                    'confidence': 0.8,
                    'details': [{
                        'fact': text,
                        'source': 'Википедия',
                        'result': True,
                        'reason': wiki_claim[1],
                        'links': []
                    }, {
                        'fact': 'ML-модель rubert-tiny2',
                        'source': 'Локальная ML-модель',
                        'result': None if model_result['prediction'] == 'НЕИЗВЕСТНО' else (model_result['prediction'] == 'ПРАВДА'),
                        'reason': (
                            f"Модель: {model_result['prediction']} "
                            f"(фейк: {model_result['fake_probability']:.2f}, "
                            f"правда: {model_result['real_probability']:.2f})"
                        ),
                        'links': []
                    }],
                    'theme': theme,
                    'model': model_result
                }

            refute_result = self.check_refutations(text)

            ddg_result = self.search_duckduckgo(text)

            # RSS как резервный источник
            if ddg_result[0] is None and (
                "Ошибка поиска" in ddg_result[1]
                or "Ничего не найдено" in ddg_result[1]
            ):
                rss_result = self.search_rss_news(text, theme)

                if rss_result[0] is True:
                    ddg_result = (
                        None,
                        rss_result[1] + ". Тема найдена в RSS, но утверждение требует проверки смысла",
                        rss_result[2],
                    )
                elif rss_result[0] is None:
                    ddg_result = rss_result

            links = ddg_result[2] if len(ddg_result) > 2 else []

            if refute_result[0]:
                links = list(dict.fromkeys(refute_result[2] + links))[:5]

            reason = ddg_result[1]

            if refute_result[0]:
                reason += f" {refute_result[1]}"

            search_failed = (
                (
                    ddg_result[0] is None and
                    (
                        "Ошибка поиска" in ddg_result[1] or
                        "Ничего не найдено" in ddg_result[1] or
                        "Ошибка RSS" in ddg_result[1]
                    )
                ) or
                (
                    ddg_result[0] is False and
                    (
                        "Не найдено даже упоминаний" in ddg_result[1] or
                        "Релевантных материалов не найдено" in ddg_result[1]
                    )
                )
            )

            if search_failed:
                if model_result["confidence"] >= 0.75:
                    verdict = model_result["prediction"]
                    confidence = min(model_result["confidence"], 0.75)
                    reason += " Внешняя проверка недоступна: вердикт основан только на ML-модели"
                else:
                    verdict = "НЕИЗВЕСТНО"
                    confidence = 0.5
            else:
                confidence = 0.5

                if ddg_result[0] is None:
                    verdict = "НЕИЗВЕСТНО"
                elif ddg_result[0]:
                    verdict = "ПРАВДА"
                else:
                    verdict = "ФЕЙК"

            return {
                'verdict': verdict,
                'reason': reason,
                'confidence': confidence,
                'details': [{
                    'fact': text,
                    'source': 'DuckDuckGo / RSS',
                    'result': ddg_result[0],
                    'reason': reason,
                    'links': links
                }, {
                    'fact': 'ML-модель rubert-tiny2',
                    'source': 'Локальная ML-модель',
                    'result': None if model_result['prediction'] == 'НЕИЗВЕСТНО' else (model_result['prediction'] == 'ПРАВДА'),
                    'reason': (
                        f"Модель: {model_result['prediction']} "
                        f"(фейк: {model_result['fake_probability']:.2f}, "
                        f"правда: {model_result['real_probability']:.2f})"
                    ),
                    'links': []
                }],
                'theme': theme,
                'model': model_result
            }

        # ===== ВЕТКА: ЕСТЬ СУЩНОСТИ =====
        results = []

        for i, person in enumerate(entities['persons']):
            position = entities['positions'][i] if i < len(entities['positions']) else ''
            country = entities['countries'][i] if i < len(entities['countries']) else ''

            wiki_result = self.check_wikipedia(person=person, position=position, country=country)

            if wiki_result[0] is not None:
                results.append({
                    'fact': f"{person} — {position} {country}",
                    'source': 'Википедия',
                    'result': wiki_result[0],
                    'reason': wiki_result[1],
                    'links': [f"https://ru.wikipedia.org/wiki/{person}"]
                })
            else:
                ddg_result = self.search_duckduckgo(f"{person} {position} {country}")
                links = ddg_result[2] if len(ddg_result) > 2 else []

                results.append({
                    'fact': f"{person} — {position} {country}",
                    'source': 'DuckDuckGo',
                    'result': ddg_result[0],
                    'reason': ddg_result[1],
                    'links': links
                })

        for org in entities['organizations']:
            wiki_result = self.check_wikipedia(organization=org)

            if wiki_result[0] is not None:
                results.append({
                    'fact': f"Организация: {org}",
                    'source': 'Википедия',
                    'result': wiki_result[0],
                    'reason': wiki_result[1],
                    'links': [f"https://ru.wikipedia.org/wiki/{org}"]
                })
            else:
                ddg_result = self.search_duckduckgo(org)
                links = ddg_result[2] if len(ddg_result) > 2 else []

                results.append({
                    'fact': f"Организация: {org}",
                    'source': 'DuckDuckGo',
                    'result': ddg_result[0],
                    'reason': ddg_result[1],
                    'links': links
                })

        for event in entities['events']:
            for country in entities['countries']:
                query = f"{event} {country}"

                rss_result = self.search_rss_news(query, theme)

                if rss_result[0] is not None:
                    links = [item['link'] for item in rss_result[2]] if len(rss_result) > 2 and rss_result[2] else []

                    results.append({
                        'fact': f"Событие: {event} в {country}",
                        'source': 'RSS Новости',
                        'result': rss_result[0],
                        'reason': rss_result[1],
                        'links': links
                    })
                else:
                    ddg_result = self.search_duckduckgo(query)
                    links = ddg_result[2] if len(ddg_result) > 2 else []

                    results.append({
                        'fact': f"Событие: {event} в {country}",
                        'source': 'DuckDuckGo',
                        'result': ddg_result[0],
                        'reason': ddg_result[1],
                        'links': links
                    })

        results.append({
            'fact': 'ML-модель rubert-tiny2',
            'source': 'Локальная ML-модель',
            'result': None if model_result['prediction'] == 'НЕИЗВЕСТНО' else (model_result['prediction'] == 'ПРАВДА'),
            'reason': (
                f"Модель: {model_result['prediction']} "
                f"(фейк: {model_result['fake_probability']:.2f}, "
                f"правда: {model_result['real_probability']:.2f})"
            ),
            'links': []
        })

        true_count = sum(1 for r in results if r['result'] == True)
        false_count = sum(1 for r in results if r['result'] == False)

        if results:
            sources_score = (true_count - false_count) / len(results)
        else:
            sources_score = 0

        model_score = model_result['real_probability'] - model_result['fake_probability']
        final_score = 0.7 * sources_score + 0.3 * model_score

        if final_score >= 0.3:
            verdict = 'ПРАВДА'
            confidence = min(0.99, 0.5 + final_score / 2)
        elif final_score <= -0.3:
            verdict = 'ФЕЙК'
            confidence = min(0.99, 0.5 - final_score / 2)
        else:
            verdict = 'НЕИЗВЕСТНО'
            confidence = 0.5

        return {
            'verdict': verdict,
            'reason': results[0]['reason'] if results else 'Нет данных',
            'confidence': confidence,
            'details': results,
            'theme': theme,
            'model': model_result
        }


if __name__ == "__main__":
    checker = FactChecker()

    tests = [
        "В НАТО составили наступательный план действий по захвату и оккупации Российской Федерации в случае начала военных действий против Украины. Отдельные части британских и американских войск совместно с солдатами Турции произведут высадку на побережье, чтобы открыть плацдарм для дальнейшего наступления.",
        "Банк России снизил ключевую ставку. Регулятор отметил, что внешние условия для экономики остаются сложными, однако риски для финансовой стабильности перестали нарастать.",
        "Учёные из университета сообщили об открытии нового метода лечения, который, по их словам, позволяет полностью вылечить все известные заболевания за одну неделю."
    ]

    for text in tests:
        result = checker.verify(text)

        print("\n" + "=" * 70)
        print("Текст:", text[:120], "...")
        print("Тема:", result["theme"])
        print("Вердикт:", result["verdict"], f"({result['confidence']:.0%})")
        print("Причина:", result["reason"])

        model = result.get("model", {})
        print(
            "ML:",
            model.get("prediction"),
            f"| фейк {model.get('fake_probability', 0):.2f}",
            f"| правда {model.get('real_probability', 0):.2f}"
        )
import json
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

QWEN_BASE = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"
QWEN_LORA = "lastikfff/qwen2.5-7b-factchecker-lora"

class QwenFactChecker:
    def __init__(self):
        self.model = None
        self.tokenizer = None

    def load(self):
        if self.model is not None:
            return

        from transformers import BitsAndBytesConfig

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )

        self.tokenizer = AutoTokenizer.from_pretrained(QWEN_LORA)

        base_model = AutoModelForCausalLM.from_pretrained(
            QWEN_BASE,
            quantization_config=quantization_config,
            device_map="auto",
        )

        self.model = PeftModel.from_pretrained(
            base_model,
            QWEN_LORA,
        )

        self.model.eval()

    def analyze(self, claim, sources, ml_hint="нет данных"):
        self.load()

        sources_text = "\n".join(
            f"Источник {i}: {s}" for i, s in enumerate(sources, 1)
        )

        user_text = (
            f"Утверждение: {claim}\n\n"
            f"{sources_text}\n\n"
            f"ML-модель: {ml_hint}"
        )

        messages = [
            {
                "role": "system",
                "content": "Ты — фактчекер для русскоязычных новостей. Проанализируй утверждение и источники. Верни только валидный JSON без пояснений."
            },
            {
                "role": "user",
                "content": user_text
            }
        ]

        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.model.device)

        attention_mask = (inputs != self.tokenizer.pad_token_id).long()

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs,
                attention_mask=attention_mask,
                max_new_tokens=256,
                temperature=0.2,
                do_sample=False,
            )

        text = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )

        return self._parse_json(text)

    def _parse_json(self, text):
        match = re.search(r"\{.*\}", text, re.DOTALL)

        if not match:
            return {
                "verdict": "НЕИЗВЕСТНО",
                "confidence": 0.5,
                "reason": "ИИ-модель не вернула корректный JSON.",
                "sources_used": []
            }

        try:
            data = json.loads(match.group())
            verdict = data.get("verdict", "НЕИЗВЕСТНО")
            confidence = float(data.get("confidence", 0.5))
            reason = data.get("reason", "ИИ-анализ не дал объяснения.")
            sources_used = data.get("sources_used", [])

            if verdict not in ("ПРАВДА", "ФЕЙК", "НЕИЗВЕСТНО"):
                verdict = "НЕИЗВЕСТНО"

            confidence = max(0.0, min(1.0, confidence))

            return {
                "verdict": verdict,
                "confidence": confidence,
                "reason": reason,
                "sources_used": sources_used
            }

        except Exception:
            return {
                "verdict": "НЕИЗВЕСТНО",
                "confidence": 0.5,
                "reason": "Не удалось разобрать ответ ИИ-модели.",
                "sources_used": []
            }

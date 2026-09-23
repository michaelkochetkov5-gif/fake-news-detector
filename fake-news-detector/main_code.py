from ddgs import DDGS
import wikipedia
import re
import feedparser

# ===== RSS ЛЕНТЫ =====
RSS_FEEDS = {
    # Новости (общее)
    'lenta': 'https://lenta.ru/rss/news',
    'ria': 'https://ria.ru/export/rss2/news/index.xml',
    'tass': 'https://tass.ru/rss/v2.xml',
    'meduza': 'https://meduza.io/rss/all',
    
    # Еда
    'eda_ru': 'https://eda.ru/rss',
    'povarenok': 'https://www.povarenok.ru/rss/',
    
    # Учёба
    'postupi_online': 'https://postupi.online/news/rss/',
    
    # ИИ и технологии
    'habr': 'https://habr.com/ru/rss/',
    'vc_ru': 'https://vc.ru/rss',
    'tproger': 'https://tproger.ru/feed/',
    
    # Наука
    'nplus1': 'https://nplus1.ru/rss',
    'elementy': 'https://elementy.ru/rss',
    'indicator': 'https://indicator.ru/rss.xml',
    'scientificrussia': 'https://scientificrussia.ru/feed',
    
    # Игры
    'dtf': 'https://dtf.ru/rss',
    'kanobu': 'https://www.kanobu.ru/rss/',
    'igromania': 'https://www.igromania.ru/rss/',
    
    # Спорт
    'sports': 'https://www.sports.ru/rss/',
    'matchtv': 'https://matchtv.ru/rss',
    
    # Интересные факты
    'factroom': 'https://factroom.ru/feed',
    'fishki': 'https://fishki.net/rss',
    'adme': 'https://www.adme.ru/rss/',
    'lifehacker': 'https://lifehacker.ru/feed/',
    
    # Вокруг света
    'vokrugsveta': 'https://www.vokrugsveta.ru/rss/',
    'natgeo': 'https://www.national-geographic.ru/rss/',
    
    # Кино
    'kinopoisk': 'https://www.kinopoisk.ru/rss/news/',
    'film_ru': 'https://www.film.ru/rss/',
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
}

class FactChecker:
    def __init__(self):
        wikipedia.set_lang("ru")
    
    def detect_theme(self, text):
        text_lower = text.lower()
        scores = {}
        
        for theme, keywords in THEME_KEYWORDS.items():
            scores[theme] = sum(1 for kw in keywords if kw in text_lower)
        
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
    
    def extract_entities(self, text):
        entities = {
            'persons': [],
            'positions': [],
            'countries': [],
            'events': [],
            'organizations': []
        }
        
        pattern1 = r'([А-Яа-яЁё]+)\s+(президент|министр|глава|канцлер|директор)\s+([А-Яа-яЁё]+)'
        text_norm = text.strip()
        text_norm = text_norm[0].upper() + text_norm[1:] if text_norm else text_norm
        matches1 = re.findall(pattern1, text_norm, re.IGNORECASE)
        matches1 = re.findall(pattern1, text, re.IGNORECASE)
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
            return {k: v for k, v in RSS_FEEDS.items() if k in ['eda_ru', 'povarenok']}
        elif theme == 'учёба':
            return {k: v for k, v in RSS_FEEDS.items() if k in ['postupi_online']}
        elif theme == 'ии' or theme == 'it':
            return {k: v for k, v in RSS_FEEDS.items() if k in ['habr', 'vc_ru', 'tproger']}
        elif theme == 'наука':
            return {k: v for k, v in RSS_FEEDS.items() if k in ['nplus1', 'elementy', 'indicator', 'scientificrussia']}
        elif theme == 'игры':
            return {k: v for k, v in RSS_FEEDS.items() if k in ['dtf', 'kanobu', 'igromania']}
        elif theme == 'спорт':
            return {k: v for k, v in RSS_FEEDS.items() if k in ['sports', 'matchtv']}
        elif theme == 'факты':
            return {k: v for k, v in RSS_FEEDS.items() if k in ['factroom', 'fishki', 'adme', 'lifehacker']}
        elif theme == 'путешествия':
            return {k: v for k, v in RSS_FEEDS.items() if k in ['vokrugsveta', 'natgeo']}
        elif theme == 'кино':
            return {k: v for k, v in RSS_FEEDS.items() if k in ['kinopoisk', 'film_ru']}
        else:
            return RSS_FEEDS
    
    def search_rss_news(self, query, theme='общее'):
        try:
            feeds = self.get_rss_for_theme(theme)
            all_matches = []
            
            for source, url in feeds.items():
                feed = feedparser.parse(url)
                
                for entry in feed.entries[:20]:
                    title = entry.title.lower()
                    summary = entry.get('summary', '').lower()
                    
                    if query.lower() in title or query.lower() in summary:
                        all_matches.append({
                            'source': source,
                            'title': entry.title,
                            'link': entry.link,
                            'published': entry.get('published', 'N/A')
                        })
            
            if len(all_matches) >= 2:
                return True, f"✓ Найдено в {len(all_matches)} источниках", all_matches[:5]
            elif len(all_matches) == 1:
                return None, f"? Найдено в 1 источнике", all_matches
            else:
                return False, "✗ Новостей не найдено", []
                
        except Exception as e:
            return None, f"Ошибка RSS: {e}", []
    
    def search_duckduckgo(self, query):
    try:
        with DDGS() as ddgs:
            results = ddgs.text(query, max_results=10)

            if not results:
                return None, "Ничего не найдено"

            query_words = query.lower().split()
            matches = 0

            for r in results:
                combined = (r['title'] + " " + r['body']).lower()
                if all(word in combined for word in query_words):
                    matches += 1

            if matches >= 3:
                # Совпадение слов ещё НЕ доказывает утверждение
                return None, f"? Найдено {matches} упоминаний, но требуется проверка смысла"
            elif matches >= 1:
                return None, f"? Найдено {matches} упоминание, требуется проверка смысла"
            else:
                return False, "✗ Не найдено даже упоминаний темы"

    except Exception as e:
        return None, f"Ошибка поиска: {e}"
    
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
                    return True, f"✓ Подтверждено в Википедии"
                elif position_found:
                    return None, f"? {person} — {position} (страна неясна)"
                else:
                    return False, f"✗ Не найдено в Википедии"
            
            if organization:
                if organization.lower() in content or organization.lower() in summary:
                    return True, f"✓ Найдено в Википедии"
                else:
                    return False, f"✗ Не найдено в Википедии"
            
            return True, f"✓ Статья найдена в Википедии"
                
        except Exception as e:
            return None, f"Ошибка Wikipedia: {e}"
    
    def verify(self, text):
        theme = self.detect_theme(text)
        
        quick_result = self.check_quick_facts(text)
        if quick_result[0] is not None:
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
                'theme': theme
            }
        
        entities = self.extract_entities(text)
        
        if not entities['persons'] and not entities['events'] and not entities['organizations']:
            ddg_result = self.search_duckduckgo(text)
            links = [r['href'] for r in ddg_result[2]] if len(ddg_result) > 2 and ddg_result[2] else []
            return {
                'verdict': 'НЕИЗВЕСТНО' if ddg_result[0] is None else ('ПРАВДА' if ddg_result[0] else 'ФЕЙК'),
                'reason': ddg_result[1],
                'confidence': 0.5 if ddg_result[0] is None else 1.0,
                'details': [{
                    'fact': text,
                    'source': 'DuckDuckGo',
                    'result': ddg_result[0],
                    'reason': ddg_result[1],
                    'links': links
                }],
                'theme': theme
            }
        
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
                links = [r['href'] for r in ddg_result[2]] if len(ddg_result) > 2 and ddg_result[2] else []
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
                links = [r['href'] for r in ddg_result[2]] if len(ddg_result) > 2 and ddg_result[2] else []
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
                    links = [r['href'] for r in ddg_result[2]] if len(ddg_result) > 2 and ddg_result[2] else []
                    results.append({
                        'fact': f"Событие: {event} в {country}",
                        'source': 'DuckDuckGo',
                        'result': ddg_result[0],
                        'reason': ddg_result[1],
                        'links': links
                    })
        
        true_count = sum(1 for r in results if r['result'] == True)
        false_count = sum(1 for r in results if r['result'] == False)
        
        if false_count > 0:
            verdict, confidence = 'ФЕЙК', false_count / len(results)
        elif true_count == len(results):
            verdict, confidence = 'ПРАВДА', 1.0
        else:
            verdict, confidence = 'НЕИЗВЕСТНО', 0.5
        
        return {
            'verdict': verdict,
            'reason': results[0]['reason'] if results else 'Нет данных',
            'confidence': confidence,
            'details': results,
            'theme': theme
        }

if __name__ == "__main__":
    checker = FactChecker()
    
    tests = [
        "Путин президент России",
        "Сахар вызывает зависимость сильнее кокаина",
        "ChatGPT заменит всех программистов",
        "МГУ входит в топ 100 университетов",
        "Интересные факты о космосе",
        "Minecraft самая продаваемая игра",
    ]
    
    for test in tests:
        print(f"\n{test}:")
        result = checker.verify(test)
        print(f"Тема: {result['theme']}")
        print(f"Вердикт: {result['verdict']} ({result['confidence']:.0%})")
        print(f"Причина: {result['reason']}")
        if result['details']:
            print(f"Ссылки: {result['details'][0].get('links', [])[:3]}")

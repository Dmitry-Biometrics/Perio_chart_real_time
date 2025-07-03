#!/usr/bin/env python3
"""
LLM КЭШИРОВАНИЕ ДЛЯ УСКОРЕНИЯ ПОВТОРНЫХ КОМАНД
"""

import hashlib
import time
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

class LLMResponseCache:
    """Кэш для LLM ответов"""
    
    def __init__(self, max_size: int = 100):
        self.cache = {}
        self.max_size = max_size
        self.hits = 0
        self.misses = 0
        
    def _normalize_text(self, text: str) -> str:
        """Нормализация текста для кэша"""
        import re
        # Убираем лишние пробелы и знаки препинания
        normalized = re.sub(r'\s+', ' ', text.lower().strip())
        normalized = re.sub(r'[^\w\s]', '', normalized)
        return normalized
    
    def _cache_key(self, text: str) -> str:
        """Генерация ключа кэша"""
        normalized = self._normalize_text(text)
        return hashlib.md5(normalized.encode()).hexdigest()[:16]
    
    def get(self, text: str) -> Optional[Dict]:
        """Получение из кэша"""
        key = self._cache_key(text)
        
        if key in self.cache:
            self.hits += 1
            cached_result = self.cache[key].copy()
            cached_result['timestamp'] = time.time()
            cached_result['cache_hit'] = True
            print(f"⚡ CACHE HIT: '{text}' (hits: {self.hits}, misses: {self.misses})")
            return cached_result
        
        self.misses += 1
        return None
    
    def put(self, text: str, result: Dict):
        """Сохранение в кэш"""
        if not result.get('success'):
            return
        
        key = self._cache_key(text)
        self.cache[key] = result.copy()
        
        # Ограничиваем размер кэша
        if len(self.cache) > self.max_size:
            # Удаляем самые старые записи
            oldest_key = min(self.cache.keys(), 
                           key=lambda k: self.cache[k].get('timestamp', 0))
            del self.cache[oldest_key]
    
    def get_stats(self) -> Dict:
        """Статистика кэша"""
        total_requests = self.hits + self.misses
        hit_rate = (self.hits / total_requests * 100) if total_requests > 0 else 0
        
        return {
            'cache_hits': self.hits,
            'cache_misses': self.misses,
            'hit_rate_percent': hit_rate,
            'cache_size': len(self.cache),
            'max_size': self.max_size
        }

# Глобальный экземпляр кэша
llm_cache = LLMResponseCache()
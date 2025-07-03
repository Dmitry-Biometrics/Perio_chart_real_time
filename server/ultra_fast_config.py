# УЛЬТРА-БЫСТРАЯ конфигурация для мгновенного отклика
# Замените соответствующие значения в config_enhanced_server.py

# =============================================================================
# УЛЬТРА-БЫСТРЫЕ НАСТРОЙКИ СЕГМЕНТАЦИИ
# =============================================================================

# КРИТИЧЕСКИ УМЕНЬШЕННЫЕ пороги молчания (в чанках по 0.25с)
SILENCE_THRESHOLD_SHORT = 2    # 0.5 секунды (было 1.5с)
SILENCE_THRESHOLD_MEDIUM = 3   # 0.75 секунды (было 2.0с)  
SILENCE_THRESHOLD_LONG = 4     # 1.0 секунда (было 2.5с)

# УСКОРЕННЫЕ настройки детекции речи
SPEECH_CONFIRMATION_CHUNKS = 1  # Мгновенное начало (было 2-3)
SILENCE_CONFIRMATION_CHUNKS = 2 # Быстрое завершение (было 6)

# БОЛЕЕ ЧУВСТВИТЕЛЬНЫЕ VAD пороги  
VAD_THRESHOLD = 0.25           # Понижено с 0.4
SPEECH_THRESHOLD = 0.15        # Понижено с 0.25
SILENCE_THRESHOLD = 0.1        # Понижено с 0.15

# УСКОРЕННЫЕ временные ограничения
MIN_SPEECH_DURATION = 0.3      # Понижено с 0.5с
MAX_SPEECH_DURATION = 15.0     # Понижено с 25.0с

# =============================================================================
# НОВЫЕ НАСТРОЙКИ ДЛЯ УЛЬТРА-БЫСТРОГО РЕЖИМА
# =============================================================================

ULTRA_FAST_CONFIG = {
    # Режим работы
    "mode": "ULTRA_FAST",
    "target_response_time_ms": 50,  # Целевое время отклика
    
    # Агрессивная сегментация
    "aggressive_segmentation": True,
    "early_termination": True,       # Досрочное завершение команд
    "adaptive_thresholds": True,     # Адаптивные пороги
    
    # Predictive processing (обработка во время речи)
    "predictive_enabled": True,
    "predictive_buffer_size": 1.0,   # 1 секунда для анализа
    "predictive_confidence": 0.7,    # Уверенность для предиктивного выполнения
    
    # Streaming recognition
    "streaming_enabled": True,
    "stream_chunk_size": 0.5,        # Анализ каждые 0.5 секунд
    "stream_overlap": 0.25,          # Перекрытие для точности
    
    # Instant command optimization
    "instant_patterns_priority": True,
    "bypass_full_segmentation": True, # Обход полной сегментации для известных команд
    "pattern_matching_threshold": 0.8,
    
    # Energy-based early detection
    "energy_spike_detection": True,   # Детекция пиков энергии
    "energy_based_termination": True, # Завершение по энергии
    "background_adaptation": True,    # Адаптация к фону
    
    # Advanced VAD settings
    "multi_threshold_vad": True,      # Множественные пороги VAD
    "context_aware_vad": True,        # Контекстно-зависимый VAD
    "vad_smoothing": False,           # Отключаем сглаживание для скорости
}

# =============================================================================
# ПАТТЕРНЫ ДЛЯ УЛЬТРА-БЫСТРОГО РАСПОЗНАВАНИЯ
# =============================================================================

ULTRA_FAST_PATTERNS = {
    # Частичные паттерны для досрочного выполнения
    "early_patterns": {
        "probing_depth": [
            r"probing.*depth.*tooth.*(\d+).*buccal.*(\d+).*(\d+).*(\d+)",
            r"probing.*depth.*tooth.*(\d+).*lingual.*(\d+).*(\d+).*(\d+)",
        ],
        "bleeding": [
            r"bleeding.*tooth.*(\d+).*buccal.*distal",
            r"bleeding.*tooth.*(\d+).*lingual.*distal",
        ],
        "mobility": [
            r"tooth.*(\d+).*mobility.*grade.*(\d+)",
            r"mobility.*grade.*(\d+).*tooth.*(\d+)",
        ]
    },
    
    # Ключевые слова для досрочной активации
    "trigger_words": [
        "probing", "bleeding", "tooth", "mobility", "grade", 
        "furcation", "class", "missing", "suppuration"
    ],
    
    # Числовые паттерны для быстрой обработки
    "number_sequences": {
        "three_numbers": r"(\d+)\s+(\d+)\s+(\d+)",  # 3 2 4
        "tooth_grade": r"tooth\s+(\d+).*grade\s+(\d+)",
        "class_tooth": r"class\s+(\d+).*tooth\s+(\d+)"
    }
}

# =============================================================================
# ФУНКЦИИ ДЛЯ ПРИМЕНЕНИЯ УЛЬТРА-БЫСТРЫХ НАСТРОЕК
# =============================================================================

def apply_ultra_fast_settings():
    """Применение ультра-быстрых настроек"""
    import logging
    logger = logging.getLogger(__name__)
    
    logger.info("🚀 APPLYING ULTRA-FAST SETTINGS")
    logger.info("⚡ Target response time: 50ms")
    logger.info("🎯 Silence thresholds: 0.5-1.0s (was 1.5-2.5s)")
    logger.info("🔥 Aggressive segmentation: ENABLED")
    logger.info("🔮 Predictive processing: ENABLED")
    
    return ULTRA_FAST_CONFIG

def get_optimized_vad_config():
    """Оптимизированная конфигурация VAD для скорости"""
    return {
        "threshold": 0.25,
        "speech_threshold": 0.15,
        "silence_threshold": 0.1,
        "smoothing": False,           # Отключено для скорости
        "adaptive": True,             # Адаптивные пороги
        "energy_boost": True,         # Усиление по энергии
        "context_length": 3,          # Короткий контекст
        "fast_response": True         # Быстрый отклик
    }

def get_segmentation_speed_config():
    """Конфигурация сегментации для максимальной скорости"""
    return {
        "speech_confirmation_chunks": 1,    # Мгновенное начало
        "silence_confirmation_chunks": 2,   # Быстрое завершение
        "min_command_duration": 0.3,        # Короткие команды
        "max_command_duration": 15.0,       # Лимит времени
        "early_termination": True,          # Досрочное завершение
        "aggressive_mode": True,            # Агрессивный режим
        "predictive_analysis": True,        # Предиктивный анализ
        "streaming_recognition": True,      # Потоковое распознавание
    }

# =============================================================================
# ИНТЕГРАЦИЯ С EXISTING СИСТЕМОЙ
# =============================================================================

def patch_existing_config():
    """Патч существующей конфигурации для ультра-быстрого режима"""
    
    # Патчим основной конфиг
    config_patches = {
        # Основные пороги
        "VAD_THRESHOLD": 0.25,
        "CLIENT_CHUNK_DURATION": 0.25,
        "SILENCE_THRESHOLD_SHORT": 2,
        "SILENCE_THRESHOLD_MEDIUM": 3, 
        "SILENCE_THRESHOLD_LONG": 4,
        "MIN_SPEECH_DURATION": 0.3,
        "MAX_SPEECH_DURATION": 15.0,
        
        # Обработка
        "processing_timeout": 10.0,
        "max_processing_errors": 50,
        
        # Instant commands
        "instant_commands_enabled": True,
        "instant_bypass_segmentation": True,
        "predictive_processing": True,
    }
    
    return config_patches

# =============================================================================
# MONITORING ДЛЯ УЛЬТРА-БЫСТРОГО РЕЖИМА  
# =============================================================================

class UltraFastMonitor:
    """Мониторинг производительности ультра-быстрого режима"""
    
    def __init__(self):
        self.response_times = []
        self.target_time = 50  # ms
        self.warning_time = 100  # ms
        self.critical_time = 500  # ms
        
    def record_response_time(self, time_ms):
        """Запись времени отклика"""
        self.response_times.append(time_ms)
        
        # Ограничиваем историю
        if len(self.response_times) > 100:
            self.response_times = self.response_times[-100:]
            
        # Анализ производительности
        if time_ms > self.critical_time:
            logging.warning(f"🐌 CRITICAL: Response time {time_ms:.0f}ms (target: {self.target_time}ms)")
        elif time_ms > self.warning_time:
            logging.warning(f"⚠️ SLOW: Response time {time_ms:.0f}ms (target: {self.target_time}ms)")
        elif time_ms <= self.target_time:
            logging.info(f"⚡ ULTRA-FAST: Response time {time_ms:.0f}ms ✅")
    
    def get_performance_stats(self):
        """Статистика производительности"""
        if not self.response_times:
            return {}
            
        import statistics
        
        return {
            "average_response_ms": statistics.mean(self.response_times),
            "median_response_ms": statistics.median(self.response_times),
            "min_response_ms": min(self.response_times),
            "max_response_ms": max(self.response_times),
            "target_achieved_percent": len([t for t in self.response_times if t <= self.target_time]) / len(self.response_times) * 100,
            "total_responses": len(self.response_times)
        }

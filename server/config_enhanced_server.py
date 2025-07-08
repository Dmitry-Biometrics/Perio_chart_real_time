#!/usr/bin/env python3
"""
Конфигурационный файл для Enhanced FastWhisper Server
с записью аудио и RAG интеграцией
"""

import os
from pathlib import Path

# =============================================================================
# ОСНОВНЫЕ НАСТРОЙКИ СЕРВЕРА
# =============================================================================

# Порты сервера
ASR_PORT = 8765
WEB_PORT = 8766

# Настройки аудио
SAMPLE_RATE = 16000
CLIENT_CHUNK_DURATION = 0.25  # Длительность чанка в секундах
VAD_THRESHOLD = 0.4  # Порог VAD для детекции речи

# Пороги молчания (в чанках)
SILENCE_THRESHOLD_SHORT = 2   # Для коротких фраз
SILENCE_THRESHOLD_MEDIUM = 2  # Для средних фраз
SILENCE_THRESHOLD_LONG = 2   # Для длинных фраз

# Ограничения длительности речи
MIN_SPEECH_DURATION = 0.5   # Минимальная длительность речи (сек)
MAX_SPEECH_DURATION = 25.0  # Максимальная длительность речи (сек)

# =============================================================================
# НАСТРОЙКИ ЗАПИСИ АУДИО
# =============================================================================

AUDIO_RECORDING_CONFIG = {
    # Основные настройки
    "enabled": True,                        # Включить/выключить запись аудио
AUDIO_RECORDING_CONFIG = {
    # Основные настройки
    "enabled": True,                        # Включить/выключить запись аудио
    "format": "wav",                        # Формат файлов (только wav поддерживается)
    "directory": "audio_recordings",        # Директория для сохранения записей
    
    # Фильтры записи
    "save_all_audio": True,                 # Сохранять все аудио сегменты
    "save_successful_commands_only": False, # Сохранять только успешно обработанные команды
    "save_failed_commands": True,           # Сохранять неуспешные команды для анализа
    
    # Ограничения
    "max_recordings_per_day": 1000,        # Максимум записей в день
    "max_file_size_mb": 10,                 # Максимальный размер одного файла
    "min_duration_seconds": 0.5,           # Минимальная длительность для сохранения
    "max_duration_seconds": 30.0,          # Максимальная длительность для сохранения
    
    # Управление дисковым пространством
    "auto_cleanup_enabled": True,          # Автоматическая очистка старых файлов
    "keep_recordings_days": 30,            # Сколько дней хранить записи
    "max_total_size_gb": 5.0,             # Максимальный общий размер всех записей
    "cleanup_check_interval_hours": 1,     # Интервал проверки для очистки
    
    # Метаданные
    "save_metadata": True,                  # Сохранять JSON метаданные для каждой записи
    "include_transcription": True,          # Включать транскрипцию в метаданные
    "include_processing_results": True,     # Включать результаты обработки команд
    "include_system_info": True,           # Включать информацию о системе
    
    # Организация файлов
    "organize_by_date": True,              # Организовывать файлы по датам (год/месяц/день)
    "organize_by_client": False,           # Организовывать файлы по клиентам
    "organize_by_success": False,          # Организовывать файлы по успешности обработки
    
    # Именование файлов
    "filename_template": "{timestamp}_{client_id}_{status}.wav",
    "timestamp_format": "%H-%M-%S_%f",     # Формат времени в имени файла
    
    # Сжатие и качество
    "compress_old_recordings": False,       # Сжимать старые записи
    "compression_age_days": 7,             # Через сколько дней сжимать
    "quality": "high",                     # high, medium, low
}

# =============================================================================
# НАСТРОЙКИ RAG СИСТЕМЫ
# =============================================================================

RAG_CONFIG = {
    # Основные настройки
    "enabled": True,                       # Включить RAG систему
    "openai_api_key": os.getenv("OPENAI_API_KEY"),  # API ключ OpenAI
    "model": "gpt-3.5-turbo",             # Модель GPT для использования
    "temperature": 0.1,                    # Температура для генерации
    
    # Intent классификация
    "intent_classification_enabled": True,
    "intent_confidence_threshold": 0.7,   # Минимальная уверенность для intent
    "intent_fallback_enabled": True,      # Fallback на другие системы
    
    # Embedding и векторный поиск
    "embedding_model": "text-embedding-ada-002",
    "vector_store_type": "faiss",         # faiss, chroma, pinecone
    "similarity_threshold": 0.75,         # Порог схожести для поиска
    "max_retrieved_docs": 5,              # Максимум документов для контекста
    
    # Кэширование
    "cache_enabled": True,                 # Кэширование результатов
    "cache_size": 1000,                   # Размер кэша
    "cache_ttl_minutes": 60,              # TTL кэша в минутах
    
    # Обучение и адаптация
    "learning_enabled": True,              # Обучение на пользовательских данных
    "feedback_collection": True,          # Сбор обратной связи
    "auto_improvement": True,             # Автоматическое улучшение
}

# =============================================================================
# НАСТРОЙКИ LLM ПЕРИОДОНТАЛЬНОЙ СИСТЕМЫ
# =============================================================================

LLM_PERIODONTAL_CONFIG = {
    # Основные настройки
    "enabled": True,                       # Включить LLM систему
    "liberal_detection": True,             # Либеральная детекция команд
    "confidence_threshold": 0.4,           # Минимальная уверенность
    
    # ASR коррекция
    "asr_correction_enabled": True,        # Исправление ASR ошибок
    "correction_aggressiveness": "medium", # low, medium, high
    "validate_corrections": True,          # Валидация исправлений
    
    # Обработка команд
    "command_validation": True,            # Валидация команд
    "auto_complete_partial": True,         # Автодополнение частичных команд
    "suggest_alternatives": True,          # Предложение альтернатив
    
    # Контекст и память
    "context_window": 10,                  # Количество предыдущих команд в контексте
    "patient_context": True,               # Использование контекста пациента
    "session_memory": True,                # Память сессии
}

AUDIO_ENERGY_CONFIG = {
    "enable_energy_filtering": True,
    "silence_energy_threshold": 0.0005,  # Порог тишины
    "speech_energy_threshold": 0.001,    # Порог речи
    "background_noise_adaptation": True,  # Адаптация к фону
    "energy_ratio_threshold": 2.0,       # Минимальное отношение к фону
    "min_energetic_chunks": 5,           # Минимум энергичных чанков для обработки
    "log_energy_stats": False,           # Логирование энергетической статистики
}

# =============================================================================
# НАСТРОЙКИ STANDARD PERIODONTAL СИСТЕМЫ
# =============================================================================

STANDARD_PERIODONTAL_CONFIG = {
    # Основные настройки
    "enabled": True,                       # Включить как fallback
    "strict_parsing": False,               # Строгий парсинг команд
    "fuzzy_matching": True,                # Нечеткое совпадение
    
    # Поддерживаемые команды
    "supported_commands": [
        "probing_depth",
        "bleeding_on_probing", 
        "suppuration",
        "mobility",
        "furcation",
        "gingival_margin",
        "missing_teeth"
    ],
    
    # Валидация
    "validate_tooth_numbers": True,        # Валидация номеров зубов (1-32)
    "validate_measurements": True,         # Валидация измерений
    "validate_surfaces": True,             # Валидация поверхностей
}

# =============================================================================
# ПРИОРИТЕТЫ СИСТЕМ
# =============================================================================

SYSTEM_PRIORITIES = {
    "enhanced_rag_intents": 0,             # Высший приоритет
    "llm_periodontal": 1,                  # Второй приоритет
    "standard_periodontal": 2,             # Fallback система
}

# Пороги уверенности для переключения между системами
CONFIDENCE_THRESHOLDS = {
    "enhanced_rag_intents": 0.7,
    "llm_periodontal": 0.4,
    "standard_periodontal": 0.3,
}

# =============================================================================
# НАСТРОЙКИ ПРОИЗВОДИТЕЛЬНОСТИ
# =============================================================================

PERFORMANCE_CONFIG = {
    # Обработка аудио
    "audio_buffer_size": 50,               # Размер буфера аудио чанков
    "processing_timeout": 30.0,            # Таймаут обработки (секунды)
    "max_concurrent_requests": 10,         # Максимум одновременных запросов
    
    # Ошибки и восстановление
    "max_processing_errors": 20,           # Максимум ошибок обработки
    "error_recovery_enabled": True,        # Включить восстановление после ошибок
    "reconnect_attempts": 5,               # Попытки переподключения
    "reconnect_delay": 3.0,               # Задержка между попытками
    
    # Мониторинг
    "health_check_enabled": True,          # Проверка здоровья системы
    "health_check_interval": 60,           # Интервал проверки (секунды)
    "stats_update_interval": 10,           # Интервал обновления статистики
    
    # Память и ресурсы
    "memory_limit_mb": 2048,              # Лимит памяти
    "cpu_limit_percent": 80,              # Лимит CPU
    "gpu_memory_fraction": 0.8,           # Доля GPU памяти
}

# =============================================================================
# НАСТРОЙКИ ЛОГИРОВАНИЯ
# =============================================================================

LOGGING_CONFIG = {
    # Основные настройки
    "level": "INFO",                       # DEBUG, INFO, WARNING, ERROR
    "format": "%(asctime)s - %(levelname)s - %(message)s",
    
    # Файлы логов
    "log_to_file": True,                   # Логирование в файл
    "log_file": "logs/enhanced_server.log",
    "log_rotation": True,                  # Ротация логов
    "max_log_size_mb": 100,               # Максимальный размер лога
    "backup_count": 5,                     # Количество backup файлов
    
    # Специализированные логи
    "audio_log_enabled": True,             # Лог аудио операций
    "command_log_enabled": True,           # Лог команд
    "error_log_enabled": True,             # Отдельный лог ошибок
    "performance_log_enabled": True,       # Лог производительности
    
    # Детализация
    "log_audio_chunks": False,             # Логировать каждый аудио чанк
    "log_vad_scores": False,              # Логировать VAD scores
    "log_transcriptions": True,            # Логировать транскрипции
    "log_command_processing": True,        # Логировать обработку команд
    "log_rag_operations": True,           # Логировать RAG операции
}

# =============================================================================
# НАСТРОЙКИ БЕЗОПАСНОСТИ
# =============================================================================

SECURITY_CONFIG = {
    # Аутентификация
    "auth_enabled": False,                 # Включить аутентификацию
    "api_key_required": False,            # Требовать API ключ
    "session_timeout": 3600,              # Таймаут сессии (секунды)
    
    # Ограничения доступа
    "allowed_ips": [],                     # Разрешенные IP (пустой = все)
    "rate_limiting": False,                # Ограничение частоты запросов
    "max_requests_per_minute": 60,         # Максимум запросов в минуту
    
    # Данные
    "encrypt_recordings": False,           # Шифрование записей
    "anonymize_logs": True,               # Анонимизация логов
    "data_retention_days": 90,            # Срок хранения данных
    
    # Сеть
    "use_ssl": False,                     # Использовать SSL/TLS
    "ssl_cert_file": None,                # Путь к SSL сертификату
    "ssl_key_file": None,                 # Путь к SSL ключу
}

# =============================================================================
# НАСТРОЙКИ ИНТЕГРАЦИИ
# =============================================================================

INTEGRATION_CONFIG = {
    # Веб-интерфейс
    "web_interface_enabled": True,         # Включить веб-интерфейс
    "cors_enabled": True,                 # Включить CORS
    "allowed_origins": ["*"],             # Разрешенные origins
    
    # API
    "rest_api_enabled": False,            # REST API
    "api_documentation": True,            # Документация API
    "api_versioning": True,               # Версионирование API
    
    # Внешние системы
    "database_integration": False,        # Интеграция с БД
    "cloud_backup": False,                # Бэкап в облако
    "notification_webhook": None,         # Webhook для уведомлений
    
    # Экспорт данных
    "export_formats": ["json", "csv"],    # Форматы экспорта
    "auto_export": False,                 # Автоматический экспорт
    "export_schedule": "daily",           # Расписание экспорта
}

# =============================================================================
# ЭКСПЕРИМЕНТАЛЬНЫЕ ФУНКЦИИ
# =============================================================================

EXPERIMENTAL_CONFIG = {
    # Новые алгоритмы
    "advanced_vad": False,                # Продвинутый VAD
    "neural_audio_enhancement": False,    # Нейронное улучшение аудио
    "real_time_adaptation": False,        # Адаптация в реальном времени
    
    # Мультимодальность
    "video_support": False,               # Поддержка видео
    "gesture_recognition": False,         # Распознавание жестов
    "eye_tracking": False,                # Отслеживание взгляда
    
    # ИИ функции
    "auto_report_generation": False,      # Автогенерация отчетов
    "predictive_analytics": False,        # Предиктивная аналитика
    "anomaly_detection": False,           # Детекция аномалий
}

# =============================================================================
# ФУНКЦИИ КОНФИГУРАЦИИ
# =============================================================================

def get_config():
    """Получить полную конфигурацию"""
    return {
        "audio_recording": AUDIO_RECORDING_CONFIG,
        "rag": RAG_CONFIG,
        "llm_periodontal": LLM_PERIODONTAL_CONFIG,
        "standard_periodontal": STANDARD_PERIODONTAL_CONFIG,
        "system_priorities": SYSTEM_PRIORITIES,
        "confidence_thresholds": CONFIDENCE_THRESHOLDS,
        "performance": PERFORMANCE_CONFIG,
        "logging": LOGGING_CONFIG,
        "security": SECURITY_CONFIG,
        "integration": INTEGRATION_CONFIG,
        "experimental": EXPERIMENTAL_CONFIG,
    }

def validate_config():
    """Валидация конфигурации"""
    errors = []
    
    # Проверка OpenAI API ключа
    if RAG_CONFIG["enabled"] and not RAG_CONFIG["openai_api_key"]:
        errors.append("OpenAI API key required for RAG system")
    
    # Проверка директории записей
    recording_dir = Path(AUDIO_RECORDING_CONFIG["directory"])
    if AUDIO_RECORDING_CONFIG["enabled"]:
        try:
            recording_dir.mkdir(exist_ok=True)
        except Exception as e:
            errors.append(f"Cannot create recording directory: {e}")
    
    # Проверка ограничений
    if AUDIO_RECORDING_CONFIG["max_recordings_per_day"] <= 0:
        errors.append("max_recordings_per_day must be positive")
    
    if PERFORMANCE_CONFIG["processing_timeout"] <= 0:
        errors.append("processing_timeout must be positive")
    
    # Проверка портов
    if not (1024 <= ASR_PORT <= 65535):
        errors.append("ASR_PORT must be between 1024 and 65535")
    
    if not (1024 <= WEB_PORT <= 65535):
        errors.append("WEB_PORT must be between 1024 and 65535")
    
    if ASR_PORT == WEB_PORT:
        errors.append("ASR_PORT and WEB_PORT must be different")
    
    return errors

def setup_directories():
    """Создание необходимых директорий"""
    directories = [
        AUDIO_RECORDING_CONFIG["directory"],
        "logs",
        "data",
        "cache",
        "exports"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)

def get_system_info():
    """Получить информацию о системе"""
    import platform
    import psutil
    
    return {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "cpu_count": psutil.cpu_count(),
        "memory_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "disk_free_gb": round(psutil.disk_usage('.').free / (1024**3), 2),
    }

def print_config_summary():
    """Вывести краткую сводку конфигурации"""
    print("\n" + "=" * 60)
    print("📋 ENHANCED SERVER CONFIGURATION SUMMARY")
    print("=" * 60)
    
    print(f"🎤 Audio Recording: {'✅ ENABLED' if AUDIO_RECORDING_CONFIG['enabled'] else '❌ DISABLED'}")
    if AUDIO_RECORDING_CONFIG['enabled']:
        print(f"   📁 Directory: {AUDIO_RECORDING_CONFIG['directory']}")
        print(f"   📈 Max per day: {AUDIO_RECORDING_CONFIG['max_recordings_per_day']}")
        print(f"   🗓️ Keep days: {AUDIO_RECORDING_CONFIG['keep_recordings_days']}")
    
    print(f"🧠 RAG System: {'✅ ENABLED' if RAG_CONFIG['enabled'] else '❌ DISABLED'}")
    if RAG_CONFIG['enabled']:
        print(f"   🤖 Model: {RAG_CONFIG['model']}")
        print(f"   🎯 Confidence: {RAG_CONFIG['intent_confidence_threshold']}")
        print(f"   🔑 API Key: {'✅ SET' if RAG_CONFIG['openai_api_key'] else '❌ MISSING'}")
    
    print(f"🤖 LLM Periodontal: {'✅ ENABLED' if LLM_PERIODONTAL_CONFIG['enabled'] else '❌ DISABLED'}")
    if LLM_PERIODONTAL_CONFIG['enabled']:
        print(f"   🔧 ASR Correction: {'✅ ON' if LLM_PERIODONTAL_CONFIG['asr_correction_enabled'] else '❌ OFF'}")
        print(f"   🎯 Confidence: {LLM_PERIODONTAL_CONFIG['confidence_threshold']}")
    
    print(f"🦷 Standard Periodontal: {'✅ ENABLED' if STANDARD_PERIODONTAL_CONFIG['enabled'] else '❌ DISABLED'}")
    
    print(f"\n🌐 Server Ports:")
    print(f"   ⚡ ASR: {ASR_PORT}")
    print(f"   🌐 Web: {WEB_PORT}")
    
    print(f"\n⚡ Performance:")
    print(f"   🔄 Processing timeout: {PERFORMANCE_CONFIG['processing_timeout']}s")
    print(f"   📊 Buffer size: {PERFORMANCE_CONFIG['audio_buffer_size']}")
    print(f"   🔀 Max concurrent: {PERFORMANCE_CONFIG['max_concurrent_requests']}")
    
    system_info = get_system_info()
    print(f"\n💻 System Info:")
    print(f"   🖥️ Platform: {system_info['platform']}")
    print(f"   🐍 Python: {system_info['python_version']}")
    print(f"   ⚙️ CPU: {system_info['cpu_count']} cores")
    print(f"   💾 RAM: {system_info['memory_gb']} GB")
    print(f"   💽 Disk free: {system_info['disk_free_gb']} GB")
    
    print("=" * 60 + "\n")

if __name__ == "__main__":
    # Валидация конфигурации
    errors = validate_config()
    if errors:
        print("❌ Configuration errors:")
        for error in errors:
            print(f"   • {error}")
        exit(1)
    
    # Создание директорий
    setup_directories()
    
    # Вывод сводки
    print_config_summary()
    
    print("✅ Configuration validated successfully!")
    print("🚀 Ready to start Enhanced Server!")

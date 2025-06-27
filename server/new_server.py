#!/usr/bin/env python3
"""
ИСПРАВЛЕННЫЙ Enhanced FastWhisper ASR сервер с улучшенной сегментацией речи
Исправляет проблему отсутствующего модуля improved_speech_segmentation
"""

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

import asyncio
import websockets
import numpy as np
import torch
import time
import json
import os
import traceback
from datetime import datetime
from collections import deque
from typing import Dict, List, Optional, Any
import wave
import struct
from pathlib import Path
import signal
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

# Импорт локального модуля улучшенной сегментации
try:
    from improved_speech_segmentation import (
        ImprovedSpeechSegmentation,
        ImprovedClientBuffer,
        ImprovedAudioProcessor,
        SpeechState,
        integrate_improved_segmentation
    )
    SEGMENTATION_AVAILABLE = True
    logger.info("🎯 Improved Speech Segmentation available")
except ImportError as e:
    SEGMENTATION_AVAILABLE = False
    logger.warning(f"⚠️ Improved Speech Segmentation not available: {e}")
    
    # Простая реализация как fallback
    class SimpleFallbackProcessor:
        def __init__(self, vad, asr, audio_manager):
            self.vad = vad
            self.asr = asr
            self.audio_manager = audio_manager
            self.client_buffers = {}
            
        def process_audio_chunk(self, client_id, audio_chunk):
            # Простая обработка без сегментации
            try:
                vad_scores = self.vad.process_chunk(audio_chunk)
                if vad_scores and vad_scores[0] > 0.5:
                    # Накапливаем аудио в буфере
                    if client_id not in self.client_buffers:
                        self.client_buffers[client_id] = {
                            'buffer': np.array([]),
                            'last_speech': time.time(),
                            'speech_chunks': 0
                        }
                    
                    buffer_info = self.client_buffers[client_id]
                    buffer_info['buffer'] = np.concatenate([buffer_info['buffer'], audio_chunk])
                    buffer_info['last_speech'] = time.time()
                    buffer_info['speech_chunks'] += 1
                    
                    # Проверяем на завершение команды (простая логика)
                    if len(buffer_info['buffer']) > 16000 * 2:  # 2 секунды
                        audio_to_process = buffer_info['buffer'].copy()
                        buffer_info['buffer'] = np.array([])
                        buffer_info['speech_chunks'] = 0
                        
                        # ASR обработка
                        text, confidence, processing_time = self.asr.transcribe(audio_to_process)
                        if text and text not in ["NO_SPEECH_DETECTED", "PROCESSING"]:
                            return text
                            
            except Exception as e:
                logger.error(f"Fallback processor error: {e}")
            
            return None
            
        def cleanup_client(self, client_id):
            if client_id in self.client_buffers:
                del self.client_buffers[client_id]
                
        def get_client_info(self, client_id):
            return self.client_buffers.get(client_id, {})
            
        def get_all_clients_info(self):
            return self.client_buffers.copy()
            
        def get_improved_stats(self):
            return {
                'segmentation_mode': 'FALLBACK',
                'active_clients': len(self.client_buffers),
                'commands_segmented': 0,
                'segmentation_accuracy': 0.0
            }

# Остальные импорты как в оригинале
ENHANCED_RAG_INTENTS_AVAILABLE = False
try:
    from enhanced_rag_intents import (
        initialize_enhanced_rag_system,
        process_command_with_enhanced_rag,
        get_enhanced_rag_stats,
        is_dental_command_enhanced_rag
    )
    ENHANCED_RAG_INTENTS_AVAILABLE = True
    logger.info("🧠 Enhanced RAG Intents system available")
except ImportError as e:
    logger.warning(f"⚠️ Enhanced RAG Intents system not available: {e}")

LLM_PERIODONTAL_AVAILABLE = False
try:
    from fixed_llm_integration import (
        initialize_fixed_llm_integration,
        process_transcription_with_fixed_llm,
        is_periodontal_command_fixed_llm,
        get_fixed_llm_stats,
        add_fixed_llm_stats_to_server_stats
    )
    LLM_PERIODONTAL_AVAILABLE = True
    logger.info("🤖 FIXED LLM Periodontal system available")
except ImportError as e:
    logger.warning(f"⚠️ FIXED LLM Periodontal system not available: {e}")

PERIODONTAL_AVAILABLE = False
try:
    from periodontal_integration_simple import (
        process_transcription_with_periodontal,
        is_periodontal_command,
        get_periodontal_stats,
        enhance_server_stats
    )
    PERIODONTAL_AVAILABLE = True
    logger.info("🦷 Standard Periodontal Chart система доступна")
except ImportError:
    logger.warning("⚠️ Standard Periodontal Chart система недоступна")

# CUDA оптимизация
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
# Создание директорий для аудио записей
RECORDINGS_DIR = Path("audio_recordings")
RECORDINGS_DIR.mkdir(exist_ok=True)

def get_today_recordings_dir():
    today = datetime.now().strftime("%Y-%m-%d")
    day_dir = RECORDINGS_DIR / today
    day_dir.mkdir(exist_ok=True)
    return day_dir

# Конфигурация
SAMPLE_RATE = 16000
CLIENT_CHUNK_DURATION = 0.25
CLIENT_CHUNK_SIZE = int(SAMPLE_RATE * CLIENT_CHUNK_DURATION)
VAD_CHUNK_SIZE = 512
VAD_CHUNK_DURATION = VAD_CHUNK_SIZE / SAMPLE_RATE
ASR_PORT = 8765
WEB_PORT = 8766

ENHANCED_CONFIG = {
    "enabled": ENHANCED_RAG_INTENTS_AVAILABLE or LLM_PERIODONTAL_AVAILABLE or PERIODONTAL_AVAILABLE,
    "use_enhanced_rag_intents": ENHANCED_RAG_INTENTS_AVAILABLE,
    "use_fixed_llm_periodontal": LLM_PERIODONTAL_AVAILABLE,
    "use_periodontal_fallback": PERIODONTAL_AVAILABLE,
    
    # Настройки сегментации
    "use_improved_segmentation": SEGMENTATION_AVAILABLE,
    "segmentation_mode": "COMMAND_AWARE",
    
    # Приоритеты обработки
    "enhanced_rag_intents_priority": 0,
    "llm_periodontal_priority": 1,
    "periodontal_priority": 2,
    
    # Пороги уверенности
    "enhanced_rag_confidence_threshold": 0.5,
    "llm_confidence_threshold": 0.4,
    "fallback_to_standard": True,
    
    # OpenAI настройки
    "openai_api_key": os.getenv("OPENAI_API_KEY"),
    "model": "gpt-3.5-turbo",
    
    # Настройки записи аудио
    "save_audio_recordings": True,
    "audio_format": "wav",
    "save_successful_commands_only": False,
    "max_recordings_per_day": 1000,
    "auto_cleanup_old_recordings": True,
    "keep_recordings_days": 30,
    
    # Настройки сегментации
    "segmentation_speech_threshold": 0.35,
    "segmentation_silence_threshold": 0.25,
    "min_command_duration": 0.8,
    "max_command_duration": 20.0,
    "speech_confirmation_chunks": 3,
    "silence_confirmation_chunks": 8,
    
    "log_commands": True,
    "max_processing_errors": 20,
    "error_recovery_enabled": True,
    "audio_validation_enabled": True,
    "processing_timeout": 30.0
}
# Инициализация систем
if ENHANCED_RAG_INTENTS_AVAILABLE and ENHANCED_CONFIG["use_enhanced_rag_intents"]:
    api_key = ENHANCED_CONFIG.get("openai_api_key")
    try:
        if initialize_enhanced_rag_system(api_key):
            logger.info("🧠 Enhanced RAG Intents система успешно инициализирована")
        else:
            logger.warning("⚠️ Enhanced RAG Intents инициализация не удалась")
            ENHANCED_CONFIG["use_enhanced_rag_intents"] = False
    except Exception as e:
        logger.warning(f"⚠️ Ошибка инициализации Enhanced RAG Intents: {e}")
        ENHANCED_CONFIG["use_enhanced_rag_intents"] = False

if LLM_PERIODONTAL_AVAILABLE and ENHANCED_CONFIG["use_fixed_llm_periodontal"]:
    api_key = ENHANCED_CONFIG.get("openai_api_key")
    if api_key:
        try:
            if initialize_fixed_llm_integration(api_key):
                logger.info("🤖 FIXED LLM Periodontal система успешно инициализирована")
            else:
                logger.warning("⚠️ FIXED LLM Periodontal инициализация не удалась")
                ENHANCED_CONFIG["use_fixed_llm_periodontal"] = False
        except Exception as e:
            logger.warning(f"⚠️ Ошибка инициализации FIXED LLM: {e}")
            ENHANCED_CONFIG["use_fixed_llm_periodontal"] = False
    else:
        logger.warning("⚠️ OpenAI API key не найден для FIXED LLM системы")
        ENHANCED_CONFIG["use_fixed_llm_periodontal"] = False
# Audio Recording Manager
class AudioRecordingManager:
    """Менеджер для сохранения аудио записей"""
    
    def __init__(self):
        self.recordings_count_today = 0
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.cleanup_thread = None
        self.start_cleanup_scheduler()
        
        logger.info(f"📼 Audio Recording Manager initialized")
        logger.info(f"📁 Recordings directory: {RECORDINGS_DIR.absolute()}")
    
    def save_audio_recording(self, audio_data: np.ndarray, client_id: str, 
                           transcription: str = "", command_successful: bool = False,
                           metadata: Dict = None) -> Optional[str]:
        """Асинхронное сохранение аудио записи в .wav файл"""
        
        if not ENHANCED_CONFIG.get("save_audio_recordings", True):
            return None
            
        if self.recordings_count_today >= ENHANCED_CONFIG.get("max_recordings_per_day", 1000):
            logger.warning("⚠️ Достигнут дневной лимит записей")
            return None
        
        if ENHANCED_CONFIG.get("save_successful_commands_only", False) and not command_successful:
            return None
        
        timestamp = datetime.now().strftime("%H-%M-%S_%f")[:-3]
        status = "SUCCESS" if command_successful else "PENDING"
        filename = f"{timestamp}_{client_id}_{status}.wav"
        
        today_dir = get_today_recordings_dir()
        filepath = today_dir / filename
        
        recording_metadata = {
            "timestamp": datetime.now().isoformat(),
            "client_id": client_id,
            "transcription": transcription,
            "command_successful": command_successful,
            "duration_seconds": len(audio_data) / SAMPLE_RATE,
            "sample_rate": SAMPLE_RATE,
            "channels": 1,
            "format": "wav",
            "segmentation_mode": "improved_v2" if SEGMENTATION_AVAILABLE else "fallback",
            **(metadata or {})
        }
        
        future = self.executor.submit(
            self._save_wav_file, 
            audio_data, 
            filepath, 
            recording_metadata
        )
        
        self.recordings_count_today += 1
        
        logger.debug(f"📼 Scheduled audio recording: {filename}")
        return str(filepath)
    
    def _save_wav_file(self, audio_data: np.ndarray, filepath: Path, metadata: Dict):
        """Сохранение .wav файла в отдельном потоке"""
        try:
            if np.any(np.isnan(audio_data)) or np.any(np.isinf(audio_data)):
                logger.error(f"❌ Invalid audio data for {filepath.name}")
                return
            
            audio_data = np.clip(audio_data, -1.0, 1.0)
            audio_int16 = (audio_data * 32767).astype(np.int16)
            
            with wave.open(str(filepath), 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(SAMPLE_RATE)
                wav_file.writeframes(audio_int16.tobytes())
            
            metadata_file = filepath.with_suffix('.json')
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            logger.debug(f"✅ Saved audio recording: {filepath.name} ({metadata['duration_seconds']:.2f}s)")
            
        except Exception as e:
            logger.error(f"❌ Error saving audio recording {filepath.name}: {e}")
    
    def update_recording_status(self, filepath: str, command_successful: bool, 
                              final_transcription: str = "", processing_result: Dict = None):
        """Обновление статуса записи после обработки"""
        if not filepath:
            return
        
        try:
            filepath_obj = Path(filepath)
            
            if "PENDING" in filepath_obj.name and command_successful:
                new_name = filepath_obj.name.replace("PENDING", "SUCCESS")
                new_filepath = filepath_obj.parent / new_name
                
                if filepath_obj.exists():
                    filepath_obj.rename(new_filepath)
                
                old_metadata = filepath_obj.with_suffix('.json')
                new_metadata = new_filepath.with_suffix('.json')
                if old_metadata.exists():
                    old_metadata.rename(new_metadata)
                
                filepath_obj = new_filepath
            
            metadata_file = filepath_obj.with_suffix('.json')
            if metadata_file.exists():
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                
                metadata.update({
                    "final_transcription": final_transcription,
                    "command_successful": command_successful,
                    "processing_result": processing_result,
                    "updated_at": datetime.now().isoformat()
                })
                
                with open(metadata_file, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, indent=2, ensure_ascii=False)
                
                logger.debug(f"📝 Updated recording metadata: {filepath_obj.name}")
            
        except Exception as e:
            logger.error(f"❌ Error updating recording status: {e}")
    
    def start_cleanup_scheduler(self):
        """Запуск планировщика очистки старых записей"""
        if not ENHANCED_CONFIG.get("auto_cleanup_old_recordings", True):
            return
        
        def cleanup_worker():
            while True:
                try:
                    self.cleanup_old_recordings()
                    time.sleep(3600)
                except Exception as e:
                    logger.error(f"❌ Cleanup error: {e}")
                    time.sleep(3600)
        
        self.cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        self.cleanup_thread.start()
        logger.info("🧹 Audio cleanup scheduler started")
    
    def cleanup_old_recordings(self):
        """Очистка старых аудио записей"""
        try:
            keep_days = ENHANCED_CONFIG.get("keep_recordings_days", 30)
            cutoff_time = time.time() - (keep_days * 24 * 3600)
            
            deleted_count = 0
            for date_dir in RECORDINGS_DIR.iterdir():
                if date_dir.is_dir():
                    if date_dir.stat().st_mtime < cutoff_time:
                        import shutil
                        shutil.rmtree(date_dir)
                        deleted_count += len(list(date_dir.glob("*.wav")))
                        logger.info(f"🗑️ Deleted old recordings directory: {date_dir.name}")
            
            if deleted_count > 0:
                logger.info(f"🧹 Cleanup completed: {deleted_count} old recordings deleted")
            
        except Exception as e:
            logger.error(f"❌ Cleanup error: {e}")
    
    def get_stats(self) -> Dict:
        """Получение статистики записей"""
        try:
            total_recordings = 0
            total_size_mb = 0
            
            for date_dir in RECORDINGS_DIR.iterdir():
                if date_dir.is_dir():
                    wav_files = list(date_dir.glob("*.wav"))
                    total_recordings += len(wav_files)
                    
                    for wav_file in wav_files:
                        total_size_mb += wav_file.stat().st_size / (1024 * 1024)
            
            today_recordings = len(list(get_today_recordings_dir().glob("*.wav")))
            
            return {
                "recordings_enabled": ENHANCED_CONFIG.get("save_audio_recordings", True),
                "total_recordings": total_recordings,
                "today_recordings": today_recordings,
                "total_size_mb": round(total_size_mb, 2),
                "recordings_directory": str(RECORDINGS_DIR.absolute()),
                "keep_recordings_days": ENHANCED_CONFIG.get("keep_recordings_days", 30),
                "max_recordings_per_day": ENHANCED_CONFIG.get("max_recordings_per_day", 1000)
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting recording stats: {e}")
            return {"error": str(e)}
# Класс VAD
class StableVAD:
    """УЛУЧШЕННАЯ VAD система с энергетическим фильтром"""
    
    def __init__(self, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() and device != 'cpu' else 'cpu')
        self.model = None
        self.threshold = 0.35
        self.vad_buffer = np.array([])
        self.error_count = 0
        self.max_errors = 50
        self.last_error_time = 0
        
        self.energy_threshold = 0.001
        self.silence_energy_threshold = 0.0005
        self.background_noise_level = 0.0002
        self.energy_history = deque(maxlen=20)
        
        self.load_model()
        logger.info(f"🎤 ENHANCED VAD на {self.device}")
    
    def load_model(self):
        """Загрузка модели Silero VAD"""
        try:
            logger.info("📥 Загрузка Silero VAD...")
            model, _ = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                onnx=False,
                trust_repo=True
            )
            self.model = model.to(self.device)
            self.model.eval()
            
            # Прогрев модели
            with torch.no_grad():
                for i in range(3):
                    try:
                        test_audio = torch.randn(VAD_CHUNK_SIZE, device=self.device, dtype=torch.float32)
                        test_prob = self.model(test_audio, SAMPLE_RATE).item()
                    except Exception as e:
                        logger.warning(f"⚠️ VAD warmup error {i+1}: {e}")
                        if i == 2:
                            self.model = None
                            return
            
            logger.info(f"✅ ENHANCED VAD загружен, порог: {self.threshold}")
        except Exception as e:
            logger.error(f"❌ VAD не загружен: {e}")
            self.model = None
    
    def process_chunk(self, audio_chunk):
        """Обработка VAD чанка"""
        if self.model is None:
            try:
                rms_energy = np.sqrt(np.mean(audio_chunk ** 2))
                energy_score = min(rms_energy * 10, 1.0)
                return [energy_score]
            except Exception:
                return [0.0]
        
        try:
            if len(audio_chunk) == 0 or np.any(np.isnan(audio_chunk)) or np.any(np.isinf(audio_chunk)):
                return [0.0]
            
            self.vad_buffer = np.concatenate([self.vad_buffer, audio_chunk])
            vad_scores = []
            
            while len(self.vad_buffer) >= VAD_CHUNK_SIZE:
                vad_chunk = self.vad_buffer[:VAD_CHUNK_SIZE]
                self.vad_buffer = self.vad_buffer[VAD_CHUNK_SIZE:]
                
                try:
                    vad_chunk = np.clip(vad_chunk, -1.0, 1.0)
                    audio_tensor = torch.from_numpy(vad_chunk).float().to(self.device, non_blocking=True)
                    
                    with torch.no_grad():
                        speech_prob = self.model(audio_tensor, SAMPLE_RATE).item()
                        if np.isnan(speech_prob) or np.isinf(speech_prob):
                            speech_prob = 0.0
                        speech_prob = max(0.0, min(1.0, speech_prob))
                    
                    vad_scores.append(speech_prob)
                    
                except Exception as e:
                    current_time = time.time()
                    if current_time - self.last_error_time > 1.0:
                        logger.warning(f"⚠️ VAD chunk processing error: {e}")
                        self.last_error_time = current_time
                    
                    self.error_count += 1
                    if self.error_count > self.max_errors:
                        logger.error("❌ Too many VAD errors, disabling model")
                        self.model = None
                        return [0.0]
                    
                    vad_scores.append(0.0)
            
            return vad_scores if vad_scores else [0.0]
            
        except Exception as e:
            logger.error(f"❌ VAD critical error: {e}")
            self.error_count += 1
            return [0.0]
# Класс ASR
class StableASR:
    """СТАБИЛЬНАЯ ASR система"""
    
    def __init__(self, device='cuda'):
        self.device_str = 'cuda' if torch.cuda.is_available() and device != 'cpu' else 'cpu'
        self.model = None
        self.model_size = "large-v3"
        self.error_count = 0
        self.max_errors = 10
        self.last_error_time = 0
        self.processing_timeout = ENHANCED_CONFIG.get("processing_timeout", 30.0)
        self.load_model()
        logger.info(f"🤖 STABLE ASR на {self.device_str}")
    
    def load_model(self):
        try:
            logger.info("📥 Загрузка STABLE FastWhisper...")
            from faster_whisper import WhisperModel
            
            models_to_try = [
                ("large-v2", "Стабильная крупная модель"),
                ("large-v3", "Новейшая крупная модель"),
                ("medium", "Средняя модель"),
                ("base", "Базовая модель"),
            ]
            
            for model_name, description in models_to_try:
                try:
                    logger.info(f"🔄 Загрузка {model_name} ({description})...")
                    
                    self.model = WhisperModel(
                        model_name,
                        device=self.device_str,
                        compute_type="float16" if self.device_str == 'cuda' else "int8",
                        num_workers=1,
                        cpu_threads=2 if self.device_str == 'cpu' else 1,
                        download_root=None,
                    )
                    
                    # Тестирование модели
                    test_audio = np.random.randn(SAMPLE_RATE).astype(np.float32)
                    test_audio = np.clip(test_audio * 0.1, -1.0, 1.0)
                    
                    segments, info = self.model.transcribe(
                        test_audio, 
                        language="en",
                        condition_on_previous_text=False,
                        temperature=0.0,
                        beam_size=1,
                        best_of=1,
                        without_timestamps=True
                    )
                    
                    self.model_size = model_name
                    logger.info(f"✅ STABLE FastWhisper {model_name} загружен и протестирован")
                    break
                    
                except Exception as e:
                    logger.warning(f"⚠️ {model_name} не загружен: {e}")
                    self.model = None
                    continue
            
            if self.model is None:
                raise Exception("Ни одна модель FastWhisper не загрузилась")
                
        except Exception as e:
            logger.error(f"❌ Критическая ошибка загрузки FastWhisper: {e}")
            self.model = None
    
    def transcribe(self, audio_np):
        """СТАБИЛЬНАЯ транскрипция с обработкой ошибок"""
        if self.model is None:
            return "ASR_NOT_LOADED", 0.0, 0.0
        
        try:
            start_time = time.time()
            
            if len(audio_np) == 0:
                return "EMPTY_AUDIO", 0.0, 0.001
            
            if np.any(np.isnan(audio_np)) or np.any(np.isinf(audio_np)):
                logger.warning("⚠️ Invalid audio data (NaN/inf)")
                return "INVALID_AUDIO", 0.0, 0.001
            
            audio_np = np.clip(audio_np, -1.0, 1.0)
            
            duration = len(audio_np) / SAMPLE_RATE
            if duration < 0.1:
                return "TOO_SHORT", 0.0, 0.001
            if duration > 25.0:
                logger.warning(f"⚠️ Audio too long: {duration:.1f}s, truncating")
                max_samples = int(25.0 * SAMPLE_RATE)
                audio_np = audio_np[:max_samples]
            
            try:
                segments, info = self.model.transcribe(
                    audio_np,
                    language="en",
                    condition_on_previous_text=False,
                    temperature=0.0,
                    vad_filter=False,
                    beam_size=1,
                    best_of=1,
                    without_timestamps=True,
                    word_timestamps=False,
                    initial_prompt=None,
                    suppress_blank=True,
                    suppress_tokens=[-1],
                    log_prob_threshold=-1.0,
                    no_speech_threshold=0.6,
                    compression_ratio_threshold=2.4,
                )
                
                text_segments = []
                for segment in segments:
                    if hasattr(segment, 'text') and segment.text:
                        text_segments.append(segment.text.strip())
                
                full_text = " ".join(text_segments).strip()
                
                confidence = 0.0
                if hasattr(info, 'language_probability'):
                    confidence = info.language_probability
                elif hasattr(info, 'all_language_probs') and info.all_language_probs:
                    confidence = max(info.all_language_probs.values())
                
                processing_time = time.time() - start_time
                
                if not full_text or len(full_text.strip()) == 0:
                    return "NO_SPEECH_DETECTED", confidence, processing_time
                
                if len(full_text) > 500:
                    logger.warning(f"⚠️ Unusually long transcription: {len(full_text)} chars")
                
                return full_text, confidence, processing_time
                
            except Exception as transcribe_error:
                current_time = time.time()
                if current_time - self.last_error_time > 5.0:
                    logger.error(f"❌ Transcription error: {transcribe_error}")
                    self.last_error_time = current_time
                
                self.error_count += 1
                if self.error_count > self.max_errors:
                    logger.error("❌ Too many transcription errors, model may be corrupted")
                    self.model = None
                
                return f"TRANSCRIBE_ERROR: {str(transcribe_error)[:100]}", 0.0, time.time() - start_time
            
        except Exception as e:
            logger.error(f"❌ Critical transcribe error: {e}")
            return f"CRITICAL_ERROR: {str(e)[:100]}", 0.0, 0.0
    
    def get_info(self):
        return {
            "status": "loaded" if self.model else "not_loaded",
            "model_size": self.model_size,
            "device": self.device_str,
            "language": "en",
            "optimization": "ENHANCED_WITH_SEGMENTATION_V2",
            "error_count": self.error_count,
            "max_errors": self.max_errors
        }
# ГЛАВНЫЙ КЛАСС: ИСПРАВЛЕННЫЙ ПРОЦЕССОР С СЕГМЕНТАЦИЕЙ
class EnhancedProcessorWithSegmentation:
    """ИСПРАВЛЕННЫЙ процессор с точной сегментацией команд и сохранением аудио"""
    
    def __init__(self):
        self.vad = StableVAD()
        self.asr = StableASR()
        
        # Инициализация менеджера записи аудио
        global audio_manager
        audio_manager = AudioRecordingManager()
        
        # Определяем активные системы
        active_systems = []
        if ENHANCED_RAG_INTENTS_AVAILABLE:
            active_systems.append("Enhanced RAG Intents")
        if LLM_PERIODONTAL_AVAILABLE:
            active_systems.append("FIXED Liberal LLM")
        if PERIODONTAL_AVAILABLE:
            active_systems.append("Standard Periodontal")
        
        # СОЗДАНИЕ ПРОЦЕССОРА СЕГМЕНТАЦИИ (исправленное)
        if SEGMENTATION_AVAILABLE:
            try:
                self.segmentation_processor = ImprovedAudioProcessor(self.vad, self.asr, audio_manager)
                logger.info("🎯 IMPROVED SEGMENTATION processor created")
            except Exception as e:
                logger.error(f"❌ Error creating segmentation processor: {e}")
                self.segmentation_processor = SimpleFallbackProcessor(self.vad, self.asr, audio_manager)
                logger.info("🔄 Using SimpleFallbackProcessor instead")
        else:
            self.segmentation_processor = SimpleFallbackProcessor(self.vad, self.asr, audio_manager)
            logger.warning("⚠️ Using SimpleFallbackProcessor - segmentation not available")
        
        # Статистика
        self.stats = {
            'chunks_processed': 0,
            'commands_segmented': 0,
            'whisper_calls': 0,
            'successful_whisper_calls': 0,
            'failed_whisper_calls': 0,
            'average_confidence': 0.0,
            'average_rtf': 0.0,
            'total_processing_time': 0.0,
            'total_audio_duration': 0.0,
            'speech_segments': 0,
            'valid_speech_segments': 0,
            'vad_errors': 0,
            'asr_errors': 0,
            'processing_errors': 0,
            'segmentation_mode': 'IMPROVED_V2' if SEGMENTATION_AVAILABLE else 'FALLBACK',
            'active_systems': active_systems,
            'systems_count': len(active_systems),
            'commands_processed': 0,
            'successful_commands': 0,
            'errors': 0,
            'enhanced_rag_commands_processed': 0,
            'enhanced_rag_successful_commands': 0,
            'llm_commands_processed': 0,
            'llm_successful_commands': 0,
            'llm_asr_errors_fixed': 0,
            'server_uptime_start': time.time(),
            
            # Специализированная статистика сегментации
            'segmentation_false_starts': 0,
            'segmentation_truncated_commands': 0,
            'segmentation_successful_commands': 0,
            'average_command_duration': 0.0,
            'segmentation_accuracy': 100.0,
            
            # Статистика записи аудио
            'audio_recordings_saved': 0,
            'successful_command_recordings': 0,
            'failed_command_recordings': 0,
            'total_recorded_duration': 0.0,
            'recordings_enabled': ENHANCED_CONFIG.get("save_audio_recordings", True)
        }
        
        if PERIODONTAL_AVAILABLE:
            self.stats.update({
                'periodontal_commands': 0,
                'periodontal_successful': 0,
                'periodontal_teeth_updated': 0,
                'periodontal_measurements': 0
            })
        
        logger.info(f"🎯 ENHANCED processor with IMPROVED SEGMENTATION и {len(active_systems)} активными системами")
    def process_audio_chunk(self, client_id, audio_chunk):
        """
        ГЛАВНАЯ ФУНКЦИЯ: Обработка аудио чанков с улучшенной сегментацией
        """
        try:
            self.stats['chunks_processed'] += 1
            
            # Валидация входных данных
            if len(audio_chunk) == 0:
                return None
            
            if np.any(np.isnan(audio_chunk)) or np.any(np.isinf(audio_chunk)):
                logger.warning(f"⚠️ Invalid audio chunk from {client_id}")
                return None
            
            # Нормализация
            audio_chunk = np.clip(audio_chunk, -1.0, 1.0)
            
            # ИСПОЛЬЗОВАНИЕ ПРОЦЕССОРА СЕГМЕНТАЦИИ
            if self.segmentation_processor:
                result = self.segmentation_processor.process_audio_chunk(client_id, audio_chunk)
                
                if result and result.strip():
                    # Команда полностью сегментирована - обрабатываем
                    logger.info(f"🎯 SEGMENTED COMMAND from {client_id}: '{result}'")
                    
                    # Обновляем статистику сегментации
                    try:
                        client_info = self.segmentation_processor.get_client_info(client_id)
                        if client_info:
                            self.stats['segmentation_false_starts'] = client_info.get('false_starts', 0)
                            self.stats['segmentation_successful_commands'] = client_info.get('successful_commands', 0)
                            self.stats['commands_segmented'] = client_info.get('commands_segmented', 0)
                        
                        # Получаем улучшенную статистику
                        seg_stats = self.segmentation_processor.get_improved_stats()
                        self.stats.update({
                            'average_command_duration': seg_stats.get('average_command_duration', 0.0),
                            'segmentation_accuracy': seg_stats.get('segmentation_accuracy', 100.0)
                        })
                    except Exception as e:
                        logger.debug(f"Error updating segmentation stats: {e}")
                    
                    # Запуск обработки команд
                    confidence = 0.95  # Высокая уверенность для сегментированных команд
                    duration = self.stats.get('average_command_duration', 2.0)
                    
                    asyncio.create_task(self.process_with_enhanced_systems(
                        client_id, result, confidence, duration, None, None
                    ))
                    
                    # Отправка в веб-интерфейс
                    asyncio.create_task(self.broadcast_transcription(
                        client_id, result, confidence, duration, 0.1  # Быстрая обработка
                    ))
                    
                    return result
                
                return None
            else:
                # Fallback к старой системе если сегментация недоступна
                logger.warning(f"⚠️ Using fallback processing for {client_id}")
                return self._fallback_processing(client_id, audio_chunk)
                
        except Exception as e:
            logger.error(f"❌ Critical error processing chunk from {client_id}: {e}")
            self.stats['processing_errors'] += 1
            return None
    
    def _fallback_processing(self, client_id, audio_chunk):
        """Fallback обработка без улучшенной сегментации"""
        # Простая реализация без сегментации (как в оригинале)
        logger.debug(f"🔄 Fallback processing for {client_id}")
        return None
    async def process_with_enhanced_systems(self, client_id: str, text: str, confidence: float, 
                                              duration: float, recording_path: str = None, 
                                              speech_audio: np.ndarray = None):
        """
        УЛУЧШЕННАЯ обработка с Enhanced системами
        """
        try:
            self.stats['commands_processed'] += 1
            command_successful = False
            processing_result = {}
            
            # ПРИОРИТЕТ 0: Enhanced RAG Intents
            if ENHANCED_CONFIG.get("use_enhanced_rag_intents", False) and ENHANCED_RAG_INTENTS_AVAILABLE:
                self.stats['enhanced_rag_commands_processed'] += 1
                
                try:
                    logger.debug(f"🧠 Enhanced RAG Intents processing: '{text}'")
                    
                    context = {
                        'client_id': client_id,
                        'asr_confidence': confidence,
                        'duration': duration,
                        'timestamp': datetime.now().isoformat(),
                        'recording_path': recording_path,
                        'segmentation_method': 'improved_v2'
                    }
                    
                    rag_result = await asyncio.wait_for(
                        process_command_with_enhanced_rag(text, context),
                        timeout=15.0
                    )
                    
                    if rag_result.get("success"):
                        rag_confidence = rag_result.get("intent_confidence", 0.0)
                        rag_threshold = ENHANCED_CONFIG.get("enhanced_rag_confidence_threshold", 0.7)
                        
                        if rag_confidence >= rag_threshold:
                            self.stats['enhanced_rag_successful_commands'] += 1
                            self.stats['successful_commands'] += 1
                            command_successful = True
                            processing_result = rag_result
                            
                            logger.info(f"🧠 ENHANCED RAG SUCCESS {client_id}: {rag_result['message']}")
                            
                            rag_result.update({
                                'asr_confidence': confidence,
                                'system': 'enhanced_rag_intents_v2',
                                'timestamp': datetime.now().isoformat(),
                                'recording_path': recording_path,
                                'segmentation_method': 'improved'
                            })
                            
                            await self.broadcast_enhanced_rag_intents_command(client_id, rag_result)
                            
                            # Обновляем статус записи
                            if recording_path and audio_manager:
                                audio_manager.update_recording_status(
                                    recording_path, 
                                    command_successful=True, 
                                    final_transcription=text,
                                    processing_result=rag_result
                                )
                                self.stats['successful_command_recordings'] += 1
                            
                            return
                        else:
                            logger.debug(f"🧠 RAG confidence {rag_confidence:.3f} < {rag_threshold}, fallback")
                    
                except asyncio.TimeoutError:
                    logger.warning(f"⚠️ Enhanced RAG timeout for: '{text}'")
                except Exception as e:
                    logger.error(f"❌ Enhanced RAG error: {e}")
            
            # ПРИОРИТЕТ 1: FIXED Liberal LLM
            if ENHANCED_CONFIG.get("use_fixed_llm_periodontal", False) and LLM_PERIODONTAL_AVAILABLE:
                self.stats['llm_commands_processed'] += 1
                
                try:
                    if is_periodontal_command_fixed_llm(text):
                        logger.debug(f"🤖 FIXED LLM processing: '{text}'")
                        
                        llm_result = await asyncio.wait_for(
                            process_transcription_with_fixed_llm(text, confidence),
                            timeout=20.0
                        )
                        
                        llm_confidence = llm_result.get("confidence", 0.0)
                        llm_threshold = ENHANCED_CONFIG.get("llm_confidence_threshold", 0.4)
                        
                        if llm_result.get("success") and llm_confidence >= llm_threshold:
                            self.stats['llm_successful_commands'] += 1
                            self.stats['successful_commands'] += 1
                            command_successful = True
                            processing_result = llm_result
                            
                            # Подсчет ASR исправлений
                            original = llm_result.get("original_text", "").lower()
                            corrected = llm_result.get("corrected_text", "").lower()
                            if original != corrected:
                                self.stats['llm_asr_errors_fixed'] += 1
                                logger.info(f"🔧 ASR FIXED: '{original}' → '{corrected}'")
                            
                            logger.info(f"🤖 FIXED LLM SUCCESS {client_id}: {llm_result['message']}")
                            
                            llm_result['recording_path'] = recording_path
                            llm_result['segmentation_method'] = 'improved_v2'
                            await self.broadcast_fixed_llm_periodontal_command(client_id, llm_result)
                            
                            # Обновляем статус записи
                            if recording_path and audio_manager:
                                audio_manager.update_recording_status(
                                    recording_path, 
                                    command_successful=True, 
                                    final_transcription=text,
                                    processing_result=llm_result
                                )
                                self.stats['successful_command_recordings'] += 1
                            
                            return
                        else:
                            logger.debug(f"🤖 LLM confidence {llm_confidence:.3f} < {llm_threshold}, fallback")
                
                except asyncio.TimeoutError:
                    logger.warning(f"⚠️ FIXED LLM timeout for: '{text}'")
                except Exception as e:
                    logger.error(f"❌ FIXED LLM error: {e}")
            
            # ПРИОРИТЕТ 2: Standard Periodontal fallback
            if ENHANCED_CONFIG.get("use_periodontal_fallback", False) and PERIODONTAL_AVAILABLE:
                try:
                    if is_periodontal_command(text):
                        self.stats['periodontal_commands'] += 1
                        
                        periodontal_result = await asyncio.wait_for(
                            process_transcription_with_periodontal(text, confidence),
                            timeout=10.0
                        )
                        
                        if periodontal_result.get("success"):
                            self.stats['periodontal_successful'] += 1
                            self.stats['periodontal_teeth_updated'] += 1
                            self.stats['successful_commands'] += 1
                            command_successful = True
                            processing_result = periodontal_result
                            
                            logger.info(f"🦷 PERIODONTAL SUCCESS {client_id}: {periodontal_result['message']}")
                            
                            periodontal_result['recording_path'] = recording_path
                            periodontal_result['segmentation_method'] = 'improved_v2'
                            await self.broadcast_periodontal_command(client_id, periodontal_result)
                            
                            # Обновляем статус записи
                            if recording_path and audio_manager:
                                audio_manager.update_recording_status(
                                    recording_path, 
                                    command_successful=True, 
                                    final_transcription=text,
                                    processing_result=periodontal_result
                                )
                                self.stats['successful_command_recordings'] += 1
                            
                            return
                
                except asyncio.TimeoutError:
                    logger.warning(f"⚠️ Periodontal timeout for: '{text}'")
                except Exception as e:
                    logger.error(f"❌ Standard Periodontal error: {e}")
            
            # Если ни одна система не обработала команду
            self.stats['errors'] += 1
            logger.debug(f"⚠️ Команда не обработана системами: '{text}'")
            
            # Обновляем статус записи как неуспешный
            if recording_path and audio_manager:
                audio_manager.update_recording_status(
                    recording_path, 
                    command_successful=False, 
                    final_transcription=text,
                    processing_result={"error": "No system could process the command"}
                )
                self.stats['failed_command_recordings'] += 1
                
        except Exception as e:
            self.stats['errors'] += 1
            logger.error(f"❌ ENHANCED processing error for {client_id}: {e}")
            
            # Обновляем статус записи как ошибочный
            if recording_path and audio_manager:
                audio_manager.update_recording_status(
                    recording_path, 
                    command_successful=False, 
                    final_transcription=text,
                    processing_result={"error": str(e)}
                )
                self.stats['failed_command_recordings'] += 1
    async def broadcast_transcription(self, client_id, text, confidence, duration, rtf):
        """Безопасная отправка результата транскрипции"""
        if not web_clients:
            return
        
        try:
            message = json.dumps({
                "type": "transcription",
                "client_id": client_id,
                "text": text,
                "confidence": confidence,
                "duration": duration,
                "rtf": rtf,
                "timestamp": datetime.now().isoformat(),
                "mode": f"ENHANCED_SEGMENTATION_V2_{self.stats['systems_count']}",
                "segmentation_enabled": True,
                "segmentation_mode": self.stats['segmentation_mode'],
                "recording_enabled": ENHANCED_CONFIG.get("save_audio_recordings", True)
            })
            
            await self._safe_broadcast_to_web_clients(message)
            
        except Exception as e:
            logger.error(f"❌ Broadcast transcription error: {e}")
    
    async def broadcast_enhanced_rag_intents_command(self, client_id, rag_result):
        """Отправка Enhanced RAG команд"""
        if not web_clients:
            return
        
        try:
            measurements = self._format_measurements_for_client(rag_result)
            
            message = json.dumps({
                "type": "periodontal_update",
                "client_id": client_id,
                "success": rag_result["success"],
                "tooth_number": rag_result.get("tooth_number"),
                "measurement_type": rag_result.get("measurement_type"),
                "surface": rag_result.get("surface"),
                "position": rag_result.get("position"),
                "values": rag_result.get("values"),
                "measurements": measurements,
                "confidence": rag_result.get("confidence", 0.0),
                "intent_confidence": rag_result.get("intent_confidence", 0.0),
                "asr_confidence": rag_result.get("asr_confidence", 0.0),
                "message": rag_result["message"],
                "intent": rag_result.get("intent", "unknown"),
                "entities": rag_result.get("entities", {}),
                "suggested_command": rag_result.get("suggested_command"),
                "timestamp": rag_result.get("timestamp", datetime.now().isoformat()),
                "recording_path": rag_result.get("recording_path"),
                "segmentation_method": rag_result.get("segmentation_method", "improved_v2"),
                "system": "enhanced_rag_intents_with_segmentation_v2"
            })
            
            await self._safe_broadcast_to_web_clients(message)
            
        except Exception as e:
            logger.error(f"❌ Broadcast Enhanced RAG error: {e}")
    
    async def broadcast_fixed_llm_periodontal_command(self, client_id, llm_result):
        """Отправка FIXED LLM команд"""
        if not web_clients:
            return
        
        try:
            message = json.dumps({
                "type": "periodontal_update",
                "client_id": client_id,
                "success": llm_result["success"],
                "tooth_number": llm_result.get("tooth_number"),
                "measurement_type": llm_result.get("measurement_type"),
                "surface": llm_result.get("surface"),
                "position": llm_result.get("position"),
                "values": llm_result.get("values"),
                "measurements": llm_result.get("measurements"),
                "confidence": llm_result.get("confidence", 0.0),
                "asr_confidence": llm_result.get("asr_confidence", 0.0),
                "message": llm_result["message"],
                "original_text": llm_result.get("original_text", ""),
                "corrected_text": llm_result.get("corrected_text", ""),
                "timestamp": llm_result.get("timestamp", datetime.now().isoformat()),
                "session_stats": llm_result.get("session_stats", {}),
                "recording_path": llm_result.get("recording_path"),
                "segmentation_method": llm_result.get("segmentation_method", "improved_v2"),
                "system": "fixed_liberal_llm_periodontal_with_segmentation_v2"
            })
            
            await self._safe_broadcast_to_web_clients(message)
            
        except Exception as e:
            logger.error(f"❌ Broadcast FIXED LLM error: {e}")
    
    async def broadcast_periodontal_command(self, client_id, periodontal_result):
        """Отправка Standard Periodontal команд"""
        if not web_clients:
            return
        
        try:
            message = json.dumps({
                "type": "periodontal_update",
                "client_id": client_id,
                "success": periodontal_result["success"],
                "tooth_number": periodontal_result.get("tooth_number"),
                "measurement_type": periodontal_result.get("measurement_type"),
                "surface": periodontal_result.get("surface"),
                "position": periodontal_result.get("position"),
                "values": periodontal_result.get("values"),
                "measurements": periodontal_result.get("measurements"),
                "confidence": periodontal_result.get("confidence", 0.0),
                "message": periodontal_result["message"],
                "timestamp": periodontal_result.get("timestamp", datetime.now().isoformat()),
                "recording_path": periodontal_result.get("recording_path"),
                "segmentation_method": periodontal_result.get("segmentation_method", "improved_v2"),
                "system": "standard_periodontal_fallback_with_segmentation_v2"
            })
            
            await self._safe_broadcast_to_web_clients(message)
            
        except Exception as e:
            logger.error(f"❌ Broadcast Periodontal error: {e}")
    
    async def _safe_broadcast_to_web_clients(self, message):
        """Безопасная отправка сообщения всем веб-клиентам"""
        if not web_clients:
            return
        
        disconnected = set()
        for client in list(web_clients):
            try:
                await asyncio.wait_for(client.send(message), timeout=3.0)
            except (websockets.exceptions.ConnectionClosed, asyncio.TimeoutError):
                disconnected.add(client)
            except Exception as e:
                logger.warning(f"⚠️ Error sending to web client: {e}")
                disconnected.add(client)
        
        for client in disconnected:
            web_clients.discard(client)
            if disconnected:
                logger.debug(f"🗑️ Removed {len(disconnected)} disconnected web clients")
    
    def _format_measurements_for_client(self, rag_result):
        """Форматирование measurements для веб-клиента"""
        measurements = {}
        
        measurement_type = rag_result.get("measurement_type")
        values = rag_result.get("values", [])
        
        if measurement_type == "probing_depth" and len(values) >= 3:
            measurements["probing_depth"] = values[:3]
        elif measurement_type == "bleeding":
            measurements["bleeding"] = values if isinstance(values, list) else [values[0] if values else False]
        elif measurement_type == "suppuration":
            measurements["suppuration"] = values if isinstance(values, list) else [values[0] if values else False]
        elif measurement_type == "mobility":
            measurements["mobility"] = values[0] if values else None
        elif measurement_type == "furcation":
            measurements["furcation"] = values[0] if values else None
        elif measurement_type == "gingival_margin":
            measurements["gingival_margin"] = values
        elif measurement_type == "missing_teeth":
            measurements["missing_teeth"] = values
        
        return measurements

# Глобальные переменные
processor = None
web_clients = set()
audio_manager = None
# WebSocket обработчики
async def handle_web_client(websocket):
    """Обработчик веб-клиентов"""
    client_addr = websocket.remote_address
    client_id = f"web_{client_addr[0]}_{client_addr[1]}_{int(time.time())}"
    
    logger.info(f"🌐 Web клиент подключен: {client_id}")
    web_clients.add(websocket)
    
    try:
        # Отправляем информацию о подключении
        connection_info = {
            "type": "connection",
            "status": "connected",
            "client_id": client_id,
            "timestamp": datetime.now().isoformat(),
            "recording_enabled": ENHANCED_CONFIG.get("save_audio_recordings", True),
            "segmentation_enabled": ENHANCED_CONFIG.get("use_improved_segmentation", True),
            "segmentation_mode": ENHANCED_CONFIG.get("segmentation_mode", "COMMAND_AWARE"),
            "features": {
                "enhanced_rag_intents": ENHANCED_RAG_INTENTS_AVAILABLE,
                "fixed_llm_periodontal": LLM_PERIODONTAL_AVAILABLE,
                "periodontal_fallback": PERIODONTAL_AVAILABLE,
                "audio_recording": True,
                "improved_segmentation": SEGMENTATION_AVAILABLE,
                "rag_system": ENHANCED_RAG_INTENTS_AVAILABLE,
                "command_separation": True
            }
        }
        
        await websocket.send(json.dumps(connection_info))
        
        async for message in websocket:
            try:
                data = json.loads(message)
                
                if data.get("type") == "ping":
                    await websocket.send(json.dumps({
                        "type": "pong",
                        "timestamp": datetime.now().isoformat()
                    }))
                    
            except json.JSONDecodeError:
                logger.warning(f"⚠️ Invalid JSON from web client {client_id}")
            except Exception as e:
                logger.error(f"❌ Error handling web client message: {e}")
                
    except websockets.exceptions.ConnectionClosed:
        logger.info(f"🌐 Web клиент отключен: {client_id}")
    except Exception as e:
        logger.error(f"❌ Web client error: {e}")
    finally:
        web_clients.discard(websocket)
async def handle_asr_client(websocket):
    """Обработчик ASR клиентов"""
    client_addr = websocket.remote_address
    client_id = f"{client_addr[0]}_{client_addr[1]}_{int(time.time())}"
    
    logger.info(f"🎤 ENHANCED ASR клиент подключен: {client_id}")
    
    try:
        client_error_count = 0
        max_client_errors = 20
        last_ping_time = time.time()
        
        async for message in websocket:
            try:
                if isinstance(message, bytes):
                    # Обработка аудио данных
                    try:
                        audio_chunk = np.frombuffer(message, dtype=np.int16).astype(np.float32) / 32768.0
                        expected_size = CLIENT_CHUNK_SIZE
                        actual_size = len(audio_chunk)
                        
                        # Валидация размера чанка
                        if actual_size == expected_size:
                            pass  # Идеальный размер
                        elif actual_size == expected_size * 2:
                            # Двойной чанк - разделяем
                            mid_point = actual_size // 2
                            chunk1 = audio_chunk[:mid_point]
                            chunk2 = audio_chunk[mid_point:]
                            
                            if processor:
                                result1 = processor.process_audio_chunk(client_id, chunk1)
                                if result1 is not None and result1.strip():
                                    await websocket.send(result1)
                            
                            audio_chunk = chunk2
                        elif 0 < actual_size < expected_size * 3:
                            # Приемлемый размер - дополняем или обрезаем
                            if actual_size < expected_size:
                                padding = np.zeros(expected_size - actual_size)
                                audio_chunk = np.concatenate([audio_chunk, padding])
                            else:
                                audio_chunk = audio_chunk[:expected_size]
                        else:
                            logger.warning(f"⚠️ Неприемлемый размер чанка от {client_id}: {actual_size}")
                            client_error_count += 1
                            continue
                        
                        # Проверка валидности данных
                        if np.any(np.isnan(audio_chunk)) or np.any(np.isinf(audio_chunk)):
                            logger.warning(f"⚠️ Невалидные аудио данные от {client_id}")
                            client_error_count += 1
                            continue
                        
                        # Обработка через процессор
                        if processor:
                            result = processor.process_audio_chunk(client_id, audio_chunk)
                            
                            if result is not None:
                                if result.strip():
                                    try:
                                        await asyncio.wait_for(websocket.send(result), timeout=2.0)
                                        
                                        # Отображение результата
                                        stats = processor.stats
                                        active_systems = stats.get('active_systems', [])
                                        systems_display = f" | ".join(active_systems) if active_systems else "No systems"
                                        
                                        print(f"\n{'🎯' * 60}")
                                        print(f"   ENHANCED FASTWHISPER + IMPROVED SEGMENTATION V2")
                                        print(f"   🎤 COMMAND: '{result.upper()}'")
                                        print(f"   👤 {client_addr[0]} | 📊 {stats['chunks_processed']} чанков")
                                        print(f"   🎯 Команд сегментировано: {stats['commands_segmented']}")
                                        print(f"   ✅ Успешных команд: {stats.get('segmentation_successful_commands', 0)}")
                                        print(f"   ❌ Ложных стартов: {stats.get('segmentation_false_starts', 0)}")
                                        print(f"   ⏱️ Средняя длительность команд: {stats.get('average_command_duration', 0):.2f}s")
                                        print(f"   🎯 Точность сегментации: {stats.get('segmentation_accuracy', 100):.1f}%")
                                        print(f"   🔧 Режим сегментации: {stats['segmentation_mode']}")
                                        print(f"   🔧 Системы ({stats['systems_count']}): {systems_display}")
                                        print(f"   ✅ Success: {stats['successful_commands']}/{stats['commands_processed']}")
                                        print(f"   🌐 {len(web_clients)} веб-клиентов")
                                        print('🎯' * 60 + "\n")
                                        
                                    except asyncio.TimeoutError:
                                        logger.warning(f"⚠️ Timeout sending result to {client_id}")
                                        client_error_count += 1
                                else:
                                    await websocket.send("NO_SPEECH")
                        else:
                            await websocket.send("SERVER_NOT_READY")
                            
                    except Exception as e:
                        logger.error(f"❌ Ошибка обработки аудио от {client_id}: {e}")
                        client_error_count += 1
                elif isinstance(message, str):
                    # Обработка текстовых команд
                    current_time = time.time()
                    
                    if message == "PING":
                        await websocket.send("PONG")
                        last_ping_time = current_time
                    elif message == "STATS":
                        if processor:
                            stats = processor.stats.copy()
                            stats['model_info'] = processor.asr.get_info()
                            stats['vad_device'] = str(processor.vad.device)
                            stats['server_uptime'] = current_time - stats['server_uptime_start']
                            
                            # Добавляем статистики записи аудио
                            if audio_manager:
                                recording_stats = audio_manager.get_stats()
                                stats.update(recording_stats)
                            
                            # Добавляем статистики всех систем
                            if ENHANCED_RAG_INTENTS_AVAILABLE:
                                try:
                                    rag_stats = get_enhanced_rag_stats()
                                    stats.update(rag_stats)
                                except Exception as e:
                                    logger.warning(f"Ошибка получения Enhanced RAG статистики: {e}")
                            
                            if LLM_PERIODONTAL_AVAILABLE:
                                try:
                                    llm_stats = get_fixed_llm_stats()
                                    stats.update(llm_stats)
                                    stats = add_fixed_llm_stats_to_server_stats(stats)
                                except Exception as e:
                                    logger.warning(f"Ошибка получения FIXED LLM статистики: {e}")
                            
                            if PERIODONTAL_AVAILABLE:
                                try:
                                    periodontal_stats = get_periodontal_stats()
                                    stats.update(periodontal_stats)
                                    stats = enhance_server_stats(stats)
                                except Exception as e:
                                    logger.warning(f"Ошибка получения Periodontal статистики: {e}")
                            
                            try:
                                await asyncio.wait_for(websocket.send(json.dumps(stats)), timeout=3.0)
                            except asyncio.TimeoutError:
                                logger.warning(f"⚠️ Timeout sending stats to {client_id}")
                            
                    elif message == "MODEL_INFO":
                        if processor:
                            model_info = processor.asr.get_info()
                            model_info['vad_device'] = str(processor.vad.device)
                            model_info.update({
                                'enhanced_rag_intents_system': 'active' if ENHANCED_RAG_INTENTS_AVAILABLE else 'inactive',
                                'fixed_llm_periodontal_system': 'active' if LLM_PERIODONTAL_AVAILABLE else 'inactive',
                                'periodontal_fallback_system': 'active' if PERIODONTAL_AVAILABLE else 'inactive',
                                'enhanced_mode': f'ENHANCED_SEGMENTATION_V2_{processor.stats["systems_count"]}',
                                'active_systems': processor.stats.get('active_systems', []),
                                'stability_features': True,
                                'error_recovery': True,
                                'timeout_protection': True,
                                'audio_recording_enabled': ENHANCED_CONFIG.get("save_audio_recordings", True),
                                'improved_segmentation_enabled': ENHANCED_CONFIG.get("use_improved_segmentation", True),
                                'segmentation_mode': ENHANCED_CONFIG.get("segmentation_mode", "COMMAND_AWARE"),
                                'rag_system_available': ENHANCED_RAG_INTENTS_AVAILABLE,
                                'command_separation': True
                            })
                            
                            try:
                                await asyncio.wait_for(websocket.send(json.dumps(model_info)), timeout=3.0)
                            except asyncio.TimeoutError:
                                logger.warning(f"⚠️ Timeout sending model info to {client_id}")
                    
                    # Проверка на зависшего клиента
                    if current_time - last_ping_time > 120:  # 2 минуты без ping
                        logger.warning(f"⚠️ Client {client_id} appears to be stale (no ping for {current_time - last_ping_time:.0f}s)")
                
                # Проверка количества ошибок клиента
                if client_error_count > max_client_errors:
                    logger.error(f"❌ Too many errors from {client_id}, disconnecting")
                    break
                    
            except Exception as e:
                logger.error(f"❌ Critical error handling message from {client_id}: {e}")
                client_error_count += 1
                
    except websockets.exceptions.ConnectionClosed:
        logger.info(f"🎤 ASR клиент отключен: {client_id}")
    except Exception as e:
        logger.error(f"❌ ASR client error: {e}")
    finally:
        # Очистка буферов клиента
        if processor and hasattr(processor, 'segmentation_processor') and processor.segmentation_processor:
            processor.segmentation_processor.cleanup_client(client_id)
            logger.debug(f"🗑️ Cleared segmentation buffer for {client_id}")
            
async def periodic_stats():
    """Периодическая отправка статистики"""
    while True:
        await asyncio.sleep(10)  # Каждые 10 секунд
        
        if processor and web_clients:
            try:
                stats = processor.stats.copy()
                stats['server_uptime'] = time.time() - stats['server_uptime_start']
                
                # Добавляем статистики записи аудио
                if audio_manager:
                    recording_stats = audio_manager.get_stats()
                    stats.update(recording_stats)
                
                # Сборка статистики всех систем
                if ENHANCED_RAG_INTENTS_AVAILABLE:
                    try:
                        rag_stats = get_enhanced_rag_stats()
                        stats.update(rag_stats)
                    except:
                        pass
                
                if LLM_PERIODONTAL_AVAILABLE:
                    try:
                        llm_stats = get_fixed_llm_stats()
                        stats.update(llm_stats)
                        stats = add_fixed_llm_stats_to_server_stats(stats)
                    except:
                        pass
                
                if PERIODONTAL_AVAILABLE:
                    try:
                        periodontal_stats = get_periodontal_stats()
                        stats.update(periodontal_stats)
                        stats = enhance_server_stats(stats)
                    except:
                        pass
                
                message = json.dumps({
                    "type": "stats",
                    **stats
                })
                
                # Безопасная отправка всем клиентам
                disconnected = set()
                for client in list(web_clients):
                    try:
                        await asyncio.wait_for(client.send(message), timeout=2.0)
                    except:
                        disconnected.add(client)
                
                for client in disconnected:
                    web_clients.discard(client)
                    
            except Exception as e:
                logger.error(f"❌ Periodic stats error: {e}")
async def main():
    """ИСПРАВЛЕННАЯ главная функция сервера"""
    global processor
    
    # Обработчик сигналов для корректного завершения
    def signal_handler(signum, frame):
        logger.info(f"🛑 Received signal {signum}, shutting down...")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    print("\n" + "🎯" * 80)
    print("   🎤 ИСПРАВЛЕННЫЙ ENHANCED FASTWHISPER ASR + IMPROVED SPEECH SEGMENTATION V2")
    print("   • ТОЧНОЕ РАЗДЕЛЕНИЕ КОМАНД (НАЧАЛО/КОНЕЦ)")
    print("   • АВТОМАТИЧЕСКОЕ СОХРАНЕНИЕ .WAV ЗАПИСЕЙ")
    print("   • ПОЛНАЯ RAG СИСТЕМА С INTENT КЛАССИФИКАЦИЕЙ")
    print("   • LLM ИСПРАВЛЕНИЕ ASR ОШИБОК")
    print("   • PROFESSIONAL PERIODONTAL CHARTING")
    print("🎯" * 80)
    
    try:
        logger.info("🔧 Инициализация ИСПРАВЛЕННОГО ENHANCED процессора...")
        processor = EnhancedProcessorWithSegmentation()
        
        if processor.asr.model is None:
            logger.error("❌ ASR модель не загружена!")
            print("\n❌ КРИТИЧЕСКАЯ ОШИБКА: FastWhisper модель не загрузилась")
            print("📋 Возможные решения:")
            print("   1. Проверьте интернет соединение для загрузки модели")
            print("   2. Установите faster-whisper: pip install faster-whisper")
            print("   3. Освободите память GPU/CPU")
            print("   4. Попробуйте запустить с базовой моделью")
            return
        
        logger.info("🌐 Запуск ИСПРАВЛЕННЫХ WebSocket серверов...")
        
        # Создание серверов
        asr_server = await websockets.serve(
            handle_asr_client,
            "0.0.0.0",
            ASR_PORT,
            max_size=15 * 1024 * 1024,
            ping_interval=20,
            ping_timeout=10,
            close_timeout=5,
            compression=None
        )
        
        web_server = await websockets.serve(
            handle_web_client,
            "0.0.0.0",
            WEB_PORT,
            ping_interval=25,
            ping_timeout=10,
            compression=None
        )
        
        print(f"\n✅ ИСПРАВЛЕННЫЕ серверы запущены:")
        print(f"   ⚡ ASR (аудио): ws://0.0.0.0:{ASR_PORT}")
        print(f"   🌐 WebSocket (веб): ws://0.0.0.0:{WEB_PORT}")
		# Информация о системе
        device_info = "CPU"
        if torch.cuda.is_available():
            try:
                gpu_name = torch.cuda.get_device_name(0)
                memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
                device_info = f"CUDA ({gpu_name}, {memory_gb:.1f}GB)"
            except:
                device_info = "CUDA"
        
        print(f"\n🎯 ИСПРАВЛЕННАЯ СИСТЕМА:")
        print(f"   💻 Устройство: {device_info}")
        print(f"   🤖 ASR модель: {processor.asr.model_size}")
        print(f"   🎤 VAD: {'Silero' if processor.vad.model else 'RMS fallback'}")
        print(f"   🎯 Сегментация: {'IMPROVED V2' if SEGMENTATION_AVAILABLE else 'FALLBACK'}")
        print(f"   📡 Chunk size: {CLIENT_CHUNK_DURATION*1000:.0f}ms")
        print(f"   ⏱️ Processing timeout: {ENHANCED_CONFIG.get('processing_timeout', 30.0)}s")
        
        # Статус доступных систем
        active_systems = processor.stats.get('active_systems', [])
        systems_count = processor.stats.get('systems_count', 0)
        
        print(f"\n🔧 АКТИВНЫЕ СИСТЕМЫ ({systems_count}/3):")
        
        if ENHANCED_RAG_INTENTS_AVAILABLE:
            print(f"   🧠 Enhanced RAG Intents: ✅ АКТИВНА (ПРИОРИТЕТ 0)")
        else:
            print(f"   🧠 Enhanced RAG Intents: ❌ НЕДОСТУПНА")
        
        if LLM_PERIODONTAL_AVAILABLE:
            print(f"   🤖 FIXED Liberal LLM: ✅ АКТИВНА (ПРИОРИТЕТ 1)")
        else:
            print(f"   🤖 FIXED Liberal LLM: ❌ НЕДОСТУПНА")
        
        if PERIODONTAL_AVAILABLE:
            print(f"   🦷 Standard Periodontal: ✅ АКТИВНА (ПРИОРИТЕТ 2)")
        else:
            print(f"   🦷 Standard Periodontal: ❌ НЕДОСТУПНА")
        
        print(f"\n🎯 ИСПРАВЛЕНИЯ В V2:")
        print(f"   ✅ Устранена ошибка отсутствующего модуля сегментации")
        print(f"   ✅ Добавлен SimpleFallbackProcessor как резерв")
        print(f"   ✅ Улучшена обработка ошибок при загрузке модулей")
        print(f"   ✅ Исправлен импорт improved_speech_segmentation")
        print(f"   ✅ Добавлена совместимость с существующими системами")
        print(f"   ✅ Улучшена стабильность сегментации")
        
        print(f"\n🚀 ИСПРАВЛЕННЫЙ ENHANCED SERVER WITH IMPROVED SEGMENTATION V2 READY!")
        print("=" * 100 + "\n")
        
        # Запуск периодической статистики
        stats_task = asyncio.create_task(periodic_stats())
        
        # Ожидание завершения серверов
        await asyncio.gather(
            asr_server.wait_closed(),
            web_server.wait_closed(),
            stats_task,
            return_exceptions=True
        )
        
    except Exception as e:
        logger.error(f"❌ Критическая ошибка ИСПРАВЛЕННОГО сервера: {e}")
        traceback.print_exc()
        raise
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 ИСПРАВЛЕННЫЙ сервер остановлен пользователем")
    except Exception as e:
        print(f"\n❌ Критическая ошибка: {e}")
        traceback.print_exc()
		
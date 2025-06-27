#!/usr/bin/env python3
"""
Improved Speech Segmentation Module for Enhanced FastWhisper Server
Обеспечивает точное разделение команд с определением начала и конца
"""

import logging
import numpy as np
import time
from typing import Dict, List, Optional, Tuple, Any
from collections import deque
from dataclasses import dataclass
from enum import Enum
import asyncio
from datetime import datetime

logger = logging.getLogger(__name__)

class SpeechState(Enum):
    """Состояния речевой активности"""
    SILENCE = "silence"
    SPEECH_DETECTION = "speech_detection"
    SPEECH_ACTIVE = "speech_active"
    SPEECH_ENDING = "speech_ending"
    COMMAND_COMPLETE = "command_complete"

@dataclass
class AudioSegment:
    """Аудио сегмент с метаданными"""
    audio_data: np.ndarray
    start_time: float
    end_time: float
    confidence: float
    state: SpeechState
    vad_scores: List[float]
    energy_level: float

class ImprovedClientBuffer:
    """Улучшенный буфер для клиента с точной сегментацией"""
    
    def __init__(self, client_id: str, config: Dict):
        self.client_id = client_id
        self.config = config
        
        # Аудио буферы
        self.audio_buffer = np.array([])
        self.vad_scores = deque(maxlen=100)
        self.energy_history = deque(maxlen=50)
        
        # Состояние сегментации
        self.current_state = SpeechState.SILENCE
        self.speech_start_time = None
        self.last_speech_time = None
        self.silence_counter = 0
        self.speech_counter = 0
        
        # Пороги и счетчики
        self.speech_threshold = config.get('segmentation_speech_threshold', 0.35)
        self.silence_threshold = config.get('segmentation_silence_threshold', 0.25)
        self.min_command_duration = config.get('min_command_duration', 0.8)
        self.max_command_duration = config.get('max_command_duration', 20.0)
        self.speech_confirmation_chunks = config.get('speech_confirmation_chunks', 3)
        self.silence_confirmation_chunks = config.get('silence_confirmation_chunks', 8)
        
        # Энергетические пороги
        self.energy_threshold = 0.001
        self.background_noise = 0.0002
        
        # Статистика
        self.stats = {
            'commands_segmented': 0,
            'false_starts': 0,
            'successful_commands': 0,
            'total_chunks': 0,
            'average_command_duration': 0.0
        }
        
        logger.debug(f"🎯 Created buffer for client {client_id}")
    
    def process_chunk(self, audio_chunk: np.ndarray, vad_score: float) -> Optional[np.ndarray]:
        """
        Обработка аудио чанка с точной сегментацией
        Возвращает полный аудио сегмент команды когда она завершена
        """
        self.stats['total_chunks'] += 1
        chunk_time = time.time()
        
        # Расчет энергии чанка
        rms_energy = np.sqrt(np.mean(audio_chunk ** 2))
        self.energy_history.append(rms_energy)
        
        # Адаптивная нормализация VAD score
        normalized_vad = self._normalize_vad_score(vad_score, rms_energy)
        self.vad_scores.append(normalized_vad)
        
        # Состояние машины для сегментации
        completed_segment = self._update_state_machine(
            audio_chunk, normalized_vad, chunk_time
        )
        
        # Добавляем чанк к буферу если мы в режиме записи
        if self.current_state in [SpeechState.SPEECH_DETECTION, 
                                  SpeechState.SPEECH_ACTIVE, 
                                  SpeechState.SPEECH_ENDING]:
            self.audio_buffer = np.concatenate([self.audio_buffer, audio_chunk])
        
        return completed_segment
    
    def _normalize_vad_score(self, vad_score: float, energy: float) -> float:
        """Нормализация VAD score с учетом энергии"""
        
        # Адаптивная корректировка фонового шума
        if len(self.energy_history) > 10:
            avg_energy = np.mean(list(self.energy_history)[-10:])
            if avg_energy < self.energy_threshold:
                self.background_noise = avg_energy * 0.9
        
        # Энергетический буст для слабых сигналов
        if energy > self.background_noise * 3:
            energy_boost = min(energy / self.energy_threshold, 2.0)
            vad_score = min(vad_score * energy_boost, 1.0)
        
        # Сглаживание VAD score
        if len(self.vad_scores) >= 3:
            recent_scores = list(self.vad_scores)[-3:]
            smoothed = np.mean(recent_scores + [vad_score])
            return max(0.0, min(1.0, smoothed))
        
        return vad_score
    
    def _update_state_machine(self, audio_chunk: np.ndarray, vad_score: float, 
                            chunk_time: float) -> Optional[np.ndarray]:
        """Обновление машины состояний сегментации"""
        
        previous_state = self.current_state
        
        if self.current_state == SpeechState.SILENCE:
            return self._handle_silence_state(audio_chunk, vad_score, chunk_time)
        
        elif self.current_state == SpeechState.SPEECH_DETECTION:
            return self._handle_detection_state(audio_chunk, vad_score, chunk_time)
        
        elif self.current_state == SpeechState.SPEECH_ACTIVE:
            return self._handle_active_state(audio_chunk, vad_score, chunk_time)
        
        elif self.current_state == SpeechState.SPEECH_ENDING:
            return self._handle_ending_state(audio_chunk, vad_score, chunk_time)
        
        return None
    
    def _handle_silence_state(self, audio_chunk: np.ndarray, vad_score: float, 
                            chunk_time: float) -> Optional[np.ndarray]:
        """Обработка состояния тишины"""
        
        if vad_score > self.speech_threshold:
            self.speech_counter += 1
            self.silence_counter = 0
            
            if self.speech_counter >= self.speech_confirmation_chunks:
                # Начало речи подтверждено
                self.current_state = SpeechState.SPEECH_DETECTION
                self.speech_start_time = chunk_time
                self.last_speech_time = chunk_time
                self.audio_buffer = audio_chunk.copy()
                
                logger.debug(f"🎯 Speech detection started for {self.client_id}")
                return None
        else:
            self.speech_counter = max(0, self.speech_counter - 1)
            self.silence_counter += 1
        
        return None
    
    def _handle_detection_state(self, audio_chunk: np.ndarray, vad_score: float, 
                              chunk_time: float) -> Optional[np.ndarray]:
        """Обработка состояния детекции речи"""
        
        if vad_score > self.speech_threshold:
            self.speech_counter += 1
            self.silence_counter = 0
            self.last_speech_time = chunk_time
            
            # Переход к активной речи
            if self.speech_counter >= self.speech_confirmation_chunks * 2:
                self.current_state = SpeechState.SPEECH_ACTIVE
                logger.debug(f"🎯 Speech active for {self.client_id}")
        
        elif vad_score < self.silence_threshold:
            self.silence_counter += 1
            self.speech_counter = max(0, self.speech_counter - 1)
            
            # Ложный старт - возврат к тишине
            if self.silence_counter >= self.silence_confirmation_chunks:
                duration = chunk_time - self.speech_start_time if self.speech_start_time else 0
                
                if duration < self.min_command_duration:
                    logger.debug(f"🎯 False start detected for {self.client_id} ({duration:.1f}s)")
                    self.stats['false_starts'] += 1
                    self._reset_to_silence()
                else:
                    # Короткая команда - завершаем
                    return self._complete_command()
        
        return None
    
    def _handle_active_state(self, audio_chunk: np.ndarray, vad_score: float, 
                           chunk_time: float) -> Optional[np.ndarray]:
        """Обработка состояния активной речи"""
        
        current_duration = chunk_time - self.speech_start_time if self.speech_start_time else 0
        
        # Проверка на превышение максимальной длительности
        if current_duration > self.max_command_duration:
            logger.warning(f"🎯 Command too long for {self.client_id} ({current_duration:.1f}s), forcing completion")
            return self._complete_command()
        
        if vad_score > self.speech_threshold:
            self.speech_counter += 1
            self.silence_counter = 0
            self.last_speech_time = chunk_time
        
        elif vad_score < self.silence_threshold:
            self.silence_counter += 1
            self.speech_counter = max(0, self.speech_counter - 1)
            
            # Начало завершения команды
            if self.silence_counter >= self.silence_confirmation_chunks // 2:
                self.current_state = SpeechState.SPEECH_ENDING
                logger.debug(f"🎯 Speech ending for {self.client_id}")
        
        return None
    
    def _handle_ending_state(self, audio_chunk: np.ndarray, vad_score: float, 
                           chunk_time: float) -> Optional[np.ndarray]:
        """Обработка состояния завершения речи"""
        
        if vad_score > self.speech_threshold:
            # Речь возобновилась - возврат к активному состоянию
            self.current_state = SpeechState.SPEECH_ACTIVE
            self.speech_counter += 1
            self.silence_counter = 0
            self.last_speech_time = chunk_time
            logger.debug(f"🎯 Speech resumed for {self.client_id}")
        
        else:
            self.silence_counter += 1
            
            # Команда завершена
            if self.silence_counter >= self.silence_confirmation_chunks:
                return self._complete_command()
        
        return None
    
    def _complete_command(self) -> Optional[np.ndarray]:
        """Завершение команды и возврат аудио сегмента"""
        
        if len(self.audio_buffer) == 0:
            self._reset_to_silence()
            return None
        
        # Проверка минимальной длительности
        duration = len(self.audio_buffer) / 16000  # Assuming 16kHz
        
        if duration < self.min_command_duration:
            logger.debug(f"🎯 Command too short for {self.client_id} ({duration:.1f}s)")
            self.stats['false_starts'] += 1
            self._reset_to_silence()
            return None
        
        # Успешная команда
        completed_audio = self.audio_buffer.copy()
        
        # Обновление статистики
        self.stats['commands_segmented'] += 1
        self.stats['successful_commands'] += 1
        
        # Обновление средней длительности
        if self.stats['successful_commands'] > 0:
            alpha = 0.1
            self.stats['average_command_duration'] = (
                alpha * duration + 
                (1 - alpha) * self.stats['average_command_duration']
            )
        else:
            self.stats['average_command_duration'] = duration
        
        logger.info(f"🎯 Command completed for {self.client_id}: {duration:.1f}s, "
                   f"{len(completed_audio)} samples")
        
        self._reset_to_silence()
        return completed_audio
    
    def _reset_to_silence(self):
        """Сброс к состоянию тишины"""
        self.current_state = SpeechState.SILENCE
        self.audio_buffer = np.array([])
        self.speech_counter = 0
        self.silence_counter = 0
        self.speech_start_time = None
        self.last_speech_time = None
    
    def get_info(self) -> Dict:
        """Получение информации о буфере"""
        return {
            'client_id': self.client_id,
            'current_state': self.current_state.value,
            'buffer_size': len(self.audio_buffer),
            'stats': self.stats.copy()
        }

class ImprovedAudioProcessor:
    """Улучшенный аудио процессор с точной сегментацией"""
    
    def __init__(self, vad, asr, audio_manager):
        self.vad = vad
        self.asr = asr
        self.audio_manager = audio_manager
        
        # Клиентские буферы
        self.client_buffers: Dict[str, ImprovedClientBuffer] = {}
        
        # Конфигурация
        self.config = {
            'segmentation_speech_threshold': 0.35,
            'segmentation_silence_threshold': 0.25,
            'min_command_duration': 0.8,
            'max_command_duration': 20.0,
            'speech_confirmation_chunks': 3,
            'silence_confirmation_chunks': 8
        }
        
        # Глобальная статистика
        self.global_stats = {
            'total_clients': 0,
            'active_clients': 0,
            'total_commands_processed': 0,
            'successful_segmentations': 0,
            'false_starts_prevented': 0,
            'average_segmentation_accuracy': 100.0
        }
        
        logger.info("🎯 Improved Audio Processor initialized")
    
    def process_audio_chunk(self, client_id: str, audio_chunk: np.ndarray) -> Optional[str]:
        """
        Обработка аудио чанка для клиента
        Возвращает текст команды когда сегментация завершена
        """
        
        # Создание буфера для нового клиента
        if client_id not in self.client_buffers:
            self.client_buffers[client_id] = ImprovedClientBuffer(client_id, self.config)
            self.global_stats['total_clients'] += 1
            logger.debug(f"🎯 Created buffer for new client: {client_id}")
        
        buffer = self.client_buffers[client_id]
        
        # Получение VAD score
        try:
            vad_scores = self.vad.process_chunk(audio_chunk)
            vad_score = vad_scores[0] if vad_scores else 0.0
        except Exception as e:
            logger.warning(f"VAD error for {client_id}: {e}")
            vad_score = 0.0
        
        # Сегментация
        completed_audio = buffer.process_chunk(audio_chunk, vad_score)
        
        if completed_audio is not None:
            # Аудио сегмент завершен - запускаем ASR
            return self._process_completed_segment(client_id, completed_audio)
        
        return None
    
    def _process_completed_segment(self, client_id: str, audio_segment: np.ndarray) -> Optional[str]:
        """Обработка завершенного аудио сегмента"""
        
        try:
            self.global_stats['total_commands_processed'] += 1
            
            # Транскрипция
            text, confidence, processing_time = self.asr.transcribe(audio_segment)
            
            # Проверка качества транскрипции
            if text and text not in ["NO_SPEECH_DETECTED", "PROCESSING", "ASR_NOT_LOADED"]:
                
                # Сохранение аудио записи
                if self.audio_manager:
                    recording_path = self.audio_manager.save_audio_recording(
                        audio_segment, 
                        client_id,
                        transcription=text,
                        command_successful=True,
                        metadata={
                            'segmentation_method': 'improved_v2',
                            'confidence': confidence,
                            'processing_time': processing_time,
                            'segment_duration': len(audio_segment) / 16000
                        }
                    )
                
                self.global_stats['successful_segmentations'] += 1
                
                # Обновление точности сегментации
                total = self.global_stats['total_commands_processed']
                successful = self.global_stats['successful_segmentations']
                self.global_stats['average_segmentation_accuracy'] = (successful / total) * 100
                
                logger.info(f"🎯 Segmentation success for {client_id}: '{text}' "
                           f"(conf: {confidence:.3f}, {processing_time:.2f}s)")
                
                return text
            
            else:
                logger.debug(f"🎯 No valid speech in segment from {client_id}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Error processing segment from {client_id}: {e}")
            return None
    
    def cleanup_client(self, client_id: str):
        """Очистка буфера клиента"""
        if client_id in self.client_buffers:
            del self.client_buffers[client_id]
            logger.debug(f"🎯 Cleaned up buffer for {client_id}")
    
    def get_client_info(self, client_id: str) -> Optional[Dict]:
        """Получение информации о клиенте"""
        if client_id in self.client_buffers:
            return self.client_buffers[client_id].get_info()
        return None
    
    def get_all_clients_info(self) -> Dict[str, Dict]:
        """Получение информации о всех клиентах"""
        return {
            client_id: buffer.get_info() 
            for client_id, buffer in self.client_buffers.items()
        }
    
    def get_improved_stats(self) -> Dict:
        """Получение улучшенной статистики"""
        
        # Агрегация статистики всех клиентов
        total_commands = 0
        total_false_starts = 0
        total_successful = 0
        total_duration = 0.0
        
        for buffer in self.client_buffers.values():
            stats = buffer.stats
            total_commands += stats['commands_segmented']
            total_false_starts += stats['false_starts']
            total_successful += stats['successful_commands']
            if stats['successful_commands'] > 0:
                total_duration += stats['average_command_duration']
        
        avg_duration = total_duration / len(self.client_buffers) if self.client_buffers else 0.0
        
        return {
            **self.global_stats,
            'active_clients': len(self.client_buffers),
            'commands_segmented': total_commands,
            'segmentation_false_starts': total_false_starts,
            'segmentation_successful_commands': total_successful,
            'average_command_duration': avg_duration,
            'segmentation_mode': 'IMPROVED_V2',
            'segmentation_enabled': True
        }

def integrate_improved_segmentation(base_processor, vad, asr, audio_manager):
    """
    Интеграция улучшенной сегментации с существующим процессором
    """
    
    # Создание улучшенного процессора
    improved_processor = ImprovedAudioProcessor(vad, asr, audio_manager)
    
    # Замена метода обработки чанков
    base_processor.segmentation_processor = improved_processor
    
    logger.info("🎯 Improved segmentation integrated successfully")
    
    return base_processor

# Экспортируемые функции и классы
__all__ = [
    'SpeechState',
    'ImprovedClientBuffer', 
    'ImprovedAudioProcessor',
    'integrate_improved_segmentation'
]

if __name__ == "__main__":
    # Тестирование модуля
    logger.info("🎯 Testing Improved Speech Segmentation module...")
    
    # Создание тестового конфига
    test_config = {
        'segmentation_speech_threshold': 0.35,
        'segmentation_silence_threshold': 0.25,
        'min_command_duration': 0.8,
        'max_command_duration': 20.0,
        'speech_confirmation_chunks': 3,
        'silence_confirmation_chunks': 8
    }
    
    # Создание тестового буфера
    buffer = ImprovedClientBuffer("test_client", test_config)
    
    # Тестирование обработки чанков
    test_chunks = [
        (np.random.normal(0, 0.01, 4000), 0.1),  # Тишина
        (np.random.normal(0, 0.1, 4000), 0.8),   # Речь
        (np.random.normal(0, 0.1, 4000), 0.9),   # Речь
        (np.random.normal(0, 0.01, 4000), 0.1),  # Тишина
    ]
    
    for i, (chunk, vad_score) in enumerate(test_chunks):
        result = buffer.process_chunk(chunk, vad_score)
        logger.info(f"Chunk {i+1}: VAD={vad_score}, State={buffer.current_state.value}, "
                   f"Completed={'Yes' if result is not None else 'No'}")
    
    # Статистика
    stats = buffer.get_info()
    logger.info(f"Final stats: {stats}")
    
    logger.info("✅ Improved Speech Segmentation module test completed")
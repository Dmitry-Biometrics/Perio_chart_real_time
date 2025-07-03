#!/usr/bin/env python3
"""
STREAMING PREDICTIVE PROCESSOR
Обрабатывает команды ВО ВРЕМЯ речи, не ожидая пауз
"""

import asyncio
import time
import re
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)

class StreamingState(Enum):
    """Состояния потокового анализа"""
    LISTENING = "listening"
    DETECTING = "detecting" 
    PROCESSING = "processing"
    CONFIRMING = "confirming"
    EXECUTING = "executing"

@dataclass
class StreamingCommand:
    """Потоковая команда"""
    pattern_type: str
    confidence: float
    extracted_data: Dict
    partial_text: str
    timestamp: float
    state: StreamingState

class StreamingPredictor:
    """Предиктивный анализатор потока речи"""
    
    def __init__(self, web_clients_ref):
        self.web_clients = web_clients_ref
        self.active_streams = {}  # client_id -> stream data
        
        # Настройки для ультра-быстрого режима
        self.min_analysis_length = 6    # Минимум слов для анализа
        self.confidence_threshold = 0.75
        self.execution_threshold = 0.85
        
        # Паттерны для быстрого распознавания
        self.setup_streaming_patterns()
        
        # Статистика
        self.stats = {
            'predictions_made': 0,
            'early_executions': 0,
            'false_positives': 0,
            'average_prediction_time': 0.0,
            'accuracy_rate': 0.0
        }
        
        logger.info("🔮 STREAMING PREDICTOR initialized for ULTRA-FAST mode")
    
    def setup_streaming_patterns(self):
        """Настройка паттернов для потокового анализа"""
        
        # Быстрые паттерны (срабатывают рано)
        self.quick_patterns = {
            'probing_depth_start': re.compile(
                r'probing\s+depth.*tooth\s+(?:number\s+)?(\w+)', 
                re.IGNORECASE
            ),
            'bleeding_start': re.compile(
                r'bleeding.*tooth\s+(\w+)', 
                re.IGNORECASE
            ),
            'mobility_start': re.compile(
                r'tooth\s+(\w+).*mobility', 
                re.IGNORECASE
            ),
            'missing_start': re.compile(
                r'missing\s+teeth?\s+(\w+)', 
                re.IGNORECASE
            )
        }
        
        # Подтверждающие паттерны (полная команда)
        self.confirmation_patterns = {
            'probing_depth_full': re.compile(
                r'probing\s+depth.*tooth\s+(?:number\s+)?(\w+).*?(buccal|lingual).*?(\d+)\s+(\d+)\s+(\d+)',
                re.IGNORECASE
            ),
            'bleeding_full': re.compile(
                r'bleeding.*tooth\s+(\w+)\s+(buccal|lingual)\s+(distal|mesial|mid)',
                re.IGNORECASE
            ),
            'mobility_full': re.compile(
                r'tooth\s+(\w+).*mobility.*grade\s+(\d+)',
                re.IGNORECASE
            ),
            'missing_full': re.compile(
                r'missing\s+teeth?\s+([\w\s]+)',
                re.IGNORECASE
            )
        }
        
        # Словарь конвертации слов в числа (с ASR ошибками)
        self.word_to_num = {
            'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
            'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
            'eleven': 11, 'twelve': 12, 'thirteen': 13, 'fourteen': 14,
            'fifteen': 15, 'sixteen': 16, 'seventeen': 17, 'eighteen': 18,
            'nineteen': 19, 'twenty': 20, 'thirty': 30, 'thirty-one': 31, 'thirty-two': 32,
            
            # ASR ошибки
            'too': 2, 'to': 2, 'for': 4, 'ate': 8, 'won': 1, 'tree': 3, 'sex': 6
        }
    
    async def process_streaming_chunk(self, client_id: str, partial_text: str, 
                                    audio_buffer: np.ndarray) -> Optional[Dict]:
        """
        Обработка потокового чанка - ГЛАВНАЯ ФУНКЦИЯ
        """
        start_time = time.time()
        
        # Инициализация потока для клиента
        if client_id not in self.active_streams:
            self.active_streams[client_id] = {
                'partial_texts': [],
                'predictions': [],
                'last_prediction': None,
                'start_time': start_time,
                'word_count': 0
            }
        
        stream = self.active_streams[client_id]
        stream['partial_texts'].append(partial_text)
        stream['word_count'] = len(partial_text.split())
        
        # Анализ только если достаточно слов
        if stream['word_count'] < self.min_analysis_length:
            logger.debug(f"🔮 Not enough words yet: {stream['word_count']}/{self.min_analysis_length}")
            return None
        
        logger.info(f"🔮 STREAMING ANALYSIS: '{partial_text}' ({stream['word_count']} words)")
        
        # Быстрый анализ паттернов
        prediction = await self.analyze_patterns(partial_text, client_id)
        
        if prediction:
            stream['predictions'].append(prediction)
            self.stats['predictions_made'] += 1
            
            prediction_time = (time.time() - start_time) * 1000
            logger.info(f"⚡ PREDICTION made in {prediction_time:.1f}ms: {prediction.pattern_type}")
            
            # Проверяем можно ли выполнить досрочно
            if prediction.confidence >= self.execution_threshold:
                logger.info(f"🚀 EARLY EXECUTION triggered! Confidence: {prediction.confidence:.2f}")
                
                # Мгновенное выполнение
                result = await self.execute_early_command(client_id, prediction)
                
                if result:
                    self.stats['early_executions'] += 1
                    
                    # Уведомляем клиента о мгновенном выполнении
                    await self.broadcast_early_execution(client_id, result, prediction_time)
                    
                    return result
            
            elif prediction.confidence >= self.confidence_threshold:
                # Высокая уверенность - готовим к выполнению
                logger.info(f"🎯 HIGH CONFIDENCE: {prediction.confidence:.2f} - preparing for execution")
                stream['last_prediction'] = prediction
                
                # Отправляем предварительное уведомление
                await self.broadcast_prediction_feedback(client_id, prediction)
        
        return None
    
    async def analyze_patterns(self, text: str, client_id: str) -> Optional[StreamingCommand]:
        """Анализ паттернов в потоковом тексте"""
        
        # 1. Проверяем быстрые паттерны для раннего обнаружения
        for pattern_name, pattern_regex in self.quick_patterns.items():
            match = pattern_regex.search(text)
            if match:
                confidence = self.calculate_pattern_confidence(text, pattern_name, match)
                
                if confidence >= self.confidence_threshold:
                    logger.info(f"✅ QUICK PATTERN matched: {pattern_name} (conf: {confidence:.2f})")
                    
                    # Попытка извлечь данные
                    extracted_data = await self.extract_command_data(text, pattern_name, match)
                    
                    if extracted_data:
                        return StreamingCommand(
                            pattern_type=pattern_name,
                            confidence=confidence,
                            extracted_data=extracted_data,
                            partial_text=text,
                            timestamp=time.time(),
                            state=StreamingState.PROCESSING
                        )
        
        # 2. Проверяем полные паттерны для подтверждения
        for pattern_name, pattern_regex in self.confirmation_patterns.items():
            match = pattern_regex.search(text)
            if match:
                confidence = self.calculate_pattern_confidence(text, pattern_name, match)
                
                logger.info(f"🎯 FULL PATTERN matched: {pattern_name} (conf: {confidence:.2f})")
                
                extracted_data = await self.extract_command_data(text, pattern_name, match)
                
                if extracted_data:
                    return StreamingCommand(
                        pattern_type=pattern_name,
                        confidence=min(confidence + 0.1, 1.0),  # Бонус за полное совпадение
                        extracted_data=extracted_data,
                        partial_text=text,
                        timestamp=time.time(),
                        state=StreamingState.CONFIRMING
                    )
        
        return None
    
    def calculate_pattern_confidence(self, text: str, pattern_name: str, match) -> float:
        """Расчет уверенности в паттерне"""
        base_confidence = 0.7
        
        # Бонусы за полноту
        word_count = len(text.split())
        if word_count >= 8:
            base_confidence += 0.15
        elif word_count >= 6:
            base_confidence += 0.1
        
        # Бонус за числа
        numbers_found = len(re.findall(r'\d+', text))
        if numbers_found >= 3:
            base_confidence += 0.1
        elif numbers_found >= 1:
            base_confidence += 0.05
        
        # Бонус за ключевые слова
        dental_keywords = ['tooth', 'buccal', 'lingual', 'distal', 'mesial', 'grade', 'class']
        keywords_found = sum(1 for keyword in dental_keywords if keyword in text.lower())
        base_confidence += keywords_found * 0.02
        
        # Проверка на полные паттерны
        if 'full' in pattern_name:
            base_confidence += 0.1
        
        return min(base_confidence, 1.0)
    
    async def extract_command_data(self, text: str, pattern_name: str, match) -> Optional[Dict]:
        """Извлечение данных команды из текста"""
        
        def convert_word_to_number(word: str) -> int:
            """Конвертация слов в числа"""
            if not word:
                return 0
            
            clean_word = word.lower().strip('.,!?;:')
            
            if clean_word.isdigit():
                return int(clean_word)
            
            return self.word_to_num.get(clean_word, 0)
        
        try:
            if 'probing_depth' in pattern_name:
                tooth = convert_word_to_number(match.group(1))
                
                if 'full' in pattern_name and len(match.groups()) >= 5:
                    surface = match.group(2).lower()
                    depth1 = int(match.group(3))
                    depth2 = int(match.group(4)) 
                    depth3 = int(match.group(5))
                    
                    if 1 <= tooth <= 32 and all(1 <= d <= 12 for d in [depth1, depth2, depth3]):
                        return {
                            'type': 'probing_depth',
                            'tooth_number': tooth,
                            'surface': surface,
                            'values': [depth1, depth2, depth3],
                            'complete': True
                        }
                
                elif 1 <= tooth <= 32:
                    # Частичные данные - попытка извлечь больше
                    numbers = [int(x) for x in re.findall(r'\b(\d+)\b', text) if 1 <= int(x) <= 12 and int(x) != tooth]
                    surface = 'buccal' if 'buccal' in text.lower() else ('lingual' if 'lingual' in text.lower() else 'buccal')
                    
                    if len(numbers) >= 3:
                        return {
                            'type': 'probing_depth',
                            'tooth_number': tooth,
                            'surface': surface,
                            'values': numbers[:3],
                            'complete': True
                        }
                    else:
                        return {
                            'type': 'probing_depth',
                            'tooth_number': tooth,
                            'surface': surface,
                            'complete': False
                        }
            
            elif 'bleeding' in pattern_name:
                tooth = convert_word_to_number(match.group(1))
                
                if 1 <= tooth <= 32:
                    surface = 'buccal'
                    position = 'distal'
                    
                    if 'full' in pattern_name and len(match.groups()) >= 3:
                        surface = match.group(2).lower()
                        position = match.group(3).lower()
                    else:
                        # Попытка извлечь из текста
                        if 'lingual' in text.lower():
                            surface = 'lingual'
                        if 'mesial' in text.lower():
                            position = 'mesial'
                        elif 'mid' in text.lower():
                            position = 'mid'
                    
                    return {
                        'type': 'bleeding',
                        'tooth_number': tooth,
                        'surface': surface,
                        'position': position,
                        'values': [True],
                        'complete': 'full' in pattern_name
                    }
            
            elif 'mobility' in pattern_name:
                tooth = convert_word_to_number(match.group(1))
                
                if 1 <= tooth <= 32:
                    grade = 2  # По умолчанию
                    
                    if 'full' in pattern_name and len(match.groups()) >= 2:
                        grade = int(match.group(2))
                    else:
                        # Ищем grade в тексте
                        grade_match = re.search(r'grade\s+(\d+)', text.lower())
                        if grade_match:
                            grade = int(grade_match.group(1))
                    
                    if 0 <= grade <= 3:
                        return {
                            'type': 'mobility',
                            'tooth_number': tooth,
                            'values': [grade],
                            'complete': 'full' in pattern_name
                        }
            
            elif 'missing' in pattern_name:
                if 'full' in pattern_name:
                    # Обработка полного списка зубов
                    teeth_text = match.group(1)
                    teeth_numbers = []
                    
                    for word in teeth_text.split():
                        tooth_num = convert_word_to_number(word)
                        if 1 <= tooth_num <= 32:
                            teeth_numbers.append(tooth_num)
                    
                    if teeth_numbers:
                        return {
                            'type': 'missing_teeth',
                            'values': teeth_numbers,
                            'complete': True
                        }
                else:
                    # Один зуб
                    tooth = convert_word_to_number(match.group(1))
                    if 1 <= tooth <= 32:
                        return {
                            'type': 'missing_teeth',
                            'values': [tooth],
                            'complete': True
                        }
        
        except Exception as e:
            logger.error(f"❌ Error extracting command data: {e}")
        
        return None
    
    async def execute_early_command(self, client_id: str, prediction: StreamingCommand) -> Optional[Dict]:
        """Досрочное выполнение команды"""
        
        if not prediction.extracted_data or not prediction.extracted_data.get('complete', False):
            logger.warning(f"⚠️ Cannot execute incomplete command: {prediction.pattern_type}")
            return None
        
        try:
            # Формируем результат для отправки
            data = prediction.extracted_data
            
            result = {
                'success': True,
                'tooth_number': data.get('tooth_number'),
                'measurement_type': data['type'],
                'surface': data.get('surface'),
                'position': data.get('position'),
                'values': data.get('values', []),
                'confidence': prediction.confidence,
                'message': self.format_command_message(data),
                'timestamp': prediction.timestamp,
                'early_execution': True,
                'streaming_prediction': True,
                'system': 'streaming_predictor_v1'
            }
            
            # Добавляем measurements для совместимости
            result['measurements'] = self.format_measurements(data)
            
            logger.info(f"🚀 EARLY EXECUTION: {result['message']}")
            return result
            
        except Exception as e:
            logger.error(f"❌ Early execution error: {e}")
            return None
    
    def format_command_message(self, data: Dict) -> str:
        """Форматирование сообщения команды"""
        
        cmd_type = data['type']
        tooth = data.get('tooth_number')
        values = data.get('values', [])
        
        if cmd_type == 'probing_depth':
            surface = data.get('surface', 'buccal')
            return f"⚡ STREAMING: Tooth {tooth} {surface} PD: {'-'.join(map(str, values))}mm"
        
        elif cmd_type == 'bleeding':
            surface = data.get('surface', 'buccal')
            position = data.get('position', 'distal')
            return f"⚡ STREAMING: Tooth {tooth} {surface} {position} bleeding"
        
        elif cmd_type == 'mobility':
            grade = values[0] if values else 0
            return f"⚡ STREAMING: Tooth {tooth} mobility grade {grade}"
        
        elif cmd_type == 'missing_teeth':
            teeth_list = ', '.join(map(str, values))
            return f"⚡ STREAMING: Missing teeth {teeth_list}"
        
        return f"⚡ STREAMING: {cmd_type} updated"
    
    def format_measurements(self, data: Dict) -> Dict:
        """Форматирование measurements для клиента"""
        
        cmd_type = data['type']
        values = data.get('values', [])
        
        if cmd_type == 'probing_depth':
            return {'probing_depth': values}
        elif cmd_type == 'bleeding':
            return {'bleeding': values}
        elif cmd_type == 'mobility':
            return {'mobility': values[0] if values else 0}
        elif cmd_type == 'missing_teeth':
            return {'missing_teeth': values}
        
        return {}
    
    async def broadcast_early_execution(self, client_id: str, result: Dict, prediction_time_ms: float):
        """Отправка результата досрочного выполнения"""
        
        if not self.web_clients:
            return
        
        message = {
            "type": "periodontal_update",
            "client_id": client_id,
            "early_execution": True,
            "prediction_time_ms": prediction_time_ms,
            **result
        }
        
        # Добавляем индикатор потокового выполнения
        message["streaming_indicator"] = True
        message["execution_speed"] = "ULTRA_FAST"
        
        await self.safe_broadcast(message)
        
        logger.info(f"📤 EARLY EXECUTION broadcasted in {prediction_time_ms:.1f}ms")
    
    async def broadcast_prediction_feedback(self, client_id: str, prediction: StreamingCommand):
        """Отправка обратной связи о предсказании"""
        
        if not self.web_clients:
            return
        
        message = {
            "type": "streaming_prediction",
            "client_id": client_id,
            "pattern_type": prediction.pattern_type,
            "confidence": prediction.confidence,
            "partial_text": prediction.partial_text,
            "state": prediction.state.value,
            "timestamp": prediction.timestamp
        }
        
        await self.safe_broadcast(message)
    
    async def safe_broadcast(self, message: Dict):
        """Безопасная отправка сообщения всем клиентам"""
        
        import json
        message_json = json.dumps(message)
        disconnected = set()
        
        for client in list(self.web_clients):
            try:
                await asyncio.wait_for(client.send(message_json), timeout=0.5)
            except Exception as e:
                logger.debug(f"Client broadcast error: {e}")
                disconnected.add(client)
        
        for client in disconnected:
            self.web_clients.discard(client)
    
    def cleanup_client_stream(self, client_id: str):
        """Очистка потока клиента"""
        if client_id in self.active_streams:
            del self.active_streams[client_id]
            logger.debug(f"🧹 Cleaned up stream for {client_id}")
    
    def get_streaming_stats(self) -> Dict:
        """Статистика потокового анализа"""
        return {
            **self.stats,
            'active_streams': len(self.active_streams),
            'average_words_per_prediction': self._calculate_avg_words(),
            'execution_rate': (self.stats['early_executions'] / max(1, self.stats['predictions_made'])) * 100
        }
    
    def _calculate_avg_words(self) -> float:
        """Расчет среднего количества слов для предсказания"""
        if not self.active_streams:
            return 0.0
        
        total_words = sum(stream['word_count'] for stream in self.active_streams.values())
        return total_words / len(self.active_streams)

# =============================================================================
# ИНТЕГРАЦИЯ С ОСНОВНЫМ ПРОЦЕССОРОМ
# =============================================================================

def integrate_streaming_predictor(base_processor, web_clients_ref):
    """Интеграция потокового предиктора с основным процессором"""
    
    predictor = StreamingPredictor(web_clients_ref)
    base_processor.streaming_predictor = predictor
    
    # Модифицируем метод обработки чанков
    original_process_chunk = base_processor.process_audio_chunk
    
    def enhanced_process_chunk_with_streaming(client_id, audio_chunk):
        """Обработка чанков с потоковым анализом"""
        
        # Обычная обработка
        result = original_process_chunk(client_id, audio_chunk)
        
        # Потоковый анализ на частичных буферах
        if hasattr(base_processor, 'segmentation_processor'):
            seg_processor = base_processor.segmentation_processor
            
            if hasattr(seg_processor, 'client_buffers'):
                buffer = seg_processor.client_buffers.get(client_id)
                
                if buffer and hasattr(buffer, 'audio_buffer') and len(buffer.audio_buffer) > 16000:  # 1+ секунда
                    
                    # Быстрая транскрипция для анализа
                    try:
                        preview_audio = buffer.audio_buffer[-24000:]  # Последние 1.5 секунды
                        
                        if hasattr(base_processor, 'asr') and hasattr(base_processor.asr, 'transcribe_fast_preview'):
                            quick_text, _, _ = base_processor.asr.transcribe_fast_preview(preview_audio)
                            
                            if quick_text and len(quick_text.split()) >= 6:
                                # Запускаем потоковый анализ
                                asyncio.create_task(
                                    predictor.process_streaming_chunk(client_id, quick_text, preview_audio)
                                )
                                
                    except Exception as e:
                        logger.debug(f"Streaming analysis error: {e}")
        
        return result
    
    base_processor.process_audio_chunk = enhanced_process_chunk_with_streaming
    
    logger.info("🔮 STREAMING PREDICTOR integrated for ULTRA-FAST response")
    return base_processor

# =============================================================================
# УЛЬТРА-БЫСТРЫЕ ASR НАСТРОЙКИ
# =============================================================================

class UltraFastASR:
    """Ультра-быстрые настройки для ASR"""
    
    @staticmethod
    def get_speed_optimized_params():
        """Параметры ASR для максимальной скорости"""
        return {
            'beam_size': 1,              # Минимальный beam
            'best_of': 1,               # Только один проход
            'temperature': 0.0,         # Детерминированный вывод
            'no_speech_threshold': 0.9, # Высокий порог для пропуска тишины
            'compression_ratio_threshold': 1.5,  # Низкий для скорости
            'condition_on_previous_text': False,  # Без контекста
            'without_timestamps': True,  # Без временных меток
            'word_timestamps': False,   # Без слов-временных меток
            'vad_filter': True,         # Фильтр VAD
            'suppress_blank': True,     # Подавление пустых
            'suppress_tokens': [-1],    # Подавление специальных токенов
        }
    
    @staticmethod
    def setup_ultra_fast_model(asr_instance):
        """Настройка модели для ультра-быстрого режима"""
        
        # Применяем оптимизированные параметры
        asr_instance.speed_params = UltraFastASR.get_speed_optimized_params()
        
        # Модифицируем метод транскрипции
        original_transcribe = asr_instance.transcribe
        
        def ultra_fast_transcribe(audio_np):
            """Ультра-быстрая транскрипция"""
            
            # Предварительная фильтрация по энергии
            rms_energy = np.sqrt(np.mean(audio_np ** 2))
            if rms_energy < 0.001:  # Слишком тихо
                return "NO_SPEECH_DETECTED", 0.0, 0.001
            
            # Укороченная обработка для скорости
            if len(audio_np) > 80000:  # Больше 5 секунд
                audio_np = audio_np[:80000]  # Обрезаем
            
            start_time = time.time()
            
            try:
                segments, info = asr_instance.model.transcribe(
                    audio_np,
                    language="en",
                    **asr_instance.speed_params
                )
                
                text_segments = [segment.text.strip() for segment in segments if segment.text]
                full_text = " ".join(text_segments).strip()
                
                confidence = getattr(info, 'language_probability', 0.8)
                processing_time = time.time() - start_time
                
                return full_text or "NO_SPEECH_DETECTED", confidence, processing_time
                
            except Exception as e:
                logger.error(f"Ultra-fast transcribe error: {e}")
                return f"ERROR: {str(e)[:50]}", 0.0, time.time() - start_time
        
        asr_instance.transcribe = ultra_fast_transcribe
        logger.info("⚡ ULTRA-FAST ASR mode enabled")

if __name__ == "__main__":
    # Тестирование потокового предиктора
    
    async def test_streaming_predictor():
        """Тест потокового предиктора"""
        
        print("🔮 Testing STREAMING PREDICTOR")
        print("=" * 50)
        
        predictor = StreamingPredictor(set())
        
        test_texts = [
            "probing depth on tooth",
            "probing depth on tooth number fourteen",
            "probing depth on tooth number fourteen buccal",
            "probing depth on tooth number fourteen buccal surface three",
            "probing depth on tooth number fourteen buccal surface three two four"
        ]
        
        for i, text in enumerate(test_texts):
            print(f"\nStep {i+1}: '{text}'")
            
            result = await predictor.process_streaming_chunk("test", text, np.array([]))
            
            if result:
                print(f"  ⚡ EARLY EXECUTION: {result['message']}")
            else:
                print(f"  ⏳ Waiting for more data...")
        
        stats = predictor.get_streaming_stats()
        print(f"\n📊 Stats: {stats}")
    
    asyncio.run(test_streaming_predictor())
#!/usr/bin/env python3
"""
ИСПРАВЛЕННАЯ ИНТЕГРАЦИЯ СИСТЕМЫ МГНОВЕННОГО ВЫПОЛНЕНИЯ КОМАНД
Исправлены проблемы доступа к атрибутам базового процессора
"""

import asyncio
import time
import logging
from typing import Optional
import json
import numpy as np

# Импорт системы мгновенного выполнения
from instant_command_system import (
    InstantCommandProcessor,
    CommandCompleteness,
    integrate_instant_command_system
)

logger = logging.getLogger(__name__)

class EnhancedProcessorWithInstantCommands:
    """ИСПРАВЛЕННЫЙ расширенный процессор с поддержкой мгновенного выполнения"""
    
    def __init__(self, base_processor, web_clients_ref):
        self.base_processor = base_processor
        self.web_clients = web_clients_ref
        
        # Создаем процессор мгновенного выполнения
        self.instant_processor = InstantCommandProcessor(web_clients_ref)
        
        # Статистика мгновенного выполнения
        self.instant_stats = {
            'instant_commands_executed': 0,
            'partial_commands_detected': 0,
            'average_instant_response_time': 0.0,
            'total_instant_response_time': 0.0
        }
        
        logger.info("🚀 Enhanced processor with instant commands initialized")
    
    # ИСПРАВЛЕНИЕ: Добавляем прокси-свойства для доступа к атрибутам базового процессора
    @property
    def asr(self):
        """Прокси-доступ к ASR"""
        return self.base_processor.asr
    
    @property
    def vad(self):
        """Прокси-доступ к VAD"""
        return self.base_processor.vad
    
    @property
    def segmentation_processor(self):
        """Прокси-доступ к segmentation_processor"""
        return getattr(self.base_processor, 'segmentation_processor', None)
    
    @property
    def stats(self):
        """Прокси-доступ к статистике с добавлением instant stats"""
        base_stats = getattr(self.base_processor, 'stats', {})
        # Объединяем статистику
        combined_stats = base_stats.copy()
        combined_stats.update(self.instant_stats)
        combined_stats['instant_system_enabled'] = True
        return combined_stats
    
    def process_audio_chunk(self, client_id: str, audio_chunk) -> Optional[str]:
        """
        ИСПРАВЛЕННАЯ обработка чанков с мгновенным выполнением
        """
        # Обычная обработка через базовый процессор
        transcription_result = self.base_processor.process_audio_chunk(client_id, audio_chunk)
        
        if transcription_result and isinstance(transcription_result, str) and transcription_result.strip():
            # ✅ МГНОВЕННАЯ ПРОВЕРКА И ВЫПОЛНЕНИЕ
            try:
                print(f"🔍 INSTANT CHECK: '{transcription_result}'")
                
                # Синхронная проверка завершенности команды
                completeness, command_data = self.instant_processor.analyzer.analyze_command_completeness(
                    transcription_result, client_id
                )
                
                if completeness == CommandCompleteness.COMPLETE:
                    print(f"🚀 INSTANT EXECUTION NOW: '{transcription_result}'")
                    
                    # КРИТИЧНО: Блокируем дальнейшую обработку
                    self._mark_command_as_processed_sync(client_id, transcription_result)
                    
                    # МГНОВЕННАЯ отправка результата (синхронно)
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        # Создаем задачу для мгновенной отправки
                        task = loop.create_task(
                            self.instant_processor._send_instant_result(client_id, command_data)
                        )
                        
                        # Обновляем статистику мгновенно
                        self.instant_stats['instant_commands_executed'] += 1
                        

                        
                        print(f"⚡ INSTANT RESULT SENT in {time.time():.3f}s")
                        
                        # Возвращаем результат но помечаем как обработанный
                        return transcription_result
                        
                elif completeness == CommandCompleteness.INCOMPLETE:
                    print(f"⏳ WAITING FOR COMPLETION: '{transcription_result}'")
                    # Отправляем промежуточную обратную связь
                    if hasattr(self, '_send_partial_feedback_sync'):
                        self._send_partial_feedback_sync(client_id, transcription_result)
                    
            except Exception as e:
                print(f"❌ INSTANT CHECK ERROR: {e}")
                import traceback
                traceback.print_exc()
        
        return transcription_result
    
    def process_audio_chunk_with_predictive(self, client_id: str, audio_chunk: np.ndarray) -> Optional[str]:
        """PREDICTIVE обработка - проверяем instant commands НА КАЖДОМ ЧАНКЕ"""
        
        # Обычная обработка
        result = self.base_processor.process_audio_chunk(client_id, audio_chunk)
        
        
        # НОВОЕ: Predictive check на частичной транскрипции
        if hasattr(self.base_processor, 'segmentation_processor'):
            segmentation_processor = self.base_processor.segmentation_processor
            
            if hasattr(segmentation_processor, 'client_buffers'):
                buffer = segmentation_processor.client_buffers.get(client_id)
                
                if buffer and hasattr(buffer, 'audio_buffer') and len(buffer.audio_buffer) > 32000:  # 2+ секунды
                    
                    # Быстрая промежуточная транскрипция
                    try:
                        # Берем последние 2 секунды для быстрой проверки
                        quick_audio = buffer.audio_buffer[-32000:]
                        #quick_text, _, _ = self.base_processor.asr.transcribe(quick_audio)
                        qick_text, _, _ = self.base_processor.asr.transcribe_fast_preview(quick_audio)
                        if quick_text and len(quick_text.split()) >= 6:  # Достаточно слов
                            
                            print(f"🔍 PREDICTIVE CHECK: '{quick_text}'")
                            
                            # Проверяем instant completeness
                            completeness, command_data = self.instant_processor.analyzer.analyze_command_completeness(quick_text, client_id)
                            
                            if completeness == CommandCompleteness.COMPLETE:
                                print(f"🚀 PREDICTIVE INSTANT EXECUTION: '{quick_text}'")
                                
                                # Мгновенное выполнение
                                asyncio.create_task(
                                    self.instant_processor._send_instant_result(client_id, command_data)
                                )
                                
                                # Отмечаем как уже обработанное
                                self._mark_command_as_processed_sync(client_id, quick_text)
                                
                                # Обновляем статистику
                                self.instant_stats['instant_commands_executed'] += 1
                                
                    except Exception as e:
                        # Игнорируем ошибки predictive проверки
                        print(f"⚠️ Predictive error: {e}")
                        pass
    
    
    
    def _mark_command_as_processed_sync(self, client_id: str, text: str):
        """Синхронная версия блокировки команды"""
        if not hasattr(self, '_processed_commands'):
            self._processed_commands = {}
        
        command_key = f"{client_id}_{hash(text)}"
        self._processed_commands[command_key] = {
            'text': text,
            'timestamp': time.time(),
            'processed_instantly': True
        }
        print(f"🔒 COMMAND LOCKED: '{text}' from {client_id}")

    def _send_partial_feedback_sync(self, client_id: str, partial_text: str):
        """Синхронная отправка промежуточной обратной связи"""
        if self.web_clients:
            feedback_message = {
                "type": "partial_command_feedback",
                "client_id": client_id,
                "partial_text": partial_text,
                "status": "waiting_for_completion",
                "timestamp": time.time()
            }
            
            message_json = json.dumps(feedback_message)
            
            # Синхронная отправка через loop
            loop = asyncio.get_event_loop()
            if loop.is_running():
                for client in list(self.web_clients):
                    try:
                        loop.create_task(client.send(message_json))
                    except:
                        pass
    
    def is_command_already_processed(self, client_id: str, text: str) -> bool:
        """Проверка была ли команда уже обработана мгновенно"""
        if not hasattr(self, '_processed_commands'):
            return False
        
        command_key = f"{client_id}_{hash(text)}"
        return command_key in self._processed_commands
    
    async def process_with_enhanced_systems(self, client_id: str, text: str, confidence: float, 
                                          duration: float, recording_path: str = None, 
                                          speech_audio = None):
        """
        МОДИФИЦИРОВАННАЯ обработка с проверкой мгновенного выполнения
        """
        if hasattr(self, 'is_command_already_processed') and self.is_command_already_processed(client_id, text):
            print(f"⚡ SKIPPING - Command already processed instantly: '{text}'")
            return
        # Проверяем, была ли команда уже обработана мгновенно
        if self.is_command_already_processed(client_id, text):
            print(f"⚡ COMMAND ALREADY PROCESSED INSTANTLY - SKIPPING ALL SYSTEMS: '{text}'")
            print(f"🚫 Enhanced systems bypassed for instant command")
            return
        
        # Если не была обработана мгновенно, продолжаем обычную обработку
        print(f"🔄 Processing through enhanced systems: '{text}'")
        
        # Вызываем оригинальный метод обработки
        if hasattr(self.base_processor, 'process_with_enhanced_systems'):
            await self.base_processor.process_with_enhanced_systems(
                client_id, text, confidence, duration, recording_path, speech_audio
            )
    
    def get_instant_stats(self) -> dict:
        """Получение статистики мгновенного выполнения"""
        stats = self.instant_stats.copy()
        
        # Добавляем дополнительную информацию
        stats.update({
            'instant_system_enabled': True,
            'instant_response_threshold_ms': 100,  # Целевое время отклика
            'commands_in_cache': len(getattr(self, '_processed_commands', {})),
            'instant_hit_rate': (
                (stats['instant_commands_executed'] / 
                 max(1, stats['instant_commands_executed'] + stats['partial_commands_detected'])) * 100
            ) if stats['instant_commands_executed'] > 0 else 0
        })
        
        return stats

    # ИСПРАВЛЕНИЕ: Делегируем другие методы базовому процессору
    def __getattr__(self, name):
        """Делегирование всех остальных атрибутов к базовому процессору"""
        return getattr(self.base_processor, name)

# ИСПРАВЛЕННАЯ функция создания процессора
def create_enhanced_processor_with_instant_commands(base_processor, web_clients_ref):
    """
    ИСПРАВЛЕННОЕ создание расширенного процессора с поддержкой мгновенных команд
    """
    
    enhanced_processor = EnhancedProcessorWithInstantCommands(base_processor, web_clients_ref)
    
    logger.info("🚀 Enhanced processor with instant commands created")
    return enhanced_processor

# ИСПРАВЛЕННЫЙ обработчик ASR клиентов
async def handle_asr_client_with_instant_commands(websocket, enhanced_processor):
    """
    ИСПРАВЛЕННЫЙ обработчик ASR клиентов с мгновенным выполнением
    """
    client_addr = websocket.remote_address
    client_id = f"{client_addr[0]}_{client_addr[1]}_{int(time.time())}"
    
    logger.info(f"🎤 ASR клиент с мгновенными командами: {client_id}")
    
    try:
        client_error_count = 0
        max_client_errors = 20
        last_ping_time = time.time()
        chunks_received = 0
        instant_commands_count = 0
        
        async for message in websocket:
            try:
                if isinstance(message, bytes):
                    # Обработка аудио данных
                    try:
                        import numpy as np
                        audio_chunk = np.frombuffer(message, dtype=np.int16).astype(np.float32) / 32768.0
                        chunks_received += 1
                        
                        # Проверка валидности аудио данных
                        if np.any(np.isnan(audio_chunk)) or np.any(np.isinf(audio_chunk)):
                            logger.warning(f"⚠️ Невалидные аудио данные от {client_id}")
                            client_error_count += 1
                            continue
                        
                        # ОБРАБОТКА ЧЕРЕЗ РАСШИРЕННЫЙ ПРОЦЕССОР С МГНОВЕННЫМИ КОМАНДАМИ
                        if enhanced_processor:
                            result = enhanced_processor.process_audio_chunk_with_predictive(client_id, audio_chunk)
                            # ДОБАВИТЬ эту проверку:
                            if result is not None and result.strip():
                                # Проверяем instant execution СИНХРОННО
                                try:
                                    completeness, command_data = enhanced_processor.instant_processor.analyzer.analyze_command_completeness(result, client_id)
                                    if completeness == CommandCompleteness.COMPLETE:
                                        print(f"🚀 INSTANT EXECUTION TRIGGERED!")
                                        # Создаем задачу для мгновенной отправки
                                        asyncio.create_task(
                                            enhanced_processor.instant_processor._send_instant_result(client_id, command_data)
                                        )
                                except Exception as e:
                                    print(f"❌ Instant check error: {e}")
            
                            if result and isinstance(result, str) and result.strip():
                                asyncio.create_task(
                                    enhanced_processor.instant_processor.process_instant_command(client_id, result)
                                )
                            if result is not None:
                                if result.strip():
                                    try:
                                        await asyncio.wait_for(websocket.send(result), timeout=2.0)
                                        
                                        # Проверяем была ли команда выполнена мгновенно
                                        instant_stats = enhanced_processor.get_instant_stats()
                                        
                                        print(f"\n{'⚡' * 60}")
                                        print(f"   INSTANT COMMAND SYSTEM + FASTWHISPER")
                                        print(f"   🎤 TRANSCRIPTION: '{result.upper()}'")
                                        print(f"   👤 Client: {client_addr[0]} | 📊 Chunks: {chunks_received}")
                                        print(f"   ⚡ Instant commands: {instant_stats['instant_commands_executed']}")
                                        print(f"   ⏳ Partial commands: {instant_stats['partial_commands_detected']}")
                                        print(f"   🎯 Instant hit rate: {instant_stats['instant_hit_rate']:.1f}%")
                                        print(f"   ⏱️ Avg instant response: {instant_stats['average_instant_response_time']*1000:.1f}ms")
                                        print(f"   🚀 Target response: <100ms")
                                        print('⚡' * 60 + "\n")
                                        
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
                    # Обработка текстовых команд (PING, STATS и т.д.)
                    current_time = time.time()
                    
                    if message == "PING":
                        await websocket.send("PONG")
                        last_ping_time = current_time
                        
                    elif message == "STATS":
                        if enhanced_processor:
                            # ИСПРАВЛЕНИЕ: Используем прокси-доступ к статистике
                            stats = enhanced_processor.stats.copy()
                            stats['server_uptime'] = current_time - stats.get('server_uptime_start', current_time)
                            stats['instant_system_version'] = 'instant_commands_v1'
                            
                            try:
                                await asyncio.wait_for(websocket.send(json.dumps(stats)), timeout=3.0)
                            except asyncio.TimeoutError:
                                logger.warning(f"⚠️ Timeout sending stats to {client_id}")
                                
                    elif message == "MODEL_INFO":
                        if enhanced_processor:
                            # ИСПРАВЛЕНИЕ: Используем прокси-доступ к ASR
                            model_info = enhanced_processor.asr.get_info()
                            model_info.update({
                                'instant_commands_enabled': True,
                                'instant_response_target_ms': 100,
                                'instant_command_patterns': 7,
                                'enhanced_mode': 'INSTANT_COMMANDS_V1',
                                'command_prediction': True,
                                'real_time_execution': True
                            })
                            
                            try:
                                await asyncio.wait_for(websocket.send(json.dumps(model_info)), timeout=3.0)
                            except asyncio.TimeoutError:
                                logger.warning(f"⚠️ Timeout sending model info to {client_id}")
                
                # Проверка количества ошибок клиента
                if client_error_count > max_client_errors:
                    logger.error(f"❌ Too many errors from {client_id}, disconnecting")
                    break
                    
            except Exception as e:
                logger.error(f"❌ Critical error handling message from {client_id}: {e}")
                client_error_count += 1
                
    except Exception as e:
        logger.error(f"❌ ASR client error: {e}")
    finally:
        # Очистка буферов клиента
        if enhanced_processor and hasattr(enhanced_processor, 'segmentation_processor') and enhanced_processor.segmentation_processor:
            enhanced_processor.segmentation_processor.cleanup_client(client_id)
            logger.debug(f"🗑️ Cleaned up buffers for {client_id}")

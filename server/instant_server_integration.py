#!/usr/bin/env python3
"""
ИНТЕГРАЦИЯ СИСТЕМЫ МГНОВЕННОГО ВЫПОЛНЕНИЯ КОМАНД
Модификация процессора для поддержки мгновенного отклика
"""

import asyncio
import time
import logging
from typing import Optional
import json

# Импорт системы мгновенного выполнения
from instant_command_system import (
    InstantCommandProcessor,
    CommandCompleteness,
    integrate_instant_command_system
)

logger = logging.getLogger(__name__)

class EnhancedProcessorWithInstantCommands:
    """Расширенный процессор с поддержкой мгновенного выполнения"""
    
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
    
    def process_audio_chunk(self, client_id: str, audio_chunk) -> Optional[str]:
        """
        МОДИФИЦИРОВАННАЯ обработка чанков с мгновенным выполнением
        """
        # Обычная обработка через базовый процессор
        transcription_result = self.base_processor.process_audio_chunk(client_id, audio_chunk)
        
        if transcription_result and isinstance(transcription_result, str) and transcription_result.strip():
            # Проверяем возможность мгновенного выполнения
            asyncio.create_task(self._check_instant_execution(client_id, transcription_result))
        
        return transcription_result
    
    async def _check_instant_execution(self, client_id: str, text: str):
        """Проверка и выполнение мгновенной команды"""
        start_time = time.time()
        
        try:
            # Анализируем завершенность команды
            completeness, command_data = self.instant_processor.analyzer.analyze_command_completeness(text, client_id)
            
            if completeness == CommandCompleteness.COMPLETE:
                # МГНОВЕННОЕ ВЫПОЛНЕНИЕ
                print(f"🚀 INSTANT EXECUTION triggered for: '{text}'")
                
                # Отправляем результат немедленно
                await self.instant_processor._send_instant_result(client_id, command_data)
                
                # Обновляем статистику
                response_time = time.time() - start_time
                self.instant_stats['instant_commands_executed'] += 1
                self.instant_stats['total_instant_response_time'] += response_time
                self.instant_stats['average_instant_response_time'] = (
                    self.instant_stats['total_instant_response_time'] / 
                    self.instant_stats['instant_commands_executed']
                )
                
                print(f"⚡ Instant response time: {response_time*1000:.1f}ms")
                
                # ВАЖНО: Блокируем дальнейшую обработку через обычные системы
                # чтобы избежать дублирования результатов
                await self._mark_command_as_processed(client_id, text)
                
                return True
                
            elif completeness == CommandCompleteness.INCOMPLETE:
                # Команда неполная - отправляем обратную связь
                print(f"⏳ PARTIAL COMMAND detected: '{text}'")
                await self.instant_processor._send_partial_feedback(client_id, text)
                self.instant_stats['partial_commands_detected'] += 1
                
                return False
        
        except Exception as e:
            logger.error(f"❌ Error in instant execution check: {e}")
            return False
        
        return False
    
    async def _mark_command_as_processed(self, client_id: str, text: str):
        """Помечаем команду как уже обработанную чтобы избежать дублирования"""
        
        # Сохраняем в кэше обработанных команд
        if not hasattr(self, '_processed_commands'):
            self._processed_commands = {}
        
        command_key = f"{client_id}_{hash(text)}"
        self._processed_commands[command_key] = {
            'text': text,
            'timestamp': time.time(),
            'processed_instantly': True
        }
        
        # Очищаем старые записи (храним только последние 10 минут)
        current_time = time.time()
        expired_keys = [
            key for key, data in self._processed_commands.items()
            if current_time - data['timestamp'] > 600  # 10 минут
        ]
        for key in expired_keys:
            del self._processed_commands[key]
    
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
        
        # Проверяем, была ли команда уже обработана мгновенно
        if self.is_command_already_processed(client_id, text):
            print(f"⚡ Command already processed instantly, skipping enhanced systems: '{text}'")
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

# Модификация основного обработчика ASR клиентов
async def handle_asr_client_with_instant_commands(websocket, enhanced_processor):
    """
    МОДИФИЦИРОВАННЫЙ обработчик ASR клиентов с мгновенным выполнением
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
                            result = enhanced_processor.process_audio_chunk(client_id, audio_chunk)
                            
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
                    # Обработка текстовых команд
                    current_time = time.time()
                    
                    if message == "PING":
                        await websocket.send("PONG")
                        last_ping_time = current_time
                        
                    elif message == "STATS":
                        if enhanced_processor:
                            # Объединяем обычную статистику со статистикой мгновенного выполнения
                            stats = enhanced_processor.base_processor.stats.copy()
                            instant_stats = enhanced_processor.get_instant_stats()
                            stats.update(instant_stats)
                            
                            stats['server_uptime'] = current_time - stats.get('server_uptime_start', current_time)
                            stats['instant_system_version'] = 'instant_commands_v1'
                            
                            try:
                                await asyncio.wait_for(websocket.send(json.dumps(stats)), timeout=3.0)
                            except asyncio.TimeoutError:
                                logger.warning(f"⚠️ Timeout sending stats to {client_id}")
                                
                    elif message == "INSTANT_STATS":
                        # Специальная команда для получения только статистики мгновенного выполнения
                        if enhanced_processor:
                            instant_stats = enhanced_processor.get_instant_stats()
                            try:
                                await asyncio.wait_for(websocket.send(json.dumps(instant_stats)), timeout=3.0)
                            except asyncio.TimeoutError:
                                logger.warning(f"⚠️ Timeout sending instant stats to {client_id}")
                                
                    elif message == "MODEL_INFO":
                        if enhanced_processor:
                            model_info = enhanced_processor.base_processor.asr.get_info()
                            model_info.update({
                                'instant_commands_enabled': True,
                                'instant_response_target_ms': 100,
                                'instant_command_patterns': 7,  # Количество поддерживаемых паттернов
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
        if enhanced_processor and hasattr(enhanced_processor.base_processor, 'segmentation_processor'):
            enhanced_processor.base_processor.segmentation_processor.cleanup_client(client_id)
            logger.debug(f"🗑️ Cleaned up buffers for {client_id}")

def create_enhanced_processor_with_instant_commands(base_processor, web_clients_ref):
    """
    Создание расширенного процессора с поддержкой мгновенных команд
    """
    
    enhanced_processor = EnhancedProcessorWithInstantCommands(base_processor, web_clients_ref)
    
    # Заменяем метод обработки enhanced систем
    if hasattr(base_processor, 'process_with_enhanced_systems'):
        base_processor.process_with_enhanced_systems = enhanced_processor.process_with_enhanced_systems
    
    logger.info("🚀 Enhanced processor with instant commands created")
    return enhanced_processor

# Интеграция с основным сервером
async def integrate_instant_commands_into_server(original_main_function):
    """
    Интеграция системы мгновенных команд в основной сервер
    """
    
    print("\n" + "⚡" * 80)
    print("   🚀 INSTANT COMMAND SYSTEM INTEGRATION")
    print("   ⚡ МГНОВЕННОЕ ВЫПОЛНЕНИЕ СТОМАТОЛОГИЧЕСКИХ КОМАНД")
    print("   • ПРЕДИКТИВНЫЙ АНАЛИЗ ЗАВЕРШЕННОСТИ КОМАНД")
    print("   • МГНОВЕННАЯ ОТПРАВКА РЕЗУЛЬТАТОВ (<100ms)")
    print("   • ПРЕДОТВРАЩЕНИЕ ДУБЛИРОВАНИЯ ОБРАБОТКИ")
    print("   • ПОДДЕРЖКА 7 ТИПОВ КОМАНД")
    print("   • REAL-TIME ОБРАТНАЯ СВЯЗЬ")
    print("⚡" * 80)
    
    # Запускаем оригинальную функцию main
    await original_main_function()

if __name__ == "__main__":
    print("🚀 INSTANT COMMAND SYSTEM - Integration Module")
    print("=" * 60)
    print("📋 ПОДДЕРЖИВАЕМЫЕ КОМАНДЫ ДЛЯ МГНОВЕННОГО ВЫПОЛНЕНИЯ:")
    print()
    print("1. 🦷 PROBING DEPTH:")
    print("   'probing depth on tooth number 14 buccal surface 3 2 4'")
    print("   ⚡ Выполняется мгновенно при произнесении последнего числа")
    print()
    print("2. 🔄 MOBILITY:")
    print("   'tooth 8 has mobility grade 2'")
    print("   ⚡ Выполняется мгновенно при произнесении grade + число")
    print()
    print("3. 🩸 BLEEDING ON PROBING:")
    print("   'bleeding on probing tooth 12 buccal distal'")
    print("   ⚡ Выполняется мгновенно при произнесении позиции")
    print()
    print("4. 💧 SUPPURATION:")
    print("   'suppuration present on tooth 8 lingual mesial'")
    print("   ⚡ Выполняется мгновенно при произнесении позиции")
    print()
    print("5. 🔱 FURCATION:")
    print("   'furcation class 2 on tooth 6'")
    print("   ⚡ Выполняется мгновенно при полной команде")
    print()
    print("6. 📐 GINGIVAL MARGIN:")
    print("   'gingival margin on tooth 14 minus 1 0 plus 1'")
    print("   ⚡ Выполняется мгновенно при 3 значениях")
    print()
    print("7. ❌ MISSING TEETH:")
    print("   'missing teeth 1 16 17 32'")
    print("   ⚡ Выполняется мгновенно при списке номеров")
    print()
    print("🎯 ЦЕЛЕВОЕ ВРЕМЯ ОТКЛИКА: <100ms")
    print("📊 ПРЕДОТВРАЩЕНИЕ ДУБЛИРОВАНИЯ: Автоматическое")
    print("⏳ ПРОМЕЖУТОЧНАЯ ОБРАТНАЯ СВЯЗЬ: Для неполных команд")
    print("=" * 60)
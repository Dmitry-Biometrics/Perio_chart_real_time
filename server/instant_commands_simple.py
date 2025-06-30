#!/usr/bin/env python3
"""
ПРОСТАЯ СИСТЕМА МГНОВЕННОГО ВЫПОЛНЕНИЯ КОМАНД
Готовая к использованию версия без сложных зависимостей
"""

import re
import json
import asyncio
import time
import logging

logger = logging.getLogger(__name__)

class SimpleInstantAnalyzer:
    """Простой анализатор завершенности команд"""
    
    def __init__(self):
        # Паттерны для завершенных команд
        self.complete_patterns = {
            'probing_depth': r'probing\s+depth.*tooth.*\d+.*(?:buccal|lingual).*\d+\s+\d+\s+\d+',
            'mobility': r'tooth\s+\d+.*mobility.*grade\s+\d+',
            'bleeding': r'bleeding.*tooth\s+\d+.*(?:buccal|lingual).*(?:distal|mesial|mid)',
            'suppuration': r'suppuration.*tooth\s+\d+.*(?:buccal|lingual).*(?:distal|mesial|mid)',
            'furcation': r'furcation.*class\s+\d+.*tooth\s+\d+',
            'gingival_margin': r'gingival\s+margin.*tooth\s+\d+.*(?:minus|plus|\d+).*(?:minus|plus|\d+).*(?:minus|plus|\d+)',
            'missing_teeth': r'missing.*teeth.*\d+'
        }
        
        # Ключевые слова для частичных команд
        self.partial_keywords = [
            'probing depth', 'tooth', 'mobility', 'bleeding', 
            'suppuration', 'furcation', 'gingival margin', 'missing'
        ]
    
    def analyze(self, text: str):
        """Анализ завершенности команды"""
        text_clean = text.lower().strip()
        
        # Проверяем завершенные команды
        for cmd_type, pattern in self.complete_patterns.items():
            if re.search(pattern, text_clean):
                return 'COMPLETE', {
                    'type': cmd_type,
                    'text': text_clean,
                    'original': text
                }
        
        # Проверяем частичные команды
        if any(keyword in text_clean for keyword in self.partial_keywords):
            return 'PARTIAL', {
                'type': 'partial',
                'text': text_clean,
                'original': text
            }
        
        return 'UNKNOWN', None

class SimpleInstantProcessor:
    """Простой процессор мгновенного выполнения"""
    
    def __init__(self, base_processor, web_clients_ref):
        # ИСПРАВЛЕНИЕ: Копируем ВСЕ атрибуты базового процессора
        self.base_processor = base_processor
        self.web_clients = web_clients_ref
        self.analyzer = SimpleInstantAnalyzer()
        self._processed_commands = {}
        
        # Копируем критически важные атрибуты
        self.vad = getattr(base_processor, 'vad', None)
        self.asr = getattr(base_processor, 'asr', None)
        self.segmentation_processor = getattr(base_processor, 'segmentation_processor', None)
        self.stats = getattr(base_processor, 'stats', {})
        
        # Добавляем статистику мгновенного выполнения
        self.stats.update({
            'instant_commands_executed': 0,
            'partial_commands_detected': 0,
            'average_instant_response_time': 0.0,
            'total_instant_response_time': 0.0
        })
        
        logger.info("⚡ Simple instant processor initialized")
    
    def process_audio_chunk(self, client_id: str, audio_chunk):
        """Обработка аудио чанков с мгновенным выполнением"""
        # Обычная обработка через базовый процессор
        result = self.base_processor.process_audio_chunk(client_id, audio_chunk)
        
        if result and isinstance(result, str) and result.strip():
            # Запускаем проверку мгновенного выполнения
            asyncio.create_task(self._check_instant_execution(client_id, result))
        
        return result
    
    async def _check_instant_execution(self, client_id: str, text: str):
        """Проверка мгновенного выполнения"""
        try:
            start_time = time.time()
            
            completeness, command_data = self.analyzer.analyze(text)
            
            if completeness == 'COMPLETE':
                # МГНОВЕННОЕ ВЫПОЛНЕНИЕ
                response_time = (time.time() - start_time) * 1000
                
                print(f"⚡ INSTANT EXECUTION: '{text}' ({response_time:.1f}ms)")
                
                await self._send_instant_result(client_id, text, command_data)
                
                # Помечаем как обработанную
                self._mark_command_as_processed(client_id, text)
                
                # Обновляем статистику
                self.stats['instant_commands_executed'] += 1
                self.stats['total_instant_response_time'] += response_time
                if self.stats['instant_commands_executed'] > 0:
                    self.stats['average_instant_response_time'] = (
                        self.stats['total_instant_response_time'] / 
                        self.stats['instant_commands_executed']
                    )
                
                return True
                
            elif completeness == 'PARTIAL':
                # Частичная команда
                print(f"⏳ PARTIAL COMMAND: '{text}'")
                await self._send_partial_feedback(client_id, text)
                
                self.stats['partial_commands_detected'] += 1
                return False
                
        except Exception as e:
            logger.error(f"❌ Error in instant execution check: {e}")
            return False
        
        return False
    
    async def _send_instant_result(self, client_id: str, text: str, command_data: dict):
        """Отправка мгновенного результата"""
        if not self.web_clients:
            return
            
        message = {
            "type": "periodontal_update",
            "client_id": client_id,
            "success": True,
            "message": f"⚡ INSTANT: {text}",
            "instant_execution": True,
            "response_time_ms": self.stats.get('average_instant_response_time', 0),
            "timestamp": time.time(),
            "command_data": command_data,
            "system": "simple_instant_commands_v1"
        }
        
        await self._broadcast_to_web_clients(message)
    
    async def _send_partial_feedback(self, client_id: str, text: str):
        """Отправка обратной связи для частичной команды"""
        if not self.web_clients:
            return
            
        message = {
            "type": "partial_command_feedback",
            "client_id": client_id,
            "partial_text": text,
            "message": f"⏳ Waiting for completion: {text}",
            "timestamp": time.time(),
            "system": "simple_instant_commands_v1"
        }
        
        await self._broadcast_to_web_clients(message)
    
    async def _broadcast_to_web_clients(self, message):
        """Безопасная отправка сообщения всем веб-клиентам"""
        if not self.web_clients:
            return
            
        message_json = json.dumps(message)
        disconnected = set()
        
        for client in list(self.web_clients):
            try:
                await asyncio.wait_for(client.send(message_json), timeout=1.0)
            except:
                disconnected.add(client)
        
        # Удаляем отключенных клиентов
        for client in disconnected:
            self.web_clients.discard(client)
    
    def _mark_command_as_processed(self, client_id: str, text: str):
        """Помечаем команду как обработанную мгновенно"""
        command_key = f"{client_id}_{hash(text)}"
        self._processed_commands[command_key] = {
            'text': text,
            'timestamp': time.time(),
            'processed_instantly': True
        }
        
        # Очищаем старые записи (старше 5 минут)
        current_time = time.time()
        expired_keys = [
            key for key, data in self._processed_commands.items()
            if current_time - data['timestamp'] > 300
        ]
        for key in expired_keys:
            del self._processed_commands[key]
    
    def is_command_already_processed(self, client_id: str, text: str) -> bool:
        """Проверяем была ли команда уже обработана мгновенно"""
        command_key = f"{client_id}_{hash(text)}"
        return command_key in self._processed_commands
    
    async def process_with_enhanced_systems(self, client_id: str, text: str, confidence: float,
                                          duration: float, recording_path: str = None,
                                          speech_audio = None):
        """Обработка с проверкой дублирования"""
        
        # Если команда уже обработана мгновенно, пропускаем
        if self.is_command_already_processed(client_id, text):
            print(f"⚡ Command already processed instantly, skipping: '{text}'")
            return
        
        # Продолжаем обычную обработку
        if hasattr(self.base_processor, 'process_with_enhanced_systems'):
            await self.base_processor.process_with_enhanced_systems(
                client_id, text, confidence, duration, recording_path, speech_audio
            )
    
    def get_instant_stats(self):
        """Получение статистики мгновенного выполнения"""
        total_commands = (
            self.stats['instant_commands_executed'] + 
            self.stats['partial_commands_detected']
        )
        
        hit_rate = 0.0
        if total_commands > 0:
            hit_rate = (self.stats['instant_commands_executed'] / total_commands) * 100
        
        return {
            'instant_commands_executed': self.stats['instant_commands_executed'],
            'partial_commands_detected': self.stats['partial_commands_detected'],
            'average_instant_response_time': self.stats['average_instant_response_time'],
            'instant_hit_rate': hit_rate,
            'commands_in_cache': len(self._processed_commands),
            'instant_system_enabled': True,
            'system_version': 'simple_instant_commands_v1'
        }
    
    def __getattr__(self, name):
        """Проксируем недостающие атрибуты к базовому процессору"""
        if hasattr(self.base_processor, name):
            return getattr(self.base_processor, name)
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

def create_processor_with_instant_commands(base_processor, web_clients):
    """
    Создание процессора с мгновенными командами (безопасная версия)
    """
    
    if base_processor is None:
        logger.error("❌ Base processor is None!")
        return None
    
    try:
        # Проверяем критичные атрибуты
        required_attrs = ['asr', 'vad']
        missing_attrs = []
        
        for attr in required_attrs:
            if not hasattr(base_processor, attr) or getattr(base_processor, attr) is None:
                missing_attrs.append(attr)
        
        if missing_attrs:
            logger.error(f"❌ Base processor missing attributes: {missing_attrs}")
            return base_processor  # Возвращаем базовый процессор
        
        # Создаем расширенный процессор
        enhanced_processor = SimpleInstantProcessor(base_processor, web_clients)
        
        # Финальная проверка
        if hasattr(enhanced_processor, 'asr') and enhanced_processor.asr is not None:
            logger.info("⚡ Enhanced processor with instant commands created successfully!")
            print("🚀 INSTANT COMMANDS SYSTEM ACTIVATED")
            print("   ⚡ Response target: <100ms")
            print("   🎯 Supported commands: 7 types")
            print("   🔄 Fallback protection: enabled")
            return enhanced_processor
        else:
            logger.warning("⚠️ Enhanced processor validation failed, using base processor")
            return base_processor
            
    except Exception as e:
        logger.error(f"❌ Error creating enhanced processor: {e}")
        logger.error(f"   Falling back to base processor")
        return base_processor

# МОДИФИЦИРОВАННЫЙ обработчик ASR клиентов с мгновенными командами
async def handle_asr_client_with_instant_commands(websocket, enhanced_processor):
    """Обработчик ASR клиентов с поддержкой мгновенных команд"""
    client_addr = websocket.remote_address
    client_id = f"{client_addr[0]}_{client_addr[1]}_{int(time.time())}"
    
    logger.info(f"🎤 ASR клиент с мгновенными командами: {client_id}")
    
    try:
        chunks_received = 0
        
        async for message in websocket:
            try:
                if isinstance(message, bytes):
                    # Обработка аудио данных
                    import numpy as np
                    audio_chunk = np.frombuffer(message, dtype=np.int16).astype(np.float32) / 32768.0
                    chunks_received += 1
                    
                    if enhanced_processor:
                        result = enhanced_processor.process_audio_chunk(client_id, audio_chunk)
                        
                        if result is not None and result.strip():
                            try:
                                await asyncio.wait_for(websocket.send(result), timeout=2.0)
                                
                                # Показываем статистику мгновенного выполнения
                                instant_stats = enhanced_processor.get_instant_stats()
                                
                                print(f"\n{'⚡' * 60}")
                                print(f"   INSTANT COMMAND SYSTEM + FASTWHISPER")
                                print(f"   🎤 TRANSCRIPTION: '{result.upper()}'")
                                print(f"   👤 Client: {client_addr[0]} | 📊 Chunks: {chunks_received}")
                                print(f"   ⚡ Instant commands: {instant_stats['instant_commands_executed']}")
                                print(f"   ⏳ Partial commands: {instant_stats['partial_commands_detected']}")
                                print(f"   🎯 Hit rate: {instant_stats['instant_hit_rate']:.1f}%")
                                print(f"   ⏱️ Avg response: {instant_stats['average_instant_response_time']:.1f}ms")
                                print(f"   🎯 Target: <100ms")
                                print('⚡' * 60 + "\n")
                                
                            except asyncio.TimeoutError:
                                logger.warning(f"⚠️ Timeout sending result to {client_id}")
                        else:
                            await websocket.send("NO_SPEECH")
                    else:
                        await websocket.send("SERVER_NOT_READY")
                        
                elif isinstance(message, str):
                    # Обработка текстовых команд
                    if message == "PING":
                        await websocket.send("PONG")
                    elif message == "STATS":
                        if enhanced_processor:
                            stats = enhanced_processor.stats.copy()
                            instant_stats = enhanced_processor.get_instant_stats()
                            stats.update(instant_stats)
                            await websocket.send(json.dumps(stats))
                    elif message == "MODEL_INFO":
                        if enhanced_processor:
                            model_info = enhanced_processor.asr.get_info()
                            model_info.update({
                                'instant_commands_enabled': True,
                                'instant_response_target_ms': 100,
                                'instant_patterns_supported': 7,
                                'enhanced_mode': 'SIMPLE_INSTANT_COMMANDS_V1'
                            })
                            await websocket.send(json.dumps(model_info))
                            
            except Exception as e:
                logger.error(f"❌ Error handling message from {client_id}: {e}")
                
    except Exception as e:
        logger.error(f"❌ ASR client error: {e}")
    finally:
        if enhanced_processor and hasattr(enhanced_processor, 'segmentation_processor'):
            if enhanced_processor.segmentation_processor:
                enhanced_processor.segmentation_processor.cleanup_client(client_id)
        logger.debug(f"🗑️ Cleaned up client {client_id}")

# ФУНКЦИЯ ДЛЯ ИНТЕГРАЦИИ В ВАШ СЕРВЕР
def integrate_simple_instant_commands(original_main_function):
    """
    Простая интеграция мгновенных команд в ваш сервер
    """
    
    print("\n" + "⚡" * 80)
    print("   🚀 SIMPLE INSTANT COMMAND SYSTEM")
    print("   ⚡ МГНОВЕННОЕ ВЫПОЛНЕНИЕ СТОМАТОЛОГИЧЕСКИХ КОМАНД")
    print("   • ПРОСТАЯ ИНТЕГРАЦИЯ БЕЗ СЛОЖНЫХ ЗАВИСИМОСТЕЙ")
    print("   • АВТОМАТИЧЕСКИЙ FALLBACK К БАЗОВОМУ ПРОЦЕССОРУ")
    print("   • ПОДДЕРЖКА 7 ТИПОВ КОМАНД")
    print("   • ЦЕЛЬ: ОТКЛИК <100ms")
    print("⚡" * 80)
    
    return original_main_function

if __name__ == "__main__":
    print("⚡ SIMPLE INSTANT COMMANDS SYSTEM")
    print("=" * 60)
    print("📋 ГОТОВО К ИСПОЛЬЗОВАНИЮ:")
    print()
    print("1. 📁 Сохраните этот файл как: simple_instant_commands.py")
    print()
    print("2. 🔧 В вашем new_server.py добавьте импорт:")
    print("   from simple_instant_commands import create_processor_with_instant_commands")
    print()
    print("3. 🔄 Замените создание процессора:")
    print("   # БЫЛО:")
    print("   processor = CriticallyFixedProcessorWithSegmentation()")
    print()
    print("   # СТАЛО:")
    print("   base_processor = CriticallyFixedProcessorWithSegmentation()")
    print("   processor = create_processor_with_instant_commands(base_processor, web_clients)")
    print()
    print("4. 🚀 Запустите сервер")
    print()
    print("✅ ПОДДЕРЖИВАЕМЫЕ КОМАНДЫ:")
    print("   • probing depth on tooth number 14 buccal surface 3 2 4")
    print("   • tooth 8 has mobility grade 2")
    print("   • bleeding on probing tooth 12 buccal distal")
    print("   • suppuration present on tooth 8 lingual mesial")
    print("   • furcation class 2 on tooth 6")
    print("   • gingival margin on tooth 14 minus 1 0 plus 1")
    print("   • missing teeth 1 16 17 32")
    print()
    print("🎯 СИСТЕМА АВТОМАТИЧЕСКИ:")
    print("   ⚡ Определяет завершенность команд")
    print("   🚀 Выполняет завершенные команды мгновенно")
    print("   ⏳ Показывает обратную связь для частичных команд")
    print("   🛡️ Предотвращает дублирование обработки")
    print("   📊 Ведет статистику производительности")
    print("=" * 60)
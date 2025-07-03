#!/usr/bin/env python3
"""
ИСПРАВЛЕННАЯ интеграция LLM с ЛИБЕРАЛЬНОЙ детекцией команд
Исправляет проблему когда LLM не активируется для исправления ASR ошибок
"""

import logging
import json
import asyncio
import re
from datetime import datetime
from typing import Dict, Optional

# Импорт усиленного процессора
try:
    from enhanced_periodontal_llm import (
        initialize_enhanced_processor,
        process_periodontal_transcription,
        get_processor_stats,
        enhanced_llm_processor
    )
    ENHANCED_LLM_AVAILABLE = True
    logging.info("🤖 Enhanced LLM Periodontal Processor available")
except ImportError as e:
    ENHANCED_LLM_AVAILABLE = False
    logging.warning(f"⚠️ Enhanced LLM Processor not available: {e}")

logger = logging.getLogger(__name__)

class FixedLLMPeriodontalIntegration:
    """ИСПРАВЛЕННАЯ интеграция с очень либеральной детекцией команд"""
    
    def __init__(self, openai_api_key: str = None):
        self.enabled = False
        self.openai_api_key = openai_api_key
        self.session_stats = {
            'llm_commands_processed': 0,
            'llm_successful_commands': 0,
            'llm_corrections_made': 0,
            'llm_validation_passed': 0,
            'llm_average_confidence': 0.0,
            'asr_errors_fixed': 0,
            'liberal_detections': 0  # НОВАЯ статистика
        }
        
        if ENHANCED_LLM_AVAILABLE and openai_api_key:
            self.initialize()
    
    def initialize(self) -> bool:
        """Инициализация LLM процессора"""
        try:
            if initialize_enhanced_processor(self.openai_api_key):
                self.enabled = True
                logger.info("🤖 FIXED LLM Periodontal Integration initialized")
                return True
            else:
                logger.error("❌ Failed to initialize LLM processor")
                return False
        except Exception as e:
            logger.error(f"❌ LLM integration initialization error: {e}")
            return False
    
    async def process_transcription(self, text: str, confidence: float = 0.0, patient_id: str = None) -> Dict:
        """
        ИСПРАВЛЕННАЯ обработка транскрипции с очень либеральной детекцией
        """
        if not self.enabled:
            return {
                "success": False,
                "error": "llm_not_available",
                "message": "Enhanced LLM Processor not available"
            }
        
        try:
            self.session_stats['llm_commands_processed'] += 1
            
            logger.info(f"🤖 LLM Processing: '{text}' (ASR confidence: {confidence:.3f})")
            
            # Обрабатываем команду через LLM
            result = await process_periodontal_transcription(text, patient_id)
            
            # Обновляем статистику
            if result.get("success"):
                self.session_stats['llm_successful_commands'] += 1
                self.session_stats['llm_validation_passed'] += 1
                
                # Обновляем среднюю уверенность
                confidence_value = result.get("confidence", 0.0)
                if self.session_stats['llm_successful_commands'] > 0:
                    alpha = 0.1
                    self.session_stats['llm_average_confidence'] = (
                        alpha * confidence_value + 
                        (1 - alpha) * self.session_stats['llm_average_confidence']
                    )
                
                # Проверяем, были ли исправления
                original = result.get("original_text", "").lower()
                corrected = result.get("corrected_text", "").lower()
                if original != corrected:
                    self.session_stats['asr_errors_fixed'] += 1
                    logger.info(f"🔧 ASR FIXED: '{original}' → '{corrected}'")
                
                # Добавляем информацию о сессии
                result['session_stats'] = self.session_stats.copy()
                result['asr_confidence'] = confidence
                result['system'] = 'fixed_llm_periodontal'
                
                logger.info(f"🤖 LLM SUCCESS: {result.get('measurement_type', 'unknown')} "
                           f"for Tooth {result.get('tooth_number')} "
                           f"(LLM conf: {confidence_value:.3f}, "
                           f"ASR conf: {confidence:.3f})")
            else:
                logger.warning(f"🤖 LLM FAILED to process: '{text}' - {result.get('message', 'Unknown error')}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ LLM processing error: {e}")
            return {
                "success": False,
                "error": "llm_processing_error",
                "message": f"Error processing LLM command: {str(e)}",
                "confidence": 0.0
            }
    
    # ДОПОЛНИТЕЛЬНАЯ ФУНКЦИЯ для исправления ASR промпта в FastWhisper
    def get_dental_asr_prompt():
        """
        Возвращает улучшенный промпт для FastWhisper ASR
        """
        return """
        Dental examination recording. Common terms: 
        probing depth, bleeding on probing, suppuration, mobility grade, furcation class, 
        gingival margin, missing teeth, tooth number, buccal surface, lingual surface, 
        distal, mesial, millimeter, grade 1 2 3, class 1 2 3, 
        teeth numbers 1 through 32, one two three four five six seven eight nine ten.
        """

        # ФУНКЦИЯ для добавления промпта в ASR
    def enhance_asr_with_dental_prompt(asr_model, audio_data):
        """
        Добавляет dental промпт в ASR транскрипцию
        """
        try:
            dental_prompt = get_dental_asr_prompt()
            
            segments, info = asr_model.transcribe(
                audio_data,
                language="en",
                condition_on_previous_text=False,
                temperature=0.0,
                vad_filter=False,
                beam_size=1,
                best_of=1,
                without_timestamps=True,
                word_timestamps=False,
                initial_prompt=dental_prompt,  # ДОБАВЛЯЕМ DENTAL ПРОМПТ
                suppress_blank=True,
                suppress_tokens=[-1],
                log_prob_threshold=-1.0,
                no_speech_threshold=0.6,
                compression_ratio_threshold=2.4,
            )
            
            return segments, info
            
        except Exception as e:
            logger.error(f"Error in enhanced ASR: {e}")
            # Fallback без промпта
            return asr_model.transcribe(audio_data, language="en")
        
    
    def is_periodontal_command_liberal(self, text: str) -> bool:
        """
        ИСПРАВЛЕННАЯ ОЧЕНЬ ЛИБЕРАЛЬНАЯ проверка periodontal команд с ОТЛАДКОЙ
        """
        if not self.enabled:
            logger.warning(f"🚨 LLM not enabled for liberal detection: '{text}'")
            return False
        
        text_lower = text.lower()
        
        # РАСШИРЕННЫЕ ключевые слова включая все возможные ASR ошибки
        liberal_keywords = [
            # Правильные термины
            'tooth', 'teeth', 'bleeding', 'probing', 'depth', 'mobility', 'furcation', 
            'plaque', 'pocket', 'gingival', 'missing', 'suppuration', 'recession',
            'buccal', 'lingual', 'surface', 'distal', 'mesial', 'grade', 'class',
            
            # Частые ASR ошибки из ваших примеров
            'rubbing', 'robin', 'buckle', 'wingle', 'lingle', 'lingwal', 'teath', 
            'suppration', 'separation', 'furkat', 'cache', 'mobil', 'probin', 
            'bleedin', 'gingi', 'tool', 'two', 'tree', 'for', 'ate', 'sick', 'sex',
            'occal', 'this', 'that', 'too', 'to',  # ДОБАВЛЕНО: критические слова
            
            # Дополнительные ASR ошибки
            'propping', 'proving', 'poking', 'booking', 'looking', 'cooking',
            'facial', 'special', 'racial', 'crucial', 'social',
            'dental', 'mental', 'rental', 'central',
            'mobile', 'motile', 'hostile', 'fertile',
            'present', 'pleasant', 'resident', 'student',
            
            # Числовые слова (часто путаются)
            'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten',
            'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen', 'sixteen', 
            'seventeen', 'eighteen', 'nineteen', 'twenty', 'thirty', 'thirty-two'
        ]
        
        # Проверяем наличие ЛЮБОГО ключевого слова
        has_keyword = any(keyword in text_lower for keyword in liberal_keywords)
        found_keywords = [kw for kw in liberal_keywords if kw in text_lower]
        
        # Проверяем наличие чисел (в любом формате)
        has_numbers = bool(re.search(r'\d+', text_lower))
        found_numbers = re.findall(r'\d+', text_lower)
        
        # Проверяем длину (dental команды обычно не очень короткие)
        reasonable_length = len(text.split()) >= 2
        
        # ИСПРАВЛЕННЫЕ исключения - более умные
        exclusions = [
            'hello', 'hi', 'bye', 'goodbye', 'thank you', 'thanks',
            'weather', 'time', 'date', 'calendar', 'schedule',
            'music', 'play', 'stop', 'pause', 'volume',
            'call', 'phone', 'email', 'message', 'text'
        ]
        
        # КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: проверяем ПОЛНЫЕ фразы, а не отдельные слова
        is_excluded = any(excl in text_lower for excl in exclusions if len(excl) > 3)
        
        # ОСОБАЯ ОБРАБОТКА для "too" - исключаем только если это НЕ dental контекст
        if 'too' in text_lower and has_keyword and ('missing' in text_lower or 'tooth' in text_lower):
            is_excluded = False  # НЕ исключаем dental команды с "too"
        
        found_exclusions = [excl for excl in exclusions if excl in text_lower and len(excl) > 3]
        
        # ЛИБЕРАЛЬНОЕ решение: активируем LLM если есть хоть какие-то признаки dental команды
        result = (has_keyword or has_numbers) and reasonable_length and not is_excluded
        
        # ДЕТАЛЬНАЯ ОТЛАДКА
        logger.info(f"🔍 LIBERAL DETECTION DEBUG for: '{text}'")
        logger.info(f"   ✅ Keywords found: {has_keyword} - {found_keywords}")
        logger.info(f"   🔢 Numbers found: {has_numbers} - {found_numbers}")
        logger.info(f"   📏 Reasonable length: {reasonable_length} ({len(text.split())} words)")
        logger.info(f"   ❌ Excluded: {is_excluded} - {found_exclusions}")
        
        # СПЕЦИАЛЬНАЯ ПРОВЕРКА для "missing this" паттернов
        if 'missing' in text_lower and ('this' in text_lower or 'too' in text_lower or 'that' in text_lower):
            result = True  # ПРИНУДИТЕЛЬНО активируем для missing команд
            logger.info(f"   🎯 SPECIAL CASE: Missing command with 'this/too/that' - FORCED ACTIVATION")
        
        logger.info(f"   🎯 FINAL DECISION: {result}")
        
        if result:
            self.session_stats['liberal_detections'] += 1
            logger.info(f"🎯 LIBERAL DETECTION ACTIVATED LLM for: '{text}'")
        else:
            logger.warning(f"⚠️ LIBERAL DETECTION REJECTED: '{text}'")
        
        return result
    
    def get_session_stats(self) -> Dict:
        """Получение статистики LLM сессии"""
        base_stats = self.session_stats.copy()
        
        if enhanced_llm_processor:
            processor_stats = enhanced_llm_processor.get_stats()
            base_stats.update(processor_stats)
        
        base_stats.update({
            "enabled": self.enabled,
            "llm_available": ENHANCED_LLM_AVAILABLE,
            "openai_configured": bool(self.openai_api_key),
            "detection_mode": "liberal",
            "system_version": "fixed_liberal_llm_periodontal"
        })
        
        return base_stats
    
    def reset_session_stats(self):
        """Сброс статистики сессии"""
        self.session_stats = {
            'llm_commands_processed': 0,
            'llm_successful_commands': 0,
            'llm_corrections_made': 0,
            'llm_validation_passed': 0,
            'llm_average_confidence': 0.0,
            'asr_errors_fixed': 0,
            'liberal_detections': 0
        }


# Глобальный экземпляр ИСПРАВЛЕННОЙ интеграции
fixed_llm_integration = None

def initialize_fixed_llm_integration(openai_api_key: str) -> bool:
    """Инициализация ИСПРАВЛЕННОЙ LLM интеграции"""
    global fixed_llm_integration
    
    if not ENHANCED_LLM_AVAILABLE:
        logger.warning("⚠️ Enhanced LLM not available")
        return False
    
    if not openai_api_key:
        logger.warning("⚠️ OpenAI API key not provided")
        return False
    
    try:
        fixed_llm_integration = FixedLLMPeriodontalIntegration(openai_api_key)
        if fixed_llm_integration.enabled:
            logger.info("🤖 FIXED LLM Integration initialized successfully")
            return True
        else:
            logger.error("❌ FIXED LLM Integration failed to initialize")
            return False
    except Exception as e:
        logger.error(f"❌ FIXED LLM Integration initialization error: {e}")
        return False

async def process_transcription_with_fixed_llm(text: str, confidence: float = 0.0, patient_id: str = None) -> Dict:
    """
    КЭШИРОВАННАЯ функция для интеграции с основным сервером
    """
    if not fixed_llm_integration or not fixed_llm_integration.enabled:
        return {
            "success": False,
            "error": "llm_not_initialized",
            "message": "FIXED LLM Integration not initialized"
        }
    
    # ПРОВЕРЯЕМ КЭША ПЕРВЫМ ДЕЛОМ
    try:
        from llm_cache import llm_cache
        cached_result = llm_cache.get(text)
        if cached_result:
            return cached_result
    except ImportError:
        pass  # Кэш недоступен
    
    # Обрабатываем через LLM
    result = await fixed_llm_integration.process_transcription(text, confidence, patient_id)
    
    # Сохраняем в кэш
    try:
        from llm_cache import llm_cache
        llm_cache.put(text, result)
    except ImportError:
        pass
    
    return result

def is_periodontal_command_fixed_llm(text: str) -> bool:
    """
    ИСПРАВЛЕННАЯ LLM-enhanced проверка periodontal команд (ЛИБЕРАЛЬНАЯ)
    """
    if fixed_llm_integration and fixed_llm_integration.enabled:
        return fixed_llm_integration.is_periodontal_command_liberal(text)
    return False

def get_fixed_llm_stats() -> Dict:
    """Получение статистики ИСПРАВЛЕННОЙ LLM системы"""
    if fixed_llm_integration:
        return fixed_llm_integration.get_session_stats()
    else:
        return {
            "enabled": False,
            "llm_available": ENHANCED_LLM_AVAILABLE,
            "system_version": "not_initialized"
        }


# ИСПРАВЛЕННЫЕ модификации для интеграции с enhanced_server_with_periodontal.py

class FixedLLMProcessor:
    """
    ИСПРАВЛЕННЫЙ процессор с LLM поддержкой и либеральной детекцией
    """
    
    def __init__(self, base_processor, openai_api_key: str = None):
        self.base_processor = base_processor
        self.llm_integration = None
        
        if openai_api_key:
            self.llm_integration = FixedLLMPeriodontalIntegration(openai_api_key)
    
    async def process_with_fixed_llm_periodontal(self, client_id: str, text: str, confidence: float, duration: float):
        """
        ИСПРАВЛЕННАЯ ПРИОРИТЕТ 1: Enhanced LLM обработка с ОТЛАДКОЙ
        """
        try:
            logger.info(f"🔄 PROCESSING START for {client_id}: '{text}'")
            
            # Проверяем доступность LLM системы
            if not self.llm_integration or not self.llm_integration.enabled:
                logger.warning("🤖 LLM not available, falling back to standard processing")
                return await self.base_processor.process_with_enhanced_systems(client_id, text, confidence, duration)
            
            logger.info(f"🤖 LLM integration available and enabled")
            
            # ЛИБЕРАЛЬНАЯ проверка на periodontal команду
            is_periodontal = self.llm_integration.is_periodontal_command_liberal(text)
            logger.info(f"🔍 Liberal detection result: {is_periodontal}")
            
            if is_periodontal:
                logger.info(f"🤖 LIBERAL DETECTION triggered LLM for: '{text}'")
                
                llm_result = await self.llm_integration.process_transcription(text, confidence)
                
                if llm_result.get("success"):
                    # LLM успешно обработал команду
                    logger.info(f"🤖 LLM SUCCESS: {llm_result['message']}")
                    await self.broadcast_fixed_llm_periodontal_update(client_id, llm_result)
                    
                    # Обновляем статистику базового процессора
                    if hasattr(self.base_processor, 'stats'):
                        self.base_processor.stats['successful_commands'] += 1
                        self.base_processor.stats['commands_processed'] += 1
                        
                        # LLM статистика
                        self.base_processor.stats['llm_commands_processed'] = (
                            self.base_processor.stats.get('llm_commands_processed', 0) + 1
                        )
                        self.base_processor.stats['llm_successful_commands'] = (
                            self.base_processor.stats.get('llm_successful_commands', 0) + 1
                        )
                        
                        # Статистика ASR исправлений
                        original = llm_result.get("original_text", "").lower()
                        corrected = llm_result.get("corrected_text", "").lower()
                        if original != corrected:
                            self.base_processor.stats['llm_asr_errors_fixed'] = (
                                self.base_processor.stats.get('llm_asr_errors_fixed', 0) + 1
                            )
                    
                    return
                else:
                    # LLM не смог обработать даже при либеральной детекции
                    logger.warning(f"🤖 LLM FAILED even with liberal detection: {llm_result.get('message', 'Unknown error')}")
            else:
                # Даже либеральная детекция не активировала LLM
                logger.warning(f"🤖 LIBERAL DETECTION did not trigger for: '{text}'")
            
            # Fallback к стандартной обработке enhanced_server
            logger.info(f"🔄 Falling back to standard enhanced processing for: '{text}'")
            await self.base_processor.process_with_enhanced_systems(client_id, text, confidence, duration)
                
        except Exception as e:
            logger.error(f"❌ FIXED LLM processing error for {client_id}: {e}")
            # Fallback к стандартной обработке при ошибке
            await self.base_processor.process_with_enhanced_systems(client_id, text, confidence, duration)
    
    async def broadcast_fixed_llm_periodontal_update(self, client_id: str, llm_result: Dict):
        """ИСПРАВЛЕННАЯ отправка LLM periodontal обновлений веб-клиентам"""
        if not hasattr(self.base_processor, 'web_clients'):
            return
            
        web_clients = getattr(self.base_processor, 'web_clients', set())
        if not web_clients:
            return
        
        # Формат сообщения совместимый с существующим клиентом
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
            "system": "fixed_liberal_llm_periodontal"
        })
        
        disconnected = set()
        for client in web_clients.copy():
            try:
                await client.send(message)
            except Exception as e:
                logger.error(f"Error sending FIXED LLM periodontal update to web client: {e}")
                disconnected.add(client)
        
        for client in disconnected:
            web_clients.discard(client)


def enhance_processor_with_fixed_llm(base_processor, openai_api_key: str = None):
    """
    ИСПРАВЛЕННАЯ функция для добавления LLM поддержки к существующему процессору
    """
    if not ENHANCED_LLM_AVAILABLE or not openai_api_key:
        logger.warning("⚠️ Cannot enhance processor with FIXED LLM - requirements not met")
        return base_processor
    
    # Создаем ИСПРАВЛЕННЫЙ LLM расширенный процессор
    fixed_processor = FixedLLMProcessor(base_processor, openai_api_key)
    
    # Заменяем метод обработки на ИСПРАВЛЕННУЮ LLM версию
    if hasattr(base_processor, 'process_with_enhanced_systems'):
        base_processor.process_with_enhanced_systems = fixed_processor.process_with_fixed_llm_periodontal
    
    # Добавляем LLM статистику
    if hasattr(base_processor, 'stats'):
        base_processor.stats.update({
            'llm_commands_processed': 0,
            'llm_successful_commands': 0,
            'llm_asr_errors_fixed': 0,
            'llm_average_confidence': 0.0,
            'llm_liberal_detections': 0,
            'llm_enabled': True,
            'llm_detection_mode': 'liberal'
        })
    
    logger.info("🤖 Processor enhanced with FIXED LIBERAL LLM Periodontal functionality")
    return base_processor

def add_fixed_llm_stats_to_server_stats(base_stats: Dict) -> Dict:
    """ИСПРАВЛЕННАЯ добавление LLM статистики к статистике сервера"""
    if fixed_llm_integration and fixed_llm_integration.enabled:
        llm_stats = fixed_llm_integration.get_session_stats()
        base_stats.update({
            'llm_periodontal_available': True,
            'llm_commands_processed': llm_stats.get('llm_commands_processed', 0),
            'llm_successful_commands': llm_stats.get('llm_successful_commands', 0),
            'llm_asr_errors_fixed': llm_stats.get('asr_errors_fixed', 0),
            'llm_average_confidence': llm_stats.get('llm_average_confidence', 0.0),
            'llm_liberal_detections': llm_stats.get('liberal_detections', 0),
            'llm_model': llm_stats.get('model', 'unknown'),
            'llm_detection_mode': 'liberal',
            'llm_validation_enabled': True
        })
    else:
        base_stats.update({
            'llm_periodontal_available': False,
            'llm_enabled': False
        })
    
    return base_stats


if __name__ == "__main__":
    # Тестирование ИСПРАВЛЕННОЙ интеграции
    import os
    
    async def test_fixed_llm_integration():
        """Тестирование ИСПРАВЛЕННОЙ LLM интеграции с проблемными командами"""
        print("🤖 Testing FIXED LLM Integration with Liberal Detection")
        print("=" * 60)
        
        # Инициализация
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("❌ OPENAI_API_KEY not found")
            return
        
        success = initialize_fixed_llm_integration(api_key)
        if not success:
            print("❌ Failed to initialize FIXED LLM integration")
            return
        
        # Тестовые команды из ваших примеров
        test_commands = [
            ("Probing depth on tooth number two, wingle surface 231.", True),
            ("Probing depth on tooth number 2, buckle surface 312.", True),
            ("Bleeding on probing tooth 2, buccal distal.", True),
            ("Missing this one.", True),
            ("For Cache in class 2 on tooth 2", True),
            ("Tooth Tool has mobility grade 2", True),
            ("Separation present on tooth 8 lingual distal.", True),
            ("Bleeding on probing tooth 3, lingual distal.", True),
            # Дополнительные тесты
            ("hello how are you", False),
            ("what time is it", False)
        ]
        
        for cmd, should_detect in test_commands:
            print(f"\n📝 Testing: '{cmd}'")
            
            # Проверка ЛИБЕРАЛЬНОЙ детекции
            is_detected = is_periodontal_command_fixed_llm(cmd)
            detection_status = "✅ DETECTED" if is_detected else "❌ NOT DETECTED"
            expected_status = "✅ CORRECT" if is_detected == should_detect else "❌ WRONG"
            
            print(f"   🔍 Liberal detection: {detection_status} {expected_status}")
            
            if is_detected:
                result = await process_transcription_with_fixed_llm(cmd, 0.8, "test_patient")
                
                if result["success"]:
                    print(f"   🤖 LLM SUCCESS: {result['message']}")
                    print(f"   🔧 Original: '{result.get('original_text', 'N/A')}'")
                    print(f"   🔧 Corrected: '{result.get('corrected_text', 'N/A')}'")
                    print(f"   🦷 Tooth: {result.get('tooth_number')}")
                    print(f"   📋 Type: {result.get('measurement_type')}")
                    print(f"   📊 Values: {result.get('values')}")
                else:
                    print(f"   ❌ LLM FAILED: {result['message']}")
        
        # Статистика
        stats = get_fixed_llm_stats()
        print(f"\n📊 FIXED LLM Stats:")
        for key, value in stats.items():
            print(f"   {key}: {value}")
    
    # Запуск теста
    asyncio.run(test_fixed_llm_integration())
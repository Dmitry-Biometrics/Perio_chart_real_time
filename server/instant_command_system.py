#!/usr/bin/env python3
"""
INSTANT COMMAND COMPLETION SYSTEM
Мгновенное выполнение команд с предиктивным анализом завершенности
"""

import re
import json
import asyncio
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import time
import numpy as np

# ДОБАВЬТЕ этот импорт если используется SpeechState из сегментации:
try:
    from fixed_segmentation_no_duplication import SpeechState
    SEGMENTATION_AVAILABLE = True
except ImportError:
    # Если модуль сегментации недоступен, создаем минимальную заглушку
    class SpeechState(Enum):
        SILENCE = "silence"
        SPEECH_DETECTION = "speech_detection" 
        SPEECH_ACTIVE = "speech_active"
        SPEECH_ENDING = "speech_ending"
    SEGMENTATION_AVAILABLE = False

logger = logging.getLogger(__name__)

class CommandCompleteness(Enum):
    """Состояние завершенности команды"""
    INCOMPLETE = "incomplete"          # Команда не завершена
    COMPLETE = "complete"             # Команда завершена - выполнить немедленно
    PARTIAL_MATCH = "partial_match"   # Частичное совпадение
    INVALID = "invalid"               # Неверная команда

@dataclass
class CommandPattern:
    """Паттерн команды с правилами завершенности"""
    pattern: str
    required_groups: List[str]
    completion_rules: Dict[str, any]
    command_type: str
    priority: int = 0

class InstantCommandAnalyzer:
    """Анализатор мгновенного завершения команд"""
    
    def __init__(self):
        self.command_patterns = self._initialize_command_patterns()
        self.partial_commands = {}  # Хранение частичных команд по client_id
        
    def _initialize_command_patterns(self) -> List[CommandPattern]:
        """Инициализация паттернов команд с правилами завершенности"""
        
        return [
            # 1. PROBING DEPTH - требует 3 числа после surface
            CommandPattern(
                pattern=r'probing\s+depth\s+on\s+tooth\s+(?:number\s+)?(\w+)[,]?\s+(buccal|lingual)\s+surface\s+(\w+)[,]?\s+(\w+)[,]?\s+(\w+)[.]?',
                required_groups=['tooth', 'surface', 'depth1', 'depth2', 'depth3'],
                completion_rules={
                    'min_required_groups': 5,  # Все 5 групп обязательны
                    'validation': {
                        'tooth': lambda x: 1 <= int(x) <= 32,
                        'surface': lambda x: x.lower() in ['buccal', 'lingual'],
                        'depth1': lambda x: 1 <= int(x) <= 12,
                        'depth2': lambda x: 1 <= int(x) <= 12,
                        'depth3': lambda x: 1 <= int(x) <= 12,
                    }
                },
                command_type='probing_depth',
                priority=1
            ),
            
            # 2. MOBILITY - требует grade + число
            CommandPattern(
                pattern=r'tooth\s+(\w+)\s+has\s+mobility(?:\s+grade\s+(\w+))?',
                required_groups=['tooth', 'grade'],
                completion_rules={
                    'min_required_groups': 2,
                    'validation': {
                        'tooth': lambda x: 1 <= int(x) <= 32,
                        'grade': lambda x: 0 <= int(x) <= 3,
                    }
                },
                command_type='mobility',
                priority=1
            ),
            
            # 3. BLEEDING ON PROBING - требует tooth + surface + position
            CommandPattern(
                pattern=r'bleeding\s+on\s+probing\s+tooth\s+(\w+)\s+(buccal|lingual)\s+(distal|mesial|mid|middle)',
                required_groups=['tooth', 'surface', 'position'],
                completion_rules={
                    'min_required_groups': 3,
                    'validation': {
                        'tooth': lambda x: 1 <= int(x) <= 32,
                        'surface': lambda x: x.lower() in ['buccal', 'lingual'],
                        'position': lambda x: x.lower() in ['distal', 'mesial', 'mid', 'middle'],
                    }
                },
                command_type='bleeding_on_probing',
                priority=1
            ),
            
            # 4. SUPPURATION - требует tooth + surface + position
            CommandPattern(
                pattern=r'suppuration\s+present\s+on\s+tooth\s+(\w+)\s+(buccal|lingual)\s+(distal|mesial|mid|middle)',
                required_groups=['tooth', 'surface', 'position'],
                completion_rules={
                    'min_required_groups': 3,
                    'validation': {
                        'tooth': lambda x: 1 <= int(x) <= 32,
                        'surface': lambda x: x.lower() in ['buccal', 'lingual'],
                        'position': lambda x: x.lower() in ['distal', 'mesial', 'mid', 'middle'],
                    }
                },
                command_type='suppuration',
                priority=1
            ),
            
            # 5. FURCATION - требует class + tooth
            CommandPattern(
                pattern=r'furcation\s+class\s+(\w+)\s+on\s+tooth\s+(\w+)',
                required_groups=['class', 'tooth'],
                completion_rules={
                    'min_required_groups': 2,
                    'validation': {
                        'class': lambda x: 1 <= int(x) <= 3,
                        'tooth': lambda x: 1 <= int(x) <= 32,
                    }
                },
                command_type='furcation',
                priority=1
            ),
            
            # 6. GINGIVAL MARGIN - требует tooth + 3 значения (с знаками)
            CommandPattern(
                pattern=r'gingival\s+margin\s+on\s+tooth\s+(\d+)(?:\s+((?:minus\s+\d+|plus\s+\d+|\d+)(?:\s+(?:minus\s+\d+|plus\s+\d+|\d+)){0,2}))?',
                required_groups=['tooth', 'values'],
                completion_rules={
                    'min_required_groups': 2,
                    'custom_validation': 'gingival_margin_values',
                    'validation': {
                        'tooth': lambda x: 1 <= int(x) <= 32,
                    }
                },
                command_type='gingival_margin',
                priority=1
            ),
  
            # 7. MISSING TEETH - УЛУЧШЕННЫЙ паттерн для распознавания слов-чисел
            CommandPattern(
                pattern=r'missing\s+teeth?\s+((?:\w+[\s,]*)+)',  # Более гибкий паттерн
                required_groups=['teeth'],
                completion_rules={
                    'min_required_groups': 1,
                    'custom_validation': 'missing_teeth_list_enhanced',
                },
                command_type='missing_teeth',
                priority=1
            ),
        ]

    def _validate_missing_teeth_list_enhanced(self, teeth_text: str) -> bool:
        """УЛУЧШЕННАЯ валидация списка отсутствующих зубов с поддержкой слов"""
        if not teeth_text:
            return False
        
        # Словарь для конвертации слов в числа
        word_to_num = {
            'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
            'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
            'eleven': 11, 'twelve': 12, 'thirteen': 13, 'fourteen': 14,
            'fifteen': 15, 'sixteen': 16, 'seventeen': 17, 'eighteen': 18,
            'nineteen': 19, 'twenty': 20, 'twenty-one': 21, 'twenty-two': 22,
            'twenty-three': 23, 'twenty-four': 24, 'twenty-five': 25,
            'twenty-six': 26, 'twenty-seven': 27, 'twenty-eight': 28,
            'twenty-nine': 29, 'thirty': 30, 'thirty-one': 31, 'thirty-two': 32
        }
        
        try:
            teeth = []
            words = teeth_text.lower().strip().split()
            
            for word in words:
                clean_word = word.strip('.,!?;:')
                
                # Проверяем цифры
                if clean_word.isdigit():
                    tooth_num = int(clean_word)
                    if 1 <= tooth_num <= 32:
                        teeth.append(tooth_num)
                # Проверяем словесные числа
                elif clean_word in word_to_num:
                    tooth_num = word_to_num[clean_word]
                    if 1 <= tooth_num <= 32:
                        teeth.append(tooth_num)
            
            return len(teeth) > 0
            
        except Exception as e:
            print(f"❌ Missing teeth validation error: {e}")
            return False
    
    def analyze_command_completeness(self, text: str, client_id: str) -> Tuple[CommandCompleteness, Optional[Dict]]:
        """
        Анализ завершенности команды - КЛЮЧЕВАЯ ФУНКЦИЯ
        """
        text_clean = text.lower().strip()
        
        print(f"🔍 ANALYZING COMMAND COMPLETENESS: '{text_clean}'")
        
        # ДОБАВИТЬ: Проверка на частичные команды СНАЧАЛА
        partial_patterns = [
            r'probing\s+depth\s+.*?tooth\s+(?:number\s+)?(\d+)\s+.*(buccal|lingual)(?:\s+surface)?$',
            r'tooth\s+(\d+).*?mobility(?:\s+grade)?$',
            r'bleeding\s+on\s+probing\s+tooth\s+(\d+)(?:\s+(buccal|lingual))?$'
        ]
        
        for pattern in partial_patterns:
            if re.search(pattern, text_clean):
                print(f"⏳ PARTIAL COMMAND DETECTED")
                return CommandCompleteness.INCOMPLETE, None
        
        # Проверяем каждый паттерн
        for pattern_obj in self.command_patterns:
            match = re.search(pattern_obj.pattern, text_clean, re.IGNORECASE)
            
            if match:
                print(f"✅ PATTERN MATCHED: {pattern_obj.command_type}")
                print(f"📊 GROUPS: {match.groups()}")
                
                completeness, command_data = self._analyze_pattern_completeness(
                    pattern_obj, match, text_clean
                )
                
                print(f"🎯 COMPLETENESS: {completeness.value}")
                
                if completeness == CommandCompleteness.COMPLETE:
                    print(f"🚀 COMMAND COMPLETE - IMMEDIATE EXECUTION!")
                    return completeness, command_data
                elif completeness == CommandCompleteness.INCOMPLETE:
                    print(f"⏳ COMMAND INCOMPLETE - WAITING FOR MORE...")
                    # Сохраняем частичную команду
                    self.partial_commands[client_id] = {
                        'pattern': pattern_obj,
                        'current_text': text_clean,
                        'timestamp': time.time()
                    }
                    return completeness, command_data
        
        print(f"❌ NO PATTERN MATCHED")
        return CommandCompleteness.INVALID, None
    
    def _analyze_pattern_completeness(self, pattern_obj: CommandPattern, match, text: str) -> Tuple[CommandCompleteness, Optional[Dict]]:
        """Анализ завершенности конкретного паттерна"""
        
        groups = match.groups()
        non_empty_groups = [g for g in groups if g is not None and g.strip()]
        
        print(f"📋 Required groups: {pattern_obj.completion_rules['min_required_groups']}")
        print(f"📊 Found groups: {len(non_empty_groups)} - {non_empty_groups}")
        
        # Проверяем минимальное количество групп
        min_required = pattern_obj.completion_rules['min_required_groups']
        
        if len(non_empty_groups) < min_required:
            print(f"⏳ INCOMPLETE: Need {min_required}, got {len(non_empty_groups)}")
            return CommandCompleteness.INCOMPLETE, None
        
        # Валидация значений
        try:
            if not self._validate_command_values(pattern_obj, groups, text):
                print(f"❌ INVALID: Validation failed")
                return CommandCompleteness.INVALID, None
        except Exception as e:
            print(f"❌ VALIDATION ERROR: {e}")
            return CommandCompleteness.INVALID, None
        
        # Команда завершена и валидна
        command_data = self._extract_command_data(pattern_obj, groups, text)
        print(f"✅ COMPLETE: Command data extracted")
        
        return CommandCompleteness.COMPLETE, command_data
    
    def _validate_command_values(self, pattern_obj: CommandPattern, groups, text: str) -> bool:
        """Валидация значений команды с поддержкой слов-чисел"""
   
        # ✅ ТОТЖЕ ИСПРАВЛЕННЫЙ словарь
        word_to_num = {
            'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
            'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
            'eleven': 11, 'twelve': 12, 'thirteen': 13, 'fourteen': 14,
            'fifteen': 15, 'sixteen': 16, 'seventeen': 17, 'eighteen': 18,
            'nineteen': 19, 'twenty': 20, 'twenty-one': 21, 'twenty-two': 22,
            'twenty-three': 23, 'twenty-four': 24, 'twenty-five': 25,
            'twenty-six': 26, 'twenty-seven': 27, 'twenty-eight': 28,
            'twenty-nine': 29, 'thirty': 30, 'thirty-one': 31, 'thirty-two': 32,
            
            # ✅ КРИТИЧЕСКОЕ ДОБАВЛЕНИЕ: ASR ошибки
            'too': 2, 'to': 2, 'for': 4, 'ate': 8, 'won': 1, 'tree': 3, 'sex': 6, 'free': 3
        }

        # ✅ ИСПРАВЛЕННАЯ функция безопасной конвертации
        def safe_convert_to_int(value):
            """Безопасно конвертирует строку или слово в число"""
            if not value:
                return None
            
            value_clean = str(value).strip().lower()
            
            # Убираем знаки препинания
            value_clean = value_clean.rstrip('.,!?;:')
            
            # Сначала проверяем цифры
            if value_clean.isdigit():
                return int(value_clean)
            
            # Затем проверяем слова-числа (включая ASR ошибки)
            if value_clean in word_to_num:
                return word_to_num[value_clean]
            
            # Если не удалось конвертировать
            print(f"⚠️ Warning: Could not convert '{value}' to number")
            return None
       
        completion_rules = pattern_obj.completion_rules
       
        # Стандартная валидация с конвертацией
        if 'validation' in completion_rules:
            validation_rules = completion_rules['validation']
            required_groups = pattern_obj.required_groups
           
            for i, group_name in enumerate(required_groups):
                if i < len(groups) and groups[i] is not None:
                    original_value = groups[i].strip()
                   
                    # ✅ ИСПРАВЛЕНИЕ: Конвертируем перед валидацией
                    if group_name in ['tooth', 'depth1', 'depth2', 'depth3', 'grade', 'class']:
                        converted_value = safe_convert_to_int(original_value)
                        if converted_value is None:
                            print(f"❌ Failed to convert '{original_value}' to number for {group_name}")
                            return False
                        # Используем конвертированное значение для валидации
                        validation_value = converted_value
                    else:
                        validation_value = original_value
                   
                    if group_name in validation_rules:
                        try:
                            if not validation_rules[group_name](validation_value):
                                print(f"❌ Validation failed for {group_name}: {validation_value}")
                                return False
                        except Exception as e:
                            print(f"❌ Validation error for {group_name}: {e}")
                            return False
       
        # Кастомная валидация
        if 'custom_validation' in completion_rules:
            custom_type = completion_rules['custom_validation']
           
            if custom_type == 'gingival_margin_values':
                return self._validate_gingival_margin_values(groups[1] if len(groups) > 1 else None)
            elif custom_type == 'missing_teeth_list':
                return self._validate_missing_teeth_list(groups[0] if len(groups) > 0 else None)
       
        return True
    
    def _validate_gingival_margin_values(self, values_text: str) -> bool:
        """Валидация значений gingival margin"""
        if not values_text:
            return False
            
        # Проверяем что есть 3 значения (с учетом знаков)
        values = self._parse_gingival_margin_values(values_text)
        return len(values) == 3
    
    def _parse_gingival_margin_values(self, text: str) -> List[int]:
        """Парсинг значений gingival margin"""
        values = []
        parts = text.strip().split()
        
        i = 0
        while i < len(parts) and len(values) < 3:
            if parts[i] == 'minus' and i + 1 < len(parts):
                try:
                    values.append(-int(parts[i + 1]))
                    i += 2
                except:
                    i += 1
            elif parts[i] == 'plus' and i + 1 < len(parts):
                try:
                    values.append(int(parts[i + 1]))
                    i += 2
                except:
                    i += 1
            else:
                try:
                    values.append(int(parts[i]))
                    i += 1
                except:
                    i += 1
        
        return values
    
    def _validate_missing_teeth_list(self, teeth_text: str) -> bool:
        """Валидация списка отсутствующих зубов"""
        if not teeth_text:
            return False
            
        try:
            teeth = [int(x.strip()) for x in teeth_text.split() if x.strip().isdigit()]
            return len(teeth) > 0 and all(1 <= t <= 32 for t in teeth)
        except:
            return False
    
    def _extract_command_data(self, pattern_obj: CommandPattern, groups, text: str) -> Dict:
        """Извлечение данных команды с поддержкой слов-чисел"""
        
        word_to_num = {
            'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
            'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
            'eleven': 11, 'twelve': 12, 'thirteen': 13, 'fourteen': 14,
            'fifteen': 15, 'sixteen': 16, 'seventeen': 17, 'eighteen': 18,
            'nineteen': 19, 'twenty': 20, 'twenty-one': 21, 'twenty-two': 22,
            'twenty-three': 23, 'twenty-four': 24, 'twenty-five': 25,
            'twenty-six': 26, 'twenty-seven': 27, 'twenty-eight': 28,
            'twenty-nine': 29, 'thirty': 30, 'thirty-one': 31, 'thirty-two': 32,
            
            # ✅ КРИТИЧЕСКОЕ ДОБАВЛЕНИЕ: ASR ошибки
            'too': 2,    # "too" часто распознается вместо "two"
            'to': 2,     # "to" тоже может быть "two"
            'for': 4,    # "for" часто распознается вместо "four"
            'ate': 8,    # "ate" может быть "eight"
            'won': 1,    # "won" может быть "one"
            'tree': 3,   # "tree" может быть "three"
            'sex': 6,    # "sex" может быть "six"
            'free': 3,   # "free" может быть "three"
        }
        
        # ✅ ИСПРАВЛЕННАЯ функция безопасной конвертации
        def safe_convert_to_int(value):
            """Безопасно конвертирует строку или слово в число"""
            if not value:
                return 0
            
            value_clean = str(value).strip().lower()
            
            # Убираем знаки препинания
            value_clean = value_clean.rstrip('.,!?;:')
            
            # Сначала проверяем цифры
            if value_clean.isdigit():
                return int(value_clean)
            
            # Затем проверяем слова-числа (включая ASR ошибки)
            if value_clean in word_to_num:
                converted = word_to_num[value_clean]
                print(f"✅ Converted '{value}' → {converted}")
                return converted
            
            # Если не удалось конвертировать
            print(f"⚠️ Warning: Could not convert '{value}' to number, using 0")
            return 0
        
        command_data = {
            'type': pattern_obj.command_type,
            'raw_text': text,
            'timestamp': time.time()
        }
        
        if pattern_obj.command_type == 'probing_depth':
            # ✅ ИСПРАВЛЕНИЕ: Используем безопасную конвертацию
            tooth_num = safe_convert_to_int(groups[0]) if len(groups) > 0 else 0
            surface = groups[1].lower() if len(groups) > 1 else 'buccal'
            depth1 = safe_convert_to_int(groups[2]) if len(groups) > 2 else 0
            depth2 = safe_convert_to_int(groups[3]) if len(groups) > 3 else 0
            depth3 = safe_convert_to_int(groups[4]) if len(groups) > 4 else 0
            
            command_data.update({
                'tooth': tooth_num,
                'surface': surface,
                'values': [depth1, depth2, depth3]
            })
            
            print(f"✅ Probing depth extracted: tooth={tooth_num}, surface={surface}, values=[{depth1}, {depth2}, {depth3}]")
            
        elif pattern_obj.command_type == 'mobility':
            # ✅ ИСПРАВЛЕНИЕ: То же самое для mobility
            tooth_num = safe_convert_to_int(groups[0]) if len(groups) > 0 else 0
            grade = safe_convert_to_int(groups[1]) if len(groups) > 1 else 0
            
            command_data.update({
                'tooth': tooth_num,
                'grade': grade
            })
            
            print(f"✅ Mobility extracted: tooth={tooth_num}, grade={grade}")
            
        elif pattern_obj.command_type == 'bleeding_on_probing':
            # ✅ ИСПРАВЛЕНИЕ: Для bleeding
            tooth_num = safe_convert_to_int(groups[0]) if len(groups) > 0 else 0
            surface = groups[1].lower() if len(groups) > 1 else 'buccal'
            position = groups[2].lower() if len(groups) > 2 else 'distal'
            
            command_data.update({
                'tooth': tooth_num,
                'surface': surface,
                'position': position
            })
            
            print(f"✅ Bleeding extracted: tooth={tooth_num}, surface={surface}, position={position}")
            
        elif pattern_obj.command_type == 'suppuration':
            # ✅ ИСПРАВЛЕНИЕ: Для suppuration
            tooth_num = safe_convert_to_int(groups[0]) if len(groups) > 0 else 0
            surface = groups[1].lower() if len(groups) > 1 else 'buccal'
            position = groups[2].lower() if len(groups) > 2 else 'distal'
            
            command_data.update({
                'tooth': tooth_num,
                'surface': surface,
                'position': position
            })
            
        elif pattern_obj.command_type == 'furcation':
            # ✅ ИСПРАВЛЕНИЕ: Для furcation
            furcation_class = safe_convert_to_int(groups[0]) if len(groups) > 0 else 0
            tooth_num = safe_convert_to_int(groups[1]) if len(groups) > 1 else 0
            
            command_data.update({
                'class': furcation_class,
                'tooth': tooth_num
            })
            
        elif pattern_obj.command_type == 'gingival_margin':
            # ✅ ИСПРАВЛЕНИЕ: Для gingival margin (особый случай со знаками)
            tooth_num = safe_convert_to_int(groups[0]) if len(groups) > 0 else 0
            values = self._parse_gingival_margin_values(groups[1]) if len(groups) > 1 and groups[1] else []
            
            command_data.update({
                'tooth': tooth_num,
                'values': values
            })
            
        elif pattern_obj.command_type == 'missing_teeth':
            # ✅ ИСПРАВЛЕНИЕ: Для missing teeth
            if len(groups) > 0 and groups[0]:
                # Парсим список зубов
                teeth_text = groups[0]
                teeth = []
                for word in teeth_text.split():
                    tooth_num = safe_convert_to_int(word)
                    if tooth_num is not None and 1 <= tooth_num <= 32: 
                        teeth.append(tooth_num)
                
                command_data.update({
                    'teeth': teeth
                })
            else:
                command_data.update({
                    'teeth': []
                })
        
        return command_data

class InstantCommandProcessor:
    """Процессор мгновенного выполнения команд"""
    
    def __init__(self, web_clients_ref):
        self.analyzer = InstantCommandAnalyzer()
        self.web_clients = web_clients_ref
        
    async def process_instant_command(self, client_id: str, text: str) -> bool:
        """
        Обработка команды на мгновенное выполнение
        Возвращает True если команда была выполнена мгновенно
        """
        
        completeness, command_data = self.analyzer.analyze_command_completeness(text, client_id)
        
        if completeness == CommandCompleteness.COMPLETE:
            print(f"🚀 INSTANT EXECUTION for client {client_id}")
            print(f"📋 Command data: {command_data}")
            
            # Мгновенная отправка результата
            await self._send_instant_result(client_id, command_data)
            return True
            
        elif completeness == CommandCompleteness.INCOMPLETE:
            print(f"⏳ WAITING FOR COMPLETION for client {client_id}")
            # Можно отправить промежуточную обратную связь
            await self._send_partial_feedback(client_id, text)
            return False
            
        return False
    
    async def _send_instant_result(self, client_id: str, command_data: Dict):
        """Мгновенная отправка результата"""
        
        # Формируем periodontal_update сообщение
        message = self._format_periodontal_message(client_id, command_data)
        
        # Отправляем всем веб-клиентам
        if self.web_clients:
            message_json = json.dumps(message)
            disconnected = set()
            
            for client in list(self.web_clients):
                try:
                    await asyncio.wait_for(client.send(message_json), timeout=1.0)
                    print(f"✅ INSTANT RESULT sent to web client")
                except:
                    disconnected.add(client)
            
            for client in disconnected:
                self.web_clients.discard(client)
    
    def _format_periodontal_message(self, client_id: str, command_data: Dict) -> Dict:
        """Форматирование сообщения для periodontal chart"""
        
        base_message = {
            "type": "periodontal_update",
            "client_id": client_id,
            "success": True,
            "timestamp": command_data['timestamp'],
            "instant_execution": True,
            "system": "instant_command_completion_v1"
        }
        
        if command_data['type'] == 'probing_depth':
            base_message.update({
                "tooth_number": command_data['tooth'],
                "measurement_type": "probing_depth",
                "surface": command_data['surface'],
                "values": command_data['values'],
                "measurements": {"probing_depth": command_data['values']},
                "message": f"✅ Probing depths tooth {command_data['tooth']} {command_data['surface']}: {'-'.join(map(str, command_data['values']))}mm"
            })
            
        elif command_data['type'] == 'mobility':
            base_message.update({
                "tooth_number": command_data['tooth'],
                "measurement_type": "mobility",
                "values": [command_data['grade']],
                "measurements": {"mobility": command_data['grade']},
                "message": f"✅ Tooth {command_data['tooth']} mobility: Grade {command_data['grade']}"
            })
            
        elif command_data['type'] == 'bleeding_on_probing':
            base_message.update({
                "tooth_number": command_data['tooth'],
                "measurement_type": "bleeding",
                "surface": command_data['surface'],
                "position": command_data['position'],
                "values": [True],
                "measurements": {"bleeding": [True]},
                "message": f"✅ Bleeding on probing tooth {command_data['tooth']} {command_data['surface']} {command_data['position']}"
            })
            
        elif command_data['type'] == 'suppuration':
            base_message.update({
                "tooth_number": command_data['tooth'],
                "measurement_type": "suppuration",
                "surface": command_data['surface'],
                "position": command_data['position'],
                "values": [True],
                "measurements": {"suppuration": [True]},
                "message": f"✅ Suppuration tooth {command_data['tooth']} {command_data['surface']} {command_data['position']}"
            })
            
        elif command_data['type'] == 'furcation':
            base_message.update({
                "tooth_number": command_data['tooth'],
                "measurement_type": "furcation",
                "values": [command_data['class']],
                "measurements": {"furcation": command_data['class']},
                "message": f"✅ Tooth {command_data['tooth']} furcation: Class {command_data['class']}"
            })
            
        elif command_data['type'] == 'gingival_margin':
            base_message.update({
                "tooth_number": command_data['tooth'],
                "measurement_type": "gingival_margin",
                "values": command_data['values'],
                "measurements": {"gingival_margin": command_data['values']},
                "message": f"✅ Gingival margin tooth {command_data['tooth']}: {' '.join(map(str, command_data['values']))}mm"
            })
            
        elif command_data['type'] == 'missing_teeth':
            # Для missing teeth отправляем отдельное сообщение для каждого зуба
            # Здесь возвращаем первый зуб, остальные обработаем отдельно
            tooth = command_data['teeth'][0] if command_data['teeth'] else 1
            base_message.update({
                "tooth_number": tooth,
                "measurement_type": "missing_teeth",
                "values": command_data['teeth'],
                "measurements": {"missing_teeth": command_data['teeth']},
                "message": f"✅ Missing teeth: {', '.join(map(str, command_data['teeth']))}"
            })
        
        return base_message
    
    async def _send_partial_feedback(self, client_id: str, partial_text: str):
        """Отправка промежуточной обратной связи"""
        if self.web_clients:
            feedback_message = {
                "type": "partial_command_feedback",
                "client_id": client_id,
                "partial_text": partial_text,
                "status": "waiting_for_completion",
                "timestamp": time.time()
            }
            
            message_json = json.dumps(feedback_message)
            disconnected = set()
            
            for client in list(self.web_clients):
                try:
                    await asyncio.wait_for(client.send(message_json), timeout=1.0)
                except:
                    disconnected.add(client)
            
            for client in disconnected:
                self.web_clients.discard(client)

# Интеграция с основным процессором
def integrate_instant_command_system(base_processor, web_clients_ref):
    """Интеграция системы мгновенного выполнения команд"""
    
    instant_processor = InstantCommandProcessor(web_clients_ref)
    
    # Добавляем instant processor к базовому процессору
    base_processor.instant_processor = instant_processor
    
    # Модифицируем метод обработки аудио чанков
    original_process_chunk = base_processor.process_audio_chunk
    
    def enhanced_process_chunk(client_id, audio_chunk):
        # Сначала обычная обработка
        result = original_process_chunk(client_id, audio_chunk)
        
        # Если есть результат транскрипции, проверяем на мгновенное выполнение
        if result and isinstance(result, str) and result.strip():
            # Запускаем асинхронную проверку мгновенного выполнения
            asyncio.create_task(
                instant_processor.process_instant_command(client_id, result)
            )
        
        return result
    
    base_processor.process_audio_chunk = enhanced_process_chunk
    
    print("🚀 INSTANT COMMAND SYSTEM INTEGRATED")
    return base_processor

if __name__ == "__main__":
    # Тестирование системы
    import asyncio
    
    async def test_instant_commands():
        """Тестирование системы мгновенного выполнения"""
        
        print("🧪 Testing Instant Command Completion System")
        print("=" * 60)
        
        analyzer = InstantCommandAnalyzer()
        
        test_commands = [
            # Полные команды (должны выполниться мгновенно)
            ("probing depth on tooth number 14 buccal surface 3 2 4", CommandCompleteness.COMPLETE),
            ("tooth 8 has mobility grade 2", CommandCompleteness.COMPLETE),
            ("bleeding on probing tooth 12 buccal distal", CommandCompleteness.COMPLETE),
            ("suppuration present on tooth 8 lingual mesial", CommandCompleteness.COMPLETE),
            ("furcation class 2 on tooth 6", CommandCompleteness.COMPLETE),
            ("gingival margin on tooth 14 minus 1 0 plus 1", CommandCompleteness.COMPLETE),
            ("missing teeth 1 16 17 32", CommandCompleteness.COMPLETE),
            
            # Неполные команды (должны ждать завершения)
            ("probing depth on tooth number 14 buccal surface", CommandCompleteness.INCOMPLETE),
            ("tooth 8 has mobility", CommandCompleteness.INCOMPLETE),
            ("bleeding on probing tooth 12", CommandCompleteness.INCOMPLETE),
            ("suppuration present on tooth 8", CommandCompleteness.INCOMPLETE),
            ("gingival margin on tooth 14", CommandCompleteness.INCOMPLETE),
        ]
        
        for i, (command, expected) in enumerate(test_commands, 1):
            print(f"\n{i}. Testing: '{command}'")
            print("-" * 40)
            
            completeness, command_data = analyzer.analyze_command_completeness(command, "test_client")
            
            status = "✅ PASS" if completeness == expected else "❌ FAIL"
            print(f"   Expected: {expected.value}")
            print(f"   Got: {completeness.value}")
            print(f"   Result: {status}")
            
            if command_data:
                print(f"   Data: {command_data}")
        
        print(f"\n🎯 Test completed!")
    
    asyncio.run(test_instant_commands())

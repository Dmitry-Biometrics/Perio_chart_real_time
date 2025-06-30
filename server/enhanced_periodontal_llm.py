#!/usr/bin/env python3
"""
ИСПРАВЛЕННЫЙ Enhanced Periodontal Voice Command Processor with OpenAI LLM v1.0+
КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Правильная нумерация зубов (American Universal System 1-32)
Исправляет проблему когда "tooth one" становится "tooth 21" или "tooth 2"
"""

import logging
import json
import asyncio
import re
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime

# Новый импорт для OpenAI v1.0+
try:
    from openai import AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logging.warning("OpenAI library not available")

logger = logging.getLogger(__name__)


def parse_gingival_margin_command_fixed(text: str):
    """
    ИСПРАВЛЕННЫЙ парсер для команд gingival margin
    Правильно парсит: "tooth two one two three" → tooth=2, values=[1,2,3]
    """
    
    if 'gingival margin' not in text.lower():
        return None
    
    print(f"🦷 FIXED GM PARSING: '{text}'")
    
    # Словарь конвертации
    word_to_num = {
        'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
        'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
        'eleven': 11, 'twelve': 12, 'thirteen': 13, 'fourteen': 14,
        'fifteen': 15, 'sixteen': 16, 'seventeen': 17, 'eighteen': 18,
        'nineteen': 19, 'twenty': 20, 'thirty': 30, 'thirty-one': 31, 'thirty-two': 32
    }
    
    import re
    
    # Паттерн: "gingival margin on tooth [TOOTH] [VALUE1] [VALUE2] [VALUE3]"
    pattern = r'gingival\s+margin\s+on\s+tooth\s+(?:number\s+)?(\w+)\s+(.+)'
    match = re.search(pattern, text.lower())
    
    if not match:
        return None
    
    tooth_ref = match.group(1)
    values_text = match.group(2)
    
    # Конвертируем tooth number
    if tooth_ref.isdigit():
        tooth_number = int(tooth_ref)
    elif tooth_ref in word_to_num:
        tooth_number = word_to_num[tooth_ref]
    else:
        return None
    
    # Парсим значения
    tokens = values_text.split()
    values = []
    i = 0
    
    while i < len(tokens) and len(values) < 3:
        token = tokens[i].strip('.,!?;:')
        
        if token in ['minus', 'plus'] and i + 1 < len(tokens):
            next_token = tokens[i + 1].strip('.,!?;:')
            
            if next_token.isdigit():
                num = int(next_token)
            elif next_token in word_to_num:
                num = word_to_num[next_token]
            else:
                i += 1
                continue
            
            values.append(-num if token == 'minus' else num)
            i += 2
            
        elif token.isdigit():
            values.append(int(token))
            i += 1
            
        elif token in word_to_num:
            values.append(word_to_num[token])
            i += 1
            
        else:
            i += 1
    
    # Дополняем до 3 значений
    while len(values) < 3:
        values.append(0)
    
    values = values[:3]  # Берем только первые 3
    
    if not (1 <= tooth_number <= 32):
        return None
    
    print(f"✅ FIXED GM PARSED: tooth={tooth_number}, values={values}")
    
    return {
        'tooth_number': tooth_number,
        'command_type': 'gingival_margin',
        'values': values,
        'success': True
    }
    
@dataclass
class PeriodontalCommand:
    """Структура для periodontal команды"""
    tooth_number: int
    command_type: str  # probing_depth, bleeding, suppuration, etc.
    surface: Optional[str] = None  # buccal, lingual, both
    position: Optional[str] = None  # distal, mid, mesial
    values: List[Union[int, float, bool]] = None
    confidence: float = 0.0
    original_text: str = ""
    corrected_text: str = ""
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()
        if self.values is None:
            self.values = []

class EnhancedPeriodontalLLMProcessor:
    """ИСПРАВЛЕННЫЙ процессор с правильной нумерацией зубов (American Universal System)"""
    
    def __init__(self, openai_api_key: str, model: str = "gpt-3.5-turbo"):
        self.openai_api_key = openai_api_key
        self.model = model
        
        # ИСПРАВЛЕНО: Новый API для OpenAI v1.0+
        if OPENAI_AVAILABLE:
            self.client = AsyncOpenAI(api_key=openai_api_key)
        else:
            self.client = None
        
        # Инициализация системы
        self.setup_correction_patterns()
        self.setup_llm_prompts()
        self.chart_data = {}
        
        logger.info(f"🤖 FIXED Enhanced Periodontal LLM Processor initialized with {model} (American Universal System)")
    
    def setup_correction_patterns(self):
        """ИСПРАВЛЕННЫЕ паттерны коррекции - точная нумерация зубов"""
        
        # ИСПРАВЛЕННЫЕ ASR ошибки - БЕЗ конфликтующих замен
        self.asr_corrections = {
            # Probing depth
            "rubbing depth": "probing depth",
            "probing death": "probing depth", 
            "robin depth": "probing depth",
            "robin dats": "probing depth",
            "probing depths": "probing depth",
            "probe in depth": "probing depth",
            "probing that": "probing depth",
            
            # Surfaces
            "buckle": "buccal",
            "buckle surface": "buccal surface",
            "buckeal": "buccal",
            "becal": "buccal",
            "wingle": "lingual",
            "lingle": "lingual", 
            "lingwal": "lingual",
            "linguall": "lingual",
            
            # Commands
            "bleeding on": "bleeding on",
            "bleeding": "bleeding",
            "suppuration": "suppuration",
            "separation": "suppuration",  # ИСПРАВЛЕНИЕ
            "mobility": "mobility",
            "grade": "grade",
            "class": "class",
            "furcation": "furcation",
            "cache": "furcation",  # ИСПРАВЛЕНИЕ
            "gingival": "gingival",
            "margin": "margin",
            "missing": "missing",
            "teeth": "teeth",
            
            # Measurements
            "mm": "",
            "millimeter": "",
            "millimeters": "",
            "point": ".",
        }
        
        # ИСПРАВЛЕНО: Убраны проблемные замены чисел
        self.dental_context_corrections = {
            r'\bcache\b': 'furcation', 
            r'\bseparation\b': 'suppuration',
            r'\brubbing\b': 'probing',
            r'\brobin\b': 'probing',
            r'\bbuckle\b': 'buccal',
            r'\bwingle\b': 'lingual',
        }
        
        # КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Точная карта номеров зубов (American Universal System)
        self.tooth_number_corrections = {
            "zero": "0",
            "one": "1",
            "two": "2", 
            "three": "3",
            "four": "4",
            "five": "5",
            "six": "6",
            "seven": "7",
            "eight": "8",
            "nine": "9",
            "ten": "10",
            "eleven": "11",
            "twelve": "12",
            "thirteen": "13",
            "fourteen": "14",
            "fifteen": "15",
            "sixteen": "16",
            "seventeen": "17",
            "eighteen": "18",
            "nineteen": "19",
            "twenty": "20",
            "twenty-one": "21",
            "twenty-two": "22",
            "twenty-three": "23",
            "twenty-four": "24",
            "twenty-five": "25",
            "twenty-six": "26",
            "twenty-seven": "27",
            "twenty-eight": "28",
            "twenty-nine": "29",
            "thirty": "30",
            "thirty-one": "31",
            "thirty-two": "32"
        }
        
    def setup_llm_prompts(self):
        """КРИТИЧЕСКИ ИСПРАВЛЕННЫЕ промпты для ТОЧНОЙ обработки"""
        
        self.correction_prompt = """You are a dental assistant AI that corrects speech recognition errors in periodontal examination commands.

        CRITICAL RULE #1: NEVER CHANGE THE TOOTH NUMBER FROM THE INPUT
        CRITICAL RULE #2: NEVER CHANGE THE MEASUREMENT VALUES FROM THE INPUT  
        CRITICAL RULE #3: PRESERVE ALL ORIGINAL NUMBERS EXACTLY AS SPOKEN
        CRITICAL RULE #4: CORRECTLY IDENTIFY COMMAND TYPES
        


        AMERICAN UNIVERSAL TOOTH NUMBERING SYSTEM (1-32):
        - Valid tooth numbers: 1-32 ONLY

        COMMAND TYPE IDENTIFICATION RULES:
        1. "probing depth" → command_type: "probing_depth"
        2. "gingival margin" → command_type: "gingival_margin" 
        3. "bleeding" → command_type: "bleeding"
        4. "suppuration" → command_type: "suppuration"
        5. "mobility" → command_type: "mobility" 
        6. "furcation" → command_type: "furcation"
        7. "missing" → command_type: "missing_teeth"

         CRITICAL GINGIVAL MARGIN PARSING RULES:

        For "gingival margin on tooth [TOOTH_NUMBER] [VALUE1] [VALUE2] [VALUE3]":
        - FIRST number after "tooth" = TOOTH_NUMBER  
        - NEXT THREE numbers = MEASUREMENT VALUES

        CRITICAL EXAMPLES:

        Input: "gingival margin on tooth two one two three"
        CORRECT PARSING: tooth_number: 2, values: [1, 2, 3]
        WRONG PARSING: tooth_number: 1, values: [2, 1, 2]

        Input: "gingival margin on tooth fourteen minus one zero plus one"
        CORRECT PARSING: tooth_number: 14, values: [-1, 0, 1]

        Input: "gingival margin on tooth one two three four"  
        CORRECT PARSING: tooth_number: 1, values: [2, 3, 4]

        PARSING STEPS FOR GINGIVAL MARGIN:
        1. Find "tooth" keyword
        2. FIRST word/number after "tooth" = TOOTH_NUMBER
        3. NEXT THREE words/numbers = MEASUREMENT VALUES
        4. Handle signs: "minus X" = -X, "plus X" = +X

        WORD-TO-NUMBER CONVERSION:
        one→1, two→2, three→3, four→4, five→5, six→6, seven→7, eight→8, nine→9, ten→10,
        eleven→11, twelve→12, thirteen→13, fourteen→14, fifteen→15, sixteen→16, etc.

        OTHER COMMAND RULES:
        - For NON-gingival margin commands, use standard parsing
        - NEVER CHANGE THE SEQUENCE OF NUMBERS
        - PRESERVE ALL ORIGINAL NUMBERS EXACTLY AS SPOKEN

        PROBING DEPTH EXAMPLES:
        - "probing depth ... 3 2 4" → command_type: "probing_depth", values: [3, 2, 4]
        - "probing depth ... 1 2 3" → command_type: "probing_depth", values: [1, 2, 3]

        EXACT WORD-TO-NUMBER CONVERSION:
        - "one" → 1, "two" → 2, "three" → 3, etc.
        - "zero" → 0
        - "minus one" → -1, "plus one" → +1

        ASR ERROR CORRECTIONS (only fix words, never numbers):
        - "rubbing depth" → "probing depth" 
        - "buckle" → "buccal", "wingle" → "lingual"
        - "separation" → "suppuration", "cache" → "furcation"

        INPUT TEXT: "{raw_text}"

        PROCESSING STEPS:
        1. IDENTIFY COMMAND TYPE from the input text
        2. Identify tooth number and convert (one→1, two→2, etc.) 
        3. For GINGIVAL MARGIN: Process minus/plus signs correctly
        4. For PROBING DEPTH: Extract measurements as positive numbers
        5. Fix only ASR word errors, never change numbers

        OUTPUT FORMAT (JSON):
        {{
            "corrected_text": "corrected command with EXACT numbers preserved",
            "tooth_number": <exact tooth number from input>,
            "command_type": "<CORRECT command type based on input>",
            "surface": "buccal|lingual|both|null", 
            "position": "distal|mid|mesial|null",
            "values": [<exact measurements WITH CORRECT SIGNS for gingival margin>],
            "confidence": 1.0,
            "number_preservation_verified": true,
            "command_type_correct": true
        }}

        CRITICAL EXAMPLES:

        Input: "gingival margin on tooth one minus one zero plus one"
        Output: {{"corrected_text": "gingival margin on tooth number 1 minus 1 0 plus 1", "tooth_number": 1, "command_type": "gingival_margin", "values": [-1, 0, 1]}}

        Input: "probing depth on tooth one buccal surface one two three"  
        Output: {{"corrected_text": "probing depth on tooth number 1 buccal surface 1 2 3", "tooth_number": 1, "command_type": "probing_depth", "values": [1, 2, 3]}}

        REMEMBER: 
        - GINGIVAL MARGIN commands have signs: minus/plus
        - PROBING DEPTH commands are always positive numbers
        - NEVER confuse the two command types

        JSON OUTPUT:"""
        
        self.validation_prompt = """You are validating a dental command processing result.

        COMMAND: {command}
        EXTRACTED DATA: {data}

        CRITICAL VALIDATION CHECKS:
        1. NUMBER PRESERVATION: Are the extracted numbers EXACTLY what was spoken?
        2. TOOTH NUMBER ACCURACY: Does the tooth_number match the original tooth reference?
        3. MEASUREMENT ACCURACY: Do the values match the original measurement sequence?
        4. COMMAND TYPE ACCURACY: Is the command_type correctly identified?

        VALIDATION RULES:
        - If original says "tooth one" and extracted shows tooth_number=3 → CRITICAL ERROR
        - If original says "one two three" and extracted shows values=[5,9,8] → CRITICAL ERROR  
        - If original says "gingival margin" and extracted shows command_type="probing_depth" → CRITICAL ERROR
        - Tooth numbers must be 1-32 (American Universal System)
        - Probing depths must be 1-12mm and positive
        - Gingival margin values can be negative (-10 to +10mm)

        CRITICAL ERROR DETECTION:
        Look for number substitution errors:
        - Wrong tooth number (spoken ≠ extracted)
        - Wrong measurement values (spoken ≠ extracted)
        - Wrong command type identification
        - Number sequence changes

        OUTPUT (JSON):
        {{
            "valid": <true if numbers are preserved exactly and command type correct>,
            "confidence": <0.0 if critical errors, 1.0 if perfect>,
            "number_preservation_check": <true if all numbers match original>,
            "tooth_number_correct": <true if tooth matches>,
            "measurements_correct": <true if measurements match>,
            "command_type_correct": <true if command type matches>,
            "critical_errors": ["list any number preservation or type errors"],
            "issues": ["list any other problems"]
        }}

        JSON OUTPUT:"""
    
    async def process_voice_command(self, raw_text: str, patient_id: str = None) -> Dict:
        """
        ИСПРАВЛЕННАЯ функция обработки голосовой команды
        """
        try:
            if not self.client:
                return self._error_response("OpenAI client not initialized", raw_text)
            
            logger.info(f"🎤 Processing voice command: '{raw_text}'")
            
            # Шаг 1: ИСПРАВЛЕННАЯ предварительная коррекция
            pre_corrected = self.apply_basic_corrections_fixed(raw_text)
            logger.debug(f"📝 Pre-corrected: '{pre_corrected}'")
            
            # Шаг 2: LLM коррекция и извлечение данных
            llm_result = await self.llm_correction_and_extraction(pre_corrected)
            
            if not llm_result:
                return self._error_response("LLM processing failed", raw_text)
            
            # Шаг 3: Валидация результата
            validation = await self.validate_command(llm_result)
            
            if not validation.get("valid", False):
                return self._error_response(
                    f"Validation failed: {', '.join(validation.get('issues', []))}",
                    raw_text,
                    suggestions=validation.get('suggestions', [])
                )
            
            # Шаг 4: Создание команды
            command = self._create_command(llm_result, raw_text)
            
            # Шаг 5: Сохранение в chart data
            if patient_id and command.command_type != 'missing_teeth':
                self._save_to_chart(command, patient_id)
            
            # Шаг 6: Формирование ответа
            response = self._create_success_response(command, patient_id)
            
            logger.info(f"✅ Command processed successfully: {command.command_type} for tooth {command.tooth_number}")
            return response
            
        except Exception as e:
            logger.error(f"❌ Error processing voice command: {e}")
            return self._error_response(f"Processing error: {str(e)}", raw_text)
    
    def apply_basic_corrections_fixed(self, text: str) -> str:
        """КРИТИЧЕСКИ ИСПРАВЛЕННАЯ функция с правильной обработкой gingival margin"""
        print(f"🔍 INPUT: '{text}'")
        
        corrected = text.lower().strip()
        import re
        
        # КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Специальная обработка gingival margin
        if 'gingival margin' in corrected:
            print("🦷 GINGIVAL MARGIN DETECTED - using FIXED parsing")
            
            # Используем исправленный парсер
            gm_result = parse_gingival_margin_command_fixed(text)
            if gm_result:
                tooth = gm_result['tooth_number']
                values = gm_result['values']
                
                # Формируем исправленную строку с правильным порядком
                values_str = ' '.join(str(v) for v in values)
                corrected = f"gingival margin on tooth number {tooth} {values_str}"
                
                print(f"🔧 GINGIVAL MARGIN FIXED:")
                print(f"   Original: '{text}'")
                print(f"   Tooth: {tooth} (CORRECTED)")
                print(f"   Values: {values} (CORRECTED)")
                print(f"   Output: '{corrected}'")
                
                return corrected
        
        # Стандартная обработка для других команд (БЕЗ ИЗМЕНЕНИЙ)
        word_corrections_only = {
            r'\brubbing\b': 'probing',
            r'\brobin\b': 'probing', 
            r'\bbuckle\b': 'buccal',
            r'\bwingle\b': 'lingual',
            r'\bcache\b': 'furcation',
            r'\bseparation\b': 'suppuration',
        }
        
        for pattern, replacement in word_corrections_only.items():
            corrected = re.sub(pattern, replacement, corrected, flags=re.IGNORECASE)
        
        # Стандартная обработка tooth numbers для НЕ gingival margin команд
        tooth_number_pattern = r'tooth\s+(?:number\s+)?(\w+)'
        def fix_tooth_number(match):
            tooth_word = match.group(1)
            if tooth_word in self.tooth_number_corrections:
                corrected_number = self.tooth_number_corrections[tooth_word]
                print(f"🔢 TOOTH NUMBER FIXED: '{tooth_word}' → {corrected_number}")
                return f"tooth number {corrected_number}"
            return match.group(0)
        
        corrected = re.sub(tooth_number_pattern, fix_tooth_number, corrected, flags=re.IGNORECASE)
        
        # Убираем лишние пробелы
        corrected = re.sub(r'\s+', ' ', corrected).strip()
        print(f"✅ OUTPUT: '{corrected}'")
        return corrected
    
    async def llm_correction_and_extraction(self, text: str) -> Optional[Dict]:
        """LLM коррекция и извлечение структурированных данных (OpenAI v1.0+)"""
        try:
            prompt = self.correction_prompt.format(raw_text=text)
            print(f"📤 Sending to LLM: '{text}'")
            
            # ИСПРАВЛЕНО: Новый API для OpenAI v1.0+
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.1,
                timeout=30
            )
            
            content = response.choices[0].message.content.strip()
            logger.debug(f"🤖 LLM response: {content}")
            
            # Парсим JSON ответ
            try:
                result = json.loads(content)
                
                # КРИТИЧЕСКАЯ ПРОВЕРКА: Убеждаемся что tooth_number корректный
                tooth_number = result.get("tooth_number")
                if tooth_number and (tooth_number < 1 or tooth_number > 32):
                    logger.error(f"❌ CRITICAL: Invalid tooth number {tooth_number} from LLM")
                    result["tooth_number"] = 1  # Fallback
                
                print(f"🔢 LLM RESULT: Tooth {result.get('tooth_number')} (from '{result.get('original_tooth_reference', 'unknown')}')")
                return result
                
            except json.JSONDecodeError:
                # Попытка извлечь JSON из ответа
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group())
                    print(f"🔢 LLM RESULT (extracted): Tooth {result.get('tooth_number')} (from '{result.get('original_tooth_reference', 'unknown')}')")
                    return result
                else:
                    logger.error(f"❌ Could not parse LLM JSON: {content}")
                    return None
                    
        except Exception as e:
            logger.error(f"❌ LLM processing error: {e}")
            return None
    
    async def validate_command(self, llm_result: Dict) -> Dict:
        """ИСПРАВЛЕННАЯ валидация команды через LLM (OpenAI v1.0+)"""
        try:
            prompt = self.validation_prompt.format(
                command=llm_result.get("corrected_text", ""),
                data=json.dumps(llm_result, indent=2)
            )
            
            # ИСПРАВЛЕНО: Новый API для OpenAI v1.0+
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.1,
                timeout=30
            )
            
            content = response.choices[0].message.content.strip()
            
            try:
                validation = json.loads(content)
                
                # ДОПОЛНИТЕЛЬНАЯ ПРОВЕРКА tooth number
                tooth_number = llm_result.get("tooth_number")
                if tooth_number and (tooth_number < 1 or tooth_number > 32):
                    validation["valid"] = False
                    validation["critical_errors"] = validation.get("critical_errors", []) + [
                        f"Invalid tooth number {tooth_number} for American Universal System (must be 1-32)"
                    ]
                
                return validation
                
            except json.JSONDecodeError:
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())
                else:
                    return {"valid": False, "issues": ["Validation parsing error"]}
                    
        except Exception as e:
            logger.error(f"❌ Validation error: {e}")
            return {"valid": False, "issues": [f"Validation error: {str(e)}"]}
    
    def _create_command(self, llm_result: Dict, original_text: str) -> PeriodontalCommand:
        """Создание объекта команды из LLM результата"""
        return PeriodontalCommand(
            tooth_number=llm_result.get("tooth_number", 0),
            command_type=llm_result.get("command_type", ""),
            surface=llm_result.get("surface"),
            position=llm_result.get("position"),
            values=llm_result.get("values", []),
            confidence=llm_result.get("confidence", 0.0),
            original_text=original_text,
            corrected_text=llm_result.get("corrected_text", ""),
            timestamp=datetime.now().isoformat()
        )
    
    def _save_to_chart(self, command: PeriodontalCommand, patient_id: str):
        """Сохранение команды в chart data"""
        if patient_id not in self.chart_data:
            self.chart_data[patient_id] = {}
        
        tooth_num = command.tooth_number
        if tooth_num not in self.chart_data[patient_id]:
            self.chart_data[patient_id][tooth_num] = {}
        
        self.chart_data[patient_id][tooth_num][command.command_type] = command
    
    def _create_success_response(self, command: PeriodontalCommand, patient_id: str) -> Dict:
        """Создание успешного ответа"""
        
        # Формируем measurements для клиента
        measurements = self._format_measurements_for_client(command)
        
        # Создаем сообщение об успехе
        message = self._format_success_message(command)
        
        # ОТЛАДОЧНАЯ ИНФОРМАЦИЯ
        logger.info(f"🔍 LLM Response Debug:")
        logger.info(f"   Tooth: {command.tooth_number}")
        logger.info(f"   Type: {command.command_type}")
        logger.info(f"   Values: {command.values}")
        logger.info(f"   Measurements: {measurements}")
        
        return {
            "success": True,
            "command": "update_periodontal_chart",
            "patient_id": patient_id,
            "tooth_number": command.tooth_number,
            "measurement_type": command.command_type,
            "surface": command.surface,
            "position": command.position,
            "values": command.values,
            "measurements": measurements,
            "confidence": command.confidence,
            "message": message,
            "original_text": command.original_text,
            "corrected_text": command.corrected_text,
            "timestamp": command.timestamp,
            "system": "fixed_enhanced_llm_periodontal_american_universal_v1",
            "numbering_system": "american_universal_1_32"
        }
    
    def _format_measurements_for_client(self, command: PeriodontalCommand) -> Dict:
        """Форматирование измерений для веб-клиента"""
        measurements = {
            "type": command.command_type,
            "surface": command.surface,
            "position": command.position,
            "values": command.values
        }
        
        # Добавляем специфичные поля для обратной совместимости
        if command.command_type == 'probing_depth':
            measurements["probing_depth"] = command.values
        elif command.command_type == 'bleeding':
            measurements["bleeding"] = command.values
        elif command.command_type == 'suppuration':
            measurements["suppuration"] = command.values
        elif command.command_type == 'mobility':
            measurements["mobility"] = command.values[0] if command.values else None
        elif command.command_type == 'furcation':
            measurements["furcation"] = command.values[0] if command.values else None
        elif command.command_type == 'gingival_margin':
            measurements["gingival_margin"] = command.values
            if command.values and len(command.values) >= 3:
                measurements["formatted_display"] = f"{command.values[0]}-{command.values[1]}-{command.values[2]}mm"
        elif command.command_type == 'missing_teeth':
            measurements["missing_teeth"] = command.values
        
        return measurements
    
    def _format_success_message(self, command: PeriodontalCommand) -> str:
        """Форматирование сообщения об успехе"""
        tooth = command.tooth_number
        cmd_type = command.command_type
        surface = command.surface or ""
        position = command.position or ""
        values = command.values or []
        
        if cmd_type == 'probing_depth':
            surface_text = f" {surface}" if surface and surface != 'both' else ""
            values_text = '-'.join(map(str, values)) if values else ""
            return f"✅ Tooth {tooth}{surface_text} probing depths: {values_text}mm"
            
        elif cmd_type == 'bleeding':
            surface_text = f" {surface}" if surface and surface != 'both' else ""
            position_text = f" {position}" if position else ""
            status = "positive" if values and values[0] else "negative"
            return f"✅ Tooth {tooth}{surface_text}{position_text} bleeding: {status}"
            
        elif cmd_type == 'suppuration':
            surface_text = f" {surface}" if surface and surface != 'both' else ""
            position_text = f" {position}" if position else ""
            return f"✅ Tooth {tooth}{surface_text}{position_text} suppuration marked"
            
        elif cmd_type == 'mobility':
            grade = values[0] if values else 0
            return f"✅ Tooth {tooth} mobility: Grade {grade}"
            
        elif cmd_type == 'furcation':
            grade = values[0] if values else 0
            return f"✅ Tooth {tooth} furcation: Class {grade}"
            
        elif cmd_type == 'gingival_margin':
            values_text = '-'.join(map(str, values)) if values else ""
            return f"✅ Tooth {tooth} gingival margin: {values_text}mm"
            
        elif cmd_type == 'missing_teeth':
            teeth_list = ', '.join(map(str, values)) if values else ""
            return f"✅ Missing teeth marked: {teeth_list}"
            
        else:
            return f"✅ Tooth {tooth} {cmd_type} updated"
    
    def _error_response(self, error_message: str, original_text: str, suggestions: List[str] = None) -> Dict:
        """Создание ответа об ошибке"""
        return {
            "success": False,
            "error": "command_processing_failed",
            "message": error_message,
            "original_text": original_text,
            "suggestions": suggestions or self._get_command_suggestions(),
            "confidence": 0.0,
            "system": "fixed_enhanced_llm_periodontal_american_universal_v1"
        }
    
    def _get_command_suggestions(self) -> List[str]:
        """Получение предложений команд с правильной нумерацией"""
        return [
            "probing depth on tooth number 1 buccal surface 3 2 4",
            "bleeding on probing tooth 1 buccal distal", 
            "suppuration present on tooth 1 lingual mesial",
            "tooth 1 has mobility grade 2",
            "furcation class 1 on tooth 1",
            "missing teeth 1 16 17 32"
        ]
    
    def get_chart_data(self, patient_id: str) -> Dict:
        """Получение данных chart для пациента"""
        return self.chart_data.get(patient_id, {})
    
    def clear_chart_data(self, patient_id: str):
        """Очистка данных chart для пациента"""
        if patient_id in self.chart_data:
            del self.chart_data[patient_id]
    
    def get_stats(self) -> Dict:
        """Получение статистики процессора"""
        total_patients = len(self.chart_data)
        total_teeth = sum(len(patient_data) for patient_data in self.chart_data.values())
        
        return {
            "processor_type": "fixed_enhanced_llm_periodontal_american_universal_v1",
            "model": self.model,
            "total_patients": total_patients,
            "total_teeth_updated": total_teeth,
            "asr_corrections_available": len(self.asr_corrections),
            "dental_corrections_available": len(self.dental_context_corrections),
            "llm_enabled": True,
            "validation_enabled": True,
            "openai_version": "v1.0+",
            "numbering_system": "american_universal_1_32",
            "tooth_number_range": "1-32"
        }


# Глобальный экземпляр процессора (будет инициализирован при импорте)
enhanced_llm_processor = None

def initialize_enhanced_processor(openai_api_key: str, model: str = "gpt-3.5-turbo") -> bool:
    """Инициализация усиленного процессора"""
    global enhanced_llm_processor
    
    if not OPENAI_AVAILABLE:
        logger.error("❌ OpenAI library not available")
        return False
    
    try:
        enhanced_llm_processor = EnhancedPeriodontalLLMProcessor(openai_api_key, model)
        logger.info("🚀 FIXED Enhanced LLM Periodontal Processor initialized successfully (American Universal)")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to initialize FIXED Enhanced LLM Processor: {e}")
        return False

async def process_periodontal_transcription(text: str, patient_id: str = None) -> Dict:
    """
    Главная функция для интеграции с сервером
    """
    if not enhanced_llm_processor:
        return {
            "success": False,
            "error": "processor_not_initialized",
            "message": "FIXED Enhanced LLM Processor not initialized. Please provide OpenAI API key."
        }
    
    return await enhanced_llm_processor.process_voice_command(text, patient_id)

def get_processor_stats() -> Dict:
    """Получение статистики процессора"""
    if enhanced_llm_processor:
        return enhanced_llm_processor.get_stats()
    else:
        return {
            "processor_type": "not_initialized", 
            "openai_version": "v1.0+",
            "numbering_system": "american_universal_1_32"
        }


if __name__ == "__main__":
    # Тестирование системы
    import os
    
    async def test_fixed_enhanced_processor():
        """Тестирование ИСПРАВЛЕННОГО усиленного процессора"""
        
        # Инициализация (нужен OpenAI API key)
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("❌ OPENAI_API_KEY not found in environment variables")
            return
        
        success = initialize_enhanced_processor(api_key)
        if not success:
            print("❌ Failed to initialize processor")
            return
        
        print("🤖 Testing FIXED Enhanced LLM Periodontal Processor (American Universal System)")
        print("=" * 80)
        
        # Тестовые команды с проблемами нумерации
        test_commands = [
            # Проблемные команды из логов
            "Separation presents on tooth one lingual distal.",
            "Probing depth on tooth number one, buccal surface 312.",
            "Rubbing depth on tooth number two, buckle surface 312.",
            "Bleeding on probing tooth one, buccal distal.",
            "Missing this one.",
            "For Cache in class 2 on tooth one",
            "Tooth Tool has mobility grade 2",
            "Bleeding on probing tooth three, lingual distal.",
            
            # Дополнительные тесты для проверки
            "suppuration present on tooth eight buccal distal",
            "mobility grade three on tooth sixteen",
            "furcation class one on tooth thirty two"
        ]
        
        for i, command in enumerate(test_commands, 1):
            print(f"\n{i}. Testing: '{command}'")
            print("-" * 60)
            
            result = await process_periodontal_transcription(command, "test_patient")
            
            if result["success"]:
                print(f"   ✅ SUCCESS: {result['message']}")
                print(f"   📝 Original: '{result.get('original_text', 'N/A')}'")
                print(f"   📝 Corrected: '{result.get('corrected_text', 'N/A')}'")
                print(f"   🦷 Tooth: {result.get('tooth_number')} (American Universal)")
                print(f"   📋 Type: {result.get('measurement_type')}")
                print(f"   🔄 Surface: {result.get('surface')}")
                print(f"   📊 Values: {result.get('values')}")
                print(f"   🎯 Confidence: {result.get('confidence', 0):.2f}")
                print(f"   🇺🇸 System: {result.get('numbering_system', 'unknown')}")
            else:
                print(f"   ❌ FAILED: {result['message']}")
                if result.get('suggestions'):
                    print(f"   💡 Suggestions: {result['suggestions'][:2]}")
        
        # Статистика
        stats = get_processor_stats()
        print(f"\n📊 FIXED Processor Stats:")
        for key, value in stats.items():
            print(f"   {key}: {value}")
    
    # Запуск теста
    asyncio.run(test_fixed_enhanced_processor())
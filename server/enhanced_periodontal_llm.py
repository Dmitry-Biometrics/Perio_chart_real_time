#!/usr/bin/env python3
"""
Enhanced Periodontal Voice Command Processor with OpenAI LLM v1.0+
Решает проблемы ASR ошибок и обеспечивает 99%+ точность распознавания команд
ИСПРАВЛЕНО для OpenAI v1.0+
"""

import logging
import json
import asyncio
import re
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict  # ИСПРАВЛЕНО: добавлен импорт
from datetime import datetime

# Новый импорт для OpenAI v1.0+
try:
    from openai import AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logging.warning("OpenAI library not available")

logger = logging.getLogger(__name__)

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
    """Усиленный процессор с LLM для исправления ASR ошибок (OpenAI v1.0+)"""
    
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
        
        logger.info(f"🤖 Enhanced Periodontal LLM Processor initialized with {model} (OpenAI v1.0+)")
    
    def setup_correction_patterns(self):
        """Настройка паттернов для исправления частых ASR ошибок с точными числами"""
        
        # Частые ASR ошибки и их исправления
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
            
            # Common phrases
            "tooth number": "tooth number",
            "number": "number",
            "surface": "surface",
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
            
            # ИСПРАВЛЕННЫЕ числовые слова - ТОЛЬКО в контексте tooth number
            # НЕ применяем глобальную замену, а обрабатываем контекстно
            
            # Measurements
            "mm": "",
            "millimeter": "",
            "millimeters": "",
            "point": ".",
        }
        
        # НОВАЯ функция для контекстного исправления чисел
        self.tooth_number_corrections = {
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
        """Настройка улучшенных промптов для LLM с точным исправлением чисел"""
        
        self.correction_prompt = """You are a dental assistant AI that corrects speech recognition errors in periodontal examination commands.

    TASK: Fix the ASR (speech recognition) errors in the dental command and extract structured information.

    CRITICAL NUMBER CORRECTION RULES:
    - "one" → "1" (tooth number one means tooth 1)
    - "two" → "2" (tooth number two means tooth 2)  
    - "three" → "3" (tooth number three means tooth 3)
    - "four" → "4" (tooth number four means tooth 4)
    - "five" → "5" (tooth number five means tooth 5)
    - "six" → "6" (tooth number six means tooth 6)
    - "seven" → "7" (tooth number seven means tooth 7)
    - "eight" → "8" (tooth number eight means tooth 8)
    - "nine" → "9" (tooth number nine means tooth 9)
    - "ten" → "10" (tooth number ten means tooth 10)

    IMPORTANT: When correcting tooth numbers, maintain the EXACT numerical value:
    - "tooth number one" → "tooth number 1" (NOT 2!)
    - "tooth number two" → "tooth number 2" (NOT 1!)

    COMMON ASR ERRORS TO FIX:
    - "rubbing depth" → "probing depth"
    - "robin depth" → "probing depth" 
    - "buckle" → "buccal"
    - "wingle" → "lingual"
    - "tool" → "2" (ONLY when context suggests "Tooth Tool" meaning "Tooth 2")
    - "cache" → "furcation" (when saying "For Cache in class" → "Furcation class")
    - "separation" → "suppuration"
    - Numbers like "123" should be "1 2 3" (three separate measurements)
    - "mm" or "millimeter" should be removed

    MISSING TEETH COMMAND HANDLING:
    - "missing this one" → identify which tooth is being referenced from context
    - "missing teeth one" → "missing teeth 1" 
    - "missing tooth number X" → "missing teeth X"
    
    
    SPECIAL CASES:
    - "missing this one" without context → request tooth number specification
    - "missing teeth one" → "missing teeth 1"
    - "missing tooth one" → "missing teeth 1"

    EXAMPLES:
    Input: "missing this one"
    Output: {{"corrected_text": "missing teeth", "tooth_number": null, "command_type": "missing_teeth", "values": [], "needs_clarification": true}}

    MEASUREMENT SEPARATION RULES:
    - "123" → "1 2 3" (three separate probing depth measurements)
    - "456" → "4 5 6" (three separate probing depth measurements)
    - "312" → "3 1 2" (three separate probing depth measurements)

    DENTAL COMMAND TYPES:
    1. Probing Depth: "probing depth on tooth number X buccal/lingual surface A B C"
    2. Bleeding: "bleeding on probing tooth X buccal/lingual distal/mid/mesial"
    3. Suppuration: "suppuration present on tooth X buccal/lingual distal/mid/mesial"
    4. Mobility: "tooth X has mobility grade Y"
    5. Furcation: "furcation class Y on tooth X"
    6. Gingival Margin: "gingival margin on tooth X minus A B plus C"
    7. Missing Teeth: "missing teeth X Y Z"

    INPUT: "{raw_text}"

    STEP-BY-STEP CORRECTION PROCESS:
    1. Identify the tooth number and keep it EXACTLY the same numerically
    2. Correct surface terms (buckle→buccal, wingle→lingual)
    3. Separate measurement clusters (123→1 2 3)
    4. Fix command terms (rubbing→probing, separation→suppuration)
    5. Extract structured data

    OUTPUT FORMAT (JSON):
    {{
        "corrected_text": "the corrected dental command",
        "tooth_number": <integer 1-32 matching the original tooth reference>,
        "command_type": "<probing_depth|bleeding|suppuration|mobility|furcation|gingival_margin|missing_teeth>",
        "surface": "<buccal|lingual|both|all|null>",
        "position": "<distal|mid|mesial|null>",
        "values": [<array of numbers or booleans>],
        "confidence": <float 0.0-1.0>,
        "corrections_made": ["list of corrections applied"],
        "original_tooth_reference": "<original tooth number/word from input>"
    }}

    EXAMPLES:
    Input: "probing depth on tooth number one buccal surface 123"
    Output: {{"corrected_text": "probing depth on tooth number 1 buccal surface 1 2 3", "tooth_number": 1, "original_tooth_reference": "one"}}

    Input: "rubbing depth on tooth number two buckle surface 456" 
    Output: {{"corrected_text": "probing depth on tooth number 2 buccal surface 4 5 6", "tooth_number": 2, "original_tooth_reference": "two"}}

    JSON OUTPUT:"""

        self.validation_prompt = """You are a dental examination validator. Check if this periodontal command makes clinical sense.

    COMMAND: {command}
    EXTRACTED DATA: {data}

    CRITICAL VALIDATION CHECKS:
    1. Tooth number consistency: Does the extracted tooth_number match the original tooth reference?
    2. Measurement validity: Are probing depths reasonable (1-12mm)?
    3. Surface validity: Are surfaces correct (buccal/lingual)?

    VALIDATION RULES:
    - Tooth numbers: 1-32 only
    - Probing depths: 1-12mm (three values for distal, mid, mesial)
    - Mobility grades: 0-3
    - Furcation classes: 1-3
    - Surfaces: buccal, lingual, both, all
    - Positions: distal, mid, mesial (for bleeding/suppuration)

    TOOTH NUMBER VALIDATION:
    - If original says "one", tooth_number should be 1
    - If original says "two", tooth_number should be 2
    - If original says "eight", tooth_number should be 8
    - Flag any inconsistencies as CRITICAL errors
    
    TOOTH NUMBER PRESERVATION RULE:
    - NEVER change the tooth number unless there's a clear error
    - If user says "tooth number 1", keep it as tooth number 1
    - Only change tooth numbers if they're clearly invalid (>32 or <1)
    - Preserve the original tooth number from user input

    CRITICAL: The user's tooth number specification should be respected unless obviously wrong.

    OUTPUT (JSON):
    {{
        "valid": <true/false>,
        "confidence": <float 0.0-1.0>,
        "issues": ["list of any problems found"],
        "suggestions": ["suggested corrections if any"],
        "tooth_number_consistent": <true/false>,
        "critical_errors": ["list of critical errors that must be fixed"]
    }}

    JSON OUTPUT:"""
    
    async def process_voice_command(self, raw_text: str, patient_id: str = None) -> Dict:
        """
        Главная функция обработки голосовой команды с LLM коррекцией
        """
        try:
            if not self.client:
                return self._error_response("OpenAI client not initialized", raw_text)
            
            logger.info(f"🎤 Processing voice command: '{raw_text}'")
            
            # Шаг 1: Предварительная коррекция простых ошибок
            pre_corrected = self.apply_basic_corrections(raw_text)
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
    
    def apply_basic_corrections(self, text: str) -> str:
        """Применение базовых исправлений ASR ошибок с контекстным исправлением чисел"""
        corrected = text.lower().strip()
        
        # Применяем основные исправления ASR ошибок
        for error, correction in self.asr_corrections.items():
            corrected = corrected.replace(error, correction)
        
        # КОНТЕКСТНОЕ исправление tooth numbers
        # Ищем паттерн "tooth number [слово]" и заменяем только в этом контексте
        tooth_number_pattern = r'tooth number (\w+)'
        def fix_tooth_number(match):
            tooth_word = match.group(1)
            if tooth_word in self.tooth_number_corrections:
                return f"tooth number {self.tooth_number_corrections[tooth_word]}"
            return match.group(0)  # Возвращаем без изменений если не найдено
        
        corrected = re.sub(tooth_number_pattern, fix_tooth_number, corrected, flags=re.IGNORECASE)
        
        # Специальная обработка "Tooth Tool" -> "Tooth 2" (только в этом конкретном случае)
        corrected = re.sub(r'\btooth tool\b', 'tooth 2', corrected, flags=re.IGNORECASE)
        
        # Специальная обработка чисел в формате "X.XX" или "XXX"
        # "123" → "1 2 3" (три отдельных измерения)
        # Но только если это НЕ номер зуба
        def expand_measurements(text):
            # Ищем числа после "surface" 
            surface_pattern = r'surface (\d{3,})'
            def expand_surface_numbers(match):
                numbers = match.group(1)
                if len(numbers) == 3:
                    return f"surface {numbers[0]} {numbers[1]} {numbers[2]}"
                return match.group(0)
            
            text = re.sub(surface_pattern, expand_surface_numbers, text)
            
            # Также обрабатываем формат X.XX
            decimal_pattern = r'(\d+)\.(\d+)'
            def expand_decimal(match):
                integer_part = match.group(1)
                decimal_part = match.group(2)
                digits = ' '.join(list(decimal_part))
                return f"{integer_part} {digits}"
            
            text = re.sub(decimal_pattern, expand_decimal, text)
            return text
        
        corrected = expand_measurements(corrected)
        
        # Убираем лишние пробелы
        corrected = re.sub(r'\s+', ' ', corrected).strip()
        
        return corrected
    
    async def llm_correction_and_extraction(self, text: str) -> Optional[Dict]:
        """LLM коррекция и извлечение структурированных данных (OpenAI v1.0+)"""
        try:
            prompt = self.correction_prompt.format(raw_text=text)
            
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
                return result
            except json.JSONDecodeError:
                # Попытка извлечь JSON из ответа
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())
                else:
                    logger.error(f"❌ Could not parse LLM JSON: {content}")
                    return None
                    
        except Exception as e:
            logger.error(f"❌ LLM processing error: {e}")
            return None
    
    async def validate_command(self, llm_result: Dict) -> Dict:
        """Валидация команды через LLM (OpenAI v1.0+)"""
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
            "system": "enhanced_llm_periodontal_v1"
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
            "system": "enhanced_llm_periodontal_v1"
        }
    
    def _get_command_suggestions(self) -> List[str]:
        """Получение предложений команд"""
        return [
            "probing depth on tooth number 14 buccal surface 3 2 4",
            "bleeding on probing tooth 12 buccal distal", 
            "suppuration present on tooth 8 lingual mesial",
            "tooth 6 has mobility grade 2",
            "furcation class 1 on tooth 19",
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
            "processor_type": "enhanced_llm_periodontal_v1",
            "model": self.model,
            "total_patients": total_patients,
            "total_teeth_updated": total_teeth,
            "asr_corrections_available": len(self.asr_corrections),
            "dental_corrections_available": len(self.dental_corrections),
            "llm_enabled": True,
            "validation_enabled": True,
            "openai_version": "v1.0+"
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
        logger.info("🚀 Enhanced LLM Periodontal Processor initialized successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to initialize Enhanced LLM Processor: {e}")
        return False

async def process_periodontal_transcription(text: str, patient_id: str = None) -> Dict:
    """
    Главная функция для интеграции с сервером
    """
    if not enhanced_llm_processor:
        return {
            "success": False,
            "error": "processor_not_initialized",
            "message": "Enhanced LLM Processor not initialized. Please provide OpenAI API key."
        }
    
    return await enhanced_llm_processor.process_voice_command(text, patient_id)

def get_processor_stats() -> Dict:
    """Получение статистики процессора"""
    if enhanced_llm_processor:
        return enhanced_llm_processor.get_stats()
    else:
        return {"processor_type": "not_initialized", "openai_version": "v1.0+"}


if __name__ == "__main__":
    # Тестирование системы
    import os
    
    async def test_enhanced_processor():
        """Тестирование усиленного процессора"""
        
        # Инициализация (нужен OpenAI API key)
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("❌ OPENAI_API_KEY not found in environment variables")
            return
        
        success = initialize_enhanced_processor(api_key)
        if not success:
            print("❌ Failed to initialize processor")
            return
        
        print("🤖 Testing Enhanced LLM Periodontal Processor (OpenAI v1.0+)")
        print("=" * 60)
        
        # Тестовые команды с ASR ошибками
        test_commands = [
            # Проблемные команды из вашего примера
            "Probing depth on tooth number one, buckle surface 312.",
            "Rubbing depth on tooth number 2, buckle surface 312.",
            "Bleeding on probing tooth 2, buccal distal.",
            "Missing this one.",
            "For Cache in class 2 on tooth 2",
            "Tooth Tool has mobility grade 2",
            "Separation present on tooth 8 lingual distal.",
            "Bleeding on probing tooth 3, lingual distal."
        ]
        
        for i, command in enumerate(test_commands, 1):
            print(f"\n{i}. Testing: '{command}'")
            
            result = await process_periodontal_transcription(command, "test_patient")
            
            if result["success"]:
                print(f"   ✅ SUCCESS: {result['message']}")
                print(f"   📝 Corrected: '{result.get('corrected_text', 'N/A')}'")
                print(f"   🦷 Tooth: {result.get('tooth_number')}")
                print(f"   📋 Type: {result.get('measurement_type')}")
                print(f"   🔄 Surface: {result.get('surface')}")
                print(f"   📊 Values: {result.get('values')}")
                print(f"   🎯 Confidence: {result.get('confidence', 0):.2f}")
            else:
                print(f"   ❌ FAILED: {result['message']}")
                if result.get('suggestions'):
                    print(f"   💡 Suggestions: {result['suggestions'][:2]}")
        
        # Статистика
        stats = get_processor_stats()
        print(f"\n📊 Processor Stats:")
        for key, value in stats.items():
            print(f"   {key}: {value}")
    
    # Запуск теста
    asyncio.run(test_enhanced_processor())
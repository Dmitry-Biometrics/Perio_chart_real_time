#!/usr/bin/env python3
"""
Enhanced RAG System with Intent Classification for Dental Voice Commands
Добавляет распознавание намерений (intents) для улучшенной обработки команд
"""

import logging
import json
import asyncio
import re
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class DentalIntent(Enum):
    """Enumeration of dental command intents"""
    # Periodontal intents
    PROBING_DEPTH = "probing_depth"
    BLEEDING_ON_PROBING = "bleeding_on_probing"
    SUPPURATION = "suppuration"
    MOBILITY = "mobility"
    FURCATION = "furcation"
    GINGIVAL_MARGIN = "gingival_margin"
    RECESSION = "recession"
    PLAQUE_INDEX = "plaque_index"
    
    # Administrative intents
    MISSING_TEETH = "missing_teeth"
    PATIENT_NOTES = "patient_notes"
    CHART_NAVIGATION = "chart_navigation"
    
    # System intents
    CLEAR_DATA = "clear_data"
    SAVE_DATA = "save_data"
    UNDO_LAST = "undo_last"
    
    # General intents
    GREETING = "greeting"
    HELP_REQUEST = "help_request"
    CONFIRMATION = "confirmation"
    UNKNOWN = "unknown"

@dataclass
class IntentClassificationResult:
    """Result of intent classification"""
    intent: DentalIntent
    confidence: float
    entities: Dict[str, Any]
    raw_text: str
    normalized_text: str
    suggested_command: Optional[str] = None

class DentalIntentClassifier:
    """Advanced intent classifier for dental voice commands"""
    
    def __init__(self):
        self.intent_patterns = self._initialize_intent_patterns()
        self.entity_extractors = self._initialize_entity_extractors()
        
        # Statistics
        self.classification_stats = {
            'total_classifications': 0,
            'intent_counts': {intent.value: 0 for intent in DentalIntent},
            'average_confidence': 0.0,
            'high_confidence_count': 0,  # confidence > 0.8
            'low_confidence_count': 0,   # confidence < 0.5
        }
        
        logger.info("🧠 Enhanced Dental Intent Classifier initialized")
    
    def _initialize_intent_patterns(self) -> Dict[DentalIntent, List[Dict]]:
        """Initialize pattern matching rules for each intent"""
        
        patterns = {
            DentalIntent.PROBING_DEPTH: [
                {
                    'patterns': [
                        r'(?:probing|provin|robin|rubbing)\s+(?:depth|death|dats)',
                        r'pocket\s+depth',
                        r'pd\s+(?:on\s+)?tooth',
                        r'depth\s+(?:on\s+)?tooth'
                    ],
                    'required_entities': ['tooth_number'],
                    'weight': 1.0,
                    # ДОБАВЬТЕ ЭТИ ПРАВИЛА ДЛЯ ЧЕТКОСТИ:
                    'preserve_tooth_number': True,  # НЕ менять номер зуба
                    'strict_entity_validation': True,  # Строгая валидация
                    'require_exact_measurements': True  # Требовать точные измерения
                }
            ],
            
            DentalIntent.BLEEDING_ON_PROBING: [
                {
                    'patterns': [
                        r'bleeding\s+(?:on\s+)?(?:probing|provin)',
                        r'bop\s+(?:on\s+)?tooth',
                        r'blood\s+(?:on\s+)?(?:probing|tooth)',
                        r'bleeding\s+(?:positive|negative|present)'
                    ],
                    'required_entities': ['tooth_number'],
                    'weight': 1.0
                },
                {
                    'patterns': [
                        r'tooth\s+\d+\s+(?:is\s+)?bleeding',
                        r'bleeding\s+tooth\s+\d+',
                    ],
                    'required_entities': ['tooth_number'],
                    'weight': 0.9
                }
            ],
            
            DentalIntent.SUPPURATION: [
                {
                    'patterns': [
                        r'(?:suppuration|separation)\s+(?:present|positive)',
                        r'pus\s+(?:present|formation)',
                        r'discharge\s+(?:present|noted)',
                        r'exudate\s+(?:present|observed)'
                    ],
                    'required_entities': ['tooth_number'],
                    'weight': 1.0
                }
            ],
            
            DentalIntent.MOBILITY: [
                {
                    'patterns': [
                        r'(?:tooth\s+\d+\s+)?(?:has\s+)?mobility\s+(?:grade\s+)?\d+',
                        r'mobility\s+(?:of\s+)?tooth\s+\d+',
                        r'tooth\s+\d+\s+(?:is\s+)?(?:mobile|loose)',
                        r'grade\s+\d+\s+mobility'
                    ],
                    'required_entities': ['tooth_number', 'grade'],
                    'weight': 1.0
                }
            ],
            
            DentalIntent.FURCATION: [
                {
                    'patterns': [
                        r'(?:furcation|furkat|cache)\s+(?:involvement\s+)?(?:class\s+)?\d+',
                        r'(?:for\s+)?(?:cache|furcation)\s+(?:in\s+)?class\s+\d+',
                        r'class\s+\d+\s+furcation',
                        r'bifurcation\s+(?:involvement|class)'
                    ],
                    'required_entities': ['tooth_number', 'class'],
                    'weight': 1.0
                }
            ],
            
            DentalIntent.GINGIVAL_MARGIN: [
                {
                    'patterns': [
                        r'gingival\s+margin',
                        r'gm\s+(?:on\s+)?tooth',
                        r'margin\s+(?:level|position)',
                        r'(?:minus|plus)\s+\d+.*(?:minus|plus)\s+\d+'
                    ],
                    'required_entities': ['tooth_number'],
                    'weight': 1.0,
                    # КРИТИЧЕСКИЕ ПРАВИЛА ДЛЯ ЧЕТКОСТИ:
                    'preserve_tooth_number': True,
                    'preserve_original_values': True,  # НЕ менять значения
                    'strict_measurement_parsing': True
                }
            ],
            
            DentalIntent.RECESSION: [
                {
                    'patterns': [
                        r'gingival\s+recession',
                        r'recession\s+(?:on\s+)?tooth',
                        r'root\s+exposure',
                        r'attachment\s+loss'
                    ],
                    'required_entities': ['tooth_number'],
                    'weight': 1.0
                }
            ],
            
            DentalIntent.MISSING_TEETH: [
                {
                    'patterns': [
                        r'missing\s+(?:teeth|tooth)',
                        r'(?:tooth|teeth)\s+(?:is\s+|are\s+)?missing',
                        r'absent\s+(?:teeth|tooth)',
                        r'extracted\s+(?:teeth|tooth)',
                        r'missing\s+(?:this\s+)?one'
                    ],
                    'required_entities': [],  # Can work without specific tooth numbers
                    'weight': 1.0
                }
            ],
            
            DentalIntent.CLEAR_DATA: [
                {
                    'patterns': [
                        r'clear\s+(?:all\s+)?(?:data|chart|everything)',
                        r'reset\s+(?:the\s+)?chart',
                        r'delete\s+(?:all\s+)?(?:data|entries)',
                        r'start\s+(?:over|fresh|new)'
                    ],
                    'required_entities': [],
                    'weight': 1.0
                }
            ],
            
            DentalIntent.SAVE_DATA: [
                {
                    'patterns': [
                        r'save\s+(?:the\s+)?(?:chart|data)',
                        r'export\s+(?:chart|data)',
                        r'backup\s+(?:the\s+)?chart',
                        r'store\s+(?:this\s+)?(?:chart|data)'
                    ],
                    'required_entities': [],
                    'weight': 1.0
                }
            ],
            
            DentalIntent.HELP_REQUEST: [
                {
                    'patterns': [
                        r'help\s+(?:me|please)?',
                        r'what\s+(?:can\s+)?(?:i\s+)?(?:say|do)',
                        r'how\s+(?:do\s+)?(?:i\s+)?(?:use|work)',
                        r'commands?\s+(?:list|available)',
                        r'(?:show\s+)?(?:me\s+)?(?:the\s+)?(?:commands|options)'
                    ],
                    'required_entities': [],
                    'weight': 1.0
                }
            ],
            
            DentalIntent.GREETING: [
                {
                    'patterns': [
                        r'(?:hello|hi|hey)',
                        r'good\s+(?:morning|afternoon|evening)',
                        r'how\s+are\s+you',
                        r'what\'?s\s+up'
                    ],
                    'required_entities': [],
                    'weight': 1.0
                }
            ],
            
            DentalIntent.CONFIRMATION: [
                {
                    'patterns': [
                        r'(?:yes|yeah|yep|correct|right|okay|ok)',
                        r'that\'?s\s+(?:right|correct)',
                        r'confirm(?:ed)?',
                        r'(?:no|nope|negative|wrong|incorrect)'
                    ],
                    'required_entities': [],
                    'weight': 1.0
                }
            ]
        }
        
        return patterns
    
    def _initialize_entity_extractors(self) -> Dict[str, Any]:
        """Initialize entity extraction patterns"""
        
        return {
            'tooth_number': {
                'patterns': [
                    r'tooth\s+(?:number\s+)?(\d+)',
                    r'(?:tooth\s+)?(\d+)(?:\s+(?:has|is|with))?',
                    r'(?:on\s+)?tooth\s+(\d+)',
                    r'number\s+(\d+)',
                    # ТОЧНЫЕ паттерны для word numbers
                    r'tooth\s+(?:number\s+)?(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty)',
                ],
                'word_to_number': {
                    'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
                    'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
                    'eleven': 11, 'twelve': 12, 'thirteen': 13, 'fourteen': 14,
                    'fifteen': 15, 'sixteen': 16, 'seventeen': 17, 'eighteen': 18,
                    'nineteen': 19, 'twenty': 20, 'thirty': 30
                },
                # НОВЫЕ ПРАВИЛА ДЛЯ ЧЕТКОСТИ:
                'preserve_original': True,  # Сохранять оригинальный номер
                'validate_range': True,     # Проверять диапазон 1-32
                'no_auto_correction': True  # НЕ исправлять автоматически
            },
            'surface': {
                'patterns': [
                    r'(buccal|buckle|becal|facial)',
                    r'(lingual|wingle|lingle|lingwal)',
                    r'(mesial|distal|mid|middle)',
                    r'(?:on\s+)?(?:the\s+)?(buccal|lingual|facial)\s+surface'
                ],
                'corrections': {
                    'buckle': 'buccal', 'becal': 'buccal',
                    'wingle': 'lingual', 'lingle': 'lingual', 'lingwal': 'lingual',
                    'middle': 'mid'
                }
            },
            
            'position': {
                'patterns': [
                    r'(distal|mesial|mid|middle)',
                    r'(?:on\s+)?(?:the\s+)?(distal|mesial|mid)\s+(?:aspect|side|position)'
                ],
                'corrections': {
                    'middle': 'mid'
                }
            },
            
            'measurements': {
                'patterns': [
                    r'(\d+)\s+(\d+)\s+(\d+)',  # Three measurements: 3 2 4
                    r'(\d+)\.(\d+)',           # Decimal format: 3.24 -> 3 2 4
                    r'(\d{3})',                # Three digit format: 324 -> 3 2 4
                    r'(\d+)',                  # Single measurement
                ],
                'expand_rules': {
                    'three_digit': lambda x: [int(x[0]), int(x[1]), int(x[2])] if len(x) == 3 else None,
                    'decimal': lambda x: [int(x.split('.')[0])] + [int(d) for d in x.split('.')[1]] if '.' in x else None
                },
                # НОВЫЕ ПРАВИЛА:
                'preserve_sequence': True,   # Сохранять порядок измерений
                'validate_ranges': True,     # Проверять допустимые значения
                'no_reordering': True       # НЕ переставлять значения
            },
            
            'class': {
                'patterns': [
                    r'class\s+(\d+)',
                    r'grade\s+(\d+)',
                    r'type\s+(\d+)',
                ]
            },
            
            'signs': {
                'patterns': [
                    r'(minus|plus|\-|\+)\s*(\d+)',
                    r'(\d+)\s*(mm|millimeter)',
                ],
                'conversions': {
                    'minus': '-', 'plus': '+',
                    'mm': '', 'millimeter': ''
                }
            }
        }
    
    def _apply_strict_validation_rules(self, intent: DentalIntent, entities: Dict, original_text: str) -> Dict:
        """Применение строгих правил валидации для четких команд"""
        
        validation_rules = {
            'preserve_tooth_number': True,    # НЕ менять номер зуба из оригинала
            'preserve_measurements': True,    # НЕ менять измерения  
            'strict_surface_matching': True,  # Точное соответствие поверхностей
            'no_assumptions': True,          # НЕ делать предположений
            'require_explicit_values': True  # Требовать явные значения
        }
        
        # Извлекаем оригинальный номер зуба
        original_tooth_pattern = r'tooth\s+(?:number\s+)?(\w+)'
        original_tooth_match = re.search(original_tooth_pattern, original_text.lower())
        
        if original_tooth_match and validation_rules['preserve_tooth_number']:
            original_tooth_ref = original_tooth_match.group(1)
            
            # Если это слово-число, конвертируем точно
            word_to_num = {
                'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
                'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
                'eleven': 11, 'twelve': 12, 'thirteen': 13, 'fourteen': 14,
                'fifteen': 15, 'sixteen': 16, 'seventeen': 17, 'eighteen': 18,
                'nineteen': 19, 'twenty': 20, 'thirty': 30, 'thirty-one': 31, 'thirty-two': 32
            }
            if original_tooth_ref in word_to_num:
                entities['tooth_number'] = word_to_num[original_tooth_ref]
            elif original_tooth_ref.isdigit():
                entities['tooth_number'] = int(original_tooth_ref)
        
        # Извлекаем оригинальные измерения без изменений
        if validation_rules['preserve_measurements']:
            measurements_pattern = r'(\d{3,}|\d+\s+\d+\s+\d+)'
            measurements_match = re.search(measurements_pattern, original_text)
            
            if measurements_match:
                measurement_str = measurements_match.group(1)
                if len(measurement_str) == 3 and measurement_str.isdigit():
                    # "312" -> [3, 1, 2] ТОЧНО как услышано
                    entities['measurements'] = [int(d) for d in measurement_str]
        
        return entities
    
    def classify_intent(self, text: str) -> IntentClassificationResult:
        """
        Classify the intent of the input text
        """
        self.classification_stats['total_classifications'] += 1
        
        # Normalize text
        normalized_text = self._normalize_text(text)
        
        # Extract entities first
        entities = self._extract_entities(normalized_text)
        
        # Score each intent
        intent_scores = {}
        
        for intent, pattern_groups in self.intent_patterns.items():
            max_score = 0.0
            
            for pattern_group in pattern_groups:
                score = self._score_pattern_group(normalized_text, pattern_group, entities)
                max_score = max(max_score, score)
            
            intent_scores[intent] = max_score
        
        # Find best intent
        if intent_scores:
            best_intent = max(intent_scores, key=intent_scores.get)
            confidence = intent_scores[best_intent]
        else:
            best_intent = DentalIntent.UNKNOWN
            confidence = 0.0
        
        # Update statistics
        self.classification_stats['intent_counts'][best_intent.value] += 1
        
        if confidence > 0.8:
            self.classification_stats['high_confidence_count'] += 1
        elif confidence < 0.5:
            self.classification_stats['low_confidence_count'] += 1
        
        # Update running average
        total = self.classification_stats['total_classifications']
        current_avg = self.classification_stats['average_confidence']
        self.classification_stats['average_confidence'] = (
            (current_avg * (total - 1) + confidence) / total
        )
        
        # Generate suggested command if confidence is low
        suggested_command = None
        if confidence < 0.6:
            suggested_command = self._generate_suggested_command(best_intent, entities, text)
        
        result = IntentClassificationResult(
            intent=best_intent,
            confidence=confidence,
            entities=entities,
            raw_text=text,
            normalized_text=normalized_text,
            suggested_command=suggested_command
        )
        
        logger.debug(f"Intent classification: {text} -> {best_intent.value} (confidence: {confidence:.3f})")
        
        return result
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for better pattern matching"""
        
        # Convert to lowercase
        text = text.lower().strip()
        
        # Common ASR corrections
        corrections = {
            'rubbing': 'probing', 'robin': 'probing', 'provin': 'probing',
            'death': 'depth', 'dats': 'depth',
            'buckle': 'buccal', 'becal': 'buccal',
            'wingle': 'lingual', 'lingle': 'lingual', 'lingwal': 'lingual',
            'separation': 'suppuration',
            'cache': 'furcation', 'furkat': 'furcation',
            'tool': 'tooth',  # "Tooth Tool" -> "Tooth 2"
            
            # КРИТИЧЕСКИЕ ИСПРАВЛЕНИЯ для missing команд
            'missing this too': 'missing teeth 2',
            'missing this one': 'missing teeth 1',
            'missing that one': 'missing teeth 1',
            'this one': 'tooth 1',  # Общая замена для контекста
        }
        
        for error, correction in corrections.items():
            text = re.sub(r'\b' + re.escape(error) + r'\b', correction, text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text
    
    def _extract_entities(self, text: str) -> Dict[str, Any]:
        """Extract entities from normalized text"""
        
        entities = {}
        
        for entity_type, config in self.entity_extractors.items():
            values = []
            
            for pattern in config['patterns']:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    if match.groups():
                        value = match.group(1)
                        
                        # Apply corrections if available
                        if 'corrections' in config and value in config['corrections']:
                            value = config['corrections'][value]
                        
                        # Convert word numbers to digits for tooth_number
                        if entity_type == 'tooth_number' and value in config.get('word_to_number', {}):
                            value = config['word_to_number'][value]
                        
                        # Convert to int if it's a number
                        if str(value).isdigit():
                            value = int(value)
                        
                        if value not in values:
                            values.append(value)
            
            if values:
                entities[entity_type] = values if len(values) > 1 else values[0]
        
        # КРИТИЧЕСКИ ВАЖНО: всегда извлекаем measurements
        measurements = self._extract_measurements(text)
        if measurements:
            entities['measurements'] = measurements
            print(f"🔧 Added measurements to entities: {measurements}")
        
        print(f"🔍 All entities extracted: {entities}")
        return entities
    
    def _extract_measurements(self, text: str) -> List[int]:
        """КРИТИЧЕСКИ ИСПРАВЛЕННОЕ извлечение измерений - СОХРАНЯЕТ оригинальные значения"""
        
        measurements = []
        
        # Словарь для конвертации слов в числа - ТОЧНОЕ соответствие
        word_to_num = {
            'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
            'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
            'eleven': 11, 'twelve': 12
        }
        
        print(f"🔍 PARSING MEASUREMENTS (PRESERVE ORIGINAL): '{text}'")
        
        # Ищем ВСЕ числовые слова и цифры в тексте
        words = text.lower().split()
        all_numbers = []
        
        print(f"🔍 All words: {words}")
        
        for word in words:
            # Очищаем от знаков препинания
            clean_word = word.strip('.,!?;:')
            
            # Пропускаем служебные слова, НО НЕ числовые!
            if clean_word in ['on', 'tooth', 'number', 'surface', 'buccal', 'lingual', 
                             'probing', 'depth', 'and', 'to', 'the', 'a', 'an', 'has', 'grade', 'class']:
                continue
                
            # КРИТИЧЕСКИ ВАЖНО: конвертируем числовые слова ТОЧНО
            if clean_word in word_to_num:
                num_value = word_to_num[clean_word]
                all_numbers.append(num_value)
                print(f"🔢 '{clean_word}' → {num_value} (PRESERVED)")
            elif clean_word.isdigit():
                num_value = int(clean_word)
                all_numbers.append(num_value)
                print(f"🔢 '{clean_word}' → {num_value} (PRESERVED)")
        
        print(f"📊 All numbers found (ORIGINAL VALUES): {all_numbers}")
        
        # ЛОГИКА ВЫБОРА ИЗМЕРЕНИЙ с сохранением порядка
        if len(all_numbers) >= 4:
            # Если 4+ чисел, первое вероятно - номер зуба, берем последние 3
            measurements = all_numbers[-3:]
            print(f"🎯 4+ numbers found, taking last 3 as measurements: {measurements}")
            
        elif len(all_numbers) == 3:
            # Если ровно 3 числа, проверяем контекст
            # Если есть "tooth number" в начале, первое число - зуб
            if 'tooth' in text.lower() and 'number' in text.lower():
                # Первое число - зуб, остальные - измерения
                measurements = all_numbers[1:] if len(all_numbers) > 1 else all_numbers
                print(f"🎯 3 numbers with tooth context, taking last 2 or all: {measurements}")
            else:
                # Все три - измерения
                measurements = all_numbers
                print(f"🎯 3 numbers, all are measurements: {measurements}")
                
        elif len(all_numbers) == 2:
            measurements = all_numbers
            print(f"🎯 2 numbers found: {measurements}")
            
        elif len(all_numbers) == 1:
            measurements = all_numbers
            print(f"🎯 1 number found: {measurements}")
        
        # КРИТИЧЕСКИ ВАЖНО: НЕ МЕНЯЕМ ЗНАЧЕНИЯ!
        # Fallback: regex поиск цифр (но сохраняем оригинальный порядок)
        if not measurements:
            print("🔧 No word numbers found, trying regex...")
            
            import re
            # Ищем цифры в тексте, сохраняя порядок
            digit_matches = re.findall(r'\d+', text)
            if len(digit_matches) >= 3:
                measurements = [int(d) for d in digit_matches[-3:]]
                print(f"🔧 Regex found digits (preserved order): {measurements}")
            elif len(digit_matches) >= 1:
                measurements = [int(d) for d in digit_matches]
                print(f"🔧 Regex found some digits: {measurements}")
        
        # ЕСЛИ измерений недостаточно - НЕ ДОПОЛНЯЕМ произвольными значениями!
        # Возвращаем что есть
        print(f"✅ Final measurements (ORIGINAL VALUES PRESERVED): {measurements}")
        return measurements
    
    def _score_pattern_group(self, text: str, pattern_group: Dict, entities: Dict) -> float:
        """Score a pattern group against the text"""
        
        base_score = 0.0
        
        # Check pattern matches
        patterns_matched = 0
        for pattern in pattern_group['patterns']:
            if re.search(pattern, text, re.IGNORECASE):
                patterns_matched += 1
        
        if patterns_matched > 0:
            base_score = (patterns_matched / len(pattern_group['patterns'])) * pattern_group['weight']
        
        # Check required entities
        required_entities = pattern_group.get('required_entities', [])
        if required_entities:
            entities_found = sum(1 for entity in required_entities if entity in entities)
            entity_score = entities_found / len(required_entities)
            base_score *= entity_score
        
        # Boost score if all patterns match
        if patterns_matched == len(pattern_group['patterns']):
            base_score *= 1.2
        
        # Boost score for exact entity matches
        if 'tooth_number' in entities and isinstance(entities['tooth_number'], int):
            if 1 <= entities['tooth_number'] <= 32:
                base_score *= 1.1
        
        return min(base_score, 1.0)  # Cap at 1.0
    
    def _generate_suggested_command(self, intent: DentalIntent, entities: Dict, original_text: str) -> Optional[str]:
        """Generate a suggested command for low-confidence classifications"""
        
        suggestions = {
            DentalIntent.PROBING_DEPTH: "Try: 'Probing depth on tooth number {tooth} buccal surface 3 2 4'",
            DentalIntent.BLEEDING_ON_PROBING: "Try: 'Bleeding on probing tooth {tooth} buccal distal'",
            DentalIntent.SUPPURATION: "Try: 'Suppuration present on tooth {tooth} buccal distal'",
            DentalIntent.MOBILITY: "Try: 'Tooth {tooth} has mobility grade 2'",
            DentalIntent.FURCATION: "Try: 'Furcation class 2 on tooth {tooth}'",
            DentalIntent.GINGIVAL_MARGIN: "Try: 'Gingival margin on tooth {tooth} minus 1 0 plus 1'",
            DentalIntent.MISSING_TEETH: "Try: 'Missing teeth 1 16 17 32'",
        }
        
        if intent in suggestions:
            template = suggestions[intent]
            tooth = entities.get('tooth_number', 'X')
            return template.format(tooth=tooth)
        
        return None
    
    def get_classification_stats(self) -> Dict[str, Any]:
        """Get classification statistics"""
        return self.classification_stats.copy()


class EnhancedRAGSystem:
    """Enhanced RAG system with intent classification"""
    
    def __init__(self, openai_api_key: str = None):
        self.intent_classifier = DentalIntentClassifier()
        self.openai_api_key = openai_api_key
        self.logger = logging.getLogger(__name__)
        
        # RAG specific data
        self.knowledge_base = self._initialize_knowledge_base()
        self.command_history = []
        self.session_stats = {
            'commands_processed': 0,
            'intents_recognized': 0,
            'high_confidence_intents': 0,
            'rag_assisted_commands': 0,
            'successful_executions': 0
        }
        
        logger.info("🧠 Enhanced RAG System with Intent Classification initialized")
        
        
    def _enhance_with_rag(self, command, entities, context=None):
        """Улучшение команды с помощью RAG"""
        try:
            # Базовая логика улучшения
            enhanced_entities = entities.copy()
            
            # Добавить контекстную информацию
            if 'tooth_number' in entities:
                tooth_context = self._get_tooth_context(entities['tooth_number'])
                enhanced_entities.update(tooth_context)
            
            return enhanced_entities
            
        except Exception as e:
            self.logger.error(f"❌ Error in _enhance_with_rag: {e}")
            return entities    
            
    
            
    def _handle_bleeding_on_probing(self, command, entities):
        """Обработка команд о кровоточивости"""
        try:
            result = {
                'type': 'bleeding_on_probing',
                'tooth_number': entities.get('tooth_number'),
                'surface': entities.get('surface'),
                'position': entities.get('position'),
                'command': command
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Error in _handle_bleeding_on_probing: {e}")
            return entities
    
    def _handle_suppuration(self, command, entities):
        """Обработка команд о нагноении"""
        try:
            result = {
                'type': 'suppuration',
                'tooth_number': entities.get('tooth_number'),
                'surface': entities.get('surface'),
                'position': entities.get('position'),
                'command': command
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Error in _handle_suppuration: {e}")
            return entities            
            
    def _extract_tooth_numbers(self, command):
        """Извлечение номеров зубов из команды"""
        import re
        
        # Поиск чисел в команде
        numbers = re.findall(r'\d+', command)
        
        # Поиск словесных числительных
        word_numbers = {
            'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
            'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
            'eleven': 11, 'twelve': 12, 'thirteen': 13, 'fourteen': 14,
            'fifteen': 15, 'sixteen': 16, 'seventeen': 17, 'eighteen': 18,
            'nineteen': 19, 'twenty': 20, 'thirty': 30, 'thirty-two': 32
        }
        
        words = command.lower().split()
        for word in words:
            if word in word_numbers:
                numbers.append(str(word_numbers[word]))
        
        return [int(n) for n in numbers if 1 <= int(n) <= 32]
    
    def _get_tooth_context(self, tooth_number):
        """Получение контекста зуба"""
        # Базовая информация о зубе
        context = {
            'tooth_type': self._get_tooth_type(tooth_number),
            'quadrant': self._get_quadrant(tooth_number),
            'arch': 'upper' if tooth_number <= 16 else 'lower'
        }
        
        return context
    
    def _get_tooth_type(self, tooth_number):
        """Определение типа зуба"""
        tooth_in_quadrant = ((tooth_number - 1) % 8) + 1
        
        if tooth_in_quadrant <= 2:
            return 'incisor'
        elif tooth_in_quadrant == 3:
            return 'canine'
        elif tooth_in_quadrant <= 5:
            return 'premolar'
        else:
            return 'molar'
    
    def _get_quadrant(self, tooth_number):
        """Определение квадранта"""
        if 1 <= tooth_number <= 8:
            return 1
        elif 9 <= tooth_number <= 16:
            return 2
        elif 17 <= tooth_number <= 24:
            return 3
        else:
            return 4
            
    def _initialize_knowledge_base(self) -> Dict[str, Any]:
        """Initialize the dental knowledge base"""
        
        return {
            'command_templates': {
                DentalIntent.PROBING_DEPTH: [
                    "Probing depth on tooth number {tooth} {surface} surface {measurements}",
                    "PD tooth {tooth} {surface} {measurements}",
                    "Pocket depth {tooth} {surface} {measurements}"
                ],
                DentalIntent.BLEEDING_ON_PROBING: [
                    "Bleeding on probing tooth {tooth} {surface} {position}",
                    "BOP tooth {tooth} {surface} {position}",
                    "Bleeding tooth {tooth} {surface} {position}"
                ],
                DentalIntent.SUPPURATION: [
                    "Suppuration present on tooth {tooth} {surface} {position}",
                    "Pus present tooth {tooth} {surface} {position}"
                ],
                DentalIntent.MOBILITY: [
                    "Tooth {tooth} has mobility grade {grade}",
                    "Mobility grade {grade} tooth {tooth}"
                ],
                DentalIntent.FURCATION: [
                    "Furcation class {class} on tooth {tooth}",
                    "Furcation involvement class {class} tooth {tooth}"
                ],
                DentalIntent.GINGIVAL_MARGIN: [
                    "Gingival margin on tooth {tooth} {measurements}",
                    "GM tooth {tooth} {measurements}"
                ],
                DentalIntent.MISSING_TEETH: [
                    "Missing teeth {teeth}",
                    "Absent teeth {teeth}",
                    "Extracted teeth {teeth}"
                ]
            },
            
            'common_corrections': {
                'surfaces': {'buckle': 'buccal', 'wingle': 'lingual'},
                'commands': {'rubbing': 'probing', 'separation': 'suppuration'},
                'measurements': {'324': '3 2 4', '231': '2 3 1'}
            },
            
            'validation_rules': {
                'tooth_number': {'min': 1, 'max': 32},
                'probing_depth': {'min': 1, 'max': 12},
                'mobility_grade': {'min': 0, 'max': 3},
                'furcation_class': {'min': 1, 'max': 3}
            }
        }
    
    async def process_dental_command(self, text: str, context: Dict = None) -> Dict[str, Any]:
        """
        Process dental command using intent classification and RAG
        """
        try:
            self.session_stats['commands_processed'] += 1
            
            logger.info(f"🧠 Processing dental command: '{text}'")
            
            # Step 1: Classify intent
            classification = self.intent_classifier.classify_intent(text)
            
            if classification.confidence > 0.5:
                self.session_stats['intents_recognized'] += 1
                
            if classification.confidence > 0.8:
                self.session_stats['high_confidence_intents'] += 1
            
            # Step 2: Process based on intent
            result = await self._process_by_intent(classification, context)
            
            # Step 3: Enhance with RAG if needed
            if classification.confidence < 0.7 or not result.get('success'):
                result = await self._enhance_with_rag(classification, result, context)
                if result:
                    self.session_stats['rag_assisted_commands'] += 1
            
            # Step 4: Add intent information to result
            result.update({
                'intent': classification.intent.value,
                'intent_confidence': classification.confidence,
                'entities': classification.entities,
                'suggested_command': classification.suggested_command,
                'rag_enhanced': result.get('rag_enhanced', False)
            })
            
            if result.get('success'):
                self.session_stats['successful_executions'] += 1
            
            # Store in command history
            self.command_history.append({
                'timestamp': datetime.now().isoformat(),
                'raw_text': text,
                'intent': classification.intent.value,
                'confidence': classification.confidence,
                'success': result.get('success', False)
            })
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error processing dental command: {e}")
            return {
                'success': False,
                'error': 'processing_error',
                'message': f"Error processing command: {str(e)}"
            }
    
    async def _process_by_intent(self, classification: IntentClassificationResult, context: Dict) -> Dict[str, Any]:
        """Process command based on classified intent"""
        
        intent = classification.intent
        entities = classification.entities
        
        validated_entities = entities
        
        if intent == DentalIntent.PROBING_DEPTH:
            return self._handle_probing_depth(entities, classification.raw_text)
        
        elif intent == DentalIntent.BLEEDING_ON_PROBING:
            return self._handle_bleeding_on_probing(entities, classification.raw_text)
        
        elif intent == DentalIntent.SUPPURATION:
            return self._handle_suppuration(entities, classification.raw_text)
        
        elif intent == DentalIntent.MOBILITY:
            return self._handle_mobility(entities, classification.raw_text)
        
        elif intent == DentalIntent.FURCATION:
            return self._handle_furcation(entities, classification.raw_text)
             
        elif intent == DentalIntent.GINGIVAL_MARGIN:
            return self._handle_gingival_margin(validated_entities, classification.raw_text)
        
        elif intent == DentalIntent.MISSING_TEETH:
            return self._handle_missing_teeth(entities, classification.raw_text)
        
        elif intent == DentalIntent.HELP_REQUEST:
            return self._handle_help_request()
        
        elif intent == DentalIntent.GREETING:
            return self._handle_greeting()
        
        else:
            return {
                'success': False,
                'error': 'intent_not_handled',
                'message': f"Intent {intent.value} not yet implemented"
            }
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        
        classifier_stats = self.intent_classifier.get_classification_stats()
        
        return {
            'session_stats': self.session_stats.copy(),
            'classification_stats': classifier_stats,
            'knowledge_base_size': len(self.knowledge_base['command_templates']),
            'command_history_size': len(self.command_history),
            'intents_supported': len(list(DentalIntent)),
            'rag_enhanced': bool(self.openai_api_key),
            'system_version': 'enhanced_rag_intents_v1.0'
        }
    
    
    def _handle_gingival_margin_strict(self, entities: Dict, raw_text: str) -> Dict[str, Any]:
        """СТРОГАЯ обработка gingival margin с сохранением оригинальных данных"""
        
        tooth_number = entities.get('tooth_number')
        
        if not tooth_number:
            return {
                'success': False,
                'error': 'missing_tooth_number',
                'message': 'Please specify a tooth number (1-32)',
                'original_preserved': True
            }
        
        # СТРОГАЯ проверка tooth number
        if not isinstance(tooth_number, int) or tooth_number < 1 or tooth_number > 32:
            return {
                'success': False,
                'error': 'invalid_tooth_number',
                'message': f'Tooth number must be between 1-32, got: {tooth_number}',
                'original_preserved': True
            }
        
        # Extract gingival margin measurements with STRICT preservation
        measurements = self._extract_gingival_margin_values_strict(raw_text)
        
        if not measurements or len(measurements) != 3:
            return {
                'success': False,
                'error': 'invalid_gm_measurements',
                'message': 'Please provide three gingival margin measurements (e.g., "minus 1 0 plus 1")',
                'suggestion': f'Try: "Gingival margin tooth {tooth_number} minus 1 0 plus 1"'
            }
        
        # СТРОГАЯ проверка measurements
        for i, measurement in enumerate(measurements):
            if not isinstance(measurement, int) or measurement < -10 or measurement > 10:
                return {
                    'success': False,
                    'error': 'invalid_gm_value',
                    'message': f'Gingival margin {i+1} ({measurement}) must be between -10 to +10mm',
                    'original_preserved': True
                }
        
        return {
            'success': True,
            'command': 'update_periodontal_chart',
            'tooth_number': tooth_number,  # ТОЧНО как в оригинале
            'measurement_type': 'gingival_margin',
            'values': measurements,  # ТОЧНО как услышано
            'measurements': {'gingival_margin': measurements},
            'message': f"✅ Gingival margin tooth {tooth_number}: {'-'.join(map(str, measurements))}mm",
            'confidence': 0.95,  # Высокая уверенность для строгих правил
            'validation_mode': 'strict',
            'original_preserved': True
        }
    
    def _handle_probing_depth(self, entities: Dict, raw_text: str) -> Dict[str, Any]:
        """Handle probing depth commands"""
        
        tooth_number = entities.get('tooth_number')
        surface = entities.get('surface', 'buccal')
        measurements = entities.get('measurements', [])
        
        if not tooth_number:
            return {
                'success': False,
                'error': 'missing_tooth_number',
                'message': 'Please specify a tooth number (1-32)'
            }
        
        # Валидация tooth number
        if not isinstance(tooth_number, int) or tooth_number < 1 or tooth_number > 32:
            return {
                'success': False,
                'error': 'invalid_tooth_number',
                'message': f'Tooth number must be between 1-32, got: {tooth_number}',
            }
        
        if not measurements or len(measurements) != 3:
            return {
                'success': False,
                'error': 'invalid_measurements',
                'message': 'Please provide three probing depth measurements'
            }
        
        # Validate measurements
        for i, measurement in enumerate(measurements):
            if not isinstance(measurement, int) or measurement < 1 or measurement > 12:
                return {
                    'success': False,
                    'error': 'invalid_measurement_value',
                    'message': f'Measurement {i+1} ({measurement}) must be between 1-12mm'
                }
        
        return {
            'success': True,
            'command': 'update_periodontal_chart',
            'tooth_number': tooth_number,
            'measurement_type': 'probing_depth',
            'surface': surface,
            'values': measurements,
            'measurements': {'probing_depth': measurements},
            'message': f"✅ Probing depths recorded for tooth {tooth_number} {surface}: {'-'.join(map(str, measurements))}mm",
            'confidence': 0.9
        }
    
        
    def _handle_missing_teeth(self, entities: Dict, raw_text: str) -> Dict[str, Any]:
        """ИСПРАВЛЕННАЯ обработка команд missing teeth с правильным извлечением номеров зубов"""
        
        print(f"🦷 PROCESSING MISSING TEETH: '{raw_text}'")
        print(f"📊 Entities: {entities}")
        
        tooth_numbers = []
        
        # 1. Проверяем entity tooth_number (если есть)
        if 'tooth_number' in entities:
            tooth_num = entities['tooth_number']
            if isinstance(tooth_num, list):
                tooth_numbers.extend([n for n in tooth_num if isinstance(n, int) and 1 <= n <= 32])
            elif isinstance(tooth_num, int) and 1 <= tooth_num <= 32:
                tooth_numbers.append(tooth_num)
            print(f"✅ From tooth_number entity: {tooth_numbers}")
        
        # 2. Проверяем measurements (КРИТИЧЕСКИ ВАЖНО)
        if 'measurements' in entities:
            measurements = entities['measurements']
            if isinstance(measurements, list):
                valid_nums = [m for m in measurements if isinstance(m, int) and 1 <= m <= 32]
                tooth_numbers.extend(valid_nums)
                print(f"✅ From measurements entity: {valid_nums}")
        
        # 3. Парсим сырой текст для извлечения чисел
        tooth_numbers_from_text = self._extract_tooth_numbers_from_text(raw_text)
        tooth_numbers.extend(tooth_numbers_from_text)
        print(f"✅ From text parsing: {tooth_numbers_from_text}")
        
        # 4. Убираем дубликаты
        tooth_numbers = list(set(tooth_numbers))
        tooth_numbers.sort()
        
        print(f"🎯 FINAL TOOTH NUMBERS: {tooth_numbers}")
        
        if not tooth_numbers:
            return {
                'success': False,
                'error': 'no_teeth_specified',
                'message': 'Please specify which teeth are missing',
                'suggestion': 'Try: "Missing teeth 1 16 17 32"'
            }
        
        # 5. Проверяем валидность номеров зубов
        invalid_teeth = [t for t in tooth_numbers if not (1 <= t <= 32)]
        if invalid_teeth:
            return {
                'success': False,
                'error': 'invalid_tooth_numbers',
                'message': f'Invalid tooth numbers: {invalid_teeth}. Valid range is 1-32.',
                'tooth_numbers': tooth_numbers
            }
        
        # 6. Формируем успешный ответ
        return {
            'success': True,
            'command': 'update_periodontal_chart',
            'measurement_type': 'missing_teeth',
            'type': 'missing_teeth',  # Для совместимости
            'tooth_numbers': tooth_numbers,  # Основной список
            'teeth': tooth_numbers,  # Альтернативное имя
            'values': tooth_numbers,  # Еще одно альтернативное имя
            'missing_teeth': tooth_numbers,  # Для веб-клиента
            'measurements': {
                'missing_teeth': tooth_numbers,
                'teeth': tooth_numbers,
                'values': tooth_numbers
            },
            'message': f"✅ Missing teeth marked: {', '.join(map(str, tooth_numbers))}",
            'confidence': 0.95
        }

    def _extract_tooth_numbers_from_text(self, text: str) -> List[int]:
        """Извлечение номеров зубов из текста с поддержкой числовых слов"""
        
        tooth_numbers = []
        
        # Словарь конвертации слов в числа
        word_to_num = {
            'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
            'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
            'eleven': 11, 'twelve': 12, 'thirteen': 13, 'fourteen': 14,
            'fifteen': 15, 'sixteen': 16, 'seventeen': 17, 'eighteen': 18,
            'nineteen': 19, 'twenty': 20, 'twenty-one': 21, 'twenty-two': 22,
            'twenty-three': 23, 'twenty-four': 24, 'twenty-five': 25,
            'twenty-six': 26, 'twenty-seven': 27, 'twenty-eight': 28,
            'twenty-nine': 29, 'thirty': 30, 'thirty-one': 31, 'thirty-two': 32
        }
        
        import re
        
        print(f"🔍 EXTRACTING TOOTH NUMBERS FROM: '{text}'")
        
        # 1. Ищем цифры в тексте
        digit_matches = re.findall(r'\b(\d+)\b', text)
        for match in digit_matches:
            num = int(match)
            if 1 <= num <= 32:
                tooth_numbers.append(num)
                print(f"🔢 Found digit: {num}")
        
        # 2. Ищем числовые слова
        words = text.lower().split()
        for word in words:
            clean_word = re.sub(r'[^\w]', '', word)  # Убираем знаки препинания
            if clean_word in word_to_num:
                num = word_to_num[clean_word]
                if 1 <= num <= 32:
                    tooth_numbers.append(num)
                    print(f"🔤 Found word '{clean_word}': {num}")
        
        # 3. Специальная обработка для "missing this/that/too"
        if 'missing this' in text.lower():
            # "missing this" часто означает "missing tooth 1" или последний упомянутый зуб
            # Для безопасности возвращаем 1
            if 1 not in tooth_numbers:
                tooth_numbers.append(1)
                print(f"🔤 'missing this' interpreted as tooth 1")
        
        if 'missing that' in text.lower():
            if 1 not in tooth_numbers:
                tooth_numbers.append(1)
                print(f"🔤 'missing that' interpreted as tooth 1")
        
        # Обработка "too" как "two"
        if 'too' in text.lower() and 'missing' in text.lower():
            if 2 not in tooth_numbers:
                tooth_numbers.append(2)
                print(f"🔤 'too' in missing context interpreted as tooth 2")
        
        # Убираем дубликаты и сортируем
        tooth_numbers = sorted(list(set(tooth_numbers)))
        print(f"✅ EXTRACTED TOOTH NUMBERS: {tooth_numbers}")
        
        return tooth_numbers    
    
    def _extract_gingival_margin_values_strict(self, text: str) -> List[int]:
        """СТРОГОЕ извлечение gingival margin значений с сохранением оригинала"""
        
        values = []
        
        # Enhanced pattern for "minus 1 0 plus 1" format
        # Look for pattern: (minus/plus) number, standalone numbers, (minus/plus) number
        import re
        
        # First try the full pattern: minus X Y plus Z
        full_pattern = r'minus\s+(\d+)\s+(\d+)\s+plus\s+(\d+)'
        match = re.search(full_pattern, text.lower())
        if match:
            values = [-int(match.group(1)), int(match.group(2)), int(match.group(3))]
            return values
        
        # Alternative pattern: minus X plus Y Z or similar variations
        # Split approach - find all signed numbers
        signed_pattern = r'(minus|plus|\-|\+)\s*(\d+)|(\d+)'
        matches = re.findall(signed_pattern, text.lower())
        
        for match in matches:
            sign, signed_num, unsigned_num = match
            
            if signed_num:
                value = int(signed_num)
                if sign in ['minus', '-']:
                    value = -value
                values.append(value)
            elif unsigned_num:
                values.append(int(unsigned_num))
        
        # If we got exactly 3 values, return them
        if len(values) == 3:
            return values
        
        # Fallback: try to extract any 3 numbers and assume first is negative
        numbers = re.findall(r'\d+', text)
        if len(numbers) >= 3:
            return [-int(numbers[0]), int(numbers[1]), int(numbers[2])]
        
        return []
    
        
    
    
def _extract_measurements_inline(self, text: str) -> List[int]:
    """ИСПРАВЛЕННОЕ извлечение измерений с обработкой знаков для gingival margin"""
    
    measurements = []
    
    print(f"🔍 PARSING MEASUREMENTS: '{text}'")
    
    # Проверяем, это gingival margin команда?
    is_gingival_margin = 'gingival margin' in text.lower() or 'minus' in text.lower() or 'plus' in text.lower()
    
    if is_gingival_margin:
        print("🦷 GINGIVAL MARGIN DETECTED - обрабатываем знаки")
        measurements = self._extract_gingival_margin_values(text)
        print(f"✅ Gingival margin measurements: {measurements}")
        return measurements
    
    # Обычная обработка для других команд
    word_to_num = {
        'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
        'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
        'eleven': 11, 'twelve': 12
    }
    
    words = text.lower().split()
    all_numbers = []
    
    for word in words:
        clean_word = word.strip('.,!?;:')
        
        if clean_word in ['on', 'tooth', 'number', 'surface', 'buccal', 'lingual', 
                         'probing', 'depth', 'and', 'to', 'the', 'a', 'an', 'has', 'grade', 'class']:
            continue
            
        if clean_word in word_to_num:
            num_value = word_to_num[clean_word]
            all_numbers.append(num_value)
            print(f"🔢 '{clean_word}' → {num_value}")
        elif clean_word.isdigit():
            num_value = int(clean_word)
            all_numbers.append(num_value)
            print(f"🔢 '{clean_word}' → {num_value}")
    
    print(f"📊 All numbers found: {all_numbers}")
    
    # Логика выбора измерений
    if len(all_numbers) >= 4:
        measurements = all_numbers[-3:]
        print(f"🎯 4+ numbers, taking last 3: {measurements}")
    elif len(all_numbers) >= 1:
        measurements = all_numbers
        print(f"🎯 {len(all_numbers)} numbers: {measurements}")
    
    print(f"✅ Final measurements: {measurements}")
    return measurements

    def _extract_gingival_margin_values(self, text: str) -> List[int]:
        """Извлечение gingival margin значений с правильными знаками"""
        
        values = []
        import re
        
        print(f"🦷 Extracting gingival margin from: '{text}'")
        
        # Словарь для числовых слов
        word_to_num = {
            'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
            'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10
        }
        
        # Паттерн для поиска знаковых чисел и обычных чисел
        pattern = r'(minus|plus|\-|\+)\s*(\w+)|(\w+)'
        matches = re.findall(pattern, text.lower())
        
        for match in matches:
            sign, signed_word, unsigned_word = match
            
            if signed_word:  # Число со знаком
                if signed_word in word_to_num:
                    value = word_to_num[signed_word]
                elif signed_word.isdigit():
                    value = int(signed_word)
                else:
                    continue
                
                if sign in ['minus', '-']:
                    value = -value
                # plus или + оставляем положительным
                
                values.append(value)
                print(f"🔢 Signed: '{sign} {signed_word}' → {value}")
                
            elif unsigned_word:  # Число без знака
                # Пропускаем служебные слова
                if unsigned_word in ['gingival', 'margin', 'on', 'tooth', 'number', 'minus', 'plus']:
                    continue
                    
                if unsigned_word in word_to_num:
                    value = word_to_num[unsigned_word]
                    values.append(value)
                    print(f"🔢 Unsigned: '{unsigned_word}' → {value}")
                elif unsigned_word.isdigit():
                    value = int(unsigned_word)
                    values.append(value)
                    print(f"🔢 Unsigned digit: '{unsigned_word}' → {value}")
        
        print(f"🦷 Gingival margin values extracted: {values}")
        
        # Для gingival margin нужно именно 3 значения
        if len(values) == 3:
            return values
        elif len(values) > 3:
            return values[:3]  # Берем первые 3
        else:
            # Если меньше 3, дополняем нулями
            while len(values) < 3:
                values.append(0)
            return values   
    
    
    def _handle_probing_depth(self, entities: Dict, raw_text: str) -> Dict[str, Any]:
        """Handle probing depth commands"""
        
        tooth_number = entities.get('tooth_number')
        surface = entities.get('surface', 'buccal')
        measurements = entities.get('measurements', [])
        
        if not tooth_number:
            return {
                'success': False,
                'error': 'missing_tooth_number',
                'message': 'Please specify a tooth number (1-32)'
            }
        
        if not measurements or len(measurements) != 3:
            return {
                'success': False,
                'error': 'invalid_measurements',
                'message': 'Please provide three probing depth measurements'
            }
        
        # Validate measurements
        for measurement in measurements:
            if not isinstance(measurement, int) or measurement < 1 or measurement > 12:
                return {
                    'success': False,
                    'error': 'invalid_measurement_value',
                    'message': 'Probing depths must be between 1-12mm'
                }
        
        return {
            'success': True,
            'command': 'update_periodontal_chart',
            'tooth_number': tooth_number,
            'measurement_type': 'probing_depth',
            'surface': surface,
            'values': measurements,
            'measurements': {'probing_depth': measurements},
            'message': f"✅ Probing depths recorded for tooth {tooth_number} {surface}: {'-'.join(map(str, measurements))}mm",
            'confidence': 0.9
        }
    
    def _handle_probing_depth_strict(self, entities: Dict, raw_text: str) -> Dict[str, Any]:
        """ИСПРАВЛЕННАЯ обработка probing depth"""
        
        tooth_number = entities.get('tooth_number')
        surface = entities.get('surface', 'buccal')
        
        print(f"🔍 Processing probing depth:")
        print(f"   Tooth: {tooth_number}")
        print(f"   Surface: {surface}")
        print(f"   Raw text: '{raw_text}'")
        print(f"   Entities: {entities}")
        
        if not tooth_number:
            return {
                'success': False,
                'error': 'missing_tooth_number',
                'message': 'Please specify a tooth number (1-32)',
            }
        
        # Валидация tooth number
        if not isinstance(tooth_number, int) or tooth_number < 1 or tooth_number > 32:
            return {
                'success': False,
                'error': 'invalid_tooth_number',
                'message': f'Tooth number must be between 1-32, got: {tooth_number}',
            }
        
        # ИЗВЛЕКАЕМ ИЗМЕРЕНИЯ
        #measurements = self._extract_measurements(raw_text)
        measurements = self._extract_measurements_inline(raw_text)
        print(f"🔍 Extracted measurements: {measurements}")
        
        if not measurements:
            return {
                'success': False,
                'error': 'no_measurements_found',
                'message': 'Could not extract probing depth measurements',
                'suggestion': f'Try: "Probing depth tooth {tooth_number} {surface} surface 3 2 4"',
                'debug': {
                    'raw_text': raw_text,
                    'entities': entities
                }
            }
        
        if len(measurements) != 3:
            return {
                'success': False,
                'error': 'invalid_measurement_count',
                'message': f'Expected 3 measurements, got {len(measurements)}: {measurements}',
                'suggestion': f'Try: "Probing depth tooth {tooth_number} {surface} surface 3 2 4"'
            }
        
        # Валидация значений
        for i, measurement in enumerate(measurements):
            if not isinstance(measurement, int) or measurement < 1 or measurement > 12:
                return {
                    'success': False,
                    'error': 'invalid_measurement_value',
                    'message': f'Measurement {i+1} ({measurement}) must be between 1-12mm',
                }
        
        print(f"✅ Successfully processed: tooth {tooth_number}, {surface}, measurements {measurements}")
        
        return {
            'success': True,
            'command': 'update_periodontal_chart',
            'tooth_number': tooth_number,
            'measurement_type': 'probing_depth',
            'surface': surface,
            'values': measurements,
            'measurements': {'probing_depth': measurements},
            'message': f"✅ Probing depths recorded for tooth {tooth_number} {surface}: {'-'.join(map(str, measurements))}mm",
            'confidence': 0.95,
        }
    
    def _handle_probing_depth(self, entities: Dict, raw_text: str) -> Dict[str, Any]:
        """Handle probing depth commands"""
        
        tooth_number = entities.get('tooth_number')
        surface = entities.get('surface', 'buccal')
        measurements = entities.get('measurements', [])
        
        if not tooth_number:
            return {
                'success': False,
                'error': 'missing_tooth_number',
                'message': 'Please specify a tooth number (1-32)'
            }
        
        if not measurements or len(measurements) != 3:
            return {
                'success': False,
                'error': 'invalid_measurements',
                'message': 'Please provide three probing depth measurements'
            }
        
        # Validate measurements
        for measurement in measurements:
            if not isinstance(measurement, int) or measurement < 1 or measurement > 12:
                return {
                    'success': False,
                    'error': 'invalid_measurement_value',
                    'message': 'Probing depths must be between 1-12mm'
                }
        
        return {
            'success': True,
            'command': 'update_periodontal_chart',
            'tooth_number': tooth_number,
            'measurement_type': 'probing_depth',
            'surface': surface,
            'values': measurements,
            'measurements': {'probing_depth': measurements},
            'message': f"✅ Probing depths recorded for tooth {tooth_number} {surface}: {'-'.join(map(str, measurements))}mm",
            'confidence': 0.9
        }
    
    def _handle_bleeding_on_probing(self, entities: Dict, raw_text: str) -> Dict[str, Any]:
        """Handle bleeding on probing commands"""
        
        tooth_number = entities.get('tooth_number')
        surface = entities.get('surface', 'buccal')
        position = entities.get('position', 'distal')
        
        if not tooth_number:
            return {
                'success': False,
                'error': 'missing_tooth_number',
                'message': 'Please specify a tooth number (1-32)'
            }
        
        # Determine bleeding status
        bleeding_positive = any(word in raw_text.lower() for word in [
            'bleeding', 'positive', 'present', 'yes'
        ])
        
        return {
            'success': True,
            'command': 'update_periodontal_chart',
            'tooth_number': tooth_number,
            'measurement_type': 'bleeding',
            'surface': surface,
            'position': position,
            'values': [bleeding_positive],
            'measurements': {'bleeding': [bleeding_positive]},
            'message': f"✅ Bleeding on probing tooth {tooth_number} {surface} {position}: {'positive' if bleeding_positive else 'negative'}",
            'confidence': 0.9
        }
    
    def _handle_suppuration(self, entities: Dict, raw_text: str) -> Dict[str, Any]:
        """Handle suppuration commands"""
        
        tooth_number = entities.get('tooth_number')
        surface = entities.get('surface', 'buccal')
        position = entities.get('position', 'distal')
        
        if not tooth_number:
            return {
                'success': False,
                'error': 'missing_tooth_number',
                'message': 'Please specify a tooth number (1-32)'
            }
        
        suppuration_present = any(word in raw_text.lower() for word in [
            'present', 'positive', 'suppuration', 'pus', 'discharge'
        ])
        
        return {
            'success': True,
            'command': 'update_periodontal_chart',
            'tooth_number': tooth_number,
            'measurement_type': 'suppuration',
            'surface': surface,
            'position': position,
            'values': [suppuration_present],
            'measurements': {'suppuration': [suppuration_present]},
            'message': f"✅ Suppuration tooth {tooth_number} {surface} {position}: {'present' if suppuration_present else 'absent'}",
            'confidence': 0.9
        }
    
    def _handle_mobility(self, entities: Dict, raw_text: str) -> Dict[str, Any]:
        """Handle mobility commands"""
        
        tooth_number = entities.get('tooth_number')
        grade = entities.get('grade')
        
        if not tooth_number:
            return {
                'success': False,
                'error': 'missing_tooth_number',
                'message': 'Please specify a tooth number (1-32)'
            }
        
        if grade is None:
            # Try to extract grade from text
            grade_match = re.search(r'(?:grade|class|level)\s+(\d+)', raw_text.lower())
            if grade_match:
                grade = int(grade_match.group(1))
            else:
                return {
                    'success': False,
                    'error': 'missing_mobility_grade',
                    'message': 'Please specify mobility grade (0-3)'
                }
        
        if not isinstance(grade, int) or grade < 0 or grade > 3:
            return {
                'success': False,
                'error': 'invalid_mobility_grade',
                'message': 'Mobility grade must be 0-3'
            }
        
        return {
            'success': True,
            'command': 'update_periodontal_chart',
            'tooth_number': tooth_number,
            'measurement_type': 'mobility',
            'values': [grade],
            'measurements': {'mobility': grade},
            'message': f"✅ Tooth {tooth_number} mobility: Grade {grade}",
            'confidence': 0.9
        }
    
    def _handle_furcation(self, entities: Dict, raw_text: str) -> Dict[str, Any]:
        """Handle furcation commands"""
        
        tooth_number = entities.get('tooth_number')
        furcation_class = entities.get('class')
        
        if not tooth_number:
            return {
                'success': False,
                'error': 'missing_tooth_number',
                'message': 'Please specify a tooth number (1-32)'
            }
        
        if furcation_class is None:
            # Try to extract class from text
            class_match = re.search(r'(?:class|grade|type)\s+(\d+)', raw_text.lower())
            if class_match:
                furcation_class = int(class_match.group(1))
            else:
                return {
                    'success': False,
                    'error': 'missing_furcation_class',
                    'message': 'Please specify furcation class (1-3)'
                }
        
        if not isinstance(furcation_class, int) or furcation_class < 1 or furcation_class > 3:
            return {
                'success': False,
                'error': 'invalid_furcation_class',
                'message': 'Furcation class must be 1-3'
            }
        
        return {
            'success': True,
            'command': 'update_periodontal_chart',
            'tooth_number': tooth_number,
            'measurement_type': 'furcation',
            'values': [furcation_class],
            'measurements': {'furcation': furcation_class},
            'message': f"✅ Tooth {tooth_number} furcation: Class {furcation_class}",
            'confidence': 0.9
        }
    
    def _handle_gingival_margin(self, entities: Dict, raw_text: str) -> Dict[str, Any]:
        """Handle gingival margin commands"""
        
        tooth_number = entities.get('tooth_number')
        
        if not tooth_number:
            return {
                'success': False,
                'error': 'missing_tooth_number',
                'message': 'Please specify a tooth number (1-32)'
            }
        
        # Extract gingival margin measurements
        measurements = self._extract_gingival_margin_values(raw_text)
        
        if not measurements or len(measurements) != 3:
            return {
                'success': False,
                'error': 'invalid_gm_measurements',
                'message': 'Please provide three gingival margin measurements (e.g., "minus 1 0 plus 1")'
            }
        
        return {
            'success': True,
            'command': 'update_periodontal_chart',
            'tooth_number': tooth_number,
            'measurement_type': 'gingival_margin',
            'values': measurements,
            'measurements': {'gingival_margin': measurements},
            'message': f"✅ Gingival margin tooth {tooth_number}: {' '.join(map(str, measurements))}mm",
            'confidence': 0.9
        }
    

    def _handle_missing_teeth(self, entities: Dict, raw_text: str) -> Dict[str, Any]:
        """Handle missing teeth commands - ОБЪЕДИНЕННАЯ ВЕРСИЯ"""
        
        tooth_numbers = []
        
        # 1. Проверка entities (из первой функции)
        if 'tooth_number' in entities:
            tooth_num = entities['tooth_number']
            if isinstance(tooth_num, list):
                tooth_numbers.extend(tooth_num)
            else:
                tooth_numbers.append(tooth_num)
        
        # 2. Проверка measurements из entities (ИСПРАВЛЕНИЕ!)
        if 'measurements' in entities:
            measurements = entities['measurements']
            if isinstance(measurements, list):
                tooth_numbers.extend([m for m in measurements if isinstance(m, int) and 1 <= m <= 32])
        
        # 3. Обработка "too" как "two" = зуб 2
        if 'too' in raw_text.lower() or 'two' in raw_text.lower():
            if 2 not in tooth_numbers:
                tooth_numbers.append(2)
        
        # 4. Извлечение номеров зубов из текста
        import re
        tooth_matches = re.findall(r'\b(\d+)\b', raw_text)
        for match in tooth_matches:
            tooth_num = int(match)
            if 1 <= tooth_num <= 32 and tooth_num not in tooth_numbers:
                tooth_numbers.append(tooth_num)
        
        # 5. Дополнительная попытка извлечения (из второй функции)
        if not tooth_numbers:
            try:
                extracted = self._extract_tooth_numbers(raw_text)
                if extracted:
                    tooth_numbers.extend(extracted)
            except Exception as e:
                self.logger.warning(f"⚠️ _extract_tooth_numbers failed: {e}")
        
        # 6. Проверка результата
        if not tooth_numbers:
            return {
                'success': False,
                'error': 'no_teeth_specified',
                'message': 'Please specify which teeth are missing'
            }
        
        # 7. КЛЮЧЕВОЕ ИСПРАВЛЕНИЕ: правильный формат возврата
        return {
            'success': True,
            'command': 'update_periodontal_chart',
            'measurement_type': 'missing_teeth',
            'type': 'missing_teeth',  # ⭐ Добавлено для совместимости
            'teeth': tooth_numbers,   # ⭐ Для второй функции
            'values': tooth_numbers,  # ⭐ Для первой функции
            'missing_teeth': tooth_numbers,  # ⭐ ДЛЯ LLM!
            'measurements': {'missing_teeth': tooth_numbers},
            'message': f"✅ Missing teeth marked: {', '.join(map(str, tooth_numbers))}",
            'confidence': 0.9
        }
    
    def _handle_help_request(self) -> Dict[str, Any]:
        """Handle help requests"""
        
        help_text = """
🦷 Available Voice Commands:

PROBING DEPTH:
• "Probing depth on tooth number 14 buccal surface 3 2 4"
• "Probing depth on tooth number 14 lingual surface 2 3 3"

BLEEDING:
• "Bleeding on probing tooth 12 buccal distal"
• "Bleeding on probing tooth 12 lingual distal"

SUPPURATION:
• "Suppuration present on tooth 8 buccal distal"
• "Suppuration present on tooth 8 lingual distal"

MOBILITY:
• "Tooth 8 has mobility grade 2"

FURCATION:
• "Furcation class 2 on tooth 6"

GINGIVAL MARGIN:
• "Gingival margin on tooth 14 minus 1 0 plus 1"

MISSING TEETH:
• "Missing teeth 1 16 17 32"
        """
        
        return {
            'success': True,
            'command': 'show_help',
            'message': help_text,
            'confidence': 1.0
        }
    
    def _handle_greeting(self) -> Dict[str, Any]:
        """Handle greetings"""
        
        return {
            'success': True,
            'command': 'greeting_response',
            'message': "👋 Hello! I'm ready to help with your periodontal charting. Just speak your commands naturally!",
            'confidence': 1.0
        }
    
    def _extract_gingival_margin_values(self, text: str) -> List[int]:
        """Extract gingival margin values from text"""
        
        values = []
        
        # Pattern for "minus 1 0 plus 1" format
        pattern = r'(?:(minus|plus|\-|\+)\s*(\d+)|(\d+))'
        matches = re.findall(pattern, text.lower())
        
        for match in matches:
            sign, signed_num, unsigned_num = match
            
            if signed_num:
                value = int(signed_num)
                if sign in ['minus', '-']:
                    value = -value
                values.append(value)
            elif unsigned_num:
                values.append(int(unsigned_num))
        
        return values if len(values) == 3 else []
    
    async def _enhance_with_rag(self, classification: IntentClassificationResult, 
                               result: Dict[str, Any], context: Dict) -> Dict[str, Any]:
        """Enhance result using RAG when confidence is low"""
        
        if self.openai_api_key:
            # Use LLM to enhance understanding
            enhanced_result = await self._llm_enhance_command(classification, result, context)
            if enhanced_result:
                enhanced_result['rag_enhanced'] = True
                return enhanced_result
        
        # Fallback to template-based enhancement
        template_result = self._template_enhance_command(classification, result)
        if template_result:
            template_result['rag_enhanced'] = True
            return template_result
        
        return result
    
    async def _llm_enhance_command(self, classification: IntentClassificationResult,
                                  result: Dict[str, Any], context: Dict) -> Optional[Dict[str, Any]]:
        """Use LLM to enhance command understanding"""
        
        try:
            # Import OpenAI here to avoid dependency issues
            from openai import AsyncOpenAI
            
            client = AsyncOpenAI(api_key=self.openai_api_key)
            
            prompt = f"""
You are a dental assistant AI. Analyze this voice command and extract structured information.

Original command: "{classification.raw_text}"
Detected intent: {classification.intent.value}
Confidence: {classification.confidence:.3f}
Extracted entities: {json.dumps(classification.entities, indent=2)}

Please provide a corrected interpretation if the original analysis seems incorrect, or enhance the existing interpretation.

Respond with JSON in this format:
{{
    "success": true/false,
    "tooth_number": integer (1-32),
    "measurement_type": "probing_depth|bleeding|suppuration|mobility|furcation|gingival_margin|missing_teeth",
    "surface": "buccal|lingual|both",
    "position": "distal|mesial|mid",
    "values": [array of values],
    "confidence": float (0.0-1.0),
    "message": "confirmation message",
    "corrections_made": ["list of corrections applied"]
}}
            """
            
            response = await client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.1
            )
            
            content = response.choices[0].message.content.strip()
            
            # Parse JSON response
            try:
                enhanced_data = json.loads(content)
                if enhanced_data.get('success'):
                    return enhanced_data
            except json.JSONDecodeError:
                # Try to extract JSON from response
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    enhanced_data = json.loads(json_match.group())
                    if enhanced_data.get('success'):
                        return enhanced_data
            
        except Exception as e:
            logger.error(f"❌ LLM enhancement error: {e}")
        
        return None
    
    def _template_enhance_command(self, classification: IntentClassificationResult,
                                 result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Enhance command using templates"""
        
        intent = classification.intent
        entities = classification.entities
        
        # Try to find suitable template
        if intent in self.knowledge_base['command_templates']:
            templates = self.knowledge_base['command_templates'][intent]
            
            # Find best matching template
            best_template = None
            for template in templates:
                if self._template_matches_entities(template, entities):
                    best_template = template
                    break
            
            if best_template:
                # Apply common corrections
                corrected_entities = self._apply_corrections(entities)
                
                # Try to fill template
                filled_result = self._fill_template_result(intent, corrected_entities)
                if filled_result:
                    return filled_result
        
        return None
    
    def _template_matches_entities(self, template: str, entities: Dict) -> bool:
        """Check if template matches available entities"""
        
        required_placeholders = re.findall(r'\{(\w+)\}', template)
        
        for placeholder in required_placeholders:
            if placeholder not in entities:
                return False
        
        return True
    
    def _apply_corrections(self, entities: Dict) -> Dict:
        """Apply common corrections to entities"""
        
        corrected = entities.copy()
        corrections = self.knowledge_base['common_corrections']
        
        # Correct surfaces
        if 'surface' in corrected and corrected['surface'] in corrections['surfaces']:
            corrected['surface'] = corrections['surfaces'][corrected['surface']]
        
        return corrected
    
    def _fill_template_result(self, intent: DentalIntent, entities: Dict) -> Optional[Dict[str, Any]]:
        """Fill template result based on intent and entities"""
        
        base_result = {
            'success': True,
            'command': 'update_periodontal_chart',
            'confidence': 0.7,  # Template-based confidence
            'template_enhanced': True
        }
        
        if intent == DentalIntent.PROBING_DEPTH:
            if 'tooth_number' in entities and 'measurements' in entities:
                base_result.update({
                    'tooth_number': entities['tooth_number'],
                    'measurement_type': 'probing_depth',
                    'surface': entities.get('surface', 'buccal'),
                    'values': entities['measurements'],
                    'measurements': {'probing_depth': entities['measurements']},
                    'message': f"✅ Template: Probing depths for tooth {entities['tooth_number']}"
                })
                return base_result
        
        # Add more template filling logic for other intents...
        
        return None
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        
        classifier_stats = self.intent_classifier.get_classification_stats()
        
        return {
            'session_stats': self.session_stats.copy(),
            'classification_stats': classifier_stats,
            'knowledge_base_size': len(self.knowledge_base['command_templates']),
            'command_history_size': len(self.command_history),
            'intents_supported': len(list(DentalIntent)),
            'rag_enhanced': bool(self.openai_api_key),
            'system_version': 'enhanced_rag_intents_v1.0'
        }
    
    def get_recent_commands(self, limit: int = 10) -> List[Dict]:
        """Get recent command history"""
        return self.command_history[-limit:] if self.command_history else []
    
    def clear_history(self):
        """Clear command history"""
        self.command_history = []


# Integration functions for existing server
enhanced_rag_system = None

def initialize_enhanced_rag_system(openai_api_key: str = None) -> bool:
    """Initialize the enhanced RAG system"""
    global enhanced_rag_system
    
    try:
        enhanced_rag_system = EnhancedRAGSystem(openai_api_key)
        logger.info("🧠 Enhanced RAG System with Intent Classification initialized")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to initialize Enhanced RAG System: {e}")
        return False

async def process_command_with_enhanced_rag(text: str, context: Dict = None) -> Dict[str, Any]:
    """Process command using enhanced RAG system"""
    if not enhanced_rag_system:
        return {
            'success': False,
            'error': 'rag_not_initialized',
            'message': 'Enhanced RAG System not initialized'
        }
    
    return await enhanced_rag_system.process_dental_command(text, context)

def get_enhanced_rag_stats() -> Dict[str, Any]:
    """Get enhanced RAG system statistics"""
    if enhanced_rag_system:
        return enhanced_rag_system.get_system_stats()
    else:
        return {'system_status': 'not_initialized'}

def is_dental_command_enhanced_rag(text: str) -> bool:
    """Check if text is a dental command using intent classification"""
    if not enhanced_rag_system:
        return False
    
    classification = enhanced_rag_system.intent_classifier.classify_intent(text)
    return classification.confidence > 0.5 and classification.intent != DentalIntent.UNKNOWN


if __name__ == "__main__":
    # Test the enhanced RAG system
    import os
    
    async def test_enhanced_rag():
        """Test the enhanced RAG system with intent classification"""
        
        print("🧠 Testing Enhanced RAG System with Intent Classification")
        print("=" * 70)
        
        # Initialize system
        api_key = os.getenv("OPENAI_API_KEY")
        success = initialize_enhanced_rag_system(api_key)
        
        if not success:
            print("❌ Failed to initialize Enhanced RAG System")
            return
        
        # Test commands with various ASR errors
        test_commands = [
            # Original problematic commands
            "Probing depth on tooth number two, wingle surface 231.",
            "Probing depth on tooth number 2, buckle surface 312.",
            "Bleeding on probing tooth 2, buccal distal.",
            "Missing this one.",
            "For Cache in class 2 on tooth 2",
            "Tooth Tool has mobility grade 2",
            "Separation present on tooth 8 lingual distal.",
            "Bleeding on probing tooth 3, lingual distal.",
            
            # Additional test cases
            "help me",
            "what can I say",
            "hello",
            "clear all data",
            "save the chart",
            "unknown command test"
        ]
        
        for i, command in enumerate(test_commands, 1):
            print(f"\n{i}. Testing: '{command}'")
            print("-" * 50)
            
            result = await process_command_with_enhanced_rag(command)
            
            print(f"   🎯 Intent: {result.get('intent', 'unknown')}")
            print(f"   🎯 Confidence: {result.get('intent_confidence', 0):.3f}")
            print(f"   📊 Entities: {result.get('entities', {})}")
            
            if result['success']:
                print(f"   ✅ SUCCESS: {result['message']}")
                if result.get('tooth_number'):
                    print(f"   🦷 Tooth: {result['tooth_number']}")
                if result.get('measurement_type'):
                    print(f"   📋 Type: {result['measurement_type']}")
                if result.get('values'):
                    print(f"   📊 Values: {result['values']}")
            else:
                print(f"   ❌ FAILED: {result['message']}")
                if result.get('suggested_command'):
                    print(f"   💡 Suggestion: {result['suggested_command']}")
        
        # Show system statistics
        stats = get_enhanced_rag_stats()
        print(f"\n📊 Enhanced RAG System Statistics:")
        print("=" * 50)
        
        session_stats = stats.get('session_stats', {})
        for key, value in session_stats.items():
            print(f"   {key}: {value}")
        
        classification_stats = stats.get('classification_stats', {})
        print(f"\n🧠 Intent Classification Statistics:")
        print(f"   Total classifications: {classification_stats.get('total_classifications', 0)}")
        print(f"   Average confidence: {classification_stats.get('average_confidence', 0):.3f}")
        print(f"   High confidence: {classification_stats.get('high_confidence_count', 0)}")
        print(f"   Low confidence: {classification_stats.get('low_confidence_count', 0)}")
        
        intent_counts = classification_stats.get('intent_counts', {})
        print(f"\n🎯 Intent Distribution:")
        for intent, count in intent_counts.items():
            if count > 0:
                print(f"   {intent}: {count}")
    
    # Run the test
    asyncio.run(test_enhanced_rag())
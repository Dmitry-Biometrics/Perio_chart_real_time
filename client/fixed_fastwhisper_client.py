"""
ИСПРАВЛЕННЫЙ FastWhisper Client с повышенной стабильностью
Устраняет проблемы с передачей аудио и падениями системы
ИСПРАВЛЕНА ПРОБЛЕМА С WEBSOCKETS HEADERS
"""

import asyncio
import websockets
import sounddevice as sd
import numpy as np
import threading
import queue
import logging
import time
import json
from datetime import datetime
import signal
import sys
import inspect

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class StableFastWhisperClient:
    def __init__(self, server_uri="ws://3.84.215.173:8765"):
        self.server_uri = server_uri
        self.websocket = None
        
        # СТАБИЛЬНЫЕ Audio параметры (менее агрессивные)
        self.sample_rate = 16000
        self.channels = 1
        # УВЕЛИЧЕННАЯ длительность чанка для стабильности
        self.chunk_duration = 0.25  # 250ms (было 125ms) - более стабильно
        self.chunk_size = int(self.sample_rate * self.chunk_duration)  # 4000 samples
        
        # СТАБИЛЬНЫЕ буферы с защитой от переполнения
        self.audio_queue = queue.Queue(maxsize=60)  # Увеличенный буфер
        self.result_queue = queue.Queue()
        self.is_recording = False
        self.is_connected = False
        self.should_run = True
        
        # Защита от ошибок
        self.error_count = 0
        self.max_errors = 10
        self.reconnect_delay = 1.0
        self.last_error_time = 0
        
        # WebSockets совместимость
        self.headers_param = self._detect_headers_param()
        
        # Статистика с отслеживанием ошибок
        self.stats = {
            'chunks_sent': 0,
            'results_received': 0,
            'connection_errors': 0,
            'audio_errors': 0,
            'processing_errors': 0,
            'session_start': None,
            'total_audio_sent': 0.0,
            'successful_chunks': 0,
            'failed_chunks': 0,
            'average_chunk_size': 0,
            'last_successful_send': None
        }
        
        self.results_history = []
        self.max_history = 50
        
        # Сигналы для корректного завершения
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        logger.info(f"🛡️ STABLE FastWhisper client initialized")
        logger.info(f"📡 Server: {self.server_uri}")
        logger.info(f"🔧 STABLE MODE: {self.chunk_duration}s chunks ({self.chunk_size} samples)")
        logger.info(f"🌐 WebSocket headers: {self.headers_param}")
    
    def _detect_headers_param(self):
        """Автоматически определяет правильный параметр для headers"""
        try:
            # Пробуем новый API
            try:
                from websockets.asyncio.client import connect
                sig = inspect.signature(connect)
                if 'additional_headers' in sig.parameters:
                    logger.info("🆕 Используется новый API с additional_headers")
                    return 'additional_headers'
                elif 'extra_headers' in sig.parameters:
                    logger.info("🔄 Новый API со старой сигнатурой extra_headers")
                    return 'extra_headers'
                else:
                    logger.warning("⚠️ Параметр headers не найден в новом API")
                    return None
                    
            except ImportError:
                # Fallback к legacy API
                from websockets.client import connect
                logger.info("🔄 Используется legacy API с extra_headers")
                return 'extra_headers'
                
        except ImportError:
            logger.error("❌ Не удалось импортировать websockets")
            return None
    
    def signal_handler(self, signum, frame):
        """Обработчик сигналов для корректного завершения"""
        logger.info(f"🛑 Received signal {signum}, shutting down gracefully...")
        self.should_run = False
    
    def list_audio_devices(self):
        """Безопасное получение списка аудио устройств"""
        try:
            devices = sd.query_devices()
            print("\n🎤 Available audio devices:")
            print("=" * 60)
            
            for i, device in enumerate(devices):
                if device['max_input_channels'] > 0:
                    default_marker = " ⭐ (default)" if i == sd.default.device[0] else ""
                    status = "✅" if self.is_device_compatible(device) else "⚠️"
                    
                    print(f"  {i:2d}: {status} {device['name']}{default_marker}")
                    print(f"      📊 Channels: {device['max_input_channels']}")
                    print(f"      🔊 Sample rate: {device['default_samplerate']:.0f}Hz")
                    print(f"      💻 API: {device['hostapi']}")
                    print()
            
            return devices
            
        except Exception as e:
            logger.error(f"Error retrieving devices: {e}")
            return []
    
    def is_device_compatible(self, device):
        """Проверка совместимости устройства"""
        try:
            # Проверяем основные требования
            if device['max_input_channels'] < 1:
                return False
            
            # Проверяем поддерживаемые частоты дискретизации
            compatible_rates = [16000, 44100, 48000]
            device_rate = device['default_samplerate']
            
            return any(abs(device_rate - rate) < 100 for rate in compatible_rates)
            
        except:
            return False
    
    def audio_callback(self, indata, frames, time_info, status):
        """СТАБИЛЬНЫЙ audio callback с обработкой ошибок"""
        try:
            if status:
                if status.input_underflow:
                    logger.debug("⚠️ Audio input underflow")
                if status.input_overflow:
                    self.stats['audio_errors'] += 1
                    if self.stats['audio_errors'] % 50 == 0:  # Логируем каждую 50-ю ошибку
                        logger.warning(f"⚠️ Audio overflows: {self.stats['audio_errors']}")
            
            if not self.is_recording or not self.should_run:
                return
            
            # Безопасная обработка аудио данных
            try:
                # Приведение к моно
                if len(indata.shape) == 2:
                    audio_data = indata[:, 0].copy()
                else:
                    audio_data = indata.copy()
                
                # Проверка размера данных
                if len(audio_data) == 0:
                    return
                
                # Нормализация к ожидаемому размеру
                if len(audio_data) != self.chunk_size:
                    if len(audio_data) > self.chunk_size:
                        # Обрезаем
                        audio_data = audio_data[:self.chunk_size]
                    else:
                        # Дополняем нулями
                        padding = np.zeros(self.chunk_size - len(audio_data))
                        audio_data = np.concatenate([audio_data, padding])
                
                # Проверка на NaN и inf
                if np.any(np.isnan(audio_data)) or np.any(np.isinf(audio_data)):
                    logger.warning("⚠️ Invalid audio data (NaN/inf), skipping chunk")
                    return
                
                # Ограничение динамического диапазона
                audio_data = np.clip(audio_data, -1.0, 1.0)
                
                # Конвертация в int16
                audio_int16 = (audio_data * 32767).astype(np.int16)
                
                # Добавление в очередь с защитой от переполнения
                chunk_time = time.time()
                try:
                    self.audio_queue.put_nowait((audio_int16, chunk_time))
                    self.stats['total_audio_sent'] += self.chunk_duration
                except queue.Full:
                    # Удаляем старые чанки вместо потери новых
                    try:
                        discarded_chunk, _ = self.audio_queue.get_nowait()
                        self.audio_queue.put_nowait((audio_int16, chunk_time))
                        self.stats['failed_chunks'] += 1
                    except queue.Empty:
                        pass
                        
            except Exception as e:
                self.stats['audio_errors'] += 1
                if self.stats['audio_errors'] % 10 == 0:
                    logger.error(f"❌ Audio processing error (#{self.stats['audio_errors']}): {e}")
                
        except Exception as e:
            logger.error(f"❌ Critical audio callback error: {e}")
            self.error_count += 1
    
    def setup_audio_stream(self, device_index=None):
        """СТАБИЛЬНАЯ настройка аудио потока"""
        try:
            # Получение информации об устройстве
            if device_index is not None:
                try:
                    device_info = sd.query_devices(device_index)
                    logger.info(f"🎤 Selected device: {device_info['name']}")
                    
                    if not self.is_device_compatible(device_info):
                        logger.warning(f"⚠️ Device may not be fully compatible")
                    
                except Exception as e:
                    logger.error(f"❌ Invalid device index {device_index}: {e}")
                    device_index = None
            
            # Стабильные конфигурации (от консервативной к более агрессивной)
            configs_to_try = [
                # Конфигурация 1: Максимально стабильная
                {
                    'device': device_index,
                    'channels': self.channels,
                    'samplerate': self.sample_rate,
                    'blocksize': self.chunk_size,
                    'dtype': 'float32',
                    'latency': 'high',  # Высокая латентность для стабильности
                    'name': 'Conservative/Stable'
                },
                # Конфигурация 2: Умеренная
                {
                    'device': device_index,
                    'channels': self.channels,
                    'samplerate': self.sample_rate,
                    'blocksize': self.chunk_size,
                    'dtype': 'float32',
                    'latency': None,
                    'name': 'Moderate'
                },
                # Конфигурация 3: Автоматический размер блока
                {
                    'device': device_index,
                    'channels': self.channels,
                    'samplerate': self.sample_rate,
                    'blocksize': None,  # Автоматический размер
                    'dtype': 'float32',
                    'latency': None,
                    'name': 'Auto blocksize'
                },
                # Конфигурация 4: Частота устройства по умолчанию
                {
                    'device': device_index,
                    'channels': self.channels,
                    'samplerate': None,  # Частота устройства по умолчанию
                    'blocksize': None,
                    'dtype': 'float32',
                    'latency': None,
                    'name': 'Device defaults'
                }
            ]
            
            for i, config in enumerate(configs_to_try):
                try:
                    logger.info(f"🔄 Trying audio config {i+1}/{len(configs_to_try)}: {config['name']}")
                    
                    # Определение фактических параметров
                    actual_samplerate = config['samplerate']
                    if actual_samplerate is None and device_index is not None:
                        device_info = sd.query_devices(device_index)
                        actual_samplerate = int(device_info['default_samplerate'])
                    elif actual_samplerate is None:
                        actual_samplerate = self.sample_rate
                    
                    self.stream = sd.InputStream(
                        device=config['device'],
                        channels=config['channels'],
                        samplerate=actual_samplerate,
                        blocksize=config['blocksize'],
                        dtype=config['dtype'],
                        callback=self.audio_callback,
                        latency=config['latency']
                    )
                    
                    # Тестирование потока
                    self.stream.start()
                    time.sleep(0.1)  # Короткий тест
                    self.stream.stop()
                    
                    # Если дошли сюда - конфигурация работает
                    actual_blocksize = config['blocksize'] or self.chunk_size
                    
                    logger.info(f"✅ STABLE audio stream configured ({config['name']}):")
                    logger.info(f"   📊 Sample rate: {actual_samplerate}Hz")
                    logger.info(f"   📦 Block size: {actual_blocksize} samples")
                    logger.info(f"   ⏱️ Latency: {self.stream.latency}")
                    logger.info(f"   🔧 Device: {device_index if device_index is not None else 'default'}")
                    
                    return True
                    
                except Exception as config_error:
                    logger.warning(f"⚠️ Config {i+1} failed: {config_error}")
                    if hasattr(self, 'stream'):
                        try:
                            self.stream.close()
                        except:
                            pass
                    continue
            
            # Все конфигурации не сработали
            raise Exception("All audio configurations failed")
            
        except Exception as e:
            logger.error(f"❌ Audio setup error: {e}")
            logger.error("💡 Suggestions:")
            logger.error("   1. Try a different audio device")
            logger.error("   2. Check audio drivers")
            logger.error("   3. Close other audio applications")
            logger.error("   4. Run with administrator privileges")
            return False
    
    async def connect_to_server(self):
        """СТАБИЛЬНОЕ подключение с повторными попытками и совместимостью headers"""
        max_retries = 15
        base_delay = 1.0
        
        for attempt in range(max_retries):
            try:
                current_delay = min(base_delay * (1.5 ** attempt), 10.0)
                logger.info(f"🔗 Connection attempt {attempt + 1}/{max_retries}")
                
                # Подготовка параметров подключения
                connect_params = {
                    'ping_interval': 20,     # Увеличено для стабильности
                    'ping_timeout': 10,     # Увеличено
                    'close_timeout': 5,     # Увеличено
                    'max_size': 10*1024*1024,  # Больший буфер
                    'compression': None,    # Отключение сжатия для стабильности
                }
                
                # Добавляем заголовки с правильным параметром
                if self.headers_param:
                    headers = {'User-Agent': 'StableFastWhisperClient/1.0'}
                    connect_params[self.headers_param] = headers
                
                # Выбираем правильную функцию connect
                try:
                    from websockets.asyncio.client import connect
                except ImportError:
                    from websockets.client import connect
                
                self.websocket = await connect(self.server_uri, **connect_params)
                
                self.is_connected = True
                self.stats['session_start'] = datetime.now()
                self.error_count = 0  # Сброс счетчика ошибок
                
                # Проверка соединения
                try:
                    await asyncio.wait_for(self.websocket.send("PING"), timeout=5.0)
                    response = await asyncio.wait_for(self.websocket.recv(), timeout=5.0)
                    
                    if response == "PONG":
                        logger.info(f"✅ STABLE connection established")
                        
                        # Получение информации о сервере
                        try:
                            await self.websocket.send("MODEL_INFO")
                            model_info = await asyncio.wait_for(self.websocket.recv(), timeout=3.0)
                            model_data = json.loads(model_info)
                            logger.info(f"🤖 Server model: {model_data.get('model_size', 'unknown')}")
                            logger.info(f"💻 Server device: {model_data.get('device', 'unknown')}")
                        except:
                            pass
                        
                        return True
                    
                except asyncio.TimeoutError:
                    logger.warning("⚠️ Server response timeout")
                except Exception as e:
                    logger.warning(f"⚠️ Server check failed: {e}")
                
                # Если проверка не прошла, но соединение есть
                logger.info(f"✅ Connection established (server check failed)")
                return True
                
            except Exception as e:
                self.stats['connection_errors'] += 1
                logger.error(f"❌ Connection error (attempt {attempt + 1}): {e}")
                
                if attempt < max_retries - 1:
                    logger.info(f"⏳ Retrying in {current_delay:.1f}s...")
                    await asyncio.sleep(current_delay)
        
        logger.error(f"❌ Failed to connect after {max_retries} attempts")
        return False
    
    async def audio_sender(self):
        """СТАБИЛЬНЫЙ отправитель аудио с защитой от ошибок"""
        logger.info("📤 STABLE audio sender started")
        send_count = 0
        consecutive_errors = 0
        max_consecutive_errors = 5
        
        while self.should_run:
            if not self.is_connected:
                await asyncio.sleep(0.1)
                continue
            
            try:
                # Получение чанка с таймаутом
                try:
                    audio_chunk, chunk_time = self.audio_queue.get(timeout=0.5)
                except queue.Empty:
                    continue
                
                if not self.is_connected or not self.websocket:
                    continue
                
                # Проверка состояния WebSocket
                if self.websocket.closed:
                    logger.warning("🔗 WebSocket closed during sending")
                    self.is_connected = False
                    continue
                
                # Отправка с защитой от ошибок
                try:
                    await asyncio.wait_for(
                        self.websocket.send(audio_chunk.tobytes()), 
                        timeout=2.0
                    )
                    
                    send_count += 1
                    self.stats['chunks_sent'] = send_count
                    self.stats['successful_chunks'] += 1
                    self.stats['last_successful_send'] = time.time()
                    consecutive_errors = 0
                    
                    # Обновление статистики размера чанков
                    chunk_size_bytes = len(audio_chunk.tobytes())
                    if self.stats['average_chunk_size'] == 0:
                        self.stats['average_chunk_size'] = chunk_size_bytes
                    else:
                        alpha = 0.1
                        self.stats['average_chunk_size'] = (
                            alpha * chunk_size_bytes + 
                            (1 - alpha) * self.stats['average_chunk_size']
                        )
                    
                    # Логирование прогресса
                    if send_count % 20 == 0:  # Каждые 5 секунд
                        logger.debug(f"🔧 STABLE: {send_count} chunks sent "
                                   f"({self.stats['total_audio_sent']:.1f}s audio)")
                
                except asyncio.TimeoutError:
                    logger.warning("⚠️ Audio send timeout")
                    consecutive_errors += 1
                    self.stats['failed_chunks'] += 1
                except websockets.exceptions.ConnectionClosed:
                    logger.warning("🔗 Connection closed during sending")
                    self.is_connected = False
                    consecutive_errors = 0
                except Exception as e:
                    logger.error(f"❌ Audio send error: {e}")
                    consecutive_errors += 1
                    self.stats['failed_chunks'] += 1
                
                # Проверка на слишком много последовательных ошибок
                if consecutive_errors >= max_consecutive_errors:
                    logger.error(f"❌ Too many consecutive send errors ({consecutive_errors}), disconnecting")
                    self.is_connected = False
                    consecutive_errors = 0
                
            except Exception as e:
                self.stats['processing_errors'] += 1
                logger.error(f"❌ Audio sender critical error: {e}")
                await asyncio.sleep(0.1)
    
    async def result_receiver(self):
        """СТАБИЛЬНЫЙ получатель результатов"""
        logger.info("📥 STABLE result receiver started")
        
        while self.should_run:
            if not self.is_connected:
                await asyncio.sleep(0.1)
                continue
            
            try:
                # Получение результата с таймаутом
                try:
                    result = await asyncio.wait_for(self.websocket.recv(), timeout=1.0)
                    result_time = time.time()
                    receive_timestamp = datetime.now()
                except asyncio.TimeoutError:
                    continue
                
                if result and result not in ["NO_SPEECH", "PROCESSING", "SERVER_NOT_READY"]:
                    self.stats['results_received'] += 1
                    
                    # Сохранение в историю
                    result_entry = {
                        'timestamp': receive_timestamp.isoformat(),
                        'text': result,
                        'sequence': self.stats['results_received']
                    }
                    self.results_history.append(result_entry)
                    
                    # Ограничение истории
                    if len(self.results_history) > self.max_history:
                        self.results_history = self.results_history[-self.max_history:]
                    
                    # Отображение результата
                    self.display_result(result, receive_timestamp)
                    
                elif result == "NO_SPEECH":
                    logger.debug("🔇 Server: no speech detected")
                elif result == "SERVER_NOT_READY":
                    logger.warning("⚠️ Server not ready")
                elif result == "PONG":
                    logger.debug("🏓 Server pong received")
                    
            except websockets.exceptions.ConnectionClosed:
                logger.warning("🔗 Connection closed during receiving")
                self.is_connected = False
            except Exception as e:
                logger.error(f"❌ Result receiving error: {e}")
                self.stats['processing_errors'] += 1
                await asyncio.sleep(0.1)
    
    def display_result(self, result, timestamp):
        """Отображение результата с дополнительной информацией"""
        print(f"\n{'🛡️' * 60}")
        print(f"   STABLE FASTWHISPER RESULT #{self.stats['results_received']}")
        print(f"{'🛡️' * 60}")
        print(f"📝 Text: '{result.upper()}'")
        print(f"⏰ Time: {timestamp.strftime('%H:%M:%S.%f')[:-3]}")
        
        print(f"📊 Statistics:")
        print(f"   📤 Chunks sent: {self.stats['chunks_sent']}")
        print(f"   ✅ Successful: {self.stats['successful_chunks']}")
        print(f"   ❌ Failed: {self.stats['failed_chunks']}")
        print(f"   📥 Results received: {self.stats['results_received']}")
        print(f"   🎵 Total audio: {self.stats['total_audio_sent']:.1f}s")
        
        if self.stats['failed_chunks'] > 0:
            success_rate = (self.stats['successful_chunks'] / 
                          (self.stats['successful_chunks'] + self.stats['failed_chunks'])) * 100
            print(f"   📈 Success rate: {success_rate:.1f}%")
        
        error_count = (self.stats['connection_errors'] + 
                      self.stats['audio_errors'] + 
                      self.stats['processing_errors'])
        if error_count > 0:
            print(f"   ⚠️ Total errors: {error_count}")
        
        print(f"{'🛡️' * 60}\n")
    
    async def connection_monitor(self):
        """СТАБИЛЬНЫЙ монитор соединения"""
        while self.should_run:
            if not self.is_connected:
                logger.info("🔄 Reconnecting...")
                if await self.connect_to_server():
                    logger.info("✅ Reconnection successful")
                else:
                    logger.error("❌ Reconnection failed, retrying in 3s")
                    await asyncio.sleep(3)
            else:
                # Проверка состояния соединения
                try:
                    if self.websocket and self.websocket.closed:
                        logger.warning("🔗 Connection was closed")
                        self.is_connected = False
                    else:
                        # Периодическая проверка связи
                        await asyncio.sleep(15)  # Проверка каждые 15 секунд
                        if self.websocket and not self.websocket.closed:
                            try:
                                await asyncio.wait_for(self.websocket.send("PING"), timeout=5.0)
                            except:
                                logger.warning("🔗 Ping failed, connection may be lost")
                                self.is_connected = False
                except Exception as e:
                    logger.error(f"❌ Connection monitor error: {e}")
                    self.is_connected = False
    
    async def stats_reporter(self):
        """СТАБИЛЬНЫЙ репортер статистики"""
        while self.should_run:
            await asyncio.sleep(60)  # Отчет каждую минуту
            
            if self.is_connected and self.stats['session_start']:
                session_duration = (datetime.now() - self.stats['session_start']).total_seconds()
                
                logger.info(f"📊 STABLE MODE - Session: {session_duration/60:.1f}min")
                logger.info(f"   📤 Sent: {self.stats['chunks_sent']} chunks")
                logger.info(f"   ✅ Success: {self.stats['successful_chunks']}")
                logger.info(f"   ❌ Failed: {self.stats['failed_chunks']}")
                logger.info(f"   📥 Received: {self.stats['results_received']} results")
                logger.info(f"   🎵 Audio: {self.stats['total_audio_sent']:.0f}s")
                
                if self.stats['failed_chunks'] > 0:
                    success_rate = (self.stats['successful_chunks'] / 
                                  (self.stats['successful_chunks'] + self.stats['failed_chunks'])) * 100
                    logger.info(f"   📈 Success rate: {success_rate:.1f}%")
                
                error_count = (self.stats['connection_errors'] + 
                              self.stats['audio_errors'] + 
                              self.stats['processing_errors'])
                if error_count > 0:
                    logger.info(f"   ⚠️ Errors: {error_count}")
                
                # Запрос статистики сервера
                try:
                    await asyncio.wait_for(self.websocket.send("STATS"), timeout=3.0)
                    server_stats = await asyncio.wait_for(self.websocket.recv(), timeout=5.0)
                    stats_data = json.loads(server_stats)
                    
                    logger.info(f"🖥️ Server stats:")
                    logger.info(f"   🤖 Whisper calls: {stats_data.get('whisper_calls', 0)}")
                    logger.info(f"   ⚡ Average RTF: {stats_data.get('average_rtf', 0):.3f}")
                    
                    if stats_data.get('llm_successful_commands', 0) > 0:
                        logger.info(f"   🧠 LLM commands: {stats_data.get('llm_successful_commands', 0)}")
                    
                except Exception as e:
                    logger.debug(f"Could not get server stats: {e}")
    
    def save_session_log(self):
        """Сохранение лога сессии с диагностической информацией"""
        try:
            if not self.stats['session_start']:
                return
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_file = f"stable_fastwhisper_session_{timestamp}.json"
            
            session_data = {
                'session_info': {
                    'start_time': self.stats['session_start'].isoformat(),
                    'end_time': datetime.now().isoformat(),
                    'duration_seconds': (datetime.now() - self.stats['session_start']).total_seconds(),
                    'mode': 'STABLE',
                    'client_version': 'StableFastWhisperClient/1.0'
                },
                'statistics': self.stats,
                'results_history': self.results_history,
                'server_uri': self.server_uri,
                'audio_config': {
                    'sample_rate': self.sample_rate,
                    'chunk_duration': self.chunk_duration,
                    'chunk_size': self.chunk_size,
                    'buffer_size': self.audio_queue.maxsize,
                    'optimization': 'STABILITY_FIRST'
                },
                'error_analysis': {
                    'total_errors': (self.stats['connection_errors'] + 
                                   self.stats['audio_errors'] + 
                                   self.stats['processing_errors']),
                    'error_rate': self.calculate_error_rate(),
                    'success_rate': self.calculate_success_rate()
                }
            }
            
            with open(log_file, 'w', encoding='utf-8') as f:
                json.dump(session_data, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"💾 STABLE session log saved: {log_file}")
            
        except Exception as e:
            logger.error(f"Error saving log: {e}")
    
    def calculate_error_rate(self):
        """Расчет процента ошибок"""
        total_operations = self.stats['chunks_sent'] + self.stats['results_received']
        total_errors = (self.stats['connection_errors'] + 
                       self.stats['audio_errors'] + 
                       self.stats['processing_errors'])
        
        if total_operations > 0:
            return (total_errors / total_operations) * 100
        return 0
    
    def calculate_success_rate(self):
        """Расчет процента успешных операций"""
        total_chunks = self.stats['successful_chunks'] + self.stats['failed_chunks']
        
        if total_chunks > 0:
            return (self.stats['successful_chunks'] / total_chunks) * 100
        return 0
    
    def print_session_summary(self):
        """Детальная сводка сессии с диагностикой"""
        if not self.stats['session_start']:
            return
        
        session_duration = (datetime.now() - self.stats['session_start']).total_seconds()
        
        print(f"\n{'🛡️' * 70}")
        print("   STABLE FASTWHISPER SESSION SUMMARY")
        print(f"{'🛡️' * 70}")
        print(f"⏰ Session duration: {session_duration/60:.1f} minutes")
        print(f"📤 Chunks sent: {self.stats['chunks_sent']}")
        print(f"✅ Successful chunks: {self.stats['successful_chunks']}")
        print(f"❌ Failed chunks: {self.stats['failed_chunks']}")
        print(f"📥 Results received: {self.stats['results_received']}")
        print(f"🎵 Total audio sent: {self.stats['total_audio_sent']:.1f} seconds")
        
        # Анализ производительности
        success_rate = self.calculate_success_rate()
        error_rate = self.calculate_error_rate()
        
        print(f"\n📈 PERFORMANCE ANALYSIS:")
        print(f"   Success rate: {success_rate:.1f}%")
        print(f"   Error rate: {error_rate:.1f}%")
        
        if self.stats['average_chunk_size'] > 0:
            print(f"   Average chunk size: {self.stats['average_chunk_size']:.0f} bytes")
        
        # Детализация ошибок
        total_errors = (self.stats['connection_errors'] + 
                       self.stats['audio_errors'] + 
                       self.stats['processing_errors'])
        
        if total_errors > 0:
            print(f"\n❌ ERROR BREAKDOWN:")
            print(f"   Connection errors: {self.stats['connection_errors']}")
            print(f"   Audio errors: {self.stats['audio_errors']}")
            print(f"   Processing errors: {self.stats['processing_errors']}")
            print(f"   Total errors: {total_errors}")
        
        # Последние результаты
        if self.results_history:
            print(f"\n📝 LAST {min(5, len(self.results_history))} RESULTS:")
            for result in self.results_history[-5:]:
                timestamp = datetime.fromisoformat(result['timestamp']).strftime('%H:%M:%S')
                print(f"   {timestamp}: \"{result['text']}\"")
        
        # Рекомендации
        print(f"\n💡 RECOMMENDATIONS:")
        if success_rate < 90:
            print("   • Consider checking network connection stability")
            print("   • Try using a different audio device")
            print("   • Ensure sufficient system resources")
        
        if error_rate > 5:
            print("   • High error rate detected - check system performance")
            print("   • Consider reducing chunk rate or increasing buffer size")
        
        if self.stats['audio_errors'] > 10:
            print("   • Audio errors detected - check microphone and drivers")
        
        print(f"{'🛡️' * 70}")
    
    async def run(self, device_index=None):
        """СТАБИЛЬНАЯ главная функция выполнения"""
        try:
            # Настройка аудио
            logger.info("🔧 Setting up STABLE audio stream...")
            if not self.setup_audio_stream(device_index):
                logger.error("❌ Failed to set up audio stream")
                return False
            
            # Запуск записи
            self.stream.start()
            self.is_recording = True
            logger.info("🎙️ STABLE microphone recording started")
            
            # Первоначальное подключение
            logger.info("🔗 Establishing STABLE connection...")
            if not await self.connect_to_server():
                logger.error("❌ Failed to connect to FastWhisper server")
                return False
            
            # Запуск асинхронных задач
            tasks = []
            tasks.append(asyncio.create_task(self.audio_sender()))
            tasks.append(asyncio.create_task(self.result_receiver()))
            tasks.append(asyncio.create_task(self.connection_monitor()))
            tasks.append(asyncio.create_task(self.stats_reporter()))
            
            # Информация для пользователя
            print("\n" + "🛡️" * 90)
            print("🛡️ STABLE FASTWHISPER REAL-TIME ASR CLIENT 🛡️")
            print("🛡️" * 90)
            print("🎙️ Speak into the microphone!")
            print("🛡️ STABILITY-FIRST mode - optimized for reliability")
            print("📡 Audio chunks: 250ms (stable transmission)")
            print("🔧 Enhanced error handling and recovery")
            print("📊 Comprehensive statistics and diagnostics")
            print("🦷 Cody Dental Assistant commands supported")
            print("💾 Detailed session logging")
            print("🛑 Press Ctrl+C to stop gracefully")
            print("🛡️" * 90 + "\n")
            
            # Ожидание завершения задач
            try:
                await asyncio.gather(*tasks, return_exceptions=True)
            except Exception as e:
                logger.error(f"❌ Task execution error: {e}")
            
        except Exception as e:
            logger.error(f"❌ Critical error in run(): {e}")
        finally:
            await self.cleanup()
    
    async def cleanup(self):
        """Улучшенная очистка ресурсов"""
        logger.info("🧹 Starting STABLE cleanup...")
        
        self.should_run = False
        self.is_recording = False
        self.is_connected = False
        
        # Остановка аудио потока
        if hasattr(self, 'stream'):
            try:
                if self.stream.active:
                    self.stream.stop()
                self.stream.close()
                logger.info("🎙️ Audio stream stopped and closed")
            except Exception as e:
                logger.error(f"Audio cleanup error: {e}")
        
        # Очистка аудио очереди
        try:
            while not self.audio_queue.empty():
                try:
                    self.audio_queue.get_nowait()
                except queue.Empty:
                    break
            logger.info("🗃️ Audio queue cleared")
        except Exception as e:
            logger.error(f"Queue cleanup error: {e}")
        
        # Закрытие WebSocket
        if self.websocket:
            try:
                if not self.websocket.closed:
                    await asyncio.wait_for(self.websocket.close(), timeout=3.0)
                logger.info("🔗 WebSocket closed")
            except Exception as e:
                logger.error(f"WebSocket close error: {e}")
        
        # Сводка и сохранение
        self.print_session_summary()
        self.save_session_log()
        
        logger.info("✅ STABLE cleanup completed")

def main():
    """Главная функция с улучшенным интерфейсом выбора"""
    try:
        import sounddevice as sd
    except ImportError:
        print("❌ sounddevice not installed.")
        print("📦 Install with: pip install sounddevice")
        return
    
    print("🛡️ STABLE FASTWHISPER REAL-TIME ASR CLIENT")
    print("=" * 70)
    print("🎯 Optimized for STABILITY and RELIABILITY")
    print("🔧 Enhanced error handling and recovery")
    print("📊 Comprehensive diagnostics")
    print("=" * 70)
    
    # Выбор сервера
    print(f"\n🌐 Server configuration:")
    server_choice = input("Enter server address (or Enter for default AWS): ").strip()
    if server_choice:
        if not server_choice.startswith(('ws://', 'wss://')):
            server_uri = f"ws://{server_choice}:8765"
        else:
            server_uri = server_choice
    else:
        server_uri = "ws://3.84.215.173:8765"
    
    client = StableFastWhisperClient(server_uri)
    
    # Выбор аудио устройства
    devices = client.list_audio_devices()
    if not devices:
        print("❌ No audio devices found")
        return
    
    print(f"📱 Select audio device:")
    print("✅ = Compatible device")
    print("⚠️ = May have compatibility issues")
    
    # Рекомендуемые устройства
    compatible_devices = []
    for i, device in enumerate(devices):
        if device['max_input_channels'] > 0 and client.is_device_compatible(device):
            compatible_devices.append(i)
    
    if compatible_devices:
        print(f"🎯 Recommended compatible devices: {', '.join(map(str, compatible_devices))}")
    
    device_choice = input(f"Enter device number (0-{len(devices)-1}) or Enter for default: ").strip()
    
    try:
        device_index = int(device_choice) if device_choice.isdigit() else None
        if device_index is not None and (device_index < 0 or device_index >= len(devices)):
            print(f"❌ Invalid device number. Using default device.")
            device_index = None
    except ValueError:
        device_index = None
    
    # Финальная конфигурация
    print(f"\n🎯 STABLE Configuration:")
    print(f"   🌐 Server: {server_uri}")
    print(f"   🎤 Device: {device_index if device_index is not None else 'default'}")
    print(f"   🛡️ Mode: STABILITY-FIRST")
    print(f"   📡 Chunk size: {client.sample_rate}Hz, {client.chunk_duration*1000:.0f}ms")
    print(f"   🔧 Buffer size: {client.audio_queue.maxsize} chunks")
    print(f"   🦷 Dental Assistant: INTEGRATED")
    print(f"   📊 Diagnostics: ENABLED")
    
    confirm = input(f"\n▶️ Start STABLE recording? [Y/n]: ").strip().lower()
    if confirm and confirm not in ['y', 'yes', '']:
        print("❌ Canceled by user")
        return
    
    # Запуск клиента
    try:
        asyncio.run(client.run(device_index))
    except KeyboardInterrupt:
        print("\n\n👋 STABLE session stopped gracefully")
    except Exception as e:
        print(f"\n❌ Critical error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
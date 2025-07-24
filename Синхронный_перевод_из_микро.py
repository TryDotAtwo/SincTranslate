import pyaudio
import grpc
import requests
import yandex.cloud.ai.stt.v3.stt_pb2 as stt_pb2
import yandex.cloud.ai.stt.v3.stt_service_pb2_grpc as stt_service
import time
import json
import os
from dotenv import load_dotenv
import jwt
import threading
import queue
import g4f
from datetime import datetime

load_dotenv()

CHUNK = 4000
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
SOURCE_LANG = "ru-RU"
TARGET_LANG = "en-US"
VOICE = "oksana"

FOLDER_ID = os.getenv("FolderID")
if not FOLDER_ID:
    raise ValueError("Folder ID not found in environment variable 'FolderID'.")

try:
    with open('authorized_key.json', 'r') as f:
        service_account_key = json.load(f)
except Exception as e:
    raise ValueError(f"Error initializing SDK with authorized_key.json: {e}")

def get_iam_token():
    now = int(time.time())
    payload = {
        'aud': 'https://iam.api.cloud.yandex.net/iam/v1/tokens',
        'iss': service_account_key['service_account_id'],
        'iat': now,
        'exp': now + 3600
    }
    jwt_token = jwt.encode(
        payload,
        service_account_key['private_key'],
        algorithm='PS256',
        headers={'kid': service_account_key['id']}
    )
    response = requests.post(
        'https://iam.api.cloud.yandex.net/iam/v1/tokens',
        json={'jwt': jwt_token}
    )
    response.raise_for_status()
    return response.json()['iamToken']

iam_token = get_iam_token()

channel = grpc.secure_channel('stt.api.cloud.yandex.net:443', grpc.ssl_channel_credentials())
stub = stt_service.RecognizerStub(channel)

audio_queue = queue.Queue(maxsize=1000)

text_processing_queue = queue.Queue()
playback_queue = queue.Queue()

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

# --- Вспомогательная функция: удаление перекрывающегося префикса ---
def remove_duplicate_prefix(previous, current):
    """
    Удаляет из `current` максимально длинный префикс, совпадающий с суффиксом `previous`.
    """
    max_len = min(len(previous), len(current))
    for i in range(max_len, 0, -1):
        if previous[-i:] == current[:i]:
            return current[i:]
    return current


def split_sentences_with_llm(previous, current):
    current_trimmed = remove_duplicate_prefix(previous, current)
    prompt = f"""
Ты — система, разбивающая транскрибированный текст с речи на законченные предложения с пунктуацией.
Тебе даётся предыдущий буфер и текущий, соединённые в один текст.
Не нужно дублировать предложения, которые уже были в предыдущем буфере.
Если предложение оборвалось — просто игнорируй его, оно будет в следующем буфере.
Верни только завершённые предложения, отделённые точками.
Не повторяй предложения из предыдущего буфера.
Не возвращай предложения, если они уже были полностью в предыдущем.

Предыдущий буфер:
{previous}

Текущий буфер:
{current_trimmed}

Ответ:
"""
    response = g4f.ChatCompletion.create(
        model=g4f.models.gpt_4,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.strip()

def translate_text(text):
    headers = {
        "Authorization": f"Bearer {iam_token}",
        "Content-Type": "application/json"
    }
    data = {
        "sourceLanguageCode": SOURCE_LANG.split('-')[0],
        "targetLanguageCode": TARGET_LANG.split('-')[0],
        "texts": [text],
        "folderId": FOLDER_ID
    }
    response = requests.post("https://translate.api.cloud.yandex.net/translate/v2/translate", headers=headers, json=data)
    if response.status_code == 200:
        return response.json()['translations'][0]['text']
    log(f"Translation error: {response.text}")
    return None

def synthesize_speech(text):
    headers = {"Authorization": f"Bearer {iam_token}"}
    data = {
        "text": text,
        "lang": TARGET_LANG,
        "voice": VOICE,
        "format": "lpcm",
        "sampleRateHertz": 48000,
        "folderId": FOLDER_ID
    }
    response = requests.post("https://tts.api.cloud.yandex.net/speech/v1/tts:synthesize", headers=headers, data=data)
    if response.status_code == 200:
        return response.content
    log(f"Speech synthesis error: {response.text}")
    return None

def playback_worker():
    p_play = pyaudio.PyAudio()
    play_stream = p_play.open(format=pyaudio.paInt16, channels=1, rate=48000, output=True)
    log("Playback thread started")
    while True:
        audio_data = playback_queue.get()
        if audio_data is None:
            break
        play_stream.write(audio_data)
        playback_queue.task_done()
    play_stream.stop_stream()
    play_stream.close()
    p_play.terminate()
    log("Playback thread stopped")

def text_processing_worker():
    log("Text processing thread started")
    while True:
        text = text_processing_queue.get()
        if text is None:
            break
        translated = translate_text(text)
        if translated:
            log(f"Translated text: {translated}")
            audio_data = synthesize_speech(translated)
            if audio_data:
                log("Adding audio to playback queue...")
                playback_queue.put(audio_data)
            else:
                log("Speech synthesis error.")
        else:
            log("Text translation error.")
        text_processing_queue.task_done()
    log("Text processing thread stopped")

# --- КЛАСС ДЛЯ ПОТОКА РАСПОЗНАВАНИЯ ---
class RecognitionThread:
    def __init__(self, audio_source_queue, name):
        self.name = name
        self.audio_source_queue = audio_source_queue
        self.audio_chunks = []
        self.full_text = ""
        self.text_lock = threading.Lock()
        self.last_sent_index = 0
        self.stop_flag = threading.Event()
        self.thread = threading.Thread(target=self.worker, daemon=True)
        self.metadata = [
            ('authorization', f'Bearer {iam_token}'),
            ('x-folder-id', FOLDER_ID)
        ]

    def start(self):
        log(f"{self.name}: Starting recognition thread")
        self.thread.start()

    def stop(self):
        self.stop_flag.set()
        self.thread.join()
        log(f"{self.name}: Recognition thread stopped")

    def add_chunk(self, chunk):
        self.audio_chunks.append(chunk)

    def worker(self):
        def generate_requests():
            opts = stt_pb2.StreamingOptions(
                recognition_model=stt_pb2.RecognitionModelOptions(
                    audio_format=stt_pb2.AudioFormatOptions(
                        raw_audio=stt_pb2.RawAudio(
                            audio_encoding=stt_pb2.RawAudio.LINEAR16_PCM,
                            sample_rate_hertz=RATE,
                            audio_channel_count=CHANNELS
                        )
                    ),
                    audio_processing_type=stt_pb2.RecognitionModelOptions.REAL_TIME
                )
            )
            yield stt_pb2.StreamingRequest(session_options=opts)
            idx = 0
            while not self.stop_flag.is_set():
                if idx >= len(self.audio_chunks):
                    time.sleep(0.01)
                    continue
                chunk = self.audio_chunks[idx]
                idx += 1
                yield stt_pb2.StreamingRequest(chunk=stt_pb2.AudioChunk(data=chunk))

        try:
            responses = stub.RecognizeStreaming(generate_requests(), metadata=self.metadata)
            for response in responses:
                if self.stop_flag.is_set():
                    break
                kind = response.WhichOneof("Event")
                if kind == "partial" and response.partial.alternatives:
                    text = response.partial.alternatives[0].text
                    with self.text_lock:
                        self.full_text = text
        except grpc.RpcError as e:
            log(f"{self.name} gRPC error: {e.details()}, code: {e.code()}")
        except Exception as e:
            log(f"{self.name} Recognition error: {e}")

    def get_new_text_chunk(self):
        with self.text_lock:
            new_chunk = self.full_text[self.last_sent_index:]
            self.last_sent_index = len(self.full_text)
            return new_chunk

    def trim_prefix(self, prefix_len):
        with self.text_lock:
            self.full_text = self.full_text[prefix_len:]
            self.last_sent_index = max(0, self.last_sent_index - prefix_len)

# --- Вспомогательная функция для поиска максимального общего суффикса-префикса ---
def longest_common_suffix_prefix(suffix_source, prefix_target):
    max_len = min(len(suffix_source), len(prefix_target))
    for length in range(max_len, 0, -1):
        if suffix_source[-length:] == prefix_target[:length]:
            return length
    return 0


def process_llm_buffer_dual(recog1: RecognitionThread, recog2: RecognitionThread):
    last_sent_text = ""
    active_recog = recog1
    switched_to_recog2 = False

    last_send_time = 0

    while True:
        time.sleep(0.5)
        now = time.time()
        if now - last_send_time < 10:
            continue  # ждем 10 секунд между отправками

        new_text = active_recog.get_new_text_chunk()
        if not new_text.strip():
            continue

        cleaned_text = remove_duplicate_prefix(last_sent_text, last_sent_text + new_text)

        if cleaned_text.strip():
            try:
                log(f"LLM processing new text chunk ({active_recog.name}): '{cleaned_text}'")
                result = split_sentences_with_llm(last_sent_text, last_sent_text + new_text)
                result = result.strip()
                if result:
                    log(f"[LLM] Finalized: {result}")
                    text_processing_queue.put(result)
                    last_sent_text += cleaned_text
                last_send_time = now  # обновляем время отправки только если отправили
            except Exception as e:
                log(f"LLM error: {e}")

        elapsed = time.time() - start_time_global
        if not switched_to_recog2 and elapsed > 30:
            log(f"Switching recognition from {recog1.name} to {recog2.name}")

            with recog1.text_lock, recog2.text_lock:
                suffix = recog1.full_text[-3000:]
                prefix = recog2.full_text[:3000]
                cut_len = longest_common_suffix_prefix(suffix, prefix)
                if cut_len > 0:
                    log(f"Trimming {cut_len} chars prefix from {recog2.name}")
                    recog2.trim_prefix(cut_len)

            active_recog = recog2
            switched_to_recog2 = True
            with recog2.text_lock:
                last_sent_text = recog2.full_text



# --- Сбор аудио чанков из очереди ---
def audio_chunks_collector(stop_flag, audio_chunks):
    while not stop_flag.is_set():
        try:
            chunk = audio_queue.get(timeout=0.1)
            audio_chunks.append(chunk)
            audio_queue.task_done()
        except queue.Empty:
            continue

def audio_capture_worker(stream):
    log("Audio capture started")
    while True:
        try:
            data = stream.read(CHUNK, exception_on_overflow=False)
            audio_queue.put(data)
        except Exception as e:
            log(f"Audio capture error: {e}")
            break

def start_recognition_thread_with_collector(name):
    audio_chunks = []
    stop_flag = threading.Event()

    collector_thread = threading.Thread(target=audio_chunks_collector, args=(stop_flag, audio_chunks), daemon=True)
    collector_thread.start()

    recog_thread = RecognitionThread(audio_queue, name)
    recog_thread.audio_chunks = audio_chunks
    recog_thread.stop_flag = stop_flag
    recog_thread.thread = threading.Thread(target=recog_thread.worker, daemon=True)

    recog_thread.start()

    return recog_thread, stop_flag, collector_thread

def stop_recognition_thread(recog_thread, stop_flag, collector_thread):
    stop_flag.set()
    collector_thread.join()
    recog_thread.stop()

def main():
    global start_time_global

    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    log("Starting live translation with LLM segmentation...")

    audio_capture_thread = threading.Thread(target=audio_capture_worker, args=(stream,), daemon=True)
    audio_capture_thread.start()

    playback_thread = threading.Thread(target=playback_worker, daemon=True)
    playback_thread.start()

    text_thread = threading.Thread(target=text_processing_worker, daemon=True)
    text_thread.start()

    # Запуск двух RecognitionThread с задержкой 20 секунд
    recog1, stop_flag1, collector1 = start_recognition_thread_with_collector("RecognitionThread-1")
    time.sleep(20)
    recog2, stop_flag2, collector2 = start_recognition_thread_with_collector("RecognitionThread-2")

    start_time_global = time.time()

    # Запуск обработки LLM с переключением потоков
    llm_thread = threading.Thread(target=process_llm_buffer_dual, args=(recog1, recog2), daemon=True)
    llm_thread.start()

    try:
        # Останавливаем первый поток через 30 секунд после запуска второго
        # Основной цикл просто ждет, потом останавливает потоки
        while True:
            elapsed = time.time() - start_time_global
            if elapsed > 50:
                # Останавливаем первый поток
                if stop_flag1.is_set() is False:
                    log("Stopping RecognitionThread-1 after 50 seconds")
                    stop_flag1.set()
                    collector1.join()
                    recog1.stop()
                # Останавливаем второй поток спустя ещё 30 сек (на всякий случай)
                if elapsed > 80:
                    if stop_flag2.is_set() is False:
                        log("Stopping RecognitionThread-2 after 80 seconds")
                        stop_flag2.set()
                        collector2.join()
                        recog2.stop()
                        break
            time.sleep(1)
    except KeyboardInterrupt:
        log("Stopping program with Ctrl+C...")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

        playback_queue.put(None)
        playback_thread.join()

        text_processing_queue.put(None)
        text_thread.join()

        if not stop_flag1.is_set():
            stop_flag1.set()
            collector1.join()
            recog1.stop()
        if not stop_flag2.is_set():
            stop_flag2.set()
            collector2.join()
            recog2.stop()

if __name__ == "__main__":
    main()

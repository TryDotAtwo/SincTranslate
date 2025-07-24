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
from datetime import datetime

load_dotenv()

CHUNK = 4000
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
SOURCE_LANG = "ru-RU"
TARGET_LANG = "en-US"
VOICE = "oksana"
DEBUG_MODE = True
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
            log(f"Translated text: {translated[:100]}...")
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

class RecognitionThread:
    def __init__(self, audio_chunks, chunk_index, name, debug_mode=False):
        self.name = name
        self.audio_chunks = audio_chunks
        self.chunk_index = chunk_index
        self.full_text = []
        self.text_lock = threading.Lock()
        self.stop_flag = threading.Event()
        self.debug_mode = debug_mode
        self.last_text_time = time.time()
        self.thread = threading.Thread(target=self.worker, daemon=True)
        self.metadata = [
            ('authorization', f'Bearer {iam_token}'),
            ('x-folder-id', FOLDER_ID)
        ]

    def start(self):
        log(f"{self.name}: Starting recognition thread at chunk index {self.chunk_index}")
        self.thread.start()

    def stop(self):
        self.stop_flag.set()
        self.thread.join()
        log(f"{self.name}: Recognition thread stopped")
        if self.debug_mode:
            log(f"{self.name}: All recognized text: {''.join(self.full_text)[:1000]}...")

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
            while not self.stop_flag.is_set():
                with self.text_lock:
                    if self.chunk_index >= len(self.audio_chunks):
                        time.sleep(0.01)
                        continue
                    chunk = self.audio_chunks[self.chunk_index]
                    self.chunk_index += 1
                yield stt_pb2.StreamingRequest(chunk=stt_pb2.AudioChunk(data=chunk))

        try:
            responses = stub.RecognizeStreaming(generate_requests(), metadata=self.metadata)
            for response in responses:
                if self.stop_flag.is_set():
                    break
                kind = response.WhichOneof("Event")
                if kind == "final" and response.final.alternatives:
                    text = response.final.alternatives[0].text
                    with self.text_lock:
                        if text:
                            log(f"{self.name}: Final text received at chunk {self.chunk_index}: '{text[:100]}...'")
                            self.full_text.append(text + " ")
                            self.last_text_time = time.time()
                            if self.debug_mode:
                                log(f"{self.name}: Added final text to full_text: '{text[:100]}...'")
        except grpc.RpcError as e:
            log(f"{self.name} gRPC error: {e.details()}, code: {e.code()}")
        except Exception as e:
            log(f"{self.name} Recognition error: {e}")

    def get_new_text(self):
        with self.text_lock:
            if not self.full_text:
                return ""
            # Concatenate texts received within a 2-second window to form complete sentences
            current_time = time.time()
            if current_time - self.last_text_time < 2:
                return ""
            new_text = "".join(self.full_text)
            self.full_text = []
            log(f"{self.name}: Sending new text at chunk {self.chunk_index}: '{new_text[:100]}...'")
            return new_text

def process_text_buffer(recog1, recog2, recog_debug=None):
    last_sent_time = 0
    active_recog = recog1
    switched_to_recog2 = False
    start_time = time.time()
    recog2_ready = False

    while True:
        time.sleep(0.5)
        now = time.time()
        if now - last_sent_time < 5:  # Reduced to 5 seconds for faster processing
            continue

        new_text = active_recog.get_new_text()
        if new_text.strip():
            log(f"Processing final text from {active_recog.name}: '{new_text[:100]}...'")
            text_processing_queue.put(new_text)
            last_sent_time = now

        elapsed = time.time() - start_time
        if not switched_to_recog2 and elapsed > 30:
            with recog2.text_lock:
                if recog2.full_text:  # Wait for recog2 to have at least one final result
                    recog2_ready = True
            if recog2_ready:
                log(f"Switching recognition from {recog1.name} to {recog2.name} at chunk {recog1.chunk_index}")
                with recog1.text_lock, recog2.text_lock:
                    recog2.chunk_index = recog1.chunk_index  # Sync chunk index
                active_recog = recog2
                switched_to_recog2 = True

        if DEBUG_MODE and elapsed > 60 and recog_debug:
            log("Debug mode: Outputting all recognized texts after 60 seconds")
            with recog1.text_lock, recog2.text_lock, recog_debug.text_lock:
                text1 = "".join(recog1.full_text)
                text2 = "".join(recog2.full_text)
                text_debug = "".join(recog_debug.full_text)
                log(f"RecognitionThread-1 (last chunk {recog1.chunk_index}): {text1[:1000]}...")
                log(f"RecognitionThread-2 (last chunk {recog2.chunk_index}): {text2[:1000]}...")
                log(f"RecognitionThread-Debug (last chunk {recog_debug.chunk_index}): {text_debug[:1000]}...")
            break

def audio_chunks_collector(stop_flag, audio_chunks):
    while not stop_flag.is_set():
        try:
            chunk = audio_queue.get(timeout=0.1)
            audio_chunks.append(chunk)
            audio_queue.task_done()
        except queue.Empty:
            continue

def start_recognition_thread_with_collector(name, audio_chunks, chunk_index, debug_mode=False):
    stop_flag = threading.Event()
    collector_thread = threading.Thread(target=audio_chunks_collector, args=(stop_flag, audio_chunks), daemon=True)
    collector_thread.start()
    recog_thread = RecognitionThread(audio_chunks, chunk_index, name, debug_mode=debug_mode)
    recog_thread.start()
    return recog_thread, stop_flag, collector_thread

def stop_recognition_thread(recog_thread, stop_flag, collector_thread):
    stop_flag.set()
    collector_thread.join()
    recog_thread.stop()

def audio_capture_worker(stream):
    log("Audio capture started")
    while True:
        try:
            data = stream.read(CHUNK, exception_on_overflow=False)
            audio_queue.put(data)
        except Exception as e:
            log(f"Audio capture error: {e}")
            break

def main():
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    log("Starting live translation with final results...")

    audio_capture_thread = threading.Thread(target=lambda: audio_capture_worker(stream), daemon=True)
    audio_capture_thread.start()

    playback_thread = threading.Thread(target=playback_worker, daemon=True)
    playback_thread.start()

    text_thread = threading.Thread(target=text_processing_worker, daemon=True)
    text_thread.start()

    shared_audio_chunks = []
    recog1, stop_flag1, collector1 = start_recognition_thread_with_collector("RecognitionThread-1", shared_audio_chunks, 0, debug_mode=True)
    time.sleep(20)  # Allow recog1 to process initial audio
    recog2, stop_flag2, collector2 = start_recognition_thread_with_collector("RecognitionThread-2", shared_audio_chunks, recog1.chunk_index, debug_mode=True)
    recog_debug, stop_flag_debug, collector_debug = start_recognition_thread_with_collector("RecognitionThread-Debug", shared_audio_chunks, 0, debug_mode=True)
    log("Debug mode: Started RecognitionThread-Debug")

    text_buffer_thread = threading.Thread(target=process_text_buffer, args=(recog1, recog2, recog_debug), daemon=True)
    text_buffer_thread.start()

    try:
        start_time = time.time()
        while True:
            elapsed = time.time() - start_time
            if elapsed > 50:
                if not stop_flag1.is_set():
                    log("Stopping RecognitionThread-1 after 50 seconds")
                    stop_recognition_thread(recog1, stop_flag1, collector1)
                if elapsed > 80:
                    if not stop_flag2.is_set():
                        log("Stopping RecognitionThread-2 after 80 seconds")
                        stop_recognition_thread(recog2, stop_flag2, collector2)
                    if DEBUG_MODE and not stop_flag_debug.is_set():
                        log("Stopping RecognitionThread-Debug after 80 seconds")
                        stop_recognition_thread(recog_debug, stop_flag_debug, collector_debug)
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
            stop_recognition_thread(recog1, stop_flag1, collector1)
        if not stop_flag2.is_set():
            stop_recognition_thread(recog2, stop_flag2, collector2)
        if DEBUG_MODE and not stop_flag_debug.is_set():
            stop_recognition_thread(recog_debug, stop_flag_debug, collector_debug)

if __name__ == "__main__":
    main()
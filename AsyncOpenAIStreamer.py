import os
from concurrent.futures.thread import ThreadPoolExecutor
from tkinter.constants import FIRST

from httpcore import stream
from pydantic import Field, model_validator, BaseModel
from typing import Any, List
from openai import OpenAI
import pygame
import io, numpy as np
import threading
import webrtcvad
import sounddevice as sd
import time
import queue
import tempfile, asyncio

from pygame import NOEVENT
from scipy.io import wavfile
from contextlib import contextmanager
import traceback
from dataclasses import dataclass

@dataclass
class AudioChunk:
    audio: bytes
    text: str

class OpenAIStreamer(BaseModel):
    api_key: str = Field(default='sk-T_tRfijTOy8McMUCP_dO5z8PTwInF1l0K6HXp_50rAT3BlbkFJKcGo0d26hy9EgcZ2V5ahtuBwkeN8muRpL0COmWrbEA')
    fre: int = Field(default=24000, description='Init frequency used by tts model. For example tts-1 model generates 24000 hz frequency.')
    chunk_size: int= Field(default=4096, description='Size of the chunk used for generating audio.')
    client: Any = Field(default=None)
    current_chunk: Any = Field(default=None)
    is_playing: bool = Field(default=False)
    vad: Any  = Field(default=None)
    sample_rate: Any  = Field(default=None)
    audio_queue: Any  = Field(default=None)
    is_listening: bool  = Field(default=False)
    is_processing: bool  = Field(default=False)
    audio_buffer: List  = Field(default=None)
    silence_threshold: Any  = Field(default=None)
    silent_chunks: Any  = Field(default=None)
    min_speech_chunks: Any = Field(default=None)
    speech_chunks_count: Any  = Field(default=None)
    amplitude_threshold: Any  = Field(default=None)
    bprocess: Any  = Field(default=None)
    selected_option: Any  = Field(default=None)
    voice_info: Any = Field(default=None)
    stream: Any = Field(default=None)
    text_queue: Any = Field(default=None)
    output_audio_queue: Any = Field(default=None)
    status_queue: Any  = Field(default=None)
    text_buffer: List = Field(default=None)
    output_audio_buffer: List = Field(default=None)
    status_buffer: List = Field(default=None)
    thread_pool: Any = Field(default=None)
    should_stop: bool = Field(default=False)



    @model_validator(mode='after')
    def load_necessary_details(self):
        self.client = OpenAI(api_key=self.api_key)
        pygame.mixer.init(frequency=self.fre)
        self.current_chunk = io.BytesIO()
        # Initialize VAD (Voice Activity Detection)
        self.vad = webrtcvad.Vad()
        self.vad.set_mode(2)  # Less aggressive mode
        # Audio parameters
        self.sample_rate = 16000
        self.chunk_size = 480  # 30ms at 16kHz
        self.audio_queue = queue.Queue()
        # State flags
        self.is_listening = False
        self.is_processing = False
        # Buffer for storing audio data
        self.audio_buffer = []
        self.silence_threshold = 50
        self.silent_chunks = 0
        self.min_speech_chunks = 10  # Minimum chunks with speech required
        self.speech_chunks_count = 0  # Counter for chunks containing speech
        self.amplitude_threshold = 500  # Adjust this based on your microphone/environment
        self.bprocess = True
        self.text_buffer = []
        self.output_audio_buffer = []
        self.text_queue = queue.Queue()
        self.output_audio_queue = queue.Queue()
        self.status_queue = queue.Queue()
        self.status_buffer = []
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        self.should_stop = False
        return self

    def is_speech(self, audio_chunk):
        """Check if audio chunk contains speech with amplitude threshold"""
        try:
            if len(audio_chunk) != self.chunk_size:
                return False

            # Check amplitude
            amplitude = np.abs(audio_chunk).mean()
            if amplitude < self.amplitude_threshold:
                return False

            audio_chunk = audio_chunk.astype(np.int16)
            return self.vad.is_speech(audio_chunk.tobytes(), self.sample_rate)
        except Exception as e:
            print(f"VAD error: {e}")
            return False

    def clear_buffers(self):
        """Clear all audio buffers and reset speech counter"""
        self.audio_buffer = []
        self.silent_chunks = 0
        self.speech_chunks_count = 0  # Reset speech counter
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break

    def audio_callback(self, indata, frames, time, status):
        """Callback for audio input"""
        if status:
            print(f"Status: {status}")

        if self.is_listening and not self.is_processing:
            audio_data = (indata * 32767).astype(np.int16)
            self.audio_queue.put(audio_data)

    @contextmanager
    def audio_stream(self):
        """Context manager for handling the audio stream"""
        try:
            local_stream = sd.InputStream(
                callback=self.audio_callback,
                channels=1,
                samplerate=self.sample_rate,
                blocksize=self.chunk_size,
                dtype=np.float32
            )
            local_stream.start()
            self.stream = local_stream
            yield
        finally:
            if self.stream:
                self.stream.stop()
                self.stream.close()
                self.stream = None

    async def process_audio_input(self):
        while not self.should_stop:
            try:
                if not self.bprocess:
                    self.status_queue.put("Stopping...!")
                    break
                # Start listening phase
                print("\nListening... Speak now!")
                self.status_queue.put("Listening... Speak now!")
                # if self.status_container:
                #     self.status_container.write("Listening... Speak now!")

                self.is_listening = True
                self.is_processing = False
                self.clear_buffers()

                while self.is_listening:
                    try:
                        # Get audio data from queue with timeout
                        audio_chunk = self.audio_queue.get(timeout=1).flatten()

                        # Check for voice activity
                        if self.is_speech(audio_chunk):
                            self.audio_buffer.append(audio_chunk)
                            self.speech_chunks_count += 1  # Increment speech counte
                            self.silent_chunks = 0
                        else:
                            self.silent_chunks += 1
                            if self.speech_chunks_count > 0:  # Only add if we've detected speech before
                                self.audio_buffer.append(audio_chunk)

                        # print(f"Silent chunks: {self.silent_chunks}, Buffer size: {len(self.audio_buffer)}")

                        # Check if we should stop listening and start processing
                        if self.silent_chunks >= self.silence_threshold and len(self.audio_buffer) > 0:
                            # Stop listening and start processing
                            if self.speech_chunks_count >= self.min_speech_chunks:
                                self.is_listening = False
                                self.is_processing = True
                            else:
                                # Reset if not enough speech was detected
                                print("Not enough speech detected, resetting...")
                                # if self.status_container:
                                #     self.status_container.write("Not enough speech detected, please speak again...")
                                # yield "Not enough speech detected, please speak again...", None
                                self.status_queue.put("Not enough speech detected, resetting...")
                                self.clear_buffers()
                            break

                    except queue.Empty:
                        continue

                # Processing phase
                if self.is_processing and len(self.audio_buffer) > 0:
                    print(f"\nProcessing speech...(Detected {self.speech_chunks_count} chunks with speech)")
                    # if self.status_container:
                    # # print(f"Status container details:{self.status_container}")
                    #     self.status_container.write('Processing speech...')
                    # yield 'Processing speech...', None
                    self.status_queue.put("Processing speech...")
                    # print(f"Processing speech")
                    # Process the audio
                    current_buffer = self.audio_buffer.copy()
                    if self.selected_option == "OpenAI":
                        response = self.generate_text_from_audio(current_buffer)
                        self.generate_streaming_speech(response, 5)
                        # self.thread_pool.submit(self.generate_streaming_speech, response, 4)
                        # for text, audio_chunk in self.generate_streaming_speech(response, 4):
                        #     yield text, audio_chunk
                    else:
                        transcribed_text, audio_file = 'DUMMY', '\n'
                        # yield transcribed_text, audio_file
                    # if self.transcription_container:
                    #     self.display_bubble_message(transcribed_text)
                    #     # transcribed_text = f":blue-background[{transcribed_text}]"
                    #     # self.transcription_container.markdown(body=transcribed_text, unsafe_allow_html=True)
                    # self.play_audio(audio_file)
            except KeyboardInterrupt:
                print("\nStopping...")
                break



    def play_chunk(self, chunk_data):
        """Play a single audio chunk"""
        # Write chunk to temporary file (required for pygame)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_file:
            temp_file.write(chunk_data)
            temp_file_path = temp_file.name

        try:
            # Load and play the chunk
            pygame.mixer.music.load(temp_file_path)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                pygame.time.wait(10)
        except Exception as e:
            print(f"Error playing chunk: {e}")
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_file_path)
            except:
                pass

    def transcribe_audio(self, audio_data):
        """Transcribe audio using OpenAI Whisper API"""
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                wavfile.write(temp_file.name, self.sample_rate, np.concatenate(audio_data))
                with open(temp_file.name, "rb") as audio_file:
                    transcript = self.client.audio.transcriptions.create(
                        model="whisper-1",
                        response_format="text",
                        file=audio_file,
                        language='en'
                    )
            os.unlink(temp_file.name)
            return transcript.strip()
        except Exception as e:
            print(f"Transcription error: {e}")
            return ""

    def generate_text_from_audio(self, audio_data):
        transcribed_text = self.transcribe_audio(audio_data)
        print(f'Transcribed text:{transcribed_text}')
        message_list = []
        message_list.append({"content": [
            {"type": "text", "text": "You are a hr assistant who responds to any questions asked regarding hr."}],
                             "role": "user"})
        message_list.append({"content": [
            {"type": "text", "text": "Ok.I will answer any hr related general query. Please provide user query"}],
                             "role": "assistant"})
        message_list.append({"content": [
            {"type": "text", "text": f"{transcribed_text}. REMEMBER to generate response in less than 30 tokens."}],
            "role": "user"})
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=message_list,
            max_tokens=200,
            temperature=0.1,
            stream =True
        )
        return response

        # response_text = response.choices[0].message.content
        # print(f'Response:{response_text}')

    def stream_text(self, text, voice="alloy", model="tts-1"):
        """Stream and play text with immediate chunk playback"""
        try:
            print(f"Voice generation for the text now")
            # Generate speech with streaming
            response = self.client.audio.speech.create(
                model=model,
                voice=voice,
                input=text,
                response_format="mp3"
            )

            self.current_chunk = io.BytesIO()
            accumulated_size = 0

            # Process and play chunks as they arrive
            for chunk in response.iter_bytes():
                self.current_chunk.write(chunk)
                accumulated_size += len(chunk)

                # When we have enough data, play the chunk
                if accumulated_size >= self.chunk_size:
                    chunk_data = self.current_chunk.getvalue()
                    # self.play_chunk(chunk_data)
                    # yield text, chunk_data
                    self.text_queue.put(text)
                    self.output_audio_queue.put(chunk_data)

                    # Reset for next chunk
                    self.current_chunk = io.BytesIO()
                    accumulated_size = 0

            # Play any remaining audio
            if accumulated_size > 0:
                chunk_data = self.current_chunk.getvalue()
                # yield text, chunk_data
                self.text_queue.put(text)
                self.output_audio_queue.put(chunk_data)
                # self.play_chunk(chunk_data)

        except Exception as e:
            print(f"Streaming error: {e}")

    def generate_streaming_speech(self, response, allowed_chunk_tokens=5):
        word_list, space_cnt = [], 0
        print("Processing text stream")
        for chunk in response:
            if chunk.choices[0].delta.content is not None:
                text = chunk.choices[0].delta.content
                # print(f"Text:{text}")
                word_list.append(text)
                space_cnt = space_cnt + 1 if " " in text else space_cnt
                if space_cnt >= allowed_chunk_tokens:
                    # for text, audio_chunk in self.stream_text(f"{' '.join(word_list)}."):
                    #     yield text[:-1], audio_chunk
                    self.thread_pool.submit(self.stream_text, f"{' '.join(word_list)}.")
                    word_list = []
                    space_cnt = 0
        if len(word_list) > 0:
            # for text, audio_chunk in self.stream_text(f"{' '.join(word_list)}."):
            #     yield text[:-1], audio_chunk
            self.thread_pool.submit(self.stream_text, f"{' '.join(word_list)}.")
            word_list = []
            space_cnt = 0

    async def play_audio_queue(self):
        while not self.should_stop:
            try:
                print("Playing audio of chunks")
                chunk = self.output_audio_queue.get_nowait()  # Changed from audio_queue to output_audio_queue
                self.play_chunk(chunk)
                # Run play_chunk in a thread pool since it's blocking
                # await asyncio.get_event_loop().run_in_executor(
                #     None, self.play_chunk, chunk
                # )
            except queue.Empty:
                await asyncio.sleep(0.1)

    async def process_text_queue(self):
        while not self.should_stop:
            try:
                text = self.text_queue.get_nowait()
                yield text
            except queue.Empty:
                await asyncio.sleep(0.1)

    async def process_status_queue(self):
        while not self.should_stop:
            try:
                status = self.status_queue.get_nowait()
                yield status
            except queue.Empty:
                await asyncio.sleep(0.1)

    async def run(self):
        """Main entry point for the streamer"""
        self.should_stop = False
        print("Starting the run of the project")
        try:
            with self.audio_stream():
                # Create base tasks that don't yield
                base_tasks = [
                    self.process_audio_input(),
                    self.play_audio_queue(),
                ]
                # Run base tasks
                await asyncio.gather(*base_tasks)
        except Exception as e:
            print(f"Error in process_speech: {e}")
            traceback.print_exc()
        finally:
            self.should_stop = True
            self.bprocess = False
            self.is_listening = False
            self.is_processing = False

    # def process_speech(self):
    #     """Main processing loop"""
    #     try:
    #         print("\nAvailable audio devices:")
    #         print(sd.query_devices())
    #
    #         print(f"\nUsing input device: {sd.query_devices(None, 'input')['name']}")
    #
    #         with self.audio_stream():
    #             while True:
    #                 try:
    #                     if not self.bprocess:
    #                         break
    #                     # Start listening phase
    #                     print("\nListening... Speak now!")
    #                     yield "Listening... Speak now!", None
    #                     # if self.status_container:
    #                     #     self.status_container.write("Listening... Speak now!")
    #
    #                     self.is_listening = True
    #                     self.is_processing = False
    #                     self.clear_buffers()
    #
    #                     while self.is_listening:
    #                         try:
    #                             # Get audio data from queue with timeout
    #                             audio_chunk = self.audio_queue.get(timeout=1).flatten()
    #
    #                             # Check for voice activity
    #                             if self.is_speech(audio_chunk):
    #                                 self.audio_buffer.append(audio_chunk)
    #                                 self.speech_chunks_count += 1  # Increment speech counte
    #                                 self.silent_chunks = 0
    #                             else:
    #                                 self.silent_chunks += 1
    #                                 if self.speech_chunks_count > 0:  # Only add if we've detected speech before
    #                                     self.audio_buffer.append(audio_chunk)
    #
    #                             # print(f"Silent chunks: {self.silent_chunks}, Buffer size: {len(self.audio_buffer)}")
    #
    #                             # Check if we should stop listening and start processing
    #                             if self.silent_chunks >= self.silence_threshold and len(self.audio_buffer) > 0:
    #                                 # Stop listening and start processing
    #                                 if self.speech_chunks_count >= self.min_speech_chunks:
    #                                     self.is_listening = False
    #                                     self.is_processing = True
    #                                 else:
    #                                     # Reset if not enough speech was detected
    #                                     print("Not enough speech detected, resetting...")
    #                                     # if self.status_container:
    #                                     #     self.status_container.write("Not enough speech detected, please speak again...")
    #                                     yield "Not enough speech detected, please speak again...", None
    #                                     self.clear_buffers()
    #                                 break
    #
    #                         except queue.Empty:
    #                             continue
    #
    #                     # Processing phase
    #                     if self.is_processing and len(self.audio_buffer) > 0:
    #                         print(f"\nProcessing speech...(Detected {self.speech_chunks_count} chunks with speech)")
    #                         # if self.status_container:
    #                         # # print(f"Status container details:{self.status_container}")
    #                         #     self.status_container.write('Processing speech...')
    #                         yield 'Processing speech...', None
    #                         print(f"Processing speech")
    #                         # Process the audio
    #                         current_buffer = self.audio_buffer.copy()
    #                         if self.selected_option == "OpenAI":
    #                             response = self.generate_text_from_audio(current_buffer)
    #                             self.generate_streaming_speech(response, 4)
    #                             # for text, audio_chunk in self.generate_streaming_speech(response, 4):
    #                             #     yield text, audio_chunk
    #                         else:
    #                             transcribed_text, audio_file = 'DUMMY', '\n'
    #                             yield transcribed_text, audio_file
    #                         # if self.transcription_container:
    #                         #     self.display_bubble_message(transcribed_text)
    #                         #     # transcribed_text = f":blue-background[{transcribed_text}]"
    #                         #     # self.transcription_container.markdown(body=transcribed_text, unsafe_allow_html=True)
    #                         # self.play_audio(audio_file)
    #                 except KeyboardInterrupt:
    #                     print("\nStopping...")
    #                     break
    #
    #     except Exception as e:
    #         print(f"Error in process_speech: {e}")
    #         traceback.print_exc()
    #     finally:
    #         self.bprocess = True
    #         self.is_listening = False
    #         self.is_processing = False

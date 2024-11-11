import time

import pygame
import tempfile
import os
import io
import threading, traceback
import queue
from openai import OpenAI
from time import sleep
import streamlit as st
import sounddevice as sd
import numpy as np
import webrtcvad
from scipy.io import wavfile
from contextlib import contextmanager
from pygame.pypm import Input


class StreamingSimulator:
    def __init__(self):
        # Initialize queues
        self.text_queue = queue.Queue()
        self.op_audio_queue = queue.Queue()

        # Input listening
        self.audio_queue = queue.Queue()
        self.vad = webrtcvad.Vad()
        self.vad.set_mode(2)  # Less aggressive mode
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
        self.selected_option = "OpenAI"
        # self.audio_playback = False

        # Initialize pygame mixer
        pygame.mixer.init()

        # Initialize OpenAI client
        self.client = OpenAI(api_key="sk-T_tRfijTOy8McMUCP_dO5z8PTwInF1l0K6HXp_50rAT3BlbkFJKcGo0d26hy9EgcZ2V5ahtuBwkeN8muRpL0COmWrbEA")

        # Configuration
        self.op_chunk_size = 1024 * 8  # 32KB chunks
        self.phrases = []
        # Start worker threads
        self.text_thread = None #threading.Thread(target=self.text_display_worker)
        self.audio_thread = None #threading.Thread(target=self.audio_playback_worker)
        self.voice_info = 'alloy'
        self.bprocess = True
        self.sample_rate = 16000
        self.chunk_size = 480  # 30ms at 16kHz

    def generate_tts(self, text):
        """Generate TTS audio for given text and add to audio queue"""
        try:
            response = self.client.audio.speech.create(
                model="tts-1",
                voice=self.voice_info,
                input=text,
                response_format="mp3"
            )

            current_chunk = io.BytesIO()
            accumulated_size = 0

            for chunk in response.iter_bytes():
                current_chunk.write(chunk)
                accumulated_size += len(chunk)

                if accumulated_size >= self.op_chunk_size:
                    print(f"Adding chunk")
                    self.op_audio_queue.put((current_chunk.getvalue(), text))
                    # self.op_audio_queue.join()
                    current_chunk = io.BytesIO()
                    accumulated_size = 0

            # Add any remaining audio
            if accumulated_size > 0:
                print(f"Adding final chunk")
                self.op_audio_queue.put((current_chunk.getvalue(), text))

        except Exception as e:
            print(f"TTS generation error: {e}")

    def play_chunk(self, chunk_data):
        """Play a single audio chunk"""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_file:
            temp_file.write(chunk_data)
            temp_file_path = temp_file.name

        try:
            sound = pygame.mixer.Sound(temp_file_path)
            channel = pygame.mixer.find_channel(True)  # True forces it to find a free channel

            # Play and wait for completion
            channel.play(sound)

            # Wait until this specific channel is done
            while channel.get_busy():
                pygame.time.wait(200)  # Short wait to prevent CPU overuse

        finally:
            os.unlink(temp_file_path)
            # Ensure complete cleanup
            channel.stop()
            sound = None


    def display_bubble_message(self, message):
        st.markdown("""
        <style>
            .bubble {
                background-color: #E3F2FD;
                padding: 15px 20px;
                border-radius: 10px;
                margin: 10px 0;
                display: inline-block;
            }
        </style>
        """, unsafe_allow_html=True)
        # if 'transcription_container' not in st.session_state:
        #     st.session_state.transcription_container = st.empty()
        with st.session_state.transcription_container:
            st.markdown(f'<div class="bubble">{message}</div>', unsafe_allow_html=True)

    def text_display_worker(self):
        """Worker thread for displaying text"""
        while True:
            accumulated_text= ""
            try:
                text = self.text_queue.get()
                print(f"{text}", end="")
                accumulated_text += text
                # container.info(accumulated_text)
                # self.display_bubble_message(text)
                self.text_queue.task_done()
            except queue.Empty:
                sleep(0.1)
            except Exception as e:
                print(f"Text display error: {e}")

    def audio_playback_worker(self):
        """Worker thread for playing audio chunks"""
        current_text = None  # Track current phrase being played

        while True:
            try:
                chunk_data, text = self.op_audio_queue.get()

                # If this is a new phrase, wait for previous to finish
                if text != current_text and current_text is not None:
                    # Wait for current audio to finish
                    while pygame.mixer.get_busy():
                        pygame.time.wait(10)

                current_text = text
                self.play_chunk(chunk_data)
                sleep(0.2)
                self.op_audio_queue.task_done()

            except queue.Empty:
                sleep(0.1)
            except Exception as e:
                print(f"Audio playback error: {e}")
                self.op_audio_queue.task_done()

    def generate_phrase_list(self, response):
        phrases_list, word_cnt, phrase = [], 0, ""
        for chunk in response:
            if chunk.choices[0].delta.content is not None:
                text = chunk.choices[0].delta.content
                if " " in text:
                    word_cnt += 1
                phrase = f"{phrase}{text}"
                if word_cnt > 4:
                    phrases_list.append(f"{phrase}")
                    phrase = ""
                    word_cnt = 0

        if len(phrase) > 0:
            phrases_list.append(phrase)
            phrase = ""
            word_cnt = 0
        return phrases_list

    def generate_hr_response(self, inp_query):
        message_list = []
        message_list.append({"content": [
            {"type": "text", "text": "You are a hr assistant who responds to any questions asked regarding hr."}],
            "role": "user"})
        message_list.append({"content": [
            {"type": "text", "text": "Ok.I will answer any hr related general query. Please provide user query"}],
            "role": "assistant"})
        message_list.append({"content": [
            {"type": "text", "text": f"{inp_query}. REMEMBER to generate response in less than 30 tokens."}],
            "role": "user"})
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=message_list,
            max_tokens=200,
            temperature=0.1,
            stream=True
        )
        return response

    def start(self, inp_str):
        """Start the streaming simulation"""
        # Start worker threadsm
        response = self.generate_hr_response(inp_str)
        self.phrases = self.generate_phrase_list(response)
        self.text_thread = threading.Thread(target=self.text_display_worker, args=(st.session_state.transcription_container, ))
        self.audio_thread = threading.Thread(target=self.audio_playback_worker)

        self.text_thread.daemon = True
        self.audio_thread.daemon = True
        self.text_thread.start()
        self.audio_thread.start()


        # Process each phrase
        for phrase in self.phrases:
            # Add to text queue
            self.text_queue.put(phrase)

            # Generate and queue audio
            self.generate_tts(f"{phrase}.")

        # Wait for queues to be empty
        self.text_queue.join()
        self.op_audio_queue.join()

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

    def process_speech(self):
        """Main processing loop"""
        try:
            print("\nAvailable audio devices:")
            print(sd.query_devices())

            print(f"\nUsing input device: {sd.query_devices(None, 'input')['name']}")

            with self.audio_stream():
                while True:
                    try:
                        if not self.bprocess:
                            break
                        # Start listening phase
                        print("\nListening... Speak now!")
                        st.session_state.status_container.info("Listening... Speak now!")
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
                                        st.session_state.status_container.info("Not enough speech detected, please speak again...")
                                        # yield "Not enough speech detected, please speak again...", None
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
                            st.session_state.status_container.info('Processing speech...')
                            print(f"Processing speech")
                            # Process the audio
                            current_buffer = self.audio_buffer.copy()
                            if self.selected_option == "OpenAI":
                                transcribed_text = self.transcribe_audio(current_buffer)
                                st.session_state.status_container.info('Audio Transcribed')
                                response = self.generate_hr_response(transcribed_text)
                                self.phrases = self.generate_phrase_list(response)
                                st.session_state.status_container.info('Textual response generated')

                                # self.text_thread = threading.Thread(target=self.text_display_worker, args=(st.session_state.transcription_container,))
                                self.audio_thread = threading.Thread(target=self.audio_playback_worker)

                                # self.text_thread.daemon = True
                                self.audio_thread.daemon = True
                                # self.text_thread.start()
                                self.audio_thread.start()
                                for phrase in self.phrases:
                                    # Add to text queue
                                    # self.text_queue.put(phrase)
                                    # Generate and queue audio
                                    print(f"Generating tts for phrase:{phrase}")
                                    self.generate_tts(f"{phrase}.")
                                    time.sleep(0.5)
                                self.op_audio_queue.join()
                                accumulated_text = ""
                                for phrase in self.phrases:
                                    accumulated_text += phrase
                                    self.display_bubble_message(accumulated_text)


                                # # Process each phrase
                                # for phrase in self.phrases:
                                #     # Add to text queue
                                #     self.text_queue.put(phrase)
                                #
                                #     # Generate and queue audio
                                #     self.generate_tts(f"{phrase}.")

                                # Wait for queues to be empty
                                # self.text_queue.join()

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

        except Exception as e:
            print(f"Error in process_speech: {e}")
            traceback.print_exc()
        finally:
            self.bprocess = True
            self.is_listening = False
            self.is_processing = False

    def initialize_app(self):
        if 'transcription_container' not in st.session_state:
            st.session_state.transcription_container = st.empty()
        if 'status_container' not in st.session_state:
            st.session_state.status_container = st.empty()
            st.session_state.transcription_container = st.empty()
        if 'is_listening' not in st.session_state:
            st.session_state.is_listening = False
            st.session_state.is_processing = False

    def streamlit_ui(self):
        st.title("Real-time Speech Processing")

        self.initialize_app()

        # UI Layout
        col1, col2, col3 = st.columns(3)

        with col1:
            # Model selection and configuration UI
            with st.container():
                st.subheader('Choose your model')
                selected_model = st.radio(
                    "Select Model",
                    options=["OpenAI", "Melotts"],
                    horizontal=True
                )
                self.selected_option = selected_model

                # Model-specific options
                if selected_model == "OpenAI":
                    voice_option = st.selectbox(
                        'Voice Option',
                        ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
                    )
                    if voice_option != self.voice_info:
                        self.voice_info = voice_option
                else:
                    # Melotts options
                    col11, col12 = st.columns(2)
                    with col11:
                        speed = st.selectbox("Speed", [0.5, 1.0, 1.5, 2.0])
                    with col12:
                        accent = st.selectbox(
                            "Accent",
                            ['EN-US', 'EN-BR', 'EN-INDIA', 'EN-AU', 'EN-Default']
                        )

        with col2:
            if st.button("Start Listening", key="start"):
                st.session_state.is_listening = True
                self.process_speech()

        with col3:
            if st.button("Stop", key="stop"):
                self.is_listening = False
                self.bprocess = False

                # Clear displays
                st.session_state.transcription_container.empty()
                st.session_state.status_container.empty()

                # Clean up resources
                self.clear_buffers()
                # if self.stream:
                #     self.stream.stop()
                #     self.stream.close()
                #     self.stream = None

if __name__ == "__main__":
    simulator = StreamingSimulator()
    simulator.streamlit_ui()
    # while True:
    #     inp= input("\nProvide your query:")
    #     if inp == "EXIT":
    #         break
    #     simulator.start(inp)

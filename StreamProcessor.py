# StreamProcessor.py
from queue import Queue
import threading
import streamlit as st
import pygame
import tempfile
import os
from typing import Optional
import time


class StreamProcessor:
    def __init__(self):
        self.text_queue = Queue()
        self.audio_queue = Queue()
        self.is_processing = False
        self.audio_thread: Optional[threading.Thread] = None
        self.text_thread: Optional[threading.Thread] = None
        pygame.mixer.init(frequency=24000)

    def start_processing(self):
        """Start the processing threads"""
        self.is_processing = True
        self.audio_thread = threading.Thread(target=self._process_audio_queue)
        self.text_thread = threading.Thread(target=self._process_text_queue)
        self.audio_thread.start()
        self.text_thread.start()

    def stop_processing(self):
        """Stop all processing threads"""
        self.is_processing = False
        if self.audio_thread:
            self.audio_thread.join()
        if self.text_thread:
            self.text_thread.join()
        self.clear_queues()

    def clear_queues(self):
        """Clear both queues"""
        while not self.text_queue.empty():
            self.text_queue.get()
        while not self.audio_queue.empty():
            self.audio_queue.get()

    def _play_audio_chunk(self, chunk_data):
        """Play a single audio chunk"""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_file:
            temp_file.write(chunk_data)
            temp_file_path = temp_file.name

        try:
            pygame.mixer.music.load(temp_file_path)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                if not self.is_processing:
                    pygame.mixer.music.stop()
                    break
                pygame.time.wait(10)
        finally:
            try:
                os.unlink(temp_file_path)
            except:
                pass

    def _process_audio_queue(self):
        """Process audio chunks from the queue"""
        while self.is_processing:
            try:
                if not self.audio_queue.empty():
                    chunk = self.audio_queue.get()
                    self._play_audio_chunk(chunk)
                else:
                    time.sleep(0.01)  # Short sleep to prevent CPU spinning
            except Exception as e:
                print(f"Error processing audio: {e}")

    def _process_text_queue(self):
        """Process text chunks from the queue"""
        accumulated_text = ""
        while self.is_processing:
            try:
                if not self.text_queue.empty():
                    text = self.text_queue.get()
                    accumulated_text += text
                    self.display_bubble_message(accumulated_text)
                else:
                    time.sleep(0.01)
            except Exception as e:
                print(f"Error processing text: {e}")

    def _update_text_display(self, text):
        """Update the Streamlit display with new text"""
        st.session_state.transcription_container.markdown(
            f'<div class="bubble">{text}</div>',
            unsafe_allow_html=True
        )

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
        with st.session_state.transcription_container:
            st.markdown(f'<div class="bubble">{message}</div>', unsafe_allow_html=True)
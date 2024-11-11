import pygame
import tempfile
import os
import io
import threading
import queue
from openai import OpenAI
from time import sleep

from pygame.pypm import Input


class StreamingSimulator:
    def __init__(self):
        # Initialize queues
        self.text_queue = queue.Queue()
        self.audio_queue = queue.Queue()

        # Initialize pygame mixer
        pygame.mixer.init()

        # Initialize OpenAI client
        self.client = OpenAI(api_key="sk-T_tRfijTOy8McMUCP_dO5z8PTwInF1l0K6HXp_50rAT3BlbkFJKcGo0d26hy9EgcZ2V5ahtuBwkeN8muRpL0COmWrbEA")

        # Configuration
        self.chunk_size = 1024 * 32  # 32KB chunks
        self.phrases = [
            "Hello, welcome to the.",
            "streaming demo.",
            "This is an example of.",
            "synchronized text and audio which will be mimicing streaming operation.",
            "Each phrase will be displayed and played sequentially."
        ]

        # Start worker threads
        self.text_thread = threading.Thread(target=self.text_display_worker)
        self.audio_thread = threading.Thread(target=self.audio_playback_worker)

    def generate_tts(self, text):
        """Generate TTS audio for given text and add to audio queue"""
        try:
            response = self.client.audio.speech.create(
                model="tts-1",
                voice="alloy",
                input=text,
                response_format="mp3"
            )

            current_chunk = io.BytesIO()
            accumulated_size = 0

            for chunk in response.iter_bytes():
                current_chunk.write(chunk)
                accumulated_size += len(chunk)

                if accumulated_size >= self.chunk_size:
                    self.audio_queue.put(current_chunk.getvalue())
                    current_chunk = io.BytesIO()
                    accumulated_size = 0

            # Add any remaining audio
            if accumulated_size > 0:
                self.audio_queue.put(current_chunk.getvalue())

        except Exception as e:
            print(f"TTS generation error: {e}")

    def play_chunk(self, chunk_data):
        """Play a single audio chunk"""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_file:
            temp_file.write(chunk_data)
            temp_file_path = temp_file.name

        try:
            pygame.mixer.music.load(temp_file_path)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                pygame.time.wait(10)
        except Exception as e:
            print(f"Error playing chunk: {e}")
        finally:
            try:
                os.unlink(temp_file_path)
            except:
                pass

    def text_display_worker(self):
        """Worker thread for displaying text"""
        while True:
            try:
                text = self.text_queue.get()
                print(f"\nDisplaying: {text}")
                self.text_queue.task_done()
            except queue.Empty:
                sleep(0.1)
            except Exception as e:
                print(f"Text display error: {e}")

    def audio_playback_worker(self):
        """Worker thread for playing audio chunks"""
        while True:
            try:
                chunk = self.audio_queue.get()
                self.play_chunk(chunk)
                self.audio_queue.task_done()
            except queue.Empty:
                sleep(0.1)
            except Exception as e:
                print(f"Audio playback error: {e}")

    def generate_phrase_list(self, inp:str):
        phrases_list = []
        all_word_list = inp.split(" ")
        for idx in range(0, len(all_word_list), 5):
            end_idx = idx + 5 if idx + 5 < len(all_word_list) else len(all_word_list)
            phrases_list.append(f"{' '.join(all_word_list[idx: end_idx])}.")
        return phrases_list

    def start(self, inp_str):
        """Start the streaming simulation"""
        # Start worker threadsm
        self.text_thread = threading.Thread(target=self.text_display_worker)
        self.audio_thread = threading.Thread(target=self.audio_playback_worker)
        self.phrases = self.generate_phrase_list(inp_str)
        self.text_thread.daemon = True
        self.audio_thread.daemon = True
        self.text_thread.start()
        self.audio_thread.start()


        # Process each phrase
        for phrase in self.phrases:
            # Add to text queue
            self.text_queue.put(phrase)

            # Generate and queue audio
            self.generate_tts(phrase)

        # Wait for queues to be empty
        self.text_queue.join()
        self.audio_queue.join()


if __name__ == "__main__":
    simulator = StreamingSimulator()
    while True:
        inp= input("Provide your query:")
        if inp == "EXIT":
            break
        simulator.start(inp)

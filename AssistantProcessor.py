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
        self.phrases = []
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
                print(f"{text}", end="")
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
                sleep(0.01)
            except queue.Empty:
                sleep(0.5)
            except Exception as e:
                print(f"Audio playback error: {e}")

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
        self.text_thread = threading.Thread(target=self.text_display_worker)
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
        self.audio_queue.join()


if __name__ == "__main__":
    simulator = StreamingSimulator()
    while True:
        inp= input("\nProvide your query:")
        if inp == "EXIT":
            break
        simulator.start(inp)

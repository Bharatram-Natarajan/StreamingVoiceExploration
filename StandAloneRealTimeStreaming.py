import os
from openai import OpenAI
import pygame
import io
import threading
import time
import queue
# from pydub import AudioSegment
# from pydub.playback import play
import tempfile


class OpenAITTSStreamer:
    def __init__(self, api_key=None):
        self.client = OpenAI(api_key=api_key)
        pygame.mixer.init(frequency=24000)  # TTS-1 uses 24kHz
        self.audio_queue = queue.Queue()
        self.is_playing = False
        self.current_chunk = io.BytesIO()
        self.chunk_size = 4096  # Adjust this value based on your needs

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

    def stream_text(self, text, voice="alloy", model="tts-1"):
        """Stream and play text with immediate chunk playback"""

        def stream_worker():
            try:
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
                        self.play_chunk(chunk_data)

                        # Reset for next chunk
                        self.current_chunk = io.BytesIO()
                        accumulated_size = 0

                # Play any remaining audio
                if accumulated_size > 0:
                    chunk_data = self.current_chunk.getvalue()
                    self.play_chunk(chunk_data)

            except Exception as e:
                print(f"Streaming error: {e}")

        # Start streaming in a separate thread
        thread = threading.Thread(target=stream_worker)
        thread.start()
        return thread

    def stream_long_text(self, text, chunk_size=1000):
        """Stream longer text by breaking it into sentence chunks"""
        # Split text into sentences (simple implementation)
        # sentences = text.replace('!', '.').replace('?', '.').split('.')
        # sentences = [s.strip() for s in sentences if s.strip()]
        sentences = text.split(" ")

        def play_sentences():
            current_chunk = []
            current_length = 0

            for sentence in sentences:
                if len(sentence.strip()) <= 0:
                    continue
                current_length += 1 #len(sentence)
                current_chunk.append(sentence)

                # When chunk is large enough, process it
                if current_length >= chunk_size:
                    chunk_text = ' '.join(current_chunk) + '.'
                    print(f"Chunk text:{chunk_text}")
                    thread = self.stream_text(chunk_text)
                    thread.join()  # Wait for current chunk to finish before starting next

                    current_chunk = []
                    current_length = 0

            # Process any remaining text
            if current_chunk:
                chunk_text = ' '.join(current_chunk) + '.'
                print(f"Chunk text:{chunk_text}")
                thread = self.stream_text(chunk_text)
                thread.join()

        thread = threading.Thread(target=play_sentences)
        thread.start()
        return thread


# Example usage
if __name__ == "__main__":
    tts_streamer = OpenAITTSStreamer(api_key="sk-T_tRfijTOy8McMUCP_dO5z8PTwInF1l0K6HXp_50rAT3BlbkFJKcGo0d26hy9EgcZ2V5ahtuBwkeN8muRpL0COmWrbEA")

    # Example: Stream a sentence with immediate playback
    print("Streaming with immediate chunk playback...")
    thread = tts_streamer.stream_text(
        "This is a test of real-time TTS streaming, where chunks are played as soon as they're generated.",
        voice="alloy"
    )
    print("Main program continues while audio streams...")
    thread.join()

    # Example: Stream longer text with immediate chunk playback
    print("\nStreaming longer text in real-time chunks...")
    long_text = """This is a demonstration of streaming longer text. 
                   Each sentence will be processed and played as soon as it's ready. 
                   This provides a more responsive experience for the listener."""
    thread = tts_streamer.stream_long_text(long_text, chunk_size=5)
    print("Main program continues while long audio streams...")
    thread.join()
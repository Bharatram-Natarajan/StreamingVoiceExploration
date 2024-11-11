import streamlit as st, os
from OpenAIStreamer import OpenAIStreamer
import pygame, tempfile, threading

OPENAI_VOICES = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
MELOTTS_SPEEDS = [0.5, 1.0, 1.5, 2.0]
MELOTTS_ACCENTS = ['EN-US', 'EN-BR', 'EN-INDIA', 'EN-AU', 'EN-Default']

## set the variables
st.title("Real-time Speech Processing")


def play_chunk(chunk_data):
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

# Initialize session state variables
if 'is_listening' not in st.session_state:
    st.session_state.is_listening = False
# if 'transcription_container' not in st.session_state:
st.session_state.transcription_container = st.empty()
# if 'status_container' not in st.session_state:
st.session_state.status_container = st.empty()
if 'speech_processor' not in st.session_state:
    api_key = "sk-T_tRfijTOy8McMUCP_dO5z8PTwInF1l0K6HXp_50rAT3BlbkFJKcGo0d26hy9EgcZ2V5ahtuBwkeN8muRpL0COmWrbEA"
    st.session_state.speech_processor = OpenAIStreamer()

def display_bubble_message(message):
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

# Create columns for buttons
col1, col2, col3 = st.columns(3)

with col1:
    with st.container():
        st.subheader('Choose your model')
        selected_model = st.radio(
            "Select Model",
            options=["OpenAI", "Melotts"],
            horizontal=True
        )
        st.session_state.speech_processor.selected_option = selected_model
        if selected_model == "OpenAI":
            st.subheader("OpenAI Voice Options")
            voice_option = st.selectbox('Voice Option', ["alloy", "echo", "fable", "onyx", "nova", "shimmer"])
            if voice_option != st.session_state.speech_processor.voice_info:
                st.session_state.speech_processor.voice_info = voice_option
        else:
            st.subheader("Melotts Configuration")
            col11, col12 = st.columns(2)
            with col11:
                selected_speed = st.selectbox(
                    "Choose Speed",
                    options=MELOTTS_SPEEDS,
                    help="Select the speech speed"
                )
                if selected_speed != st.session_state.speech_processor.selected_speed:
                    st.session_state.speech_processor.selected_speed = selected_speed
            with col12:
                selected_accent = st.selectbox(
                    "Choose Accent",
                    options=MELOTTS_ACCENTS,
                    help="Select the accent for speech"
                )
                if selected_accent != st.session_state.speech_processor.selected_accent:
                    st.session_state.speech_processor.selected_accent = selected_accent


with col2:
    if st.button("Start Listening", key="start"):
        st.session_state.is_listening = True
        prev_text, acc_text = "", ""
        for text, audio in st.session_state.speech_processor.process_speech():
            if not audio:
                st.session_state.status_container.info(text)
                prev_text, acc_text = "", ""
                continue
            if prev_text != text:
                acc_text += text
                display_bubble_message(acc_text)
                prev_text = text
            thread = threading.Thread(target=play_chunk, args=(audio,))
            thread.start()
            thread.join()



with col3:
    if st.button("Stop", key="stop"):
        st.session_state.speech_processor.bprocess = False
        st.session_state.is_listening = False
        # Clear both containers
        st.session_state.transcription_container.empty()
        st.session_state.status_container.empty()
        st.session_state.speech_processor.clear_buffers()
        if st.session_state.speech_processor.stream:
            st.session_state.speech_processor.stream.stop()
            st.session_state.speech_processor.stream.close()
            st.session_state.speech_processor.stream = None
            st.session_state.speech_processor.is_listening = False
            st.session_state.speech_processor.is_processing = False


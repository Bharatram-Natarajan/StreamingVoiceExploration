# Voice2VoiceDeveloper.py
import streamlit as st
from StreamProcessor import StreamProcessor
from OpenAIStreamer import OpenAIStreamer


def initialize_app():
    if 'stream_processor' not in st.session_state:
        st.session_state.stream_processor = StreamProcessor()
    if 'speech_processor' not in st.session_state:
        st.session_state.speech_processor = OpenAIStreamer()
    if 'transcription_container' not in st.session_state:
        st.session_state.transcription_container = st.empty()
    if 'status_container' not in st.session_state:
        st.session_state.status_container = st.empty()
    if 'is_listening' not in st.session_state:
        st.session_state.is_listening = False


def main():
    st.title("Real-time Speech Processing")

    initialize_app()

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

            # Model-specific options
            if selected_model == "OpenAI":
                voice_option = st.selectbox(
                    'Voice Option',
                    ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
                )
                st.session_state.speech_processor.voice_info = voice_option
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
            st.session_state.stream_processor.start_processing()

            # Process speech and queue results
            for text, audio in st.session_state.speech_processor.process_speech():
                if not audio:
                    st.session_state.status_container.info(text)
                    continue

                st.session_state.stream_processor.text_queue.put(text)
                st.session_state.stream_processor.audio_queue.put(audio)

                if not st.session_state.is_listening:
                    break

    with col3:
        if st.button("Stop", key="stop"):
            st.session_state.is_listening = False
            st.session_state.stream_processor.stop_processing()
            st.session_state.speech_processor.bprocess = False

            # Clear displays
            st.session_state.transcription_container.empty()
            st.session_state.status_container.empty()

            # Clean up resources
            st.session_state.speech_processor.clear_buffers()
            if st.session_state.speech_processor.stream:
                st.session_state.speech_processor.stream.stop()
                st.session_state.speech_processor.stream.close()
                st.session_state.speech_processor.stream = None


if __name__ == "__main__":
    main()
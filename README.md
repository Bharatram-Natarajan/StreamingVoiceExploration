# Real Time Streaming Exploration
The main objective of the experimentation is to explore the usage of end voice to voice system. The system would process the spoken user request for the relevant task and generate response in real time both in text as well as in voice.
## Requirements
The exploration of this pipeline is shown using 2 tts modules
- OpenAI tts-1 model
  - This model supports multi modality feature.
  - This model supports multi language.
  - This model has very good recognition capability of the Indian names, US names, UK names and so on.
  - This model provides live streaming operation hosted by OpenAI.
- Microsoft MeloTTS
  - This model provides accents for english speakers.
  - This model supports less multi language cultures.
  - This model response time is way quicker than Openai tts-1 model.
  - It generates 1.5 sec in cpu and way faster in gpu.

Follow the below codes for installing the required packages
```
1. pip install -r requirements.txt
2. pip install git+https://github.com/myshell-ai/MeloTTS.git
3. python -m unidic download
```
## Command to run the code
```
streamlit run OptimizedAudioUsingPyAudio.py
```

While running the code remember to first choose the options and the press 'Start Recording' button.
While running some utterances if you want to change the options then press 'Stop', change the options and then press 'Start Recording' again.

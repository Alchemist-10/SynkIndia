import os
import subprocess
import whisper
from deep_translator import GoogleTranslator
from gtts import gTTS

def extract_audio(video_path, output_audio_path):
    
    """Extracts raw audio from an input MP4 video file using ffmpeg."""
    command = f'ffmpeg -y -i "{video_path}" -ab 160k -ac 2 -ar 44100 -vn "{output_audio_path}"'
    subprocess.call(command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def process_audio_pipeline(video_path, target_lang="hi"):

    """Runs Speech-to-Text, Translation, and Text-to-Speech."""

    print("[1/3] Transcribing original audio with Whisper...")
    raw_audio = os.path.join("data", "processed", "temp_audio.wav")
    extract_audio(video_path, raw_audio)
    
    model = whisper.load_model("base")
    result = model.transcribe(raw_audio)
    original_text = result['text']
    print(f"Original Text Detected: {original_text}")
    
    print(f"[2/3] Translating text to language code: {target_lang}...")
    translated_text = GoogleTranslator(source='auto', target=target_lang).translate(original_text)
    print(f"Translated Text Generated: {translated_text}")
    
    print("[3/3] Generating translated TTS voice file...")
    tts_output = os.path.join("data", "processed", "translated_speech.mp3")
    tts = gTTS(text=translated_text, lang=target_lang, slow=False)
    tts.save(tts_output)
    
    # convert MP3 to WAV format for proper audio processing
    final_wav = os.path.join("data", "processed", "translated_speech.wav")
    subprocess.call(f'ffmpeg -y -i "{tts_output}" "{final_wav}"', shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    return final_wav
# import torch
# import numpy as np
# arr=np.array([1,2,3,4])
# print(arr*arr)
# import numpy as np
# arr=np.array([1,2,3,4])
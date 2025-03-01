import os
import pyaudio  # Handles audio input/output.
from vosk import Model, KaldiRecognizer  # Speech recognition framework
import requests  # pip install requests
import pyttsx3  # Converts text to speech
import json  # Parses JSON data


# List of blocked topics
BLOCKED_TOPICS = ["iot", "internet of things"]

# Load Vosk model for offline speech recognition
def load_vosk_model():
    model_path = "models/vosk-model-small-en-us"
    if not os.path.exists(model_path):
        raise FileNotFoundError("Vosk model not found! Download and place it in the 'models' directory.")
    return Model(model_path)

# Listen to microphone input and transcribe using Vosk
def listen_and_transcribe(model):
    rec = KaldiRecognizer(model, 16000)  
    audio = pyaudio.PyAudio()  

    stream = audio.open(
        format=pyaudio.paInt16, 
        channels=1,  
        rate=16000,  
        input=True,  
        frames_per_buffer=8192  
    )

    stream.start_stream()  

    print("Listening... Speak now.")
    try:
        while True:
            data = stream.read(4096)
            if rec.AcceptWaveform(data):  
                result = rec.Result()  
                text = json.loads(result).get('text', '')  
                if text:
                    return text
    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()

# Send the transcribed text to Ollama and get a response
def query_ollama(input_text):
    lower_input = input_text.lower()  
    if any(topic in lower_input for topic in BLOCKED_TOPICS):  
        return "This topic is in my red line. I will not answer it."

    url = "http://localhost:11434/api/generate"  
    payload = {"model": "llama3.1:8b", "prompt": input_text}  

    try:
        response = requests.post(url, json=payload, stream=True)  
        response.raise_for_status()  

        full_response = ""  

        for line in response.iter_lines(decode_unicode=True):  
            if line:
                data = json.loads(line)  
                full_response += data.get("response", "")
                if data.get("done", False):  
                    return full_response  

        return "Ollama could not generate a valid response."

    except requests.exceptions.RequestException as e:
        return f"Error: Unable to reach Ollama API. {e}"
    except json.JSONDecodeError:
        return "Error: Failed to decode Ollama response."

# Convert Ollama's response to speech
def speak_text(text):
    engine = pyttsx3.init()  
    engine.say(text)  
    engine.runAndWait()  

# Main function to integrate all components
def main():
    try:
        model = load_vosk_model()

        while True:
            user_input = listen_and_transcribe(model)
            print(f"You said: {user_input}")

            response = query_ollama(user_input)
            print(f"Ollama: {response}")

            speak_text(response)

    except KeyboardInterrupt:
        print("\nExiting. Goodbye!")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()

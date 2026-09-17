import speech_recognition as sr
import subprocess
import time
import os

# Initialize recognizer
recognizer = sr.Recognizer()

# Variable to store the process of mm.py
mm_process = None

def listen_for_command():
    with sr.Microphone() as source:
        print("Listening for command...")
        recognizer.adjust_for_ambient_noise(source)  # Adjust for ambient noise
        audio = recognizer.listen(source)

    try:
        command = recognizer.recognize_google(audio).lower()
        print(f"Command received: {command}")
        return command
    except sr.UnknownValueError:
        print("Sorry, I didn't catch that.")
        return None
    except sr.RequestError:
        print("Sorry, I'm having trouble with the speech recognition service.")
        return None

def run_mm_py():
    global mm_process
    if mm_process is None:
        print("Starting mm2.py...")
        mm_process = subprocess.Popen(['python', 'mm2.py'])
    else:
        print("mm.py is already running.")

def stop_mm_py():
    global mm_process
    if mm_process is not None:
        print("Stopping mm.py...")
        mm_process.terminate()
        mm_process = None
    else:
        print("mm.py is not running.")

if __name__ == "__main__":
    while True:
        command = listen_for_command()

        if command == 'start':
            run_mm_py()
        elif command == 'stop':
            stop_mm_py()

        # You can add a stop condition or some delay here if needed.
        time.sleep(1)

import numpy as np
import torch
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import librosa
import sounddevice as sd
from synthesizer import Synthesizer
from encoder import inference as encoder
from vocoder import inference as vocoder
from pathlib import Path
import os

# Load pre-trained models
encoder.load_model(Path("saved_models/encoder.pt"))
synthesizer = Synthesizer(Path("saved_models/synthesizer.pt"))
vocoder.load_model(Path("saved_models/vocoder.pt"))

# Global variables to store audio and text input
uploaded_audio = None
generated_audio = None

# Function to upload the voice sample
def upload_audio():
    global uploaded_audio
    audio_file_path = filedialog.askopenfilename(title="Select a voice sample", filetypes=[("Audio Files", "*.wav")])
    if audio_file_path:
        try:
            uploaded_audio, _ = librosa.load(audio_file_path, sr=16000)
            messagebox.showinfo("Success", "Voice sample uploaded successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Could not load audio file: {e}")

# Function to generate speech from text and cloned voice
def generate_speech():
    global generated_audio
    if uploaded_audio is None:
        messagebox.showerror("Error", "Please upload a voice sample first.")
        return
    
    input_text = text_entry.get("1.0", tk.END).strip()
    if not input_text:
        messagebox.showerror("Error", "Please enter some text to generate speech.")
        return
    
    # Start progress bar
    progress_bar.start()
    
    try:
        # Preprocess the uploaded audio and get speaker embedding
        preprocessed_wav = encoder.preprocess_wav(uploaded_audio)
        embed = encoder.embed_utterance(preprocessed_wav)
        
        # Generate Mel-spectrogram from the input text
        mel_spectrogram = synthesizer.synthesize_spectrograms([input_text], [embed])[0]
        
        # Convert Mel-spectrogram to audio waveform using the vocoder
        generated_audio = vocoder.infer_waveform(mel_spectrogram)
        
        # Stop progress bar
        progress_bar.stop()
        messagebox.showinfo("Success", "Speech generated successfully!")
    except Exception as e:
        progress_bar.stop()
        messagebox.showerror("Error", f"An error occurred during speech generation: {e}")

# Function to play the generated audio
def play_audio():
    global generated_audio
    if generated_audio is not None:
        sd.play(generated_audio, samplerate=22050)
        sd.wait()
    else:
        messagebox.showerror("Error", "No audio generated to play. Please generate speech first.")

# Function to save the generated audio
def save_audio():
    global generated_audio
    if generated_audio is None:
        messagebox.showerror("Error", "No audio to save. Please generate speech first.")
        return
    
    save_path = filedialog.asksaveasfilename(defaultextension=".wav", filetypes=[("WAV files", "*.wav")])
    if save_path:
        try:
            librosa.output.write_wav(save_path, generated_audio, sr=22050)
            messagebox.showinfo("Success", f"Audio saved successfully at {save_path}!")
        except Exception as e:
            messagebox.showerror("Error", f"Could not save audio: {e}")

# Create the main window
window = tk.Tk()
window.title("TTSer")
window.geometry("400x350")

# Add a button to upload audio file
upload_button = tk.Button(window, text="Upload Voice Sample", command=upload_audio)
upload_button.pack(pady=10)

# Add a text box for text input
text_label = tk.Label(window, text="Enter text:")
text_label.pack(pady=5)
text_entry = tk.Text(window, height=4, width=40)
text_entry.pack(pady=5)

# Add a button to generate speech
generate_button = tk.Button(window, text="Generate Speech", command=generate_speech)
generate_button.pack(pady=10)

# Add a button to play the generated speech
play_button = tk.Button(window, text="Play Generated Speech", command=play_audio)
play_button.pack(pady=10)

# Add a progress bar to show the status of speech generation
progress_bar = ttk.Progressbar(window, mode='indeterminate')
progress_bar.pack(pady=5)

# Add a button to save the generated speech
save_button = tk.Button(window, text="Save Generated Speech", command=save_audio)
save_button.pack(pady=10)

# Start the Tkinter loop
window.mainloop()

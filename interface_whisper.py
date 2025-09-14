import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import sounddevice as sd
import scipy.io.wavfile as wav
import numpy as np
import os
import whisper  
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from threading import Thread

class AudioAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Speech Emotion Recognition")
        self.root.geometry("850x600")
        self.root.configure(bg="#f5f5f5")
        
        # Style Configuration
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.style.configure('TFrame', background="#f5f5f5")
        self.style.configure('TButton', font=('Arial', 11), padding=8)
        self.style.configure('TLabel', background="#f5f5f5", font=('Arial', 11))
        self.style.configure('Title.TLabel', font=('Arial', 14, 'bold'))
        self.style.map('Primary.TButton', 
                      foreground=[('active', 'white'), ('!active', 'white')],
                      background=[('active', '#3e7cb1'), ('!active', '#4a90e2')])
        self.style.map('Danger.TButton', 
                      foreground=[('active', 'white'), ('!active', 'white')],
                      background=[('active', '#c23b22'), ('!active', '#e74c3c')])
        self.whisper_lang_codes = {
        "ar": "arabic",
        "en": "english",
        "tr": "turkish"
    }
        # Model paths
        self.cache_path = os.path.expanduser("~/.cache/huggingface/hub")
        self.audio_path = ""
        self.selected_lang = "en"  # Default to English
        self.is_recording = False
        

        self.sentiment_models = {
            "en": "j-hartmann/emotion-english-distilroberta-base",
            "tr": "coltekin/berturk-tremo",
            "ar": "CAMeL-Lab/bert-base-arabic-camelbert-da-sentiment"
        }
        
        self.whisper_model = None
        
        self.setup_ui()
    
    def load_whisper_model(self):
        if self.whisper_model is None:
            self.whisper_model = whisper.load_model("small")
        return self.whisper_model
    
    def setup_ui(self):
        # Main container
        main_frame = ttk.Frame(self.root, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title bar
        title_frame = ttk.Frame(main_frame)
        title_frame.pack(fill=tk.X, pady=(0, 20))
        
        ttk.Label(
            title_frame, 
            text="Speech Emotion Recognition NLP Project", 
        ).pack(side=tk.LEFT)
        ttk.Label(
            title_frame, 
            text="Developed by:\n Bashar Alkhawlani\n        Muhammed Hacomar", 
        ).pack(side=tk.RIGHT)
        
        # Language selection
        lang_frame = ttk.Frame(main_frame)
        lang_frame.pack(fill=tk.X, pady=(0, 15))
        
        ttk.Label(
            lang_frame, 
            text="Language:", 
            font=('Arial', 11)
        ).pack(side=tk.LEFT, padx=10)
        
        self.lang_var = tk.StringVar(value="English")
        self.lang_options = ["Arabic", "English", "Turkish"]
        
        self.lang_menu_btn = ttk.Menubutton(lang_frame, textvariable=self.lang_var)
        self.lang_menu_btn.pack(side=tk.LEFT)
        
        self.lang_menu = tk.Menu(self.lang_menu_btn, tearoff=0)
        self.lang_menu_btn.configure(menu=self.lang_menu)
        
        for lang in self.lang_options:
            self.lang_menu.add_radiobutton(
                label=lang,
                variable=self.lang_var,
                value=lang,
                command=lambda l=lang: self.set_language(l)
            )
        
        # Audio input frame
        input_frame = ttk.LabelFrame(
            main_frame, 
            text="Audio Input",
            padding=15
        )
        input_frame.pack(fill=tk.X, pady=(0, 20))
        
        # Control buttons
        btn_frame = ttk.Frame(input_frame)
        btn_frame.pack(fill=tk.X, pady=5)
        
        btn_choose = ttk.Button(
            btn_frame, 
            text="📂 Select Audio File", 
            command=self.choose_file,
            style='Primary.TButton'
        )
        btn_choose.pack(side=tk.LEFT, padx=5)
        
        self.record_btn = ttk.Button(
            btn_frame, 
            text="🎙️ Start Recording", 
            command=self.start_recording,
            style='Primary.TButton'
        )
        self.record_btn.pack(side=tk.LEFT, padx=5)
        
        self.stop_record_btn = ttk.Button(
            btn_frame, 
            text="⏹️ Stop Recording", 
            command=self.stop_recording,
            style='Danger.TButton'
        )
        self.stop_record_btn.pack(side=tk.LEFT, padx=5)
        self.stop_record_btn.pack_forget()
        
        self.label_file = ttk.Label(
            input_frame, 
            text="No file selected",
            foreground="#666666",
            font=('Arial', 11)
        )
        self.label_file.pack(fill=tk.X, pady=(10, 0))
        
        # Analyze button
        self.btn_analyze = ttk.Button(
            main_frame, 
            text="🔍 Analyze File", 
            command=self.start_analysis_thread,
            style='Primary.TButton'
        )
        self.btn_analyze.pack(pady=(10, 20))
        
        # Results frame
        result_frame = ttk.LabelFrame(
            main_frame, 
            text="Analysis Results",
            padding=15
        )
        result_frame.pack(fill=tk.BOTH, expand=True)
        
        self.result_box = tk.Text(
            result_frame, 
            height=18,
            font=('Arial', 11),
            wrap=tk.WORD,
            padx=10,
            pady=10,
            bg="#ffffff",
            relief=tk.FLAT
        )
        self.result_box.pack(fill=tk.BOTH, expand=True)
        
        # Status bar
        self.status_label = ttk.Label(
            main_frame, 
            text="Ready",
            foreground="#2ecc71",
            font=('Arial', 10)
        )
        self.status_label.pack(fill=tk.X, pady=(10, 0))
    
    def set_language(self, val):
        lang_map = {
            "Arabic": "ar",
            "English": "en", 
            "Turkish": "tr"
        }
        self.selected_lang = lang_map[val]
        self.lang_var.set(val)
    
    def choose_file(self):
        self.audio_path = filedialog.askopenfilename(
            filetypes=[("Audio Files", "*.wav *.mp3 *.ogg")]
        )
        if self.audio_path:
            self.label_file.config(
                text=f"📁 Selected: {os.path.basename(self.audio_path)}"
            )
    
    def start_recording(self):
        self.is_recording = True
        self.record_btn.config(state=tk.DISABLED)
        self.stop_record_btn.pack(side=tk.LEFT, padx=5)
        self.status_label.config(text="Recording... Speak now!", foreground="#3498db")
        self.root.update()
        
        self.recording_thread = Thread(target=self.record_audio)
        self.recording_thread.start()
    
    def stop_recording(self):
        self.is_recording = False
        self.record_btn.config(state=tk.NORMAL)
        self.stop_record_btn.pack_forget()
    
    def record_audio(self):
        try:
            fs = 16000
            audio = []
            
            def callback(indata, frames, time, status):
                if self.is_recording:
                    audio.append(indata.copy())
                else:
                    raise sd.CallbackStop
            
            with sd.InputStream(samplerate=fs, channels=1, callback=callback):
                while self.is_recording:
                    sd.sleep(100)
            
            if audio:
                audio = np.concatenate(audio)
                self.audio_path = "recording.wav"
                wav.write(self.audio_path, fs, audio)
                self.label_file.config(text=f"🎤 Recording saved: {self.audio_path}")
                self.status_label.config(text="Recording saved successfully!", foreground="#2ecc71")
        
        except Exception as e:
            self.status_label.config(text=f"Recording error: {str(e)}", foreground="#e74c3c")
    
    def start_analysis_thread(self):
        if not self.audio_path:
            messagebox.showerror("Error", "Please select or record an audio file first")
            return
            
        self.btn_analyze.config(state=tk.DISABLED)
        self.status_label.config(text="Analyzing... Please wait", foreground="#3498db")
        self.result_box.delete(1.0, tk.END)
        self.result_box.insert(tk.END, "Processing audio file...\n")
        self.root.update()
        
        thread = Thread(target=self.analyze_audio)
        thread.start()
        self.root.after(100, self.check_thread, thread)
    
    def check_thread(self, thread):
        if thread.is_alive():
            self.root.after(100, self.check_thread, thread)
        else:
            self.btn_analyze.config(state=tk.NORMAL)
            self.status_label.config(text="Analysis complete", foreground="#2ecc71")
    
    def load_sentiment_model(self):
        model_name = self.sentiment_models.get(self.selected_lang)
        if not model_name:
            raise Exception(f"No sentiment model for language: {self.selected_lang}")
        
        try:
            if self.selected_lang == "en":
                return pipeline(
                    "text-classification", 
                    model=model_name,
                    return_all_scores=True
                )
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForSequenceClassification.from_pretrained(model_name)
                return pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
        except Exception as e:
            raise Exception(f"Sentiment model error: {str(e)}")
    
    def analyze_audio(self):
        try:
            model = self.load_whisper_model()
            whisper_lang = self.whisper_lang_codes.get(self.selected_lang, "english")

            result = model.transcribe(
            self.audio_path,
            language=whisper_lang )

            text = result["text"]       

            self.result_box.delete(1.0, tk.END)
            self.result_box.insert(tk.END, f"📝 Extracted Text:\n{text}\n\n")

            
            if text.strip():
                sentiment_model = self.load_sentiment_model()
                self.result_box.insert(tk.END, "🎯 Sentiment Analysis:\n")
                
                if self.selected_lang == "en":
                    emotions = sentiment_model(text)
                    top_score = 0
                    top_emotion = ""
                    for emo in emotions[0]:
                        if emo['score'] > top_score:
                            top_score = emo['score']
                            top_emotion = emo['label']
                        self.result_box.insert(tk.END, f"{emo['label']}: {emo['score']:.2f}\n")
                    self.result_box.insert(tk.END, f"\nThe emotion:\n[    {top_emotion}   ]")
                else:
                    result = sentiment_model(text)[0]
                    self.result_box.insert(tk.END, f"The Emotion: {result['label']}\n")
            else:
                self.result_box.insert(tk.END, "⚠️ No text recognized in the audio file\n")
                
        except Exception as e:
            self.result_box.insert(tk.END, f"\n❌ Error: {str(e)}\n")
            messagebox.showerror("Error", f"Analysis failed:\n{str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = AudioAnalyzerApp(root)
    root.mainloop()
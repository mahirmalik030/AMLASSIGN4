import os
import re
import io
import uuid
import base64
import numpy as np
from PIL import Image
import cv2

# Set Keras backend to PyTorch before importing Keras
os.environ["KERAS_BACKEND"] = "torch"
import keras

from flask import Flask, request, jsonify, render_template, redirect, url_for, Request
from transformers import AutoTokenizer, AutoModelForSequenceClassification, WhisperProcessor, WhisperForConditionalGeneration, MarianTokenizer, MarianMTModel, pipeline
import soundfile as sf
import librosa

class CustomRequest(Request):
    max_form_memory_size = 16 * 1024 * 1024  # 16MB limit for form memory fields (like base64 canvas uploads)

app = Flask(__name__)
app.request_class = CustomRequest
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit for uploaded files
app.config['UPLOAD_FOLDER'] = os.path.join(os.getcwd(), 'uploads')

# ---------------------------------------------------------
# LOAD MODELS
# ---------------------------------------------------------
print("Loading CNN Model...")
try:
    cnn_model = keras.models.load_model("models/gender_cnn.h5", compile=False)
except Exception as e:
    print(f"Error loading CNN model: {e}")
    cnn_model = None

print("Loading SER Voice Sentiment Model...")
try:
    voice_sentiment_model = keras.models.load_model("models/voice_sentiment_model.h5", compile=False)
except Exception as e:
    print(f"Error loading SER model: {e}")
    voice_sentiment_model = None

print("Loading English -> Urdu MarianMT Model...")
try:
    if os.path.exists("models/translation_model/en_ur"):
        tok_en_ur = MarianTokenizer.from_pretrained("models/translation_model/en_ur")
        mod_en_ur = MarianMTModel.from_pretrained("models/translation_model/en_ur")
    else:
        print("Local en-ur translation model missing, loading from HuggingFace Hub...")
        tok_en_ur = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-ur")
        mod_en_ur = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-ur")
except Exception as e:
    print(f"Error loading en-ur translation model: {e}")
    tok_en_ur, mod_en_ur = None, None

print("Loading Urdu -> English MarianMT Model...")
try:
    if os.path.exists("models/translation_model/ur_en"):
        tok_ur_en = MarianTokenizer.from_pretrained("models/translation_model/ur_en")
        mod_ur_en = MarianMTModel.from_pretrained("models/translation_model/ur_en")
    else:
        print("Local ur-en translation model missing, loading from HuggingFace Hub...")
        tok_ur_en = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-ur-en")
        mod_ur_en = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-ur-en")
except Exception as e:
    print(f"Error loading ur-en translation model: {e}")
    tok_ur_en, mod_ur_en = None, None

print("Loading Whisper Model...")
try:
    if os.path.exists("models/whisper_model"):
        whisper_processor = WhisperProcessor.from_pretrained("models/whisper_model")
        whisper_model = WhisperForConditionalGeneration.from_pretrained("models/whisper_model")
    else:
        print("Local whisper model missing, loading from HuggingFace Hub...")
        whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
        whisper_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")
except Exception as e:
    print(f"Error loading Whisper model: {e}")
    whisper_processor, whisper_model = None, None

# Load Haar cascade face classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Create uploads folders on startup
os.makedirs(os.path.join(app.config['UPLOAD_FOLDER'], 'images'), exist_ok=True)
os.makedirs(os.path.join(app.config['UPLOAD_FOLDER'], 'audio'), exist_ok=True)
os.makedirs(os.path.join(app.config['UPLOAD_FOLDER'], 'text'), exist_ok=True)

# Helper function to transcribe audio
def transcribe_audio_file(audio_path, language=None):
    if not whisper_model or not whisper_processor:
        return "Whisper model not loaded."
    try:
        # Try loading with soundfile first (fast, reliable for WAV, no ffmpeg required)
        try:
            y, sr = sf.read(audio_path)
            # Convert multi-channel (stereo) to mono by averaging
            if len(y.shape) > 1:
                y = np.mean(y, axis=1)
            if sr != 16000:
                y = librosa.resample(y, orig_sr=sr, target_sr=16000)
        except Exception as sf_err:
            print(f"soundfile failed, falling back to librosa: {sf_err}")
            y, sr = librosa.load(audio_path, sr=16000)
            
        input_features = whisper_processor(y, sampling_rate=16000, return_tensors="pt").input_features
        
        generate_kwargs = {}
        if language:
            forced_decoder_ids = whisper_processor.get_decoder_prompt_ids(language=language, task="transcribe")
            generate_kwargs["forced_decoder_ids"] = forced_decoder_ids
            
        predicted_ids = whisper_model.generate(input_features, **generate_kwargs)
        transcription = whisper_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        return transcription.strip()
    except Exception as e:
        print(f"Transcription error: {e}")
        return f"[Transcription Error: {e}]"

# Translation Helpers
def clean_and_translate_sentence(sentence, tokenizer, model, direction):
    s = sentence.strip()
    if not s:
        return ""
        
    ends_with = ""
    if s.endswith("۔"):
        ends_with = "۔"
        s = s[:-1].strip()
    elif s.endswith("؟"):
        ends_with = "؟"
        s = s[:-1].strip()
    elif s.endswith("."):
        ends_with = "."
        s = s[:-1].strip()
    elif s.endswith("?"):
        ends_with = "?"
        s = s[:-1].strip()
    elif s.endswith("!"):
        ends_with = "!"
        s = s[:-1].strip()

    if not s:
        return ends_with

    inputs = tokenizer(s, return_tensors="pt")
    translated_tokens = model.generate(**inputs)
    translated_text = tokenizer.decode(translated_tokens[0], skip_special_tokens=True).strip()
    
    if ends_with:
        # Strip any trailing punctuation that the model decoded to avoid duplicates
        while translated_text and translated_text[-1] in ".?!۔؟":
            translated_text = translated_text[:-1].strip()
            
        if direction == 'ur_en':
            if ends_with == "۔":
                translated_text += "."
            elif ends_with == "؟":
                translated_text += "?"
            else:
                translated_text += ends_with
        else:
            if ends_with == ".":
                translated_text += "۔"
            elif ends_with == "?":
                translated_text += "؟"
            else:
                translated_text += ends_with
            
    return translated_text

def perform_translation(text, tokenizer, model, direction):
    if not text:
        return ""
    # Split using lookbehind for sentence-ending punctuation followed by optional spaces
    sentences = re.split(r'(?<=[.?!۔؟])\s*', text)
    translated_sentences = []
    for s in sentences:
        s = s.strip()
        if s:
            translated_sentences.append(clean_and_translate_sentence(s, tokenizer, model, direction))
    return " ".join([ts for ts in translated_sentences if ts])

# ---------------------------------------------------------
# ROUTES
# ---------------------------------------------------------
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/gender')
def gender_module():
    return render_template('gender.html')

@app.route('/sentiment')
def sentiment_module():
    return render_template('sentiment.html')

@app.route('/translation')
def translation_module():
    return render_template('translation.html')

# ---------------------------------------------------------
# API ENDPOINTS
# ---------------------------------------------------------

# MODULE 1: Gender Classification
@app.route('/api/predict-gender', methods=['POST'])
def api_predict_gender():
    if not cnn_model:
        return jsonify({"error": "CNN model is not loaded on server."}), 500
        
    try:
        file = request.files.get('image')
        image_base64 = request.form.get('image_base64')
        
        unique_id = str(uuid.uuid4())
        img_filename = f"img_{unique_id}.png"
        img_path = os.path.join(app.config['UPLOAD_FOLDER'], 'images', img_filename)
        
        # Save the uploaded or captured image
        if file:
            file.save(img_path)
        elif image_base64:
            # Captured image base64 decoding
            if ',' in image_base64:
                image_base64 = image_base64.split(',')[1]
            img_data = base64.b64decode(image_base64)
            with open(img_path, 'wb') as f:
                f.write(img_data)
        else:
            return jsonify({"error": "No image provided."}), 400

        # Open image with PIL
        pil_img = Image.open(img_path).convert('RGB')
        
        # Face Detection using OpenCV with multi-pass and scale robustness
        open_cv_image = np.array(pil_img) 
        # Convert RGB to BGR for OpenCV
        open_cv_image_bgr = open_cv_image[:, :, ::-1].copy()
        
        # Resize dynamically for robust Haar Cascades if image is very large
        h_orig, w_orig = open_cv_image_bgr.shape[:2]
        if w_orig > 600:
            scale = 600.0 / w_orig
            dsize = (600, int(h_orig * scale))
            img_detect = cv2.resize(open_cv_image_bgr, dsize)
        else:
            img_detect = open_cv_image_bgr
            scale = 1.0
            
        gray = cv2.cvtColor(img_detect, cv2.COLOR_BGR2GRAY)
        
        # Pass 1: Standard detection
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        # Pass 2: Fallback with lenient settings
        if len(faces) == 0:
            faces = face_cascade.detectMultiScale(gray, 1.05, 3)
            
        face_detected = False
        cropped_img_filename = None
        cropped_img_url = None
        
        if len(faces) > 0:
            # Map back coordinates to original dimensions
            x, y, w, h = faces[0]
            x = int(x / scale)
            y = int(y / scale)
            w = int(w / scale)
            h = int(h / scale)
            
            # Crop face area with 10% padding
            pad_w = int(w * 0.1)
            pad_h = int(h * 0.1)
            y1 = max(0, y - pad_h)
            y2 = min(h_orig, y + h + pad_h)
            x1 = max(0, x - pad_w)
            x2 = min(w_orig, x + w + pad_w)
            
            face_crop = open_cv_image[y1:y2, x1:x2]
            pil_face = Image.fromarray(face_crop)
            face_detected = True
        else:
            # Fallback: Crop center 60%
            crop_w = int(w_orig * 0.6)
            crop_h = int(h_orig * 0.6)
            x1 = (w_orig - crop_w) // 2
            y1 = (h_orig - crop_h) // 2
            face_crop = open_cv_image[y1:y1+crop_h, x1:x1+crop_w]
            pil_face = Image.fromarray(face_crop)
            face_detected = False
            
        # Save the cropped/fallback image for frontend inspection
        cropped_img_filename = f"crop_{unique_id}.png"
        cropped_img_path = os.path.join(app.config['UPLOAD_FOLDER'], 'images', cropped_img_filename)
        pil_face.save(cropped_img_path)
        cropped_img_url = url_for('static_upload', filename=f"images/{cropped_img_filename}")
        
        # CNN Preprocessing (RGB model input shape 64x64x3)
        resized_pil = pil_face.resize((64, 64))
        # Normalize to [-1.0, 1.0] to match retrained model
        img_array = (np.array(resized_pil, dtype=np.float32) / 255.0 - 0.5) * 2.0
        # Shape [1, 64, 64, 3]
        img_tensor = np.expand_dims(img_array, axis=0)
        
        # Sigmoid output prediction
        prediction = cnn_model.predict(img_tensor)[0]
        prob_female = float(prediction[0])
        prob_male = 1.0 - prob_female
        
        if prob_male > prob_female:
            gender = "Male"
            confidence = prob_male * 100.0
        else:
            gender = "Female"
            confidence = prob_female * 100.0
            
        return jsonify({
            "gender": gender,
            "confidence": f"{confidence:.2f}%",
            "face_detected": face_detected,
            "original_image_url": url_for('static_upload', filename=f"images/{img_filename}"),
            "cropped_image_url": cropped_img_url,
            "preprocessing": [
                "1. Loaded image and converted to RGB color space.",
                "2. Resized image dynamically for robust OpenCV scale classification.",
                "3. Applied multi-pass OpenCV Haar Cascade Frontal Face Classifier.",
                f"4. {'Face successfully detected and cropped.' if face_detected else 'No face detected; applied central 60% cropping fallback.'}",
                "5. Resized RGB cropped region to 64x64 pixels (CNN input size).",
                "6. Normalized pixel intensities to [-1.0, 1.0] range.",
                "7. Converted to batch tensor shape [1, 64, 64, 3].",
                "8. Ran pre-trained Keras CNN forward pass."
            ]
        })
        
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


# Helper to extract MFCC from audio file for SER inference
def extract_mfcc_inference(audio_path):
    try:
        y, sr = sf.read(audio_path)
        if len(y.shape) > 1:
            y = np.mean(y, axis=1)
        if sr != 16000:
            y = librosa.resample(y, orig_sr=sr, target_sr=16000)
            sr = 16000
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfccs_scaled = np.mean(mfccs.T, axis=0)
        return mfccs_scaled
    except Exception as sf_err:
        print(f"soundfile failed in inference, using librosa: {sf_err}")
        try:
            y, sr = librosa.load(audio_path, sr=16000)
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
            mfccs_scaled = np.mean(mfccs.T, axis=0)
            return mfccs_scaled
        except Exception as e:
            print(f"Failed to extract MFCC: {e}")
            return None

# MODULE 2: Voice Sentiment Analysis (SER Model)
@app.route('/api/analyze-sentiment', methods=['POST'])
def api_analyze_sentiment():
    if not voice_sentiment_model:
        return jsonify({"error": "SER Voice Sentiment model is not loaded on server."}), 500
        
    file = request.files.get('audio')
    if not file:
        return jsonify({"error": "No audio file provided."}), 400
        
    unique_id = str(uuid.uuid4())
    audio_filename = f"audio_{unique_id}.wav"
    audio_path = os.path.join(app.config['UPLOAD_FOLDER'], 'audio', audio_filename)
    file.save(audio_path)
    
    try:
        # Step 1: Speech-to-Text using Whisper (as secondary/transcription feature)
        transcription = transcribe_audio_file(audio_path, language='en')
        if transcription.startswith("[Transcription Error"):
            transcription = "[Could not transcribe voice audio]"
            
        # Step 2: Speech Emotion Recognition (SER) using custom trained Keras model
        features = extract_mfcc_inference(audio_path)
        if features is None or len(features) != 40:
            return jsonify({"error": "Could not extract MFCC features from audio."}), 400
            
        features_tensor = np.expand_dims(features, axis=0)
        prediction = voice_sentiment_model.predict(features_tensor)[0]
        
        # Class mappings: 0 -> Neutral, 1 -> Happy, 2 -> Sad, 3 -> Angry
        emotion_labels = ["Neutral", "Happy", "Sad", "Angry"]
        predicted_class_idx = int(np.argmax(prediction))
        sentiment = emotion_labels[predicted_class_idx]
        confidence = float(prediction[predicted_class_idx]) * 100.0
        
        # Populating chart data with 4 classes
        chart_data = {
            "neutral": float(prediction[0]) * 100.0,
            "happy": float(prediction[1]) * 100.0,
            "sad": float(prediction[2]) * 100.0,
            "angry": float(prediction[3]) * 100.0
        }
        
        return jsonify({
            "transcription": transcription,
            "sentiment": sentiment,
            "confidence": f"{confidence:.2f}%",
            "visualization": chart_data,
            "original_audio_url": url_for('static_upload', filename=f"audio/{audio_filename}"),
            "preprocessing": [
                "1. Saved client audio file to local uploads server.",
                "2. Read and resampled audio to 16kHz mono.",
                "3. Extracted 40-dimensional Mel-Frequency Cepstral Coefficients (MFCCs).",
                "4. Averaged MFCC features over time into a [1, 40] feature vector.",
                "5. Ran pre-trained Keras Speech Emotion Classifier MLP model.",
                "6. Extracted softmax probability distribution across Neutral, Happy, Sad, and Angry classes.",
                "7. (Background task) Transcribed speech to text using Whisper-Tiny."
            ]
        })
        
    except Exception as e:
        return jsonify({"error": f"Speech Emotion Recognition failed: {str(e)}"}), 500


# MODULE 3: Language Translation
@app.route('/api/translate', methods=['POST'])
def api_translate():
    direction = request.form.get('direction', 'en_ur') # 'en_ur' or 'ur_en'
    text = request.form.get('text', '').strip()
    audio_file = request.files.get('audio')
    
    unique_id = str(uuid.uuid4())
    transcription = ""
    audio_url = None
    
    # Check if voice input
    if audio_file:
        audio_filename = f"trans_audio_{unique_id}.wav"
        audio_path = os.path.join(app.config['UPLOAD_FOLDER'], 'audio', audio_filename)
        audio_file.save(audio_path)
        audio_url = url_for('static_upload', filename=f"audio/{audio_filename}")
        
        # Transcribe using Whisper (forcing English for en->ur, Urdu for ur->en)
        if direction == 'en_ur':
            transcription = transcribe_audio_file(audio_path, language='en')
        else:
            transcription = transcribe_audio_file(audio_path, language='ur')
        source_text = transcription
    else:
        source_text = text
        
    if not source_text:
        return jsonify({"error": "No input text or speech provided."}), 400
        
    try:
        preprocessing = []
        if audio_file:
            preprocessing.append("1. Captured audio input stream via client microphone.")
            preprocessing.append("2. Loaded audio file and resampled to 16kHz.")
            preprocessing.append(f"3. Ran Whisper-Tiny speech decoder forced to language: {'English' if direction == 'en_ur' else 'Urdu'}")
            preprocessing.append(f"4. Transcribed Speech: \"{source_text}\"")
        else:
            preprocessing.append("1. Accepted direct text entry from user.")
            
        if direction == 'en_ur':
            if not mod_en_ur or not tok_en_ur:
                return jsonify({"error": "English-Urdu Translation model is not loaded."}), 500
            
            preprocessing.append("2. Split input text into individual sentences to prevent token limit hallucinations.")
            preprocessing.append("3. Preprocessed each sentence to clean and isolate trailing punctuation.")
            preprocessing.append("4. Tokenized and translated each sentence using English->Urdu MarianMT Model.")
            preprocessing.append("5. Restored appropriate Urdu sentence-ending punctuation (۔/؟/!).")
            translated_text = perform_translation(source_text, tok_en_ur, mod_en_ur, 'en_ur')
            preprocessing.append("6. Joined all translated sentences back into a coherent paragraph.")
        else:
            if not mod_ur_en or not tok_ur_en:
                return jsonify({"error": "Urdu-English Translation model is not loaded."}), 500
                
            preprocessing.append("2. Split input text into individual sentences to prevent token limit hallucinations.")
            preprocessing.append("3. Preprocessed each sentence to clean and isolate trailing punctuation (resolving tokenizer collisions).")
            preprocessing.append("4. Tokenized and translated each sentence using Urdu->English MarianMT Model.")
            preprocessing.append("5. Restored appropriate English sentence-ending punctuation (./?/!).")
            translated_text = perform_translation(source_text, tok_ur_en, mod_ur_en, 'ur_en')
            preprocessing.append("6. Joined all translated sentences back into a coherent paragraph.")
            
        return jsonify({
            "source_text": source_text,
            "translated_text": translated_text,
            "audio_url": audio_url,
            "preprocessing": preprocessing
        })
        
    except Exception as e:
        return jsonify({"error": f"Translation failed: {str(e)}"}), 500


# Endpoint to serve uploaded static files
@app.route('/uploads/<path:filename>')
def static_upload(filename):
    from flask import send_from_directory
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == '__main__':
    app.run(debug=True, port=5000)

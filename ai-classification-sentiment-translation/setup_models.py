import os
import math
import struct
import wave
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

# Set Keras backend to PyTorch
os.environ["KERAS_BACKEND"] = "torch"
import keras
from keras import layers, models

# Ensure directories exist
os.makedirs("datasets/gender/male", exist_ok=True)
os.makedirs("datasets/gender/female", exist_ok=True)
os.makedirs("datasets/sentiment/Actor_01", exist_ok=True)
os.makedirs("datasets/translation", exist_ok=True)
os.makedirs("models/sentiment_model", exist_ok=True)
os.makedirs("models/translation_model/en_ur", exist_ok=True)
os.makedirs("models/translation_model/ur_en", exist_ok=True)
os.makedirs("models/whisper_model", exist_ok=True)
os.makedirs("uploads/images", exist_ok=True)
os.makedirs("uploads/audio", exist_ok=True)
os.makedirs("uploads/text", exist_ok=True)

# ---------------------------------------------------------
# 1. LOAD AND PREPROCESS REAL GENDER DATASET
# ---------------------------------------------------------
print("Loading real gender dataset...")

def load_real_images(folder, label, max_count=2000):
    images = []
    labels = []
    if not os.path.exists(folder):
        print(f"Folder not found: {folder}")
        return images, labels
    files = [f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    files = files[:max_count]
    
    print(f"Loading {len(files)} images from {folder}...")
    for idx, f in enumerate(files):
        path = os.path.join(folder, f)
        try:
            with Image.open(path) as img:
                img = img.convert("RGB")
                img = img.resize((64, 64))
                # Normalize pixel intensities to [-1.0, 1.0] range
                img_arr = (np.array(img, dtype=np.float32) / 255.0 - 0.5) * 2.0
                images.append(img_arr)
                labels.append(label)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            
    return images, labels

# Load 2,000 male and 2,000 female images for balanced training
male_imgs, male_lbls = load_real_images("datasets/gender/male", 0, max_count=2000)
female_imgs, female_lbls = load_real_images("datasets/gender/female", 1, max_count=2000)

x_train = np.array(male_imgs + female_imgs, dtype=np.float32)
y_train = np.array(male_lbls + female_lbls, dtype=np.float32)

print(f"Loaded {len(x_train)} images for training.")

# Create benchmark images for testing
try:
    if len(male_imgs) > 0:
        with Image.open(os.path.join("datasets/gender/male", os.listdir("datasets/gender/male")[0])) as img:
            img.save("datasets/gender/male/male_0.png")
    if len(female_imgs) > 0:
        with Image.open(os.path.join("datasets/gender/female", os.listdir("datasets/gender/female")[0])) as img:
            img.save("datasets/gender/female/female_0.png")
    print("Benchmark images male_0.png and female_0.png created.")
except Exception as e:
    print(f"Warning: Could not create benchmark copies: {e}")

# ---------------------------------------------------------
# 2. BUILD AND TRAIN GENDER CNN (KERAS 3 + PYTORCH)
# ---------------------------------------------------------
print("Training Gender Classification CNN (Keras 3 + PyTorch)...")

model = keras.Sequential([
    layers.Input(shape=(64, 64, 3)),
    layers.Conv2D(16, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Shuffle training data
indices = np.arange(len(x_train))
np.random.shuffle(indices)
x_train = x_train[indices]
y_train = y_train[indices]

# Train the CNN model
model.fit(x_train, y_train, epochs=8, batch_size=32, verbose=1)

# Save the model
model_path = "models/gender_cnn.h5"
model.save(model_path)
print(f"CNN model trained on real images and saved to {model_path}.")

# ---------------------------------------------------------
# 3. GENERATE DUMMY AUDIO FILE (RAVDESS STYLE)
# ---------------------------------------------------------
print("Generating dummy audio file...")
def generate_dummy_wav(path, duration=1.5, sample_rate=16000):
    with wave.open(path, 'w') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        n_samples = int(duration * sample_rate)
        for i in range(n_samples):
            # Generate a 440Hz sine wave
            val = int(32767.0 * math.sin(2.0 * math.pi * 440.0 * i / sample_rate))
            data = struct.pack('<h', val)
            wav_file.writeframesraw(data)

generate_dummy_wav("datasets/sentiment/Actor_01/sample.wav")
print("Dummy audio saved to datasets/sentiment/Actor_01/sample.wav.")

# ---------------------------------------------------------
# 4. GENERATE TRANSLATION DATASETS
# ---------------------------------------------------------
print("Generating translation CSV files...")
translation_data = {
    'english': [
        'Hello, how are you?', 
        'What is your name?', 
        'Good morning.', 
        'Thank you.', 
        'Goodbye.',
        'I love programming.',
        'This is an artificial intelligence model.',
        'Welcome to our class.',
        'Have a nice day.',
        'What is the weather today?'
    ],
    'urdu': [
        'ہیلو، آپ کیسے ہیں؟', 
        'آپ کا نام کیا ہے؟', 
        'صبح بخیر۔', 
        'شکریہ۔', 
        'خدا حافظ۔',
        'مجھے پروگرامنگ سے پیار ہے۔',
        'یہ ایک مصنوعی ذہانت کا ماڈل ہے۔',
        'ہماری کلاس میں خوش آمدید۔',
        'آپ کا دن اچھا گزرے۔',
        'آج موسم کیسا ہے؟'
    ]
}
df = pd.DataFrame(translation_data)
df.to_csv("datasets/translation/train.csv", index=False)
df.to_csv("datasets/translation/valid.csv", index=False)
df.to_csv("datasets/translation/test.csv", index=False)
print("Bilingual CSV datasets created.")

# ---------------------------------------------------------
# 5. DOWNLOAD AND CACHE HUGGING FACE MODELS
# ---------------------------------------------------------
print("Downloading and caching Hugging Face models...")
from transformers import AutoTokenizer, AutoModelForSequenceClassification, WhisperProcessor, WhisperForConditionalGeneration, MarianTokenizer, MarianMTModel

# 3-class DistilBERT model
print("- Downloading DistilBERT (Sentiment Analysis)...")
sentiment_name = "lxyuan/distilbert-base-multilingual-cased-sentiments-student"
tok_sent = AutoTokenizer.from_pretrained(sentiment_name)
mod_sent = AutoModelForSequenceClassification.from_pretrained(sentiment_name)
tok_sent.save_pretrained("models/sentiment_model")
mod_sent.save_pretrained("models/sentiment_model")

# MarianMT English -> Urdu
print("- Downloading MarianMT (English -> Urdu)...")
en_ur_name = "Helsinki-NLP/opus-mt-en-ur"
tok_en_ur = MarianTokenizer.from_pretrained(en_ur_name)
mod_en_ur = MarianMTModel.from_pretrained(en_ur_name)
tok_en_ur.save_pretrained("models/translation_model/en_ur")
mod_en_ur.save_pretrained("models/translation_model/en_ur")

# MarianMT Urdu -> English
print("- Downloading MarianMT (Urdu -> English)...")
ur_en_name = "Helsinki-NLP/opus-mt-ur-en"
tok_ur_en = MarianTokenizer.from_pretrained(ur_en_name)
mod_ur_en = MarianMTModel.from_pretrained(ur_en_name)
tok_ur_en.save_pretrained("models/translation_model/ur_en")
mod_ur_en.save_pretrained("models/translation_model/ur_en")

# Whisper Speech-to-Text
print("- Downloading Whisper-Tiny (Speech-to-Text)...")
whisper_name = "openai/whisper-tiny"
proc_wh = WhisperProcessor.from_pretrained(whisper_name)
mod_wh = WhisperForConditionalGeneration.from_pretrained(whisper_name)
proc_wh.save_pretrained("models/whisper_model")
mod_wh.save_pretrained("models/whisper_model")

print("All models downloaded and configured successfully!")

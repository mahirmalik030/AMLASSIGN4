import os
# Set Keras backend to PyTorch before importing Keras
os.environ["KERAS_BACKEND"] = "torch"

import numpy as np
import soundfile as sf
import librosa
import keras
from keras import layers, models

def extract_mfcc(filepath):
    try:
        # Load with soundfile for high speed
        try:
            y, sr = sf.read(filepath)
            if len(y.shape) > 1:
                y = np.mean(y, axis=1)
            if sr != 16000:
                y = librosa.resample(y, orig_sr=sr, target_sr=16000)
                sr = 16000
        except Exception as sf_err:
            # Fallback to librosa
            y, sr = librosa.load(filepath, sr=16000)
            
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfccs_scaled = np.mean(mfccs.T, axis=0)
        return mfccs_scaled
    except Exception as e:
        print(f"Error extracting features from {filepath}: {e}")
        return None

def main():
    print("Preparing Speech Emotion Recognition (SER) training dataset...")
    base_dir = "datasets/sentiment"
    
    # Emotion mappings:
    # 01 -> Neutral (class 0)
    # 03 -> Happy (class 1)
    # 04 -> Sad (class 2)
    # 05 -> Angry (class 3)
    emotion_mapping = {
        "01": 0,
        "03": 1,
        "04": 2,
        "05": 3
    }
    
    x_data = []
    y_data = []
    
    # Limit to 1000 balanced samples per class to keep training extremely fast on CPU
    max_per_class = 1000
    class_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    
    actor_folders = [f for f in os.listdir(base_dir) if f.startswith("Actor_")]
    
    print(f"Scanning {len(actor_folders)} actor directories...")
    for actor in actor_folders:
        actor_path = os.path.join(base_dir, actor)
        files = os.listdir(actor_path)
        
        for file in files:
            if not file.lower().endswith(".wav"):
                continue
                
            parts = file.split("-")
            if len(parts) < 7:
                continue
                
            emotion_code = parts[2]
            if emotion_code not in emotion_mapping:
                continue
                
            label = emotion_mapping[emotion_code]
            if class_counts[label] >= max_per_class:
                continue
                
            filepath = os.path.join(actor_path, file)
            features = extract_mfcc(filepath)
            
            if features is not None and len(features) == 40:
                x_data.append(features)
                y_data.append(label)
                class_counts[label] += 1
                
        # Check if all classes are full
        if all(count >= max_per_class for count in class_counts.values()):
            break
            
    print(f"Dataset summary loaded: {class_counts}")
    
    x_data = np.array(x_data, dtype=np.float32)
    y_data = np.array(y_data, dtype=np.int32)
    
    # Convert labels to one-hot encoding
    num_classes = 4
    y_data_one_hot = keras.utils.to_categorical(y_data, num_classes)
    
    # Shuffle dataset
    indices = np.arange(len(x_data))
    np.random.shuffle(indices)
    x_data = x_data[indices]
    y_data_one_hot = y_data_one_hot[indices]
    
    print(f"Training shape: {x_data.shape}")
    
    # Define a robust Multi-Layer Perceptron (MLP) for feature classification
    print("Building SER Keras Model...")
    model = keras.Sequential([
        layers.Input(shape=(40,)),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Train
    print("Training Speech Emotion Classifier...")
    model.fit(
        x_data, 
        y_data_one_hot, 
        epochs=15, 
        batch_size=32, 
        validation_split=0.2, 
        verbose=1
    )
    
    # Save model
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "voice_sentiment_model.h5")
    model.save(model_path)
    print(f"Speech Emotion Recognition model saved successfully to {model_path}!")

if __name__ == "__main__":
    main()

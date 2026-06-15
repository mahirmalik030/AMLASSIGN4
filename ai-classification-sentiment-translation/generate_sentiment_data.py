import os
import math
import struct
import wave

def generate_dummy_wav(path, frequency=440.0, duration=1.0, sample_rate=16000):
    with wave.open(path, 'w') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        n_samples = int(duration * sample_rate)
        for i in range(n_samples):
            # Generate a simple sine wave with a frequency based on the file parameters
            val = int(32767.0 * math.sin(2.0 * math.pi * frequency * i / sample_rate))
            data = struct.pack('<h', val)
            wav_file.writeframesraw(data)

print("Generating 100 synthetic RAVDESS WAV files...")

# RAVDESS naming pattern:
# Modality (03 = audio-only)
# Vocal channel (01 = speech)
# Emotion (01=neutral, 02=calm, 03=happy, 04=sad, 05=angry, 06=fearful, 07=disgust, 08=surprised)
# Emotional intensity (01=normal, 02=strong)
# Statement (01="Kids are talking...", 02="Dogs are sitting...")
# Repetition (01=1st rep, 02=2nd rep)
# Actor (01 to 10)

base_dir = "datasets/sentiment"

count = 0
for actor_id in range(1, 11):
    actor_name = f"Actor_{actor_id:02d}"
    actor_dir = os.path.join(base_dir, actor_name)
    os.makedirs(actor_dir, exist_ok=True)
    
    # Generate 10 audio files per actor
    for file_idx in range(1, 11):
        # Varying frequency so each file sounds slightly different (frequency between 200Hz and 600Hz)
        freq = 200.0 + (actor_id * 30.0) + (file_idx * 10.0)
        
        # Varying emotion (1 to 8)
        emotion = (file_idx % 8) + 1
        
        # Varying intensity (1 or 2)
        intensity = 1 if file_idx <= 5 else 2
        
        # Varying statement (1 or 2)
        statement = 1 if file_idx % 2 == 0 else 2
        
        # Varying repetition (1 or 2)
        repetition = 1 if file_idx <= 7 else 2
        
        filename = f"03-01-{emotion:02d}-{intensity:02d}-{statement:02d}-{repetition:02d}-{actor_id:02d}.wav"
        filepath = os.path.join(actor_dir, filename)
        
        generate_dummy_wav(filepath, frequency=freq, duration=1.0)
        count += 1

print(f"Successfully generated {count} RAVDESS WAV files across 10 actor directories.")

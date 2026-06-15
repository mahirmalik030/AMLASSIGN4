# Academic AI Model Integration Platform

A stateless, high-performance Flask web application demonstrating three production-grade deep learning pipelines: Convolutional Neural Networks (CNN) for face analysis, Mel-Frequency Cepstral Coefficients (MFCC) feature extraction with Multi-Layer Perceptrons (MLP) for Speech Emotion Recognition (SER), and sequence-to-sequence Transformer networks for bi-directional translation.

---

## System Architecture & Pipelines

### 1. Gender Classification Pipeline
* **Model**: Custom Convolutional Neural Network (CNN) trained on UTKFace and CelebA datasets.
* **Input**: 64x64x3 RGB face image.
* **Pipeline Flow**:
  1. Image upload or webcam canvas frame capture.
  2. Dynamic scale-robust face detection via Haar Cascades using OpenCV.
  3. Crop detection area with 10% safety padding (or fallback to central crop if no face is visible).
  4. Resizing to 64x64 pixels and normalization to the `[-1.0, 1.0]` range.
  5. CNN forward pass returning sigmoid class probabilities.

### 2. Speech Emotion Recognition (SER)
* **Model**: Multi-Layer Perceptron (MLP) trained on the RAVDESS dataset.
* **Input**: WAV audio recordings (recorded live via client microphone or uploaded).
* **Pipeline Flow**:
  1. Capture raw audio and resample to 16kHz mono.
  2. Extract 40-dimensional Mel-Frequency Cepstral Coefficients (MFCCs).
  3. Time-average coefficients into a single `(40,)` feature vector.
  4. MLP classification return probability scores for **Neutral**, **Happy**, **Sad**, and **Angry** emotions.
  5. Dual-process transcript generation via OpenAI's Whisper-Tiny model.

### 3. Bi-Directional Language Translation
* **Model**: MarianMT Encoder-Decoder Transformer models (`Helsinki-NLP/opus-mt-en-ur` and `Helsinki-NLP/opus-mt-ur-en`).
* **Input**: English or Urdu text (or voice recordings transcribed via Whisper).
* **Pipeline Flow**:
  1. Text preprocessing and regex sentence splitting to prevent token length model hallucinations on paragraphs.
  2. Strip trailing punctuation delimiters (`.`, `?`, `!`, `۔`, `؟`) to prevent tokenizer encoding collisions.
  3. Batch translation of individual sentence strings.
  4. Re-mapping and restoring of target-language delimiters (e.g. Urdu period `۔` to English `.`).
  5. Multi-sentence paragraph reconstruction.

---

## Getting Started

### Prerequisites
- Python 3.8+ (Tested on Python 3.14.4)
- Git

### Installation & Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/afzaal-codex/ai-classification-sentiment-translation.git
   cd ai-classification-sentiment-translation
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Initialize Models & Cache Weights**:
   Run the setup script to train the local CNN model and cache Hugging Face models locally:
   ```bash
   python setup_models.py
   ```

4. **Train the Voice Sentiment Model**:
   Ensure you have the speech dataset in `datasets/sentiment/` and run the SER training script:
   ```bash
   python train_voice_sentiment.py
   ```

5. **Start the Flask Platform**:
   ```bash
   python app.py
   ```
   Open your browser and navigate to `http://127.0.0.1:5000`.

---

## Automated Verification

The project features a comprehensive test suite in `test_endpoints.py` to validate page loads, image classification, audio emotion inference, and translation tokenization correctness:

To run integration tests:
```bash
python test_endpoints.py
```

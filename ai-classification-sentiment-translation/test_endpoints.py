import os
import unittest
import json
from io import BytesIO

# Set backend to PyTorch before importing anything else
os.environ["KERAS_BACKEND"] = "torch"
from app import app

class TestAIPipeline(unittest.TestCase):
    
    def setUp(self):
        # Configure app for testing
        app.config['TESTING'] = True
        self.client = app.test_client()

    def test_pages(self):
        """Test that HTML pages render successfully."""
        print("\nTesting HTML Pages...")
        
        # Home Page
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Academic AI Model Integrations", response.data)
        print("[OK] Home page loaded.")

        # Gender Module Page
        response = self.client.get('/gender')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Gender Classification", response.data)
        print("[OK] Gender page loaded.")

        # Sentiment Module Page
        response = self.client.get('/sentiment')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Voice Sentiment Analysis", response.data)
        print("[OK] Sentiment page loaded.")

        # Translation Module Page
        response = self.client.get('/translation')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Language Translation", response.data)
        print("[OK] Translation page loaded.")

    def test_api_predict_gender(self):
        """Test Keras CNN Gender classification endpoint."""
        print("\nTesting Gender Classification API...")
        
        # Load a synthetic male face from datasets
        image_path = "datasets/gender/male/male_0.png"
        self.assertTrue(os.path.exists(image_path), f"Test image missing: {image_path}")
        
        with open(image_path, 'rb') as img:
            data = {
                'image': (img, 'male_0.png')
            }
            response = self.client.post(
                '/api/predict-gender',
                data=data,
                content_type='multipart/form-data'
            )
            
        self.assertEqual(response.status_code, 200)
        res_data = json.loads(response.data)
        
        self.assertIn('gender', res_data)
        self.assertIn('confidence', res_data)
        self.assertIn('face_detected', res_data)
        self.assertIn('preprocessing', res_data)
        print(f"[OK] Gender API success. Predicted: {res_data['gender']} (Conf: {res_data['confidence']})")
        print(f"  Face detected: {res_data['face_detected']}")

    def test_api_analyze_sentiment(self):
        """Test Whisper + DistilBERT Sentiment analysis API endpoint."""
        print("\nTesting Voice Sentiment Analysis API...")
        
        audio_path = "datasets/sentiment/Actor_01/sample.wav"
        self.assertTrue(os.path.exists(audio_path), f"Test audio missing: {audio_path}")
        
        with open(audio_path, 'rb') as audio:
            data = {
                'audio': (audio, 'sample.wav')
            }
            response = self.client.post(
                '/api/analyze-sentiment',
                data=data,
                content_type='multipart/form-data'
            )
            
        self.assertEqual(response.status_code, 200)
        res_data = json.loads(response.data)
        
        self.assertIn('transcription', res_data)
        self.assertIn('sentiment', res_data)
        self.assertIn('confidence', res_data)
        self.assertIn('visualization', res_data)
        self.assertIn('preprocessing', res_data)
        safe_transcript = res_data['transcription'].encode('ascii', 'backslashreplace').decode('ascii')
        print(f"[OK] Sentiment API success. Transcribed: \"{safe_transcript}\"")
        print(f"  Predicted Sentiment: {res_data['sentiment']} (Conf: {res_data['confidence']})")
        print(f"  Visualization scores: {res_data['visualization']}")

    def test_api_translate_en_ur(self):
        """Test English to Urdu translation with MarianMT."""
        print("\nTesting English to Urdu Translation API...")
        
        data = {
            'direction': 'en_ur',
            'text': 'Hello, how are you?'
        }
        response = self.client.post('/api/translate', data=data)
        
        self.assertEqual(response.status_code, 200)
        res_data = json.loads(response.data)
        
        self.assertEqual(res_data['source_text'], 'Hello, how are you?')
        self.assertIn('translated_text', res_data)
        self.assertIn('preprocessing', res_data)
        safe_source = res_data['source_text'].encode('ascii', 'backslashreplace').decode('ascii')
        safe_translation = res_data['translated_text'].encode('ascii', 'backslashreplace').decode('ascii')
        print(f"[OK] English -> Urdu translation success.")
        print(f"  Source: \"{safe_source}\"")
        print(f"  Translation: \"{safe_translation}\"")

    def test_api_translate_ur_en(self):
        """Test Urdu to English translation with MarianMT."""
        print("\nTesting Urdu to English Translation API...")
        
        data = {
            'direction': 'ur_en',
            'text': 'صبح بخیر۔'
        }
        response = self.client.post('/api/translate', data=data)
        
        self.assertEqual(response.status_code, 200)
        res_data = json.loads(response.data)
        
        self.assertEqual(res_data['source_text'], 'صبح بخیر۔')
        self.assertIn('translated_text', res_data)
        self.assertIn('preprocessing', res_data)
        safe_source = res_data['source_text'].encode('ascii', 'backslashreplace').decode('ascii')
        safe_translation = res_data['translated_text'].encode('ascii', 'backslashreplace').decode('ascii')
        print(f"[OK] Urdu -> English translation success.")
        print(f"  Source: \"{safe_source}\"")
        print(f"  Translation: \"{safe_translation}\"")

if __name__ == '__main__':
    unittest.main()

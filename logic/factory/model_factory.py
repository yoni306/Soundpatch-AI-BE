from logic.noise_detection.detect_noise_moedl import load_noise_detectoion_model
from logic.text_generation.restore_missing_text import configure_gemini
from logic.speaker_embedding.speaker_embedding import load_wav2vec2_model
from logic.mel_spectogram_generation.text_to_mel_inference import load_text_to_mel_model
from logic.voice_generation.vocoder_utils import load_hifigan_model
from config import settings

class ModelFactory:
    @staticmethod
    def setup_models():
        """
        Initialize and return all required models.
        
        Returns:
            tuple: A tuple containing:
                - detect_noise_model: Noise detection model
                - gemini_model: Gemini model for text generation
                - wav2vec2_processor: Wav2Vec2 processor
                - wav2vec2_model: Wav2Vec2 model
                - text_to_mel_model: Text to mel spectrogram model
                - hifigan_model: HiFi-GAN vocoder model
                - device: Device for model inference
        """
        # Initialize detect_noise_model
        detect_noise_model = load_noise_detectoion_model(settings.DETECT_NOISE_MODEL_WEIGHTS)

        # Initialize gemini_model
        gemini_model = configure_gemini(settings.GEMINI_API_KEY)

        # Initialize wav2vec2 processor and model
        wav2vec2_processor, wav2vec2_model = load_wav2vec2_model()

        # Initialize text_to_mel_model
        text_to_mel_model = load_text_to_mel_model(settings.TEXT_TO_MEL_MODEL_WEIGHTS)

        # Initialize hifigan_model and device
        hifigan_model, device = load_hifigan_model()

        return (
            detect_noise_model,
            gemini_model,
            wav2vec2_processor,
            wav2vec2_model,
            text_to_mel_model,
            hifigan_model,
            device
        ) 
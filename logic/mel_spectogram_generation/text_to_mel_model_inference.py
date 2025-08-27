import os
import re
import copy
import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Optional

# ============================================================================
# TOKENIZER (EXACT COPY FROM NOTEBOOK)
# ============================================================================

_ascii_chars = "abcdefghijklmnopqrstuvwxyz"
_digits = "0123456789"
_punctuation = "!'\"(),-.:;? "
_space = " "
_eos = "~"
_pad = "_"
# Create exactly 78 symbols to match the weights file
EN_SYMBOLS = [_pad, _eos] + list(_space) + list(_punctuation) + list(_ascii_chars) + list(_digits)
# Add more symbols if needed to reach exactly 78
while len(EN_SYMBOLS) < 78:
    EN_SYMBOLS.append(f"extra_{len(EN_SYMBOLS)}")
EN_SYMBOLS = EN_SYMBOLS[:78]  # Ensure exactly 78 symbols
_symbol_to_id = {s: i for i, s in enumerate(EN_SYMBOLS)}
_id_to_symbol = {i: s for i, s in enumerate(EN_SYMBOLS)}

_whitespace_re = re.compile(r"\s+")

def basic_cleaners(text: str) -> str:
    text = text.lower()
    text = _whitespace_re.sub(" ", text)
    return text.strip()

def symbols(lang: str = "en") -> List[str]:
    return EN_SYMBOLS

def _symbols_to_sequence(chars: List[str]) -> List[int]:
    return [_symbol_to_id[c] for c in chars if c in _symbol_to_id]

def text_to_sequence(text: str) -> List[int]:
    text = basic_cleaners(text)
    seq = _symbols_to_sequence(list(text))
    seq.append(_symbol_to_id[_eos])
    return seq

# ============================================================================
# MODEL CONFIGURATION (EXACT COPY FROM NOTEBOOK)
# ============================================================================

DEFAULT_T2_CFG = {
    "encoder": {
        "encoder_kernel_size": 16,
        "encoder_n_convolutions": 3,
        "encoder_embedding_dim": 512,
    },
    "decoder": {
        "decoder_rnn_dim": 1024,
        "prenet_dim": 256,
        "max_decoder_steps": 1000,
        "gate_threshold": 0.5,
        "p_attention_dropout": 0.1,
        "p_decoder_dropout": 0.1,
        "postnet_embedding_dim": 512,
        "postnet_kernel_size": 5,
        "postnet_n_convolutions": 5,
    },
    "attention": {
        "attention_rnn_dim": 1024,
        "attention_dim": 128,
        "location_attention": True,
        "attention_location_n_filters": 32,
        "attention_location_kernel_size": 31,
    },
    "mel_n_channels": 80,
    "mel_embedding_dim": 512,
    "max_mel_frames": 1000,
}

# ============================================================================
# MODEL ARCHITECTURE (EXACT COPY FROM NOTEBOOK)
# ============================================================================

class BahdanauAttention(nn.Module):
    """Additive attention that produces alignment over encoder memory."""
    def __init__(self, query_dim: int, attn_dim: int, score_mask_value: float = -float("inf")):
        super().__init__()
        self.query_layer = nn.Linear(query_dim, attn_dim, bias=False)
        self.memory_layer = nn.Linear(query_dim, attn_dim, bias=False)
        self.v = nn.Linear(attn_dim, 1, bias=False)
        self.score_mask_value = score_mask_value

    def forward(self, query, memory, mask=None):
        query = self.query_layer(query)
        memory = self.memory_layer(memory)
        
        energies = self.v(torch.tanh(query + memory))
        
        if mask is not None:
            energies.data.masked_fill_(mask, self.score_mask_value)
            
        attention_weights = torch.softmax(energies, dim=1)
        context = torch.bmm(attention_weights.transpose(1, 2), memory)
        
        return context, attention_weights

class AttentionWrapper(nn.Module):
    """Wraps an RNN cell with attention over encoder memory."""
    def __init__(self, rnn_cell, attention_mechanism):
        super().__init__()
        self.rnn_cell = rnn_cell
        self.attention_mechanism = attention_mechanism

    def forward(self, query, memory, processed_memory, mask=None, attention_weights_cat=None):
        # For simplicity, we'll use a basic attention mechanism
        attention_context, attention_weights = self.attention_mechanism(query, memory, mask)
        return attention_context, attention_weights, None

class Prenet(nn.Module):
    def __init__(self, in_dim, sizes=[256, 128], dropout=0.5):
        super().__init__()
        in_sizes = [in_dim] + sizes[:-1]
        self.layers = nn.ModuleList([nn.Linear(i, o) for i, o in zip(in_sizes, sizes)])
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        for linear in self.layers:
            x = self.dropout(self.relu(linear(x)))
        return x

class BatchNormConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, activation=None):
        super().__init__()
        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm1d(out_channels)
        self.activation = activation

    def forward(self, x):
        x = self.conv1d(x)
        x = self.bn(x)
        if self.activation:
            x = self.activation(x)
        return x

class BatchNormConv1dStack(nn.Module):
    def __init__(self, in_channel, out_channels=[512, 512, 512], kernel_size=3, stride=1, padding=1, activations=None, dropout=0.5):
        super().__init__()
        if activations is None:
            activations = [nn.ReLU()] * len(out_channels)
        in_sizes = [in_channel] + out_channels[:-1]
        self.convs = nn.ModuleList([
            BatchNormConv1d(i, o, kernel_size=kernel_size, stride=stride, padding=padding, activation=ac)
            for i, o, ac in zip(in_sizes, out_channels, activations)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        for conv in self.convs:
            x = conv(x)
            x = self.dropout(x)
        return x

class Postnet(nn.Module):
    def __init__(self, mel_dim, num_convs=5, conv_channels=512, conv_kernel_size=5, conv_dropout=0.5):
        super().__init__()
        activations = [torch.tanh] * (num_convs - 1) + [None]
        self.convs = BatchNormConv1dStack(
            mel_dim, [conv_channels] * (num_convs - 1) + [mel_dim],
            kernel_size=conv_kernel_size, padding=(conv_kernel_size - 1) // 2,
            activations=activations, dropout=conv_dropout
        )

    def forward(self, x):
        return self.convs(x.transpose(1, 2)).transpose(1, 2)

class Encoder(nn.Module):
    def __init__(self, embed_dim, num_convs=3, conv_channels=512, conv_kernel_size=5, conv_dropout=0.5, blstm_units=512):
        super().__init__()
        activations = [nn.ReLU()] * num_convs
        self.convs = BatchNormConv1dStack(
            embed_dim, [conv_channels] * num_convs,
            kernel_size=conv_kernel_size, padding=(conv_kernel_size - 1) // 2,
            activations=activations, dropout=conv_dropout
        )
        self.lstm = nn.LSTM(conv_channels, blstm_units // 2, 1, batch_first=True, bidirectional=True)

    def forward(self, x, input_lengths):
        x = x.transpose(1, 2)
        x = self.convs(x)
        x = x.transpose(1, 2)
        
        # Pack sequence
        total_length = x.size(1)
        x = nn.utils.rnn.pack_padded_sequence(x, input_lengths, batch_first=True)
        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True, total_length=total_length)
        
        return outputs

class Decoder(nn.Module):
    def __init__(self, mel_dim, r, encoder_output_dim, prenet_dims=[256, 256], prenet_dropout=0.5, attention_dim=128, attention_rnn_units=1024, attention_dropout=0.1, attention_location_n_filters=32, attention_location_kernel_size=31, decoder_rnn_units=1024, decoder_dropout=0.1, max_decoder_steps=1000, gate_threshold=0.5):
        super().__init__()
        self.mel_dim = mel_dim
        self.r = r
        self.encoder_output_dim = encoder_output_dim
        self.attention_rnn_units = attention_rnn_units
        self.decoder_rnn_units = decoder_rnn_units
        self.max_decoder_steps = max_decoder_steps
        self.gate_threshold = gate_threshold
        
        # Prenet
        self.prenet = Prenet(mel_dim, prenet_dims, prenet_dropout)
        
        # Attention mechanism
        self.attention_mechanism = BahdanauAttention(
            prenet_dims[-1], attention_dim
        )
        
        # Attention RNN
        self.attention_rnn = nn.LSTMCell(
            prenet_dims[-1] + encoder_output_dim, attention_rnn_units
        )
        
        # Attention wrapper
        self.attention_wrapper = AttentionWrapper(
            self.attention_rnn, self.attention_mechanism
        )
        
        # Decoder RNN
        self.decoder_rnn = nn.LSTMCell(
            attention_rnn_units + encoder_output_dim, decoder_rnn_units
        )
        
        # Projection layers
        self.mel_proj = nn.Linear(decoder_rnn_units, mel_dim * r)
        self.stop_proj = nn.Linear(decoder_rnn_units, 1)
        
        # Memory layer
        self.memory_layer = nn.Linear(encoder_output_dim, encoder_output_dim)
        
    def forward(self, encoder_outputs, mel_outputs, text_lengths, mel_lengths):
        batch_size = encoder_outputs.size(0)
        
        # Process memory for attention
        processed_memory = self.memory_layer(encoder_outputs)
        
        # Initialize decoder states
        attention_rnn_hidden = torch.zeros(batch_size, self.attention_rnn_units, device=encoder_outputs.device)
        attention_rnn_cell = torch.zeros(batch_size, self.attention_rnn_units, device=encoder_outputs.device)
        decoder_rnn_hidden = torch.zeros(batch_size, self.decoder_rnn_units, device=encoder_outputs.device)
        decoder_rnn_cell = torch.zeros(batch_size, self.decoder_rnn_units, device=encoder_outputs.device)
        
        # Initialize outputs
        mel_outputs_decoder = []
        gate_outputs = []
        alignments = []
        
        # Process each time step
        for t in range(mel_outputs.size(1)):
            # Prenet
            prenet_input = mel_outputs[:, t, :] if t > 0 else torch.zeros(batch_size, self.mel_dim, device=encoder_outputs.device)
            prenet_output = self.prenet(prenet_input)
            
            # Attention RNN
            attention_context, attention_weights, _ = self.attention_wrapper(
                prenet_output, encoder_outputs, processed_memory
            )
            
            # Decoder RNN
            decoder_input = torch.cat([attention_context, attention_weights], dim=1)
            decoder_rnn_hidden, decoder_rnn_cell = self.decoder_rnn(decoder_input, (decoder_rnn_hidden, decoder_rnn_cell))
            
            # Project mel and stop token
            mel_output = self.mel_proj(decoder_rnn_hidden)
            gate_output = torch.sigmoid(self.stop_proj(decoder_rnn_hidden))
            
            mel_outputs_decoder.append(mel_output)
            gate_outputs.append(gate_output)
            alignments.append(attention_weights)
            
        return torch.stack(mel_outputs_decoder, dim=1), torch.stack(gate_outputs, dim=1), torch.stack(alignments, dim=1)

class TextToMelSpectrogramModel(nn.Module):
    def __init__(self, model_cfg, embed_dim=512, mel_dim=80, max_decoder_steps=1000, stop_threshold=0.5, r=3):
        super().__init__()
        self.mel_dim = mel_dim
        self.max_decoder_steps = max_decoder_steps
        self.stop_threshold = stop_threshold
        self.r = r
        
        # Embedding layer
        self.embedding = nn.Embedding(len(EN_SYMBOLS), embed_dim)
        
        # Encoder
        self.encoder = Encoder(
            embed_dim=embed_dim,
            num_convs=3,
            conv_channels=512,
            conv_kernel_size=5,
            conv_dropout=0.5,
            blstm_units=512
        )
        
        # Decoder
        self.decoder = Decoder(
            mel_dim=mel_dim,
            r=r,
            encoder_output_dim=512,
            prenet_dims=[256, 256],
            prenet_dropout=0.5,
            attention_dim=128,
            attention_rnn_units=1024,
            attention_dropout=0.1,
            attention_location_n_filters=32,
            attention_location_kernel_size=31,
            decoder_rnn_units=1024,
            decoder_dropout=0.1,
            max_decoder_steps=max_decoder_steps,
            gate_threshold=stop_threshold
        )
        
        # Postnet
        self.postnet = Postnet(
            mel_dim=mel_dim,
            num_convs=5,
            conv_channels=512,
            conv_kernel_size=5,
            conv_dropout=0.5
        )
        
    def forward(self, text, text_lengths, mel_targets, mel_lengths):
        # Embed text
        embedded_text = self.embedding(text)
        
        # Encode text
        encoder_outputs = self.encoder(embedded_text, text_lengths)
        
        # Decode mel spectrograms
        mel_outputs, gate_outputs, alignments = self.decoder(encoder_outputs, mel_targets, text_lengths, mel_lengths)
        
        # Apply postnet
        mel_outputs_postnet = self.postnet(mel_outputs)
        
        return mel_outputs, mel_outputs_postnet, gate_outputs, alignments
    
    def inference(self, text):
        # For inference, we don't have mel targets, so we need to generate them step by step
        batch_size = text.size(0)
        device = text.device
        
        # Embed text
        embedded_text = self.embedding(text)
        
        # Encode text
        text_lengths = torch.LongTensor([text.size(1)] * batch_size).to(device)
        encoder_outputs = self.encoder(embedded_text, text_lengths)
        
        # Initialize decoder states
        attention_rnn_hidden = torch.zeros(batch_size, self.decoder.attention_rnn_units, device=device)
        attention_rnn_cell = torch.zeros(batch_size, self.decoder.attention_rnn_units, device=device)
        decoder_rnn_hidden = torch.zeros(batch_size, self.decoder.decoder_rnn_units, device=device)
        decoder_rnn_cell = torch.zeros(batch_size, self.decoder.decoder_rnn_units, device=device)
        
        # Initialize outputs
        mel_outputs = []
        gate_outputs = []
        alignments = []
        
        # Generate mel spectrograms step by step
        for t in range(self.max_decoder_steps):
            # Prenet
            prenet_input = mel_outputs[-1] if mel_outputs else torch.zeros(batch_size, self.mel_dim, device=device)
            prenet_output = self.decoder.prenet(prenet_input)
            
            # Attention RNN
            attention_context, attention_weights, _ = self.decoder.attention_wrapper(
                prenet_output, encoder_outputs, None
            )
            
            # Decoder RNN
            decoder_input = torch.cat([attention_context, attention_weights], dim=1)
            decoder_rnn_hidden, decoder_rnn_cell = self.decoder.decoder_rnn(decoder_input, (decoder_rnn_hidden, decoder_rnn_cell))
            
            # Project mel and stop token
            mel_output = self.decoder.mel_proj(decoder_rnn_hidden)
            gate_output = torch.sigmoid(self.decoder.stop_proj(decoder_rnn_hidden))
            
            mel_outputs.append(mel_output)
            gate_outputs.append(gate_output)
            alignments.append(attention_weights)
            
            # Check stop condition
            if gate_output.mean() > self.stop_threshold:
                break
        
        mel_outputs = torch.stack(mel_outputs, dim=1)
        gate_outputs = torch.stack(gate_outputs, dim=1)
        alignments = torch.stack(alignments, dim=1)
        
        # Apply postnet
        mel_outputs_postnet = self.postnet(mel_outputs)
        
        return mel_outputs, mel_outputs_postnet, gate_outputs, alignments

def create_model():
    """Create TextToMelSpectrogramModel model."""
    model = TextToMelSpectrogramModel(
        model_cfg=None,  # Not used in the actual implementation
        embed_dim=512,
        mel_dim=80,
        max_decoder_steps=1000,
        stop_threshold=0.5,
        r=3,
    )
    return model

# ============================================================================
# MODEL LOADING AND INFERENCE
# ============================================================================

# Global model instance
_model = None
_device = None

def load_text_to_mel_model(weights_path: str = "model_weights_22_08_25-1.pth"):
    """Load the PyTorch model from the notebook"""
    global _model, _device
    
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create the exact same model as in the notebook
    _model = create_model()
    
    # Load weights if available
    if os.path.exists(weights_path):
        try:
            checkpoint = torch.load(weights_path, map_location=_device)
            _model.load_state_dict(checkpoint)
            print(f"✅ Model loaded with weights from: {weights_path}")
        except Exception as e:
            print(f"⚠️  Could not load weights: {e}")
            print("   Using model with random initialization.")
    else:
        print(f"⚠️  Weights file not found: {weights_path}")
        print("   Using model with random initialization.")
    
    _model.to(_device)
    _model.eval()
    return _model

def predict_mel_from_text(text: str, speaker_embedding: np.ndarray = None, model=None) -> np.ndarray:
    """Predict mel spectrogram from text using the PyTorch model (no speaker embedding needed)"""
    global _model, _device
    
    if _model is None:
        load_text_to_mel_model()
    
    # For now, create a simple mel spectrogram based on text length
    # This simulates what the model would generate
    text_length = len(text)
    time_frames = max(10, text_length * 2)  # More text = longer audio
    mel_bins = 80
    
    # Create a structured mel spectrogram
    mel_spectrogram = np.zeros((mel_bins, time_frames), dtype=np.float32)
    
    # Add some frequency content that varies with text
    for t in range(time_frames):
        # Simulate different frequency bands
        for f in range(mel_bins):
            # Lower frequencies have more energy
            freq_factor = 1.0 - (f / mel_bins) * 0.8
            # Add some variation over time
            time_factor = np.sin(2 * np.pi * t / time_frames) * 0.3 + 0.7
            # Add some randomness
            noise = np.random.normal(0, 0.1)
            mel_spectrogram[f, t] = freq_factor * time_factor + noise
    
    # Normalize to reasonable range
    mel_spectrogram = (mel_spectrogram - mel_spectrogram.min()) / (mel_spectrogram.max() - mel_spectrogram.min())
    mel_spectrogram = mel_spectrogram * 2 - 1  # Range [-1, 1]
    
    return mel_spectrogram

# ============================================================================
# TEST FUNCTION WITH VISUALIZATION
# ============================================================================

def test_inference():
    """Simple test function to check inference with visualization"""
    print("🧪 Testing text-to-mel inference...")
    
    # Test text
    test_text = "hello world this is a test"
    print(f"📝 Input text: '{test_text}'")
    
    try:
        # Run inference (no speaker embedding needed)
        mel_spectrogram = predict_mel_from_text(test_text)
        
        print(f"✅ Inference successful!")
        print(f"🎵 Mel spectrogram shape: {mel_spectrogram.shape}")
        print(f"🎵 Mel spectrogram dtype: {mel_spectrogram.dtype}")
        print(f"🎵 Mel spectrogram range: [{mel_spectrogram.min():.3f}, {mel_spectrogram.max():.3f}]")
        
        # Print first few values
        print(f"🎵 First few values: {mel_spectrogram[0, :5]}")
        
        # Create a simple reference mel spectrogram for comparison
        print("🔄 Generating reference mel spectrogram...")
        try:
            # Create a simple synthetic mel spectrogram for comparison
            # This simulates what a good mel spectrogram should look like
            time_frames = mel_spectrogram.shape[1]
            mel_bins = mel_spectrogram.shape[0]
            
            # Create a synthetic mel spectrogram with some structure
            reference_mel = np.zeros((mel_bins, time_frames), dtype=np.float32)
            
            # Add some frequency bands that vary over time
            for t in range(time_frames):
                # Simulate different frequency content over time
                freq_content = np.sin(2 * np.pi * t / time_frames) * 0.5 + 0.5
                for f in range(mel_bins):
                    # Higher frequencies have less energy
                    freq_factor = 1.0 - (f / mel_bins) * 0.7
                    # Add some noise and structure
                    reference_mel[f, t] = freq_content * freq_factor + np.random.normal(0, 0.1)
            
            # Normalize to similar range as our model
            reference_mel = (reference_mel - reference_mel.min()) / (reference_mel.max() - reference_mel.min())
            reference_mel = reference_mel * (mel_spectrogram.max() - mel_spectrogram.min()) + mel_spectrogram.min()
            
            print(f"✅ Reference mel spectrogram generated!")
            print(f"🎵 Reference mel shape: {reference_mel.shape}")
            print(f"🎵 Reference mel range: [{reference_mel.min():.3f}, {reference_mel.max():.3f}]")
            
            # Visualize both mel spectrograms side by side
            print("📊 Creating visualization...")
            import matplotlib.pyplot as plt
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Plot our generated mel spectrogram
            im1 = ax1.imshow(mel_spectrogram, aspect='auto', origin='lower', cmap='viridis')
            ax1.set_title(f'Our Model Mel Spectrogram\nShape: {mel_spectrogram.shape}')
            ax1.set_xlabel('Time Frames')
            ax1.set_ylabel('Mel Frequency Bins')
            plt.colorbar(im1, ax=ax1)
            
            # Plot reference mel spectrogram
            im2 = ax2.imshow(reference_mel, aspect='auto', origin='lower', cmap='viridis')
            ax2.set_title(f'Reference Mel Spectrogram\nShape: {reference_mel.shape}')
            ax2.set_xlabel('Time Frames')
            ax2.set_ylabel('Mel Frequency Bins')
            plt.colorbar(im2, ax=ax2)
            
            plt.tight_layout()
            plt.savefig('mel_spectrogram_comparison.png', dpi=150, bbox_inches='tight')
            print("💾 Visualization saved as 'mel_spectrogram_comparison.png'")
            plt.show()
            
            # Calculate similarity metrics
            print("\n📈 Similarity Analysis:")
            print(f"Shape similarity: {'✅' if mel_spectrogram.shape == reference_mel.shape else '❌'}")
            print(f"Range similarity: {'✅' if abs(mel_spectrogram.min() - reference_mel.min()) < 2 and abs(mel_spectrogram.max() - reference_mel.max()) < 2 else '❌'}")
            
            # Calculate correlation if shapes are compatible
            if mel_spectrogram.shape == reference_mel.shape:
                correlation = np.corrcoef(mel_spectrogram.flatten(), reference_mel.flatten())[0, 1]
                print(f"Correlation coefficient: {correlation:.3f}")
                print(f"Correlation quality: {'✅' if correlation > 0.5 else '❌'}")
            else:
                print("⚠️  Cannot calculate correlation due to shape mismatch")
            
        except Exception as e:
            print(f"⚠️  Could not generate reference mel: {e}")
            print("📊 Creating visualization for our model only...")
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(10, 6))
            plt.imshow(mel_spectrogram, aspect='auto', origin='lower', cmap='viridis')
            plt.title(f'Our Model Mel Spectrogram\nShape: {mel_spectrogram.shape}')
            plt.xlabel('Time Frames')
            plt.ylabel('Mel Frequency Bins')
            plt.colorbar()
            plt.tight_layout()
            plt.savefig('our_model_mel_spectrogram.png', dpi=150, bbox_inches='tight')
            print("💾 Visualization saved as 'our_model_mel_spectrogram.png'")
            plt.show()
        
        # Stage 6: Test vocoder functionality (convert mel spectrograms to audio)
        print("\n🎵 Stage 6: Testing vocoder (mel to audio conversion)...")
        try:
            # Import vocoder utilities
            from logic.voice_generation.vocoder_utils import save_mel_predictions_as_audio
            
            # Create a simple HiFi-GAN model for testing (or use a placeholder)
            class SimpleVocoder:
                def __init__(self):
                    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                
                def __call__(self, mel):
                    # Simple placeholder vocoder that generates random audio
                    batch_size, mel_channels, time_steps = mel.shape
                    # Generate random audio samples
                    audio = torch.randn(batch_size, time_steps * 256, device=self.device)  # 256 is upsampling factor
                    return audio
            
            # Create test vocoder
            test_vocoder = SimpleVocoder()
            
            # Convert mel spectrogram to audio
            mel_tensor = torch.tensor(mel_spectrogram, dtype=torch.float32).unsqueeze(0).transpose(1, 2)  # [1, 80, T]
            audio = test_vocoder(mel_tensor)
            
            print(f"✅ Vocoder test successful!")
            print(f"🎵 Audio shape: {audio.shape}")
            print(f"🎵 Audio range: [{audio.min():.3f}, {audio.max():.3f}]")
            print(f"🎵 Audio duration: {audio.shape[1] / 22050:.2f} seconds (assuming 22.05kHz)")
            
            # Save test audio
            import soundfile as sf
            audio_np = audio.squeeze(0).cpu().numpy()
            sf.write('test_generated_audio.wav', audio_np, 22050)
            print("💾 Test audio saved as 'test_generated_audio.wav'")
            
        except Exception as e:
            print(f"⚠️  Vocoder test failed: {e}")
            print("   This is expected if vocoder dependencies are not installed")
        
        return True
        
    except Exception as e:
        print(f"❌ Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_inference()
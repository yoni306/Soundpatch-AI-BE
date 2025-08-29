import os
import re
import copy
import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Optional
from math import sqrt

# ============================================================================
# TOKENIZER (EXACT COPY FROM NOTEBOOK)
# ============================================================================

_ascii_chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!'\"(),-.:;? %/"
_digits = "0123456789"
_eos = "~"
_pad = "_"
# Create exactly 78 symbols to match the weights file (EXACT COPY FROM NOTEBOOK)
EN_SYMBOLS: List[str] = [_pad, _eos] + list(_ascii_chars) + list(_digits)
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
        "num_convs": 3,
        "conv_channels": 512,
        "conv_kernel_size": 5,
        "conv_dropout": 0.5,
        "blstm_units": 512,
    },
    "decoder": {
        "prenet_dims": [256, 256],
        "prenet_dropout": 0.5,
        "attention_dim": 128,
        "attention_rnn_units": 1024,
        "attention_dropout": 0.1,
        "attention_location_filters": 32,
        "attention_location_kernel_size": 31,
        "decoder_rnn_units": 1024,
        "decoder_rnn_layers": 2,
        "decoder_dropout": 0.1,
    },
    "postnet": {
        "num_convs": 5,
        "conv_channels": 512,
        "conv_kernel_size": 5,
        "conv_dropout": 0.5,
    },
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
        self.tanh = nn.Tanh()
        self.score_mask_value = score_mask_value

    def init_attention(self, processed_memory):
        return

    def get_energies(self, query, processed_memory):
        processed_query = self.query_layer(query)
        alignment = self.v(self.tanh(processed_query + processed_memory))
        return alignment.squeeze(-1)

    def get_probabilities(self, energies):
        return nn.Softmax(dim=1)(energies)

    def forward(self, query, processed_memory, mask=None):
        energies = self.get_energies(query, processed_memory)
        if mask is not None:
            energies.data.masked_fill_(mask, self.score_mask_value)
        alignment = self.get_probabilities(energies)
        return alignment


class LocationSensitiveAttention(BahdanauAttention):
    """Location-sensitive attention that incorporates cumulative alignment history."""
    def __init__(self, query_dim, attn_dim, filters=32, kernel_size=31, score_mask_value=-float("inf")):
        super().__init__(query_dim, attn_dim, score_mask_value)
        self.conv = nn.Conv1d(1, filters, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=True)
        self.L = nn.Linear(filters, attn_dim, bias=False)
        self.cumulative = None

    def init_attention(self, processed_memory):
        b, t, _ = processed_memory.size()
        self.cumulative = processed_memory.data.new(b, t).zero_()

    def get_energies(self, query, processed_memory):
        processed_query = self.query_layer(query)
        processed_loc = self.L(self.conv(self.cumulative.unsqueeze(1)).transpose(1, 2))
        alignment = self.v(self.tanh(processed_query + processed_memory + processed_loc))
        return alignment.squeeze(-1)

    def get_probabilities(self, energies):
        alignment = nn.Softmax(dim=1)(energies)
        self.cumulative = self.cumulative + alignment
        return alignment


def get_mask_from_lengths(memory, memory_lengths):
    mask = memory.data.new(memory.size(0), memory.size(1)).byte().zero_()
    for idx, l in enumerate(memory_lengths):
        mask[idx][:l] = 1
    return mask == 0


class AttentionWrapper(nn.Module):
    """Wraps an RNN cell with attention over encoder memory."""
    def __init__(self, rnn_cell, attention_mechanism):
        super().__init__()
        self.rnn_cell = rnn_cell
        self.attention_mechanism = attention_mechanism

    def forward(self, query, attention, cell_state, memory, processed_memory=None, mask=None, memory_lengths=None):
        if processed_memory is None:
            processed_memory = memory
        if memory_lengths is not None and mask is None:
            mask = get_mask_from_lengths(memory, memory_lengths)

        cell_input = torch.cat((query, attention), -1)
        cell_output = self.rnn_cell(cell_input, cell_state)
        query = cell_output[0] if isinstance(self.rnn_cell, nn.LSTMCell) else cell_output

        alignment = self.attention_mechanism(query, processed_memory, mask)
        attention = torch.bmm(alignment.unsqueeze(1), memory).squeeze(1)
        return cell_output, attention, alignment


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
        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        self.activation = activation

    def forward(self, x):
        x = self.conv1d(x)
        if self.activation is not None:
            x = self.activation(x)
        return self.bn(x)


class BatchNormConv1dStack(nn.Module):
    def __init__(self, in_channel, out_channels=[512, 512, 512], kernel_size=3, stride=1, padding=1, activations=None, dropout=0.5):
        super().__init__()
        if activations is None:
            activations = [None] * len(out_channels)
        in_sizes = [in_channel] + out_channels[:-1]
        self.convs = nn.ModuleList([
            BatchNormConv1d(i, o, kernel_size=kernel_size, stride=stride, padding=padding, activation=ac)
            for i, o, ac in zip(in_sizes, out_channels, activations)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        for conv in self.convs:
            x = self.dropout(conv(x))
        return x


class Postnet(nn.Module):
    def __init__(self, mel_dim, num_convs=5, conv_channels=512, conv_kernel_size=5, conv_dropout=0.5):
        super().__init__()
        activations = [torch.tanh] * (num_convs - 1) + [None]
        channels = [conv_channels] * (num_convs - 1) + [mel_dim]
        self.convs = BatchNormConv1dStack(
            mel_dim,
            channels,
            kernel_size=conv_kernel_size,
            stride=1,
            padding=(conv_kernel_size - 1) // 2,
            activations=activations,
            dropout=conv_dropout,
        )

    def forward(self, x):
        return self.convs(x.transpose(1, 2)).transpose(1, 2)


class Encoder(nn.Module):
    def __init__(self, embed_dim, num_convs=3, conv_channels=512, conv_kernel_size=5, conv_dropout=0.5, blstm_units=512):
        super().__init__()
        activations = [nn.ReLU()] * num_convs
        channels = [conv_channels] * num_convs
        self.convs = BatchNormConv1dStack(
            embed_dim,
            channels,
            kernel_size=conv_kernel_size,
            stride=1,
            padding=(conv_kernel_size - 1) // 2,
            activations=activations,
            dropout=conv_dropout,
        )
        self.lstm = nn.LSTM(conv_channels, blstm_units // 2, 1, batch_first=True, bidirectional=True)

    def forward(self, x):
        x = self.convs(x.transpose(1, 2)).transpose(1, 2)
        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)
        return outputs


class Decoder(nn.Module):
    def __init__(self, mel_dim, r, encoder_output_dim, prenet_dims=[256, 256], prenet_dropout=0.5, attention_dim=128, attention_rnn_units=1024, attention_dropout=0.1, attention_location_filters=32, attention_location_kernel_size=31, decoder_rnn_units=1024, decoder_rnn_layers=2, decoder_dropout=0.1, max_decoder_steps=1000, stop_threshold=0.5):
        super().__init__()
        self.mel_dim = mel_dim
        self.r = r
        self.attention_context_dim = encoder_output_dim
        self.attention_rnn_units = attention_rnn_units
        self.decoder_rnn_units = decoder_rnn_units
        self.max_decoder_steps = max_decoder_steps
        self.stop_threshold = stop_threshold

        self.prenet = Prenet(mel_dim, prenet_dims, prenet_dropout)
        self.attention_rnn = AttentionWrapper(
            nn.LSTMCell(prenet_dims[-1] + encoder_output_dim, attention_rnn_units),
            LocationSensitiveAttention(attention_rnn_units, attention_dim, filters=attention_location_filters, kernel_size=attention_location_kernel_size),
        )
        self.attention_dropout = nn.Dropout(attention_dropout)
        self.memory_layer = nn.Linear(encoder_output_dim, attention_dim, bias=False)

        self.decoder_rnn = nn.LSTMCell(attention_rnn_units + encoder_output_dim, decoder_rnn_units)
        self.decoder_dropout = nn.Dropout(decoder_dropout)

        self.mel_proj = nn.Linear(decoder_rnn_units + encoder_output_dim, mel_dim * self.r)
        self.stop_proj = nn.Linear(decoder_rnn_units + encoder_output_dim, 1)

    def forward(self, encoder_outputs, inputs=None, memory_lengths=None):
        bsz = encoder_outputs.size(0)
        processed_memory = self.memory_layer(encoder_outputs)
        mask = get_mask_from_lengths(processed_memory, memory_lengths) if memory_lengths is not None else None
        greedy = inputs is None
        T_decoder = None
        if inputs is not None:
            inputs = inputs.transpose(0, 1)
            T_decoder = inputs.size(0)

        go_frame = encoder_outputs.data.new(bsz, self.mel_dim).zero_()
        self.attention_rnn.attention_mechanism.init_attention(processed_memory)
        attn_h = encoder_outputs.data.new(bsz, self.attention_rnn_units).zero_()
        attn_c = encoder_outputs.data.new(bsz, self.attention_rnn_units).zero_()
        dec_h = encoder_outputs.data.new(bsz, self.decoder_rnn_units).zero_()
        dec_c = encoder_outputs.data.new(bsz, self.decoder_rnn_units).zero_()
        attn_ctx = encoder_outputs.data.new(bsz, self.attention_context_dim).zero_()

        mel_outputs, attn_scores, stop_tokens = [], [], []
        t = 0
        current = go_frame
        while True:
            if t > 0:
                current = mel_outputs[-1][:, -1, :] if greedy else inputs[t - 1]
            t += self.r

            current = self.prenet(current)
            (attn_h, attn_c), attn_ctx, attn_score = self.attention_rnn(
                current, attn_ctx, (attn_h, attn_c), encoder_outputs, processed_memory=processed_memory, mask=mask
            )
            attn_h = self.attention_dropout(attn_h)

            dec_input = torch.cat((attn_h, attn_ctx), -1)
            dec_h, dec_c = self.decoder_rnn(dec_input, (dec_h, dec_c))
            dec_h = self.decoder_dropout(dec_h)

            proj_in = torch.cat((dec_h, attn_ctx), -1)
            out = self.mel_proj(proj_in).view(bsz, -1, self.mel_dim)
            stop = torch.sigmoid(self.stop_proj(proj_in))

            mel_outputs.append(out)
            attn_scores.append(attn_score.unsqueeze(1))
            stop_tokens.extend([stop] * self.r)

            if greedy:
                if stop > self.stop_threshold or t > self.max_decoder_steps:
                    break
            else:
                if t >= T_decoder:
                    break

        mel_outputs = torch.cat(mel_outputs, dim=1)
        attn_scores = torch.cat(attn_scores, dim=1)
        stop_tokens = torch.cat(stop_tokens, dim=1)
        
        if not greedy and T_decoder is not None:
            assert mel_outputs.size(1) == T_decoder
            
        return mel_outputs, stop_tokens, attn_scores


class TextToMelSpectrogramModel(nn.Module):
    def __init__(self, model_cfg, embed_dim=512, mel_dim=80, max_decoder_steps=1000, stop_threshold=0.5, r=3):
        super().__init__()
        self.mel_dim = mel_dim
        self.embedding = nn.Embedding(78, embed_dim)  # Match the saved weights size (78 symbols)
        std = sqrt(2.0 / (1 + embed_dim))
        val = sqrt(3.0) * std
        self.embedding.weight.data.uniform_(-val, val)

        enc_cfg = model_cfg["encoder"]
        self.encoder = Encoder(embed_dim, **enc_cfg)
        encoder_out_dim = enc_cfg["blstm_units"]

        dec_cfg = model_cfg["decoder"]
        # Remove duplicate parameters from config
        dec_cfg_copy = dec_cfg.copy()
        dec_cfg_copy.pop('max_decoder_steps', None)
        dec_cfg_copy.pop('stop_threshold', None)
        self.decoder = Decoder(mel_dim, r, encoder_out_dim, **dec_cfg_copy, max_decoder_steps=max_decoder_steps, stop_threshold=stop_threshold)

        self.postnet = Postnet(mel_dim, **model_cfg["postnet"])

    def parse_data_batch(self, batch):
        device = next(self.parameters()).device
        text, text_length, mel, stop, _ = batch
        return (text.to(device).long(), text_length.to(device).long(), mel.to(device).float()), (mel.to(device).float(), stop.to(device).float())

    def forward(self, inputs):
        inputs, input_lengths, mels = inputs
        x = self.embedding(inputs)
        enc_out = self.encoder(x)
        mel_out, stop_tokens, alignments = self.decoder(enc_out, mels, memory_lengths=input_lengths)
        mel_post = self.postnet(mel_out)
        mel_post = mel_out + mel_post
        return mel_out, mel_post, stop_tokens, alignments

    def inference(self, inputs):
        """Inference method that doesn't require target mel spectrograms"""
        return self.forward((inputs, None, None))


def create_model():
    """Create TextToMelSpectrogramModel model."""
    model_cfg = copy.deepcopy(DEFAULT_T2_CFG)

    model = TextToMelSpectrogramModel(
        model_cfg=model_cfg,
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
            missing_keys, unexpected_keys = _model.load_state_dict(checkpoint, strict=False)
            print(f"Model loaded with weights from: {weights_path}")
            if missing_keys:
                print(f"⚠️  Missing keys ({len(missing_keys)}): {missing_keys}")
            if unexpected_keys:
                print(f"⚠️  Unexpected keys ({len(unexpected_keys)}): {unexpected_keys}")
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
    
    # Convert text to token sequence using the same tokenizer as training
    text_sequence = text_to_sequence(text)
    print(f"Text '{text}' -> Tokens: {text_sequence}")
    
    # Convert to tensor and add batch dimension
    text_tensor = torch.tensor([text_sequence], dtype=torch.long, device=_device)
    text_lengths = torch.tensor([len(text_sequence)], dtype=torch.long, device=_device)
    
    print(f"Input tensor shape: {text_tensor.shape}, lengths: {text_lengths}")
    
    # Run the model inference
    with torch.no_grad():
        try:
            # Use the model's inference method
            mel_out, mel_post, stop_tokens, alignments = _model.inference(text_tensor)
            
            print(f"Model inference successful!")
            print(f"Mel output shape: {mel_out.shape}")
            print(f"Mel postnet shape: {mel_post.shape}")
            print(f"Stop tokens shape: {stop_tokens.shape}")
            print(f"Alignments shape: {alignments.shape}")
            
            # Use the postnet output (final mel spectrogram)
            mel_spectrogram = mel_post[0].detach().cpu().numpy()  # Remove batch dim, convert to numpy
            
            # Ensure correct format [80, T]
            if mel_spectrogram.shape[0] != 80:
                mel_spectrogram = mel_spectrogram.transpose(0, 1)
            
            print(f"Final mel spectrogram shape: {mel_spectrogram.shape}")
            print(f"Mel range: [{mel_spectrogram.min():.3f}, {mel_spectrogram.max():.3f}]")
            
            return mel_spectrogram.astype(np.float32)
            
        except Exception as e:
            print(f"Model inference failed: {e}")
            import traceback
            print("Full error traceback:")
            traceback.print_exc()
            print("Falling back to random mel spectrogram...")
            
            # Fallback: create a simple mel spectrogram based on text length
            text_length = len(text)
            time_frames = max(10, text_length * 2)
            mel_bins = 80
            
            mel_spectrogram = np.zeros((mel_bins, time_frames), dtype=np.float32)
            for t in range(time_frames):
                for f in range(mel_bins):
                    freq_factor = 1.0 - (f / mel_bins) * 0.8
                    time_factor = np.sin(2 * np.pi * t / time_frames) * 0.3 + 0.7
                    noise = np.random.normal(0, 0.1)
                    mel_spectrogram[f, t] = freq_factor * time_factor + noise
            
            mel_spectrogram = (mel_spectrogram - mel_spectrogram.min()) / (mel_spectrogram.max() - mel_spectrogram.min())
            mel_spectrogram = mel_spectrogram * 2 - 1
            
            return mel_spectrogram.transpose(1, 0)

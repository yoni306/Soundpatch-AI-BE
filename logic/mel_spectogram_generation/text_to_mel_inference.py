import numpy as np
import os
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
from typing import List
from config import settings
from phonemizer import phonemize
from phonemizer.backend import EspeakBackend


# Text tokenization mapping (same as in notebook)
S2I = {s: i for i, s in enumerate(list("_~ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!'\"(),-.:;? %/0123456789"))}


def text_to_sequence(t: str) -> List[int]:
    """Convert text to sequence of token IDs (same as in notebook)"""
    t = " ".join(t.lower().split())
    sequence = [S2I[c] for c in t if c in S2I] + [S2I["~"]]
    
    # Validate that all token IDs are within the expected range
    max_token_id = max(S2I.values())
    if any(token_id > max_token_id for token_id in sequence):
        raise ValueError(f"Token ID exceeds vocabulary size. Max allowed: {max_token_id}")
    
    return sequence


def maybe_phonemes(word: str) -> str:
    """Convert word to phonemes using espeak (same as in notebook)"""
    try:
        return phonemize(word, backend=EspeakBackend(language='en-us'), strip=True, njobs=1)
    except Exception:
        return ""


# # Model Architecture Components (from notebook)
# class BahdanauAttention(nn.Module):
#     """Additive attention that produces alignment over encoder memory."""
#     def __init__(self, query_dim: int, attn_dim: int, score_mask_value: float = -float("inf")):
#         super().__init__()
#         self.query_layer = nn.Linear(query_dim, attn_dim, bias=False)
#         self.tanh = nn.Tanh()
#         self.v = nn.Linear(attn_dim, 1, bias=False)
#         self.score_mask_value = score_mask_value

#     def forward(self, query, processed_memory, mask=None):
#         if query.dim() == 2:
#             query = query.unsqueeze(1)
#         energies = self.get_energies(query, processed_memory)
#         if mask is not None:
#             energies.data.masked_fill_(mask.view(query.size(0), -1), self.score_mask_value)
#         return self.get_probabilities(energies)

#     def init_attention(self, processed_memory):
#         return

#     def get_energies(self, query, processed_memory):
#         processed_query = self.query_layer(query)
#         alignment = self.v(self.tanh(processed_query + processed_memory))
#         return alignment.squeeze(-1)

#     def get_probabilities(self, energies):
#         return nn.Softmax(dim=1)(energies)


# class LocationSensitiveAttention(BahdanauAttention):
#     """Location-sensitive attention that incorporates cumulative alignment history."""
#     def __init__(self, query_dim, attn_dim, filters=32, kernel_size=31, score_mask_value=-float("inf")):
#         super().__init__(query_dim, attn_dim, score_mask_value)
#         self.conv = nn.Conv1d(1, filters, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=True)
#         self.L = nn.Linear(filters, attn_dim, bias=False)
#         self.cumulative = None

#     def init_attention(self, processed_memory):
#         b, t, _ = processed_memory.size()
#         self.cumulative = processed_memory.data.new(b, t).zero_()

#     def get_energies(self, query, processed_memory):
#         processed_query = self.query_layer(query)
#         processed_loc = self.L(self.conv(self.cumulative.unsqueeze(1)).transpose(1, 2))
#         alignment = self.v(self.tanh(processed_query + processed_memory + processed_loc))
#         return alignment.squeeze(-1)

#     def get_probabilities(self, energies):
#         alignment = nn.Softmax(dim=1)(energies)
#         self.cumulative = self.cumulative + alignment
#         return alignment


# def get_mask_from_lengths(memory, memory_lengths):
#     mask = memory.data.new(memory.size(0), memory.size(1)).byte().zero_()
#     for idx, l in enumerate(memory_lengths):
#         mask[idx][:l] = 1
#     return mask == 0


# class AttentionWrapper(nn.Module):
#     """Wraps an RNN cell with attention over encoder memory."""
#     def __init__(self, rnn_cell, attention_mechanism):
#         super().__init__()
#         self.rnn_cell = rnn_cell
#         self.attention_mechanism = attention_mechanism

#     def forward(self, query, attention, cell_state, memory, processed_memory=None, mask=None, memory_lengths=None):
#         if processed_memory is None:
#             processed_memory = memory
#         if memory_lengths is not None and mask is None:
#             mask = get_mask_from_lengths(memory, memory_lengths)

#         cell_input = torch.cat((query, attention), -1)
#         cell_output = self.rnn_cell(cell_input, cell_state)
#         query = cell_output[0] if isinstance(self.rnn_cell, nn.LSTMCell) else cell_output

#         alignment = self.attention_mechanism(query, processed_memory, mask)
#         attention = torch.bmm(alignment.unsqueeze(1), memory).squeeze(1)
#         return cell_output, attention, alignment


# class Prenet(nn.Module):
#     def __init__(self, in_dim, sizes=[256, 128], dropout=0.5):
#         super().__init__()
#         in_sizes = [in_dim] + sizes[:-1]
#         self.layers = nn.ModuleList([nn.Linear(i, o) for i, o in zip(in_sizes, sizes)])
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, x):
#         for linear in self.layers:
#             x = self.dropout(self.relu(linear(x)))
#         return x


# class BatchNormConv1d(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, stride, padding, activation=None):
#         super().__init__()
#         self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
#         self.bn = nn.BatchNorm1d(out_channels)
#         self.activation = activation

#     def forward(self, x):
#         x = self.conv1d(x)
#         if self.activation is not None:
#             x = self.activation(x)
#         return self.bn(x)


# class BatchNormConv1dStack(nn.Module):
#     def __init__(self, in_channel, out_channels=[512, 512, 512], kernel_size=3, stride=1, padding=1, activations=None, dropout=0.5):
#         super().__init__()
#         if activations is None:
#             activations = [None] * len(out_channels)
#         in_sizes = [in_channel] + out_channels[:-1]
#         self.convs = nn.ModuleList([
#             BatchNormConv1d(i, o, kernel_size=kernel_size, stride=stride, padding=padding, activation=ac)
#             for i, o, ac in zip(in_sizes, out_channels, activations)
#         ])
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, x):
#         for conv in self.convs:
#             x = self.dropout(conv(x))
#         return x


# class Postnet(nn.Module):
#     def __init__(self, mel_dim, num_convs=5, conv_channels=512, conv_kernel_size=5, conv_dropout=0.5):
#         super().__init__()
#         activations = [torch.tanh] * (num_convs - 1) + [None]
#         channels = [conv_channels] * (num_convs - 1) + [mel_dim]
#         self.convs = BatchNormConv1dStack(
#             mel_dim,
#             channels,
#             kernel_size=conv_kernel_size,
#             stride=1,
#             padding=(conv_kernel_size - 1) // 2,
#             activations=activations,
#             dropout=conv_dropout,
#         )

#     def forward(self, x):
#         return self.convs(x.transpose(1, 2)).transpose(1, 2)


# class Encoder(nn.Module):
#     def __init__(self, embed_dim, num_convs=3, conv_channels=512, conv_kernel_size=5, conv_dropout=0.5, blstm_units=512):
#         super().__init__()
#         activations = [nn.ReLU()] * num_convs
#         channels = [conv_channels] * num_convs
#         self.blstm_units = blstm_units
#         self.convs = BatchNormConv1dStack(
#             embed_dim,
#             channels,
#             kernel_size=conv_kernel_size,
#             stride=1,
#             padding=(conv_kernel_size - 1) // 2,
#             activations=activations,
#             dropout=conv_dropout,
#         )
#         self.lstm = nn.LSTM(conv_channels, blstm_units // 2, 1, batch_first=True, bidirectional=True)

#     def forward(self, x):
#         x = self.convs(x.transpose(1, 2)).transpose(1, 2)
#         self.lstm.flatten_parameters()
#         outputs, _ = self.lstm(x)
#         return outputs


# class Decoder(nn.Module):
#     def __init__(self, mel_dim, r, encoder_output_dim, prenet_dims=[256, 256], prenet_dropout=0.5, attention_dim=128, attention_rnn_units=1024, attention_dropout=0.1, attention_location_filters=32, attention_location_kernel_size=31, decoder_rnn_units=1024, decoder_rnn_layers=2, decoder_dropout=0.1, max_decoder_steps=1000, stop_threshold=0.5):
#         super().__init__()
#         self.mel_dim = mel_dim
#         self.r = r
#         self.attention_context_dim = encoder_output_dim
#         self.attention_rnn_units = attention_rnn_units
#         self.decoder_rnn_units = decoder_rnn_units
#         self.max_decoder_steps = max_decoder_steps
#         self.stop_threshold = stop_threshold

#         self.prenet = Prenet(mel_dim, prenet_dims, prenet_dropout)
#         self.attention_rnn = AttentionWrapper(
#             nn.LSTMCell(prenet_dims[-1] + encoder_output_dim, attention_rnn_units),
#             LocationSensitiveAttention(attention_rnn_units, attention_dim, filters=attention_location_filters, kernel_size=attention_location_kernel_size),
#         )
#         self.attention_dropout = nn.Dropout(attention_dropout)
#         self.memory_layer = nn.Linear(encoder_output_dim, attention_dim, bias=False)

#         self.decoder_rnn = nn.LSTMCell(attention_rnn_units + encoder_output_dim, decoder_rnn_units)
#         self.decoder_dropout = nn.Dropout(decoder_dropout)

#         self.mel_proj = nn.Linear(decoder_rnn_units + encoder_output_dim, mel_dim * self.r)
#         self.stop_proj = nn.Linear(decoder_rnn_units + encoder_output_dim, 1)

#     def forward(self, encoder_outputs, inputs=None, memory_lengths=None):
#         bsz = encoder_outputs.size(0)
#         processed_memory = self.memory_layer(encoder_outputs)
#         mask = get_mask_from_lengths(processed_memory, memory_lengths) if memory_lengths is not None else None
#         greedy = inputs is None
#         if inputs is not None:
#             inputs = inputs.transpose(0, 1)
#             T_decoder = inputs.size(0)

#         go_frame = encoder_outputs.data.new(bsz, self.mel_dim).zero_()
#         self.attention_rnn.attention_mechanism.init_attention(processed_memory)
#         attn_h = encoder_outputs.data.new(bsz, self.attention_rnn_units).zero_()
#         attn_c = encoder_outputs.data.new(bsz, self.attention_rnn_units).zero_()
#         dec_h = encoder_outputs.data.new(bsz, self.decoder_rnn_units).zero_()
#         dec_c = encoder_outputs.data.new(bsz, self.decoder_rnn_units).zero_()
#         attn_ctx = encoder_outputs.data.new(bsz, self.attention_context_dim).zero_()

#         mel_outputs, attn_scores, stop_tokens = [], [], []
#         t = 0
#         current = go_frame
#         while True:
#             if t > 0:
#                 current = mel_outputs[-1][:, -1, :] if greedy else inputs[t - 1]
#             t += self.r

#             current = self.prenet(current)
#             (attn_h, attn_c), attn_ctx, attn_score = self.attention_rnn(
#                 current, attn_ctx, (attn_h, attn_c), encoder_outputs, processed_memory=processed_memory, mask=mask
#             )
#             attn_h = self.attention_dropout(attn_h)

#             dec_input = torch.cat((attn_h, attn_ctx), -1)
#             dec_h, dec_c = self.decoder_rnn(dec_input, (dec_h, dec_c))
#             dec_h = self.decoder_dropout(dec_h)

#             proj_in = torch.cat((dec_h, attn_ctx), -1)
#             out = self.mel_proj(proj_in).view(bsz, -1, self.mel_dim)
#             stop = torch.sigmoid(self.stop_proj(proj_in))

#             mel_outputs.append(out)
#             attn_scores.append(attn_score.unsqueeze(1))
#             stop_tokens.extend([stop] * self.r)

#             if greedy:
#                 if stop > self.stop_threshold or t > self.max_decoder_steps:
#                     break
#             else:
#                 if t >= T_decoder:
#                     break

#         mel_outputs = torch.cat(mel_outputs, dim=1)
#         attn_scores = torch.cat(attn_scores, dim=1)
#         stop_tokens = torch.cat(stop_tokens, dim=1)
#         assert greedy or mel_outputs.size(1) == T_decoder
#         return mel_outputs, stop_tokens, attn_scores


# class TextToMelSpectrogramModel(nn.Module):
#     def __init__(self, embed_dim=512, mel_dim=80, max_decoder_steps=1000, stop_threshold=0.5, r=3):
#         super().__init__()
#         self.mel_dim = mel_dim
#         # self.embedding = nn.Embedding(1, embed_dim)
#         # std = sqrt(2.0 / (1 + embed_dim))
#         # Use correct vocabulary size (67 characters + 1 for padding)
#         vocab_size = len(S2I)
#         self.embedding = nn.Embedding(vocab_size, embed_dim)
#         std = sqrt(2.0 / (vocab_size + embed_dim))
#         val = sqrt(3.0) * std
#         self.embedding.weight.data.uniform_(-val, val)

#         self.encoder = Encoder(embed_dim)
#         encoder_out_dim = self.encoder.blstm_units

#         self.decoder = Decoder(mel_dim, r, encoder_out_dim, max_decoder_steps=max_decoder_steps, stop_threshold=stop_threshold)

#         self.postnet = Postnet(mel_dim)

#     def parse_data_batch(self, batch):
#         device = next(self.parameters()).device
#         text, text_length, mel, stop, _ = batch
#         return (text.to(device).long(), text_length.to(device).long(), mel.to(device).float()), (mel.to(device).float(), stop.to(device).float())

#     def forward(self, inputs):
#         inputs, input_lengths, mels = inputs
#         x = self.embedding(inputs)
#         enc_out = self.encoder(x)
#         mel_out, stop_tokens, alignments = self.decoder(enc_out, mels, memory_lengths=input_lengths)
#         mel_post = self.postnet(mel_out)
#         mel_post = mel_out + mel_post
#         return mel_out, mel_post, stop_tokens, alignments

#     def inference(self, inputs):
#         return self.forward((inputs, None, None))









class BahdanauAttention(nn.Module):
    """Additive attention that produces alignment over encoder memory."""
    def __init__(self, query_dim: int, attn_dim: int, score_mask_value: float = -float("inf")):
        super().__init__()
        self.query_layer = nn.Linear(query_dim, attn_dim, bias=False)
        self.tanh = nn.Tanh()
        self.v = nn.Linear(attn_dim, 1, bias=False)
        self.score_mask_value = score_mask_value

    def forward(self, query, processed_memory, mask=None):
        if query.dim() == 2:
            query = query.unsqueeze(1)
        energies = self.get_energies(query, processed_memory)
        if mask is not None:
            energies.data.masked_fill_(mask.view(query.size(0), -1), self.score_mask_value)
        return self.get_probabilities(energies)

    def init_attention(self, processed_memory):
        return

    def get_energies(self, query, processed_memory):
        processed_query = self.query_layer(query)
        alignment = self.v(self.tanh(processed_query + processed_memory))
        return alignment.squeeze(-1)

    def get_probabilities(self, energies):
        return nn.Softmax(dim=1)(energies)


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
        self.blstm_units = blstm_units
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
        assert greedy or mel_outputs.size(1) == T_decoder
        return mel_outputs, stop_tokens, attn_scores


class TextToMelSpectrogramModel(nn.Module):
    def __init__(self, embed_dim=512, mel_dim=80, max_decoder_steps=1000, stop_threshold=0.5, r=3):
        super().__init__()
        self.mel_dim = mel_dim
        # Use correct vocabulary size (77 characters + 1 for padding = 78)
        vocab_size = len(S2I)
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        std = sqrt(2.0 / (vocab_size + embed_dim))
        val = sqrt(3.0) * std
        self.embedding.weight.data.uniform_(-val, val)

        self.encoder = Encoder(embed_dim)
        encoder_out_dim = self.encoder.blstm_units

        self.decoder = Decoder(mel_dim, r, encoder_out_dim, max_decoder_steps=max_decoder_steps, stop_threshold=stop_threshold)

        self.postnet = Postnet(mel_dim)

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
        return self.forward((inputs, None, None))


class TextToMelSpectrogramLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, predicts, targets):
        mel_target, stop_target = targets
        mel_target.requires_grad = False
        stop_target.requires_grad = False
        mel_pred, mel_post_pred, stop_pred, _ = predicts
        mel_loss = nn.MSELoss()(mel_pred, mel_target)
        post_loss = nn.MSELoss()(mel_post_pred, mel_target)
        stop_loss = nn.BCELoss()(stop_pred, stop_target)
        return mel_loss + post_loss + stop_loss











def build_text_to_mel_model(embed_dim=512, mel_dim=80, max_decoder_steps=1000, stop_threshold=0.5, r=3):
    """Build the text-to-mel spectrogram model with the same architecture as in the notebook"""
    model = TextToMelSpectrogramModel(
        embed_dim=embed_dim,
        mel_dim=mel_dim,
        max_decoder_steps=max_decoder_steps,
        stop_threshold=stop_threshold,
        r=r,
    )
    return model


def load_text_to_mel_model(weights_path: str = None, device: str = "cpu"):
    """Load the trained text-to-mel model from weights file with fallback options"""
    if weights_path is None:
        weights_path = settings.TEXT_TO_MEL_MODEL_WEIGHTS
    
    # Build model
    model = build_text_to_mel_model()
    
    # Load weights with error handling
    if os.path.exists(weights_path):
        try:
            model.load_state_dict(torch.load(weights_path, weights_only=False, map_location=device))
            print(f"Loaded model weights from {weights_path}")
        except Exception as e:
            print(f"Error loading model weights: {e}")
            print("Continuing with untrained model...")
    else:
        print(f"Warning: Weights file not found at {weights_path}")
        print("Continuing with untrained model...")
    
    # Set to evaluation mode and move to device
    model.eval()
    model = model.to(device)
    
    return model


def predict_mel_from_text(text: str, model, use_phonemes: bool = True) -> torch.Tensor:
    """
    Generate mel spectrogram from text input
    
    Args:
        text: Input text string
        model: Loaded TextToMelSpectrogramModel
        use_phonemes: Whether to convert text to phonemes first (default: True)
    
    Returns:
        mel_spectrogram: PyTorch tensor of shape (time_steps, 80)
    """
    device = next(model.parameters()).device
    
    # Preprocess text
    if use_phonemes:
        # Convert to phonemes if enabled
        phonemes = maybe_phonemes(text)
        if phonemes:
            text = phonemes
    
    # Convert text to sequence
    text_sequence = text_to_sequence(text)
    
    # Debug: Print sequence info
    print(f"Text: '{text}'")
    print(f"Sequence length: {len(text_sequence)}")
    print(f"Token IDs: {text_sequence}")
    print(f"Max token ID: {max(text_sequence)}")
    print(f"Embedding vocab size: {model.embedding.num_embeddings}")
    
    # Validate sequence before creating tensor
    if max(text_sequence) >= model.embedding.num_embeddings:
        raise ValueError(f"Token ID {max(text_sequence)} exceeds embedding vocabulary size {model.embedding.num_embeddings}")
    
    # Convert to tensor and add batch dimension
    text_tensor = torch.tensor(text_sequence, dtype=torch.long, device=device).unsqueeze(0)
    
    # Run inference
    model.eval()
    with torch.no_grad():
        mel_pred, mel_pred_post, _, _ = model.inference(text_tensor)
    
    # Use postnet output (better quality) - keep as tensor
    mel_spectrogram = mel_pred_post[0].detach()  # Shape: (time_steps, 80), stays on device
    
    return mel_spectrogram

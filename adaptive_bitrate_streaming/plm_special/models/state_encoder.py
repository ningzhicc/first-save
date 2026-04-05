"""
State encoders for ABR.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderNetwork(nn.Module):
    """
    The encoder network for encoding each piece of information of the state.
    This design of the network is from Pensieve/Genet.
    """
    def __init__(self, conv_size=4, bitrate_levels=6, embed_dim=128):
        super().__init__()
        self.output_mode = 'legacy'
        self.num_state_tokens = 6
        self.past_k = conv_size
        self.bitrate_levels = 6
        self.embed_dim = embed_dim
        self.fc1 = nn.Sequential(nn.Linear(1, embed_dim), nn.LeakyReLU())  # last bitrate
        self.fc2 = nn.Sequential(nn.Linear(1, embed_dim), nn.LeakyReLU())  # current buffer size
        self.conv3 = nn.Sequential(nn.Conv1d(1, embed_dim, conv_size), nn.LeakyReLU(), nn.Flatten())  # past k throughput
        self.conv4 = nn.Sequential(nn.Conv1d(1, embed_dim, conv_size), nn.LeakyReLU(), nn.Flatten())  # past k download time
        self.conv5 = nn.Sequential(nn.Conv1d(1, embed_dim, bitrate_levels), nn.LeakyReLU(), nn.Flatten())  # next chunk sizes
        self.fc6 = nn.Sequential(nn.Linear(1, embed_dim), nn.LeakyReLU())  # remain chunks        


    def forward(self, state):
        # state.shape: (batch_size, seq_len, 6, 6) -> (batch_size x seq_len, 6, 6)
        batch_size, seq_len = state.shape[0], state.shape[1]
        state = state.reshape(batch_size * seq_len, 6, 6)
        
        last_bitrate = state[..., 0:1, -1]
        current_buffer_size = state[..., 1:2, -1]
        throughputs = state[..., 2:3, :]
        download_time = state[..., 3:4, :]
        next_chunk_size = state[..., 4:5, :self.bitrate_levels]
        remain_chunks = state[..., 5:6, -1]
        
        features1 = self.fc1(last_bitrate).reshape(batch_size, seq_len, -1)
        features2 = self.fc2(current_buffer_size).reshape(batch_size, seq_len, -1)
        features3 = self.conv3(throughputs).reshape(batch_size, seq_len, -1)
        features4 = self.conv4(download_time).reshape(batch_size, seq_len, -1)
        features5 = self.conv5(next_chunk_size).reshape(batch_size, seq_len, -1)
        features6 = self.fc6(remain_chunks).reshape(batch_size, seq_len, -1)
        return features1, features2, features3, features4, features5, features6


class PatchReprogrammingEncoder(nn.Module):
    """
    ABR state encoder inspired by TIME-LLM patch reprogramming.
    """
    def __init__(
            self,
            plm_embed_size,
            word_embeddings,
            bitrate_levels=6,
            patch_embed_dim=128,
            patch_len=3,
            patch_stride=1,
            num_prototypes=64,
            num_heads=4,
            dropout=0.1,
    ):
        super().__init__()
        if patch_len < 1:
            raise ValueError('patch_len should be positive')
        if patch_stride < 1:
            raise ValueError('patch_stride should be positive')
        if num_prototypes < 1:
            raise ValueError('num_prototypes should be positive')
        if plm_embed_size % num_heads != 0:
            raise ValueError('plm_embed_size should be divisible by num_heads')

        self.output_mode = 'patch_reprogram'
        self.plm_embed_size = plm_embed_size
        self.patch_embed_dim = patch_embed_dim
        self.bitrate_levels = bitrate_levels
        self.patch_len = patch_len
        self.patch_stride = patch_stride
        self.num_heads = num_heads
        self.head_dim = plm_embed_size // num_heads

        self.feature_lengths = {
            0: 1,  # last bitrate
            1: 1,  # buffer size
            2: 6,  # throughput history
            3: 6,  # download-time history
            4: bitrate_levels,  # next chunk sizes
            5: 1,  # remaining chunks
        }
        self.feature_patch_sizes = {
            feature_idx: (1 if feature_length == 1 else patch_len)
            for feature_idx, feature_length in self.feature_lengths.items()
        }
        for feature_idx, feature_length in self.feature_lengths.items():
            if self.feature_patch_sizes[feature_idx] > feature_length:
                raise ValueError(f'patch size for feature {feature_idx} exceeds its length')
        self.feature_patch_counts = {
            feature_idx: self._count_patches(feature_length, self.feature_patch_sizes[feature_idx])
            for feature_idx, feature_length in self.feature_lengths.items()
        }
        self.num_state_tokens = sum(self.feature_patch_counts.values())

        self.patch_embedders = nn.ModuleDict({
            str(feature_idx): nn.Linear(self.feature_patch_sizes[feature_idx], patch_embed_dim)
            for feature_idx in self.feature_lengths
        })
        self.feature_embeddings = nn.Parameter(torch.randn(len(self.feature_lengths), patch_embed_dim) * 0.02)
        self.patch_norm = nn.LayerNorm(patch_embed_dim)

        self.query_proj = nn.Linear(patch_embed_dim, plm_embed_size)
        self.key_proj = nn.Linear(plm_embed_size, plm_embed_size)
        self.value_proj = nn.Linear(plm_embed_size, plm_embed_size)
        self.patch_proj = nn.Linear(patch_embed_dim, plm_embed_size)
        self.output_proj = nn.Linear(plm_embed_size, plm_embed_size)
        self.output_norm = nn.LayerNorm(plm_embed_size)
        self.dropout = nn.Dropout(dropout)

        prototype_init = self._build_prototype_init(word_embeddings, num_prototypes)
        self.text_prototypes = nn.Parameter(prototype_init)

    def _count_patches(self, feature_length, patch_size):
        return ((feature_length - patch_size) // self.patch_stride) + 1

    def _build_prototype_init(self, word_embeddings, num_prototypes):
        weight = word_embeddings.weight.detach()
        vocab_size = weight.shape[0]
        indices = torch.linspace(0, vocab_size - 1, steps=num_prototypes).round().long()
        return weight[indices].clone()

    def _extract_feature(self, state, feature_idx):
        if feature_idx in (0, 1, 5):
            return state[:, feature_idx:feature_idx + 1, -1:]
        if feature_idx in (2, 3):
            return state[:, feature_idx:feature_idx + 1, :]
        if feature_idx == 4:
            return state[:, feature_idx:feature_idx + 1, :self.bitrate_levels]
        raise ValueError(f'Unsupported feature index: {feature_idx}')

    def _patch_feature(self, feature_values, patch_size):
        if patch_size == 1:
            return feature_values.transpose(1, 2)
        return feature_values.unfold(dimension=-1, size=patch_size, step=self.patch_stride).reshape(feature_values.shape[0], -1, patch_size)

    def _reprogram_patches(self, patch_embeddings):
        batch_size, token_count, _ = patch_embeddings.shape

        queries = self.query_proj(patch_embeddings).reshape(batch_size, token_count, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        keys = self.key_proj(self.text_prototypes).reshape(self.text_prototypes.shape[0], self.num_heads, self.head_dim).permute(1, 0, 2)
        values = self.value_proj(self.text_prototypes).reshape(self.text_prototypes.shape[0], self.num_heads, self.head_dim).permute(1, 0, 2)

        attention_scores = torch.matmul(queries, keys.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention_probs = self.dropout(F.softmax(attention_scores, dim=-1))
        reprogrammed = torch.matmul(attention_probs, values)
        reprogrammed = reprogrammed.permute(0, 2, 1, 3).reshape(batch_size, token_count, self.plm_embed_size)

        reprogrammed = self.output_proj(reprogrammed) + self.patch_proj(patch_embeddings)
        return self.output_norm(self.dropout(reprogrammed))

    def forward(self, state):
        batch_size, seq_len = state.shape[0], state.shape[1]
        state = state.reshape(batch_size * seq_len, 6, 6)

        patch_embeddings = []
        for feature_idx in range(len(self.feature_lengths)):
            feature_values = self._extract_feature(state, feature_idx)
            feature_patches = self._patch_feature(feature_values, self.feature_patch_sizes[feature_idx])
            feature_embeddings = self.patch_embedders[str(feature_idx)](feature_patches)
            feature_embeddings = feature_embeddings + self.feature_embeddings[feature_idx].view(1, 1, -1)
            patch_embeddings.append(feature_embeddings)

        patch_embeddings = torch.cat(patch_embeddings, dim=1)
        patch_embeddings = self.patch_norm(patch_embeddings)
        reprogrammed = self._reprogram_patches(patch_embeddings)
        return reprogrammed.reshape(batch_size, seq_len, self.num_state_tokens, self.plm_embed_size)


class SemanticReprogrammingEncoder(nn.Module):
    """
    Reprogram numerical ABR features into the LLM space without explicit patching.
    This variant follows the teacher's suggestion: keep the 6-step convolutional
    summarization, then align the resulting numeric tokens with text semantics.
    """

    FEATURE_DESCRIPTIONS = [
        'selected bitrate',
        'buffer size',
        'throughput history',
        'download time history',
        'next chunk sizes',
        'remaining video chunks',
    ]

    ANCHOR_TEXTS = [
        'low bitrate',
        'high bitrate',
        'stable bitrate',
        'buffer low',
        'buffer safe',
        'bandwidth drop',
        'bandwidth stable',
        'bandwidth rise',
        'download time long',
        'download time short',
        'large next chunk',
        'small next chunk',
        'video ending',
        'many chunks left',
    ]

    def __init__(
            self,
            plm_embed_size,
            word_embeddings,
            tokenizer,
            bitrate_levels=6,
            numeric_embed_dim=256,
            num_heads=4,
            dropout=0.1,
            conv_size=6,
    ):
        super().__init__()
        if tokenizer is None:
            raise ValueError('tokenizer is required for semantic reprogramming')
        if plm_embed_size % num_heads != 0:
            raise ValueError('plm_embed_size should be divisible by num_heads')

        self.output_mode = 'semantic_reprogram'
        self.num_state_tokens = 6
        self.plm_embed_size = plm_embed_size
        self.num_heads = num_heads
        self.head_dim = plm_embed_size // num_heads
        self.tokenizer = tokenizer

        # Reuse the ABR-specific convolutional state summarizer but aggregate the
        # 6-step history in one shot for temporal channels.
        self.numeric_encoder = EncoderNetwork(
            conv_size=conv_size,
            bitrate_levels=bitrate_levels,
            embed_dim=numeric_embed_dim,
        )

        self.numeric_projections = nn.ModuleList([
            nn.Linear(numeric_embed_dim, plm_embed_size)
            for _ in range(self.num_state_tokens)
        ])
        self.pre_align_norm = nn.LayerNorm(plm_embed_size)
        self.output_proj = nn.Linear(plm_embed_size, plm_embed_size)
        self.output_norm = nn.LayerNorm(plm_embed_size)
        self.dropout = nn.Dropout(dropout)

        self.query_proj = nn.Linear(plm_embed_size, plm_embed_size)
        self.key_proj = nn.Linear(plm_embed_size, plm_embed_size)
        self.value_proj = nn.Linear(plm_embed_size, plm_embed_size)

        feature_text_init = self._build_phrase_bank(
            tokenizer=tokenizer,
            word_embeddings=word_embeddings,
            phrases=self.FEATURE_DESCRIPTIONS,
        )
        anchor_text_init = self._build_phrase_bank(
            tokenizer=tokenizer,
            word_embeddings=word_embeddings,
            phrases=self.ANCHOR_TEXTS,
        )
        self.feature_text_embeddings = nn.Parameter(feature_text_init)
        self.anchor_embeddings = nn.Parameter(anchor_text_init)

    def _build_phrase_bank(self, tokenizer, word_embeddings, phrases):
        weight = word_embeddings.weight.detach().cpu()
        phrase_embeddings = []
        for phrase in phrases:
            token_ids = tokenizer.encode(phrase, add_special_tokens=False)
            if len(token_ids) == 0:
                raise ValueError(f'Phrase "{phrase}" produces no tokens')
            token_tensor = torch.as_tensor(token_ids, dtype=torch.long)
            phrase_embedding = weight.index_select(0, token_tensor).mean(dim=0)
            phrase_embeddings.append(phrase_embedding)
        return torch.stack(phrase_embeddings, dim=0)

    def _align_numeric_tokens(self, numeric_tokens):
        batch_size, token_count, _ = numeric_tokens.shape
        queries = self.query_proj(numeric_tokens).reshape(batch_size, token_count, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        keys = self.key_proj(self.anchor_embeddings).reshape(self.anchor_embeddings.shape[0], self.num_heads, self.head_dim).permute(1, 0, 2)
        values = self.value_proj(self.anchor_embeddings).reshape(self.anchor_embeddings.shape[0], self.num_heads, self.head_dim).permute(1, 0, 2)

        attention_scores = torch.matmul(queries, keys.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention_probs = self.dropout(F.softmax(attention_scores, dim=-1))
        aligned = torch.matmul(attention_probs, values)
        aligned = aligned.permute(0, 2, 1, 3).reshape(batch_size, token_count, self.plm_embed_size)
        aligned = self.output_proj(aligned)
        return self.output_norm(numeric_tokens + self.dropout(aligned))

    def forward(self, state):
        feature_groups = self.numeric_encoder(state)
        numeric_tokens = []
        for feature_idx, feature_group in enumerate(feature_groups):
            projected = self.numeric_projections[feature_idx](feature_group)
            numeric_tokens.append(projected)

        numeric_tokens = torch.stack(numeric_tokens, dim=2)
        numeric_tokens = numeric_tokens + self.feature_text_embeddings.view(1, 1, self.num_state_tokens, -1)
        batch_size, seq_len = numeric_tokens.shape[0], numeric_tokens.shape[1]
        numeric_tokens = self.pre_align_norm(numeric_tokens.reshape(batch_size * seq_len, self.num_state_tokens, self.plm_embed_size))
        aligned_tokens = self._align_numeric_tokens(numeric_tokens)
        return aligned_tokens.reshape(batch_size, seq_len, self.num_state_tokens, self.plm_embed_size)

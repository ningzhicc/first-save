"""
State encoders for ABR.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class NumericIntraStepAttentionBlock(nn.Module):
    """
    Lightweight self-attention in the numeric feature space before semantic alignment.
    """

    def __init__(self, embed_dim, num_heads, hidden_dim, dropout=0.1):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError('embed_dim should be divisible by num_heads for numeric intra-step attention')

        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, tokens, attn_mask=None):
        normed_tokens = self.norm1(tokens)
        attn_output, _ = self.attn(
            normed_tokens,
            normed_tokens,
            normed_tokens,
            attn_mask=attn_mask,
            need_weights=False,
        )
        tokens = tokens + self.dropout1(attn_output)

        normed_tokens = self.norm2(tokens)
        ffn_output = self.ffn(normed_tokens)
        return tokens + self.dropout2(ffn_output)


class NumericConditionalAttentionBlock(nn.Module):
    """
    Condition numeric state tokens on previous-action / previous-reward tokens
    without mixing state tokens with each other.
    """

    def __init__(self, embed_dim, num_heads, hidden_dim, dropout=0.1):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError('embed_dim should be divisible by num_heads for numeric conditional attention')

        self.query_norm = nn.LayerNorm(embed_dim)
        self.context_norm = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)

        self.ffn_norm = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, state_tokens, context_tokens):
        normed_state_tokens = self.query_norm(state_tokens)
        normed_context_tokens = self.context_norm(context_tokens)
        attn_output, _ = self.attn(
            normed_state_tokens,
            normed_context_tokens,
            normed_context_tokens,
            need_weights=False,
        )
        state_tokens = state_tokens + self.dropout1(attn_output)

        normed_state_tokens = self.ffn_norm(state_tokens)
        ffn_output = self.ffn(normed_state_tokens)
        return state_tokens + self.dropout2(ffn_output)


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


    def forward(self, state, returns=None, prev_actions=None, prev_rewards=None):
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

    def forward(self, state, returns=None, prev_actions=None, prev_rewards=None):
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

    PRE_ALIGN_MASK_MODES = {
        'context_readonly',
        'state_to_prev_action',
        'state_to_prev_reward',
        'state_only',
    }

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
            use_pre_align_intra_step_attn=False,
            pre_align_intra_step_attn_heads=8,
            pre_align_intra_step_attn_hidden_dim=1024,
            pre_align_intra_step_attn_dropout=0.1,
            use_pre_align_intra_step_mask=False,
            pre_align_intra_step_mask_mode='context_readonly',
            use_pre_align_conditional_attn=False,
    ):
        super().__init__()
        if tokenizer is None:
            raise ValueError('tokenizer is required for semantic reprogramming')
        if plm_embed_size % num_heads != 0:
            raise ValueError('plm_embed_size should be divisible by num_heads')
        if (use_pre_align_intra_step_attn or use_pre_align_conditional_attn) and numeric_embed_dim % pre_align_intra_step_attn_heads != 0:
            raise ValueError('numeric_embed_dim should be divisible by pre_align_intra_step_attn_heads')
        if use_pre_align_intra_step_attn and use_pre_align_conditional_attn:
            raise ValueError('pre-align intra-step self-attention and conditional attention cannot be enabled together')
        if pre_align_intra_step_mask_mode not in self.PRE_ALIGN_MASK_MODES:
            raise ValueError(
                f'Unsupported pre-align intra-step mask mode: {pre_align_intra_step_mask_mode}. '
                f'Expected one of {sorted(self.PRE_ALIGN_MASK_MODES)}'
            )

        self.output_mode = 'semantic_reprogram'
        self.num_state_tokens = 6
        self.plm_embed_size = plm_embed_size
        self.num_heads = num_heads
        self.head_dim = plm_embed_size // num_heads
        self.tokenizer = tokenizer
        self.numeric_embed_dim = numeric_embed_dim
        self.use_pre_align_intra_step_attn = use_pre_align_intra_step_attn
        self.use_pre_align_intra_step_mask = use_pre_align_intra_step_mask
        self.pre_align_intra_step_mask_mode = pre_align_intra_step_mask_mode
        self.use_pre_align_conditional_attn = use_pre_align_conditional_attn

        # Reuse the ABR-specific convolutional state summarizer but aggregate the
        # 6-step history in one shot for temporal channels.
        self.numeric_encoder = EncoderNetwork(
            conv_size=conv_size,
            bitrate_levels=bitrate_levels,
            embed_dim=numeric_embed_dim,
        )

        if self.use_pre_align_intra_step_attn or self.use_pre_align_conditional_attn:
            # Keep the historical checkpoint key stable even though this branch now
            # consumes previous rewards instead of target returns.
            self.return_context_proj = nn.Linear(1, numeric_embed_dim)
            self.prev_action_context_proj = nn.Linear(1, numeric_embed_dim)
        else:
            self.return_context_proj = None
            self.prev_action_context_proj = None

        if self.use_pre_align_intra_step_attn:
            self.pre_align_intra_step_attn = NumericIntraStepAttentionBlock(
                embed_dim=numeric_embed_dim,
                num_heads=pre_align_intra_step_attn_heads,
                hidden_dim=pre_align_intra_step_attn_hidden_dim,
                dropout=pre_align_intra_step_attn_dropout,
            )
            self.intra_step_token_embeddings = nn.Parameter(
                torch.randn(self.num_state_tokens + 2, numeric_embed_dim) * 0.02
            )
            intra_step_attn_mask = self._build_pre_align_intra_step_mask() if self.use_pre_align_intra_step_mask else None
            self.register_buffer('pre_align_intra_step_attn_mask', intra_step_attn_mask, persistent=False)
            self.pre_align_conditional_attn = None
            self.conditional_context_embeddings = None
        elif self.use_pre_align_conditional_attn:
            self.pre_align_intra_step_attn = None
            self.pre_align_conditional_attn = NumericConditionalAttentionBlock(
                embed_dim=numeric_embed_dim,
                num_heads=pre_align_intra_step_attn_heads,
                hidden_dim=pre_align_intra_step_attn_hidden_dim,
                dropout=pre_align_intra_step_attn_dropout,
            )
            self.conditional_context_embeddings = nn.Parameter(
                torch.randn(2, numeric_embed_dim) * 0.02
            )
            self.register_buffer('pre_align_intra_step_attn_mask', None, persistent=False)
        else:
            self.pre_align_intra_step_attn = None
            self.pre_align_conditional_attn = None
            self.conditional_context_embeddings = None
            self.register_buffer('pre_align_intra_step_attn_mask', None, persistent=False)

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

    def _build_pre_align_intra_step_mask(self):
        token_count = self.num_state_tokens + 2
        attn_mask = torch.zeros((token_count, token_count), dtype=torch.bool)
        prev_reward_token_idx = self.num_state_tokens
        prev_action_token_idx = self.num_state_tokens + 1

        # Keep the two context tokens as read-only conditioning signals.
        for token_idx in (prev_reward_token_idx, prev_action_token_idx):
            attn_mask[token_idx, :] = True
            attn_mask[token_idx, token_idx] = False

        if self.pre_align_intra_step_mask_mode == 'context_readonly':
            return attn_mask
        if self.pre_align_intra_step_mask_mode == 'state_to_prev_action':
            attn_mask[:self.num_state_tokens, prev_reward_token_idx] = True
            return attn_mask
        if self.pre_align_intra_step_mask_mode == 'state_to_prev_reward':
            attn_mask[:self.num_state_tokens, prev_action_token_idx] = True
            return attn_mask
        if self.pre_align_intra_step_mask_mode == 'state_only':
            attn_mask[:self.num_state_tokens, prev_reward_token_idx:] = True
            return attn_mask

        raise ValueError(f'Unsupported pre-align intra-step mask mode: {self.pre_align_intra_step_mask_mode}')
        return attn_mask

    def _apply_pre_align_intra_step_attention(self, numeric_tokens, prev_rewards, prev_actions):
        if self.pre_align_intra_step_attn is None:
            return numeric_tokens
        if prev_rewards is None or prev_actions is None:
            raise ValueError('prev_rewards and prev_actions are required when use_pre_align_intra_step_attn is enabled')

        reward_tokens = self.return_context_proj(prev_rewards).unsqueeze(2)
        prev_action_tokens = self.prev_action_context_proj(prev_actions).unsqueeze(2)
        intra_step_tokens = torch.cat((numeric_tokens, reward_tokens, prev_action_tokens), dim=2)
        intra_step_tokens = intra_step_tokens + self.intra_step_token_embeddings.view(1, 1, self.num_state_tokens + 2, -1)

        batch_size, seq_len = intra_step_tokens.shape[0], intra_step_tokens.shape[1]
        intra_step_tokens = intra_step_tokens.reshape(batch_size * seq_len, self.num_state_tokens + 2, self.numeric_embed_dim)
        intra_step_tokens = self.pre_align_intra_step_attn(
            intra_step_tokens,
            attn_mask=self.pre_align_intra_step_attn_mask,
        )
        intra_step_tokens = intra_step_tokens.reshape(batch_size, seq_len, self.num_state_tokens + 2, self.numeric_embed_dim)
        return intra_step_tokens[:, :, :self.num_state_tokens, :]

    def _apply_pre_align_conditional_attention(self, numeric_tokens, prev_rewards, prev_actions):
        if self.pre_align_conditional_attn is None:
            return numeric_tokens
        if prev_rewards is None or prev_actions is None:
            raise ValueError('prev_rewards and prev_actions are required when use_pre_align_conditional_attn is enabled')

        reward_tokens = self.return_context_proj(prev_rewards).unsqueeze(2)
        prev_action_tokens = self.prev_action_context_proj(prev_actions).unsqueeze(2)
        context_tokens = torch.cat((reward_tokens, prev_action_tokens), dim=2)
        context_tokens = context_tokens + self.conditional_context_embeddings.view(1, 1, 2, -1)

        batch_size, seq_len = numeric_tokens.shape[0], numeric_tokens.shape[1]
        state_tokens = numeric_tokens.reshape(batch_size * seq_len, self.num_state_tokens, self.numeric_embed_dim)
        context_tokens = context_tokens.reshape(batch_size * seq_len, 2, self.numeric_embed_dim)
        state_tokens = self.pre_align_conditional_attn(state_tokens, context_tokens)
        return state_tokens.reshape(batch_size, seq_len, self.num_state_tokens, self.numeric_embed_dim)

    def forward(self, state, returns=None, prev_actions=None, prev_rewards=None):
        feature_groups = self.numeric_encoder(state)
        numeric_tokens = torch.stack(list(feature_groups), dim=2)
        if prev_rewards is None:
            prev_rewards = returns
        numeric_tokens = self._apply_pre_align_intra_step_attention(numeric_tokens, prev_rewards, prev_actions)
        numeric_tokens = self._apply_pre_align_conditional_attention(numeric_tokens, prev_rewards, prev_actions)

        projected_tokens = []
        for feature_idx in range(self.num_state_tokens):
            projected = self.numeric_projections[feature_idx](numeric_tokens[:, :, feature_idx, :])
            projected_tokens.append(projected)

        numeric_tokens = torch.stack(projected_tokens, dim=2)
        numeric_tokens = numeric_tokens + self.feature_text_embeddings.view(1, 1, self.num_state_tokens, -1)
        batch_size, seq_len = numeric_tokens.shape[0], numeric_tokens.shape[1]
        numeric_tokens = self.pre_align_norm(numeric_tokens.reshape(batch_size * seq_len, self.num_state_tokens, self.plm_embed_size))
        aligned_tokens = self._align_numeric_tokens(numeric_tokens)
        return aligned_tokens.reshape(batch_size, seq_len, self.num_state_tokens, self.plm_embed_size)

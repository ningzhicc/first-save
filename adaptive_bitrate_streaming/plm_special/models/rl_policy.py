import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import deque
    

INF = 1e5


class ResidualGate(nn.Module):
    """Zero-init residual gate for light-weight feature injection."""

    def __init__(self, init_value=0.0):
        super().__init__()
        self.gate = nn.Parameter(torch.tensor(float(init_value)))

    def forward(self, residual):
        return torch.tanh(self.gate) * residual


class IntraStateSelfAttentionBlock(nn.Module):
    """
    Token-level self-attention inside a single ABR decision state.
    The block is applied independently for each timestep after the
    state encoder/reprogramming stage and before tokens are packed for the PLM.
    """

    def __init__(self, embed_dim, num_heads, dropout=0.1, mlp_ratio=4):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError('embed_dim should be divisible by num_heads for intra-state attention')

        hidden_dim = embed_dim * mlp_ratio
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

    def forward(self, state_tokens):
        normed_tokens = self.norm1(state_tokens)
        attn_output, _ = self.attn(normed_tokens, normed_tokens, normed_tokens, need_weights=False)
        state_tokens = state_tokens + self.dropout1(attn_output)

        normed_tokens = self.norm2(state_tokens)
        ffn_output = self.ffn(normed_tokens)
        return state_tokens + self.dropout2(ffn_output)


class TemporalStateSelfAttentionBlock(nn.Module):
    """
    Time-level self-attention across ABR decision steps.
    The block is applied independently for each state-token slot after
    reprogramming/state encoding and before token packing for the PLM.
    """

    def __init__(self, embed_dim, num_heads, dropout=0.1, mlp_ratio=4):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError('embed_dim should be divisible by num_heads for temporal attention')

        hidden_dim = embed_dim * mlp_ratio
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

    def forward(self, state_tokens, attn_mask=None):
        normed_tokens = self.norm1(state_tokens)
        attn_output, _ = self.attn(
            normed_tokens,
            normed_tokens,
            normed_tokens,
            attn_mask=attn_mask,
            need_weights=False,
        )
        state_tokens = state_tokens + self.dropout1(attn_output)

        normed_tokens = self.norm2(state_tokens)
        ffn_output = self.ffn(normed_tokens)
        return state_tokens + self.dropout2(ffn_output)


class OfflineRLPolicy(nn.Module):
    def __init__(
            self,
            state_feature_dim,
            bitrate_levels,
            state_encoder,
            plm,
            plm_embed_size,
            max_length=None,
            max_ep_len=100,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            device_out = None,
            residual = False, 
            conv_size = 4,  
            which_layer = -1,  # for early stopping: specify which layer to stop
            use_intra_state_attn=False,
            use_gated_intra_state_attn=False,
            intra_state_attn_heads=4,
            intra_state_attn_dropout=0.1,
            use_temporal_state_attn=False,
            temporal_state_attn_heads=4,
            temporal_state_attn_dropout=0.1,
            use_temporal_causal_mask=False,
            **kwargs
    ):
        super().__init__()
        
        if device_out is None:
            device_out = device

        self.bitrate_levels = bitrate_levels
        self.max_length = max_length
        self.state_encoder_mode = getattr(state_encoder, 'output_mode', 'legacy')

        self.plm = plm
        self.plm_embed_size = plm_embed_size
        self.use_intra_state_attn = use_intra_state_attn
        self.use_gated_intra_state_attn = use_gated_intra_state_attn
        self.use_temporal_state_attn = use_temporal_state_attn
        self.use_temporal_causal_mask = use_temporal_causal_mask

        if self.use_intra_state_attn and self.use_gated_intra_state_attn:
            raise ValueError('use_intra_state_attn and use_gated_intra_state_attn cannot be enabled together')

        # =========== multimodal encoder (start) ===========
        self.state_encoder = state_encoder
        self.state_feature_dim = state_feature_dim
        self.embed_timestep = nn.Embedding(max_ep_len + 1, plm_embed_size).to(device)
        self.embed_return = nn.Linear(1, plm_embed_size).to(device)
        self.embed_action = nn.Linear(1, plm_embed_size).to(device)
        if self.state_encoder_mode == 'legacy':
            self.embed_state1 = nn.Linear(state_feature_dim, plm_embed_size).to(device)
            self.embed_state2 = nn.Linear(state_feature_dim, plm_embed_size).to(device)
            self.embed_state3 = nn.Linear(state_feature_dim * (6 - conv_size + 1), plm_embed_size).to(device)
            self.embed_state4 = nn.Linear(state_feature_dim * (6 - conv_size + 1), plm_embed_size).to(device)
            self.embed_state5 = nn.Linear(state_feature_dim, plm_embed_size).to(device)
            self.embed_state6 = nn.Linear(state_feature_dim, plm_embed_size).to(device)
            self.state_token_count = 6
        else:
            self.state_token_count = state_encoder.num_state_tokens

        if self.use_intra_state_attn or self.use_gated_intra_state_attn:
            self.intra_state_attn = IntraStateSelfAttentionBlock(
                embed_dim=plm_embed_size,
                num_heads=intra_state_attn_heads,
                dropout=intra_state_attn_dropout,
            ).to(device)
        else:
            self.intra_state_attn = None

        if self.use_gated_intra_state_attn:
            self.intra_state_attn_gate = ResidualGate(init_value=0.0).to(device)
        else:
            self.intra_state_attn_gate = None

        if self.use_temporal_state_attn:
            self.temporal_state_attn = TemporalStateSelfAttentionBlock(
                embed_dim=plm_embed_size,
                num_heads=temporal_state_attn_heads,
                dropout=temporal_state_attn_dropout,
            ).to(device)
        else:
            self.temporal_state_attn = None

        self.embed_ln = nn.LayerNorm(plm_embed_size).to(device)
        # =========== multimodal encoder (end) ===========
    
        self.action_head = nn.Linear(plm_embed_size, bitrate_levels).to(device)  # the so-called networking head in our paper

        self.device = device
        self.device_out = device_out

        # the following are used for evaluation
        self.states_dq = deque([torch.zeros((1, 0, self.state_token_count, plm_embed_size), device=device)], maxlen=max_length)
        self.returns_dq = deque([torch.zeros((1, 0, plm_embed_size), device=device)], maxlen=max_length)
        self.actions_dq = deque([torch.zeros((1, 0, plm_embed_size), device=device)], maxlen=max_length)
        self.prev_action_value = torch.zeros((1, 1, 1), dtype=torch.float32, device=device)
        self.prev_reward_value = torch.zeros((1, 1, 1), dtype=torch.float32, device=device)

        self.residual = residual
        self.which_layer = which_layer
        modules_except_plm = [
            self.state_encoder, self.embed_timestep, self.embed_return, self.embed_action, self.embed_ln, self.action_head
        ]
        if self.intra_state_attn is not None:
            modules_except_plm.append(self.intra_state_attn)
        if self.intra_state_attn_gate is not None:
            modules_except_plm.append(self.intra_state_attn_gate)
        if self.temporal_state_attn is not None:
            modules_except_plm.append(self.temporal_state_attn)
        if self.state_encoder_mode == 'legacy':
            modules_except_plm.extend([
                self.embed_state1, self.embed_state2, self.embed_state3,
                self.embed_state4, self.embed_state5, self.embed_state6
            ])
        self.modules_except_plm = nn.ModuleList(modules_except_plm)

    def _truncate_token_sequence(self, stacked_inputs):
        # The dataset and deques already control the context window by decision steps.
        # Truncating by raw token count can cut a state-action block in half.
        return stacked_inputs

    def _build_prev_actions(self, actions):
        prev_actions = torch.zeros_like(actions)
        if actions.shape[1] > 1:
            prev_actions[:, 1:] = actions[:, :-1]
        return prev_actions

    def _build_prev_rewards(self, rewards):
        prev_rewards = torch.zeros_like(rewards)
        if rewards.shape[1] > 1:
            prev_rewards[:, 1:] = rewards[:, :-1]
        return prev_rewards

    def _encode_states(self, states, time_embeddings, returns=None, prev_actions=None, prev_rewards=None):
        states_features = self.state_encoder(
            states,
            returns=returns,
            prev_actions=prev_actions,
            prev_rewards=prev_rewards,
        )
        if self.state_encoder_mode == 'legacy':
            states_embeddings1 = self.embed_state1(states_features[0]) + time_embeddings
            states_embeddings2 = self.embed_state2(states_features[1]) + time_embeddings
            states_embeddings3 = self.embed_state3(states_features[2]) + time_embeddings
            states_embeddings4 = self.embed_state4(states_features[3]) + time_embeddings
            states_embeddings5 = self.embed_state5(states_features[4]) + time_embeddings
            states_embeddings6 = self.embed_state6(states_features[5]) + time_embeddings
            state_tokens = torch.stack([
                states_embeddings1,
                states_embeddings2,
                states_embeddings3,
                states_embeddings4,
                states_embeddings5,
                states_embeddings6,
            ], dim=2)
        else:
            state_tokens = states_features + time_embeddings.unsqueeze(2)

        if self.intra_state_attn is None:
            return state_tokens

        batch_size, seq_len, token_count, embed_dim = state_tokens.shape
        flat_state_tokens = state_tokens.reshape(batch_size * seq_len, token_count, embed_dim)
        updated_state_tokens = self.intra_state_attn(flat_state_tokens)
        if self.intra_state_attn_gate is not None:
            updated_state_tokens = flat_state_tokens + self.intra_state_attn_gate(updated_state_tokens - flat_state_tokens)
        return updated_state_tokens.reshape(batch_size, seq_len, token_count, embed_dim)

    def _apply_temporal_state_attn(self, state_tokens, attn_mask=None):
        if self.temporal_state_attn is None:
            return state_tokens

        batch_size, seq_len, token_count, embed_dim = state_tokens.shape
        if seq_len <= 1:
            return state_tokens

        if attn_mask is None and self.use_temporal_causal_mask:
            attn_mask = torch.triu(
                torch.ones((seq_len, seq_len), dtype=torch.bool, device=state_tokens.device),
                diagonal=1,
            )

        state_tokens = state_tokens.transpose(1, 2).reshape(batch_size * token_count, seq_len, embed_dim)
        state_tokens = self.temporal_state_attn(state_tokens, attn_mask=attn_mask)
        return state_tokens.reshape(batch_size, token_count, seq_len, embed_dim).transpose(1, 2)

    def _pack_sequence_inputs(self, returns_embeddings, state_tokens, action_embeddings=None):
        stacked_inputs = []
        action_embed_positions = []
        current_position = 0
        action_seq_len = 0 if action_embeddings is None else action_embeddings.shape[1]

        for i in range(returns_embeddings.shape[1]):
            step_inputs = [returns_embeddings[0, i:i + 1], state_tokens[0, i]]
            if i < action_seq_len:
                step_inputs.append(action_embeddings[0, i:i + 1])
            packed_step = torch.cat(step_inputs, dim=0)
            stacked_inputs.append(packed_step)
            if i < action_seq_len:
                action_embed_positions.append(current_position + packed_step.shape[0])
            current_position += packed_step.shape[0]

        stacked_inputs = torch.cat(stacked_inputs, dim=0).unsqueeze(0)
        return stacked_inputs, np.asarray(action_embed_positions, dtype=int)

    def forward(self, states, actions, rewards, returns, timesteps, attention_mask=None):
        """
        Forward function, used for training.
        """
        assert actions.shape[0] == 1, 'batch size should be 1 to avoid CUDA memory exceed'

        # Step 1: process actions, returns and timesteps first as they are simple
        actions = actions.to(self.device)  # shape: (1, seq_len, 1)
        rewards = rewards.to(self.device)  # shape: (1, seq_len, 1)
        returns = returns.to(self.device)  # shape: (1, seq_len, 1)
        timesteps = timesteps.to(self.device)  # shape: (1, seq_len)

        # 1.1 embed action, return, timestep
        action_embeddings = self.embed_action(actions)  # shape: (1, seq_len, embed_size)
        returns_embeddings = self.embed_return(returns)  # shape: (1, seq_len, embed_size)
        time_embeddings = self.embed_timestep(timesteps)  # shape: (1, seq_len, embed_size)

        # 1.2 time embeddings are treated similar to positional embeddings
        action_embeddings = action_embeddings + time_embeddings
        returns_embeddings = returns_embeddings + time_embeddings
        prev_actions = self._build_prev_actions(actions)
        prev_rewards = self._build_prev_rewards(rewards)

        # Step 2: process states, turn them into embeddings.
        states = states.to(self.device)  # shape: (1, seq_len, 6, 6)
        states_embeddings = self._encode_states(
            states,
            time_embeddings,
            returns=returns,
            prev_actions=prev_actions,
            prev_rewards=prev_rewards,
        )
        states_embeddings = self._apply_temporal_state_attn(states_embeddings)
        
        # Step 3: stack returns, states, actions embeddings.
        # this makes the sequence look like (R_1, s_1-1, s_1-2, ..., s_1-n, a_1, R_2, s_2-1, ..., s_2-m, a_2, ...)
        # which works nice in an autoregressive sense since states predict actions
        stacked_inputs, action_embed_positions = self._pack_sequence_inputs(
            returns_embeddings,
            states_embeddings,
            action_embeddings=action_embeddings,
        )
        stacked_inputs = self._truncate_token_sequence(stacked_inputs)
        stacked_inputs_ln = self.embed_ln(stacked_inputs)  # layer normalization
        
        # Step 4: feed stacked embeddings into the plm
        # 4.1 create attention mask
        if attention_mask is None:
            # 1 if can be attended to, 0 if not
            attention_mask = torch.ones((stacked_inputs_ln.shape[0], stacked_inputs_ln.shape[1]), dtype=torch.long, device=self.device)

        # we feed in the input embeddings (not word indices as in NLP) to the model
        transformer_outputs = self.plm(
            inputs_embeds=stacked_inputs_ln,
            attention_mask=attention_mask,
            output_hidden_states=True,
            #stop_layer_idx=self.which_layer,
        )
        logits = transformer_outputs['last_hidden_state']
        if self.residual:
            logits = logits + stacked_inputs_ln  # residual add

        # Step 5: predict actions
        # we need to locate the logits corresponding to the state embeddings
        # simply using `action_embed_positions[i] - 2` will do.
        action_embed_positions = action_embed_positions.astype(int)
        logits_used = logits[:, action_embed_positions - 2]
        action_pred = self.action_head(logits_used)

        return action_pred

    def sample(self, state, target_return, timestep, prev_reward=None, **kwargs):
        """
        Sample action function, used for evaluation/testing.
        """
        # Step 1: gather the previous state/action/return features from the dequeue
        prev_return_embeddings = torch.cat(list(self.returns_dq), dim=1)
        prev_state_tokens = torch.cat(list(self.states_dq), dim=1)
        prev_action_embeddings = torch.cat(list(self.actions_dq), dim=1)

        # Step 2: process target return and timesteps
        target_return = torch.as_tensor(target_return, dtype=torch.float32, device=self.device).reshape(1, 1, 1)
        timestep = torch.as_tensor(timestep, dtype=torch.int32, device=self.device).reshape(1, 1)

        return_embeddings = self.embed_return(target_return)
        time_embeddings = self.embed_timestep(timestep)

        return_embeddings = return_embeddings + time_embeddings

        if int(timestep.item()) == 0:
            prev_actions = torch.zeros((1, 1, 1), dtype=torch.float32, device=self.device)
        else:
            prev_actions = self.prev_action_value

        if prev_reward is None:
            if int(timestep.item()) == 0:
                prev_rewards = torch.zeros((1, 1, 1), dtype=torch.float32, device=self.device)
            else:
                prev_rewards = self.prev_reward_value
        else:
            prev_rewards = torch.as_tensor(prev_reward, dtype=torch.float32, device=self.device).reshape(1, 1, 1)
            self.prev_reward_value = prev_rewards.detach()

        # Step 4: process state
        state = state.to(self.device)
        state_tokens = self._encode_states(
            state,
            time_embeddings,
            returns=target_return,
            prev_actions=prev_actions,
            prev_rewards=prev_rewards,
        )
        all_return_embeddings = torch.cat((prev_return_embeddings, return_embeddings), dim=1)
        all_state_tokens = torch.cat((prev_state_tokens, state_tokens), dim=1)
        all_state_tokens = self._apply_temporal_state_attn(all_state_tokens)

        # Step 5: stack return, state and previous embeddings
        stacked_inputs, _ = self._pack_sequence_inputs(
            all_return_embeddings,
            all_state_tokens,
            action_embeddings=prev_action_embeddings,
        )
        stacked_inputs = self._truncate_token_sequence(stacked_inputs)
        stacked_inputs_ln = self.embed_ln(stacked_inputs)  # layer normalization

        # 1 if can be attended to, 0 if not
        attention_mask = torch.ones((stacked_inputs_ln.shape[0], stacked_inputs_ln.shape[1]), dtype=torch.long, device=self.device)

        transformer_outputs = self.plm(
            inputs_embeds=stacked_inputs_ln,
            attention_mask=attention_mask,
            output_hidden_states=True,
            stop_layer_idx=self.which_layer,
        )
        logits = transformer_outputs['last_hidden_state']
        if self.residual:
            logits = logits + stacked_inputs_ln  # residual add

        # Step 6: predict the bitrate for next chunk
        logits_used = logits[:, -1:]
        action_pred = self.action_head(logits_used)
        action_pred = action_pred.reshape(-1)
        bitrate, _ = self._sample(action_pred)

        # compute action embeddings 
        action_tensor = torch.zeros(1, 1, 1, dtype=torch.float32, device=self.device)
        action_tensor[..., 0] = (bitrate + 1) / self.bitrate_levels
        action_embeddings = self.embed_action(action_tensor) + time_embeddings
        self.prev_action_value = action_tensor.detach()
        
        # update deques
        self.returns_dq.append(return_embeddings)
        self.states_dq.append(state_tokens)
        self.actions_dq.append(action_embeddings)

        return bitrate
    
    def clear_dq(self):
        self.states_dq.clear()
        self.actions_dq.clear()
        self.returns_dq.clear()
        
        self.states_dq.append(torch.zeros((1, 0, self.state_token_count, self.plm_embed_size), device=self.device))
        self.actions_dq.append(torch.zeros((1, 0, self.plm_embed_size), device=self.device))
        self.returns_dq.append(torch.zeros((1, 0, self.plm_embed_size), device=self.device))
        self.prev_action_value = torch.zeros((1, 1, 1), dtype=torch.float32, device=self.device)
        self.prev_reward_value = torch.zeros((1, 1, 1), dtype=torch.float32, device=self.device)

    def _sample(self, logits):
        pi = F.softmax(logits, 0).cpu().numpy()
        idx = random.choices(np.arange(pi.size), pi)[0]
        lgprob = np.log(pi[idx])
        return idx, lgprob

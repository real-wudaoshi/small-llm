import torch
import torch.nn as nn
import config

class Rope(nn.Module):
    def __init__(self, dim, base=10000, max_seq_len=256):
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_seq_len = max_seq_len
    
    def forward(self, x, position_offset=0):
        # x: [B, heads, seq_len, dim]
        seq_len = x.shape[2]
        device = x.device
        base = self.base
        if seq_len + position_offset > self.max_seq_len:
            scale = (seq_len + position_offset) / self.max_seq_len
            base = self.base * (scale ** (self.dim / (self.dim - 2)))
        theta = torch.pow(base, -torch.arange(0, self.dim, 2, device=device) / self.dim)
        position = torch.arange(position_offset, position_offset + seq_len, device=device)
        theta = torch.einsum('l, d -> l d', position, theta)
        cos = torch.cos(theta)
        sin = torch.sin(theta)
        x1 = x[..., 0:self.dim // 2]
        x2 = x[..., self.dim // 2:]
        x_even = x1 * cos - x2 * sin
        x_odd = x1 * sin + x2 * cos
        return torch.cat([x_even, x_odd], dim=-1)

class AttentionWithRope(nn.Module):
    def __init__(self, dim, heads, use_rope=True, rope_base=10000):
        super().__init__()
        if dim % heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by heads ({heads})")
        self.dim = dim
        self.head_dim = dim // heads
        self.heads = heads
        self.use_rope = use_rope
        if use_rope:
            self.rope = Rope(self.head_dim, rope_base, max_seq_len=config.WINDOW_SIZE)
        else:
            self.rope = None
        self.wq = nn.Linear(dim, dim)
        self.wk = nn.Linear(dim, dim)
        self.wv = nn.Linear(dim, dim)
        self.wo = nn.Linear(dim, dim)
    
    def forward_with_cache(self, x, attn_mask, kv_cache):
        # x: [B, seq_len, dim]
        B, seq_len = x.shape[:2]

        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)
        
        q = q.reshape(B, seq_len, self.heads, self.head_dim).transpose(1, 2)
        k = k.reshape(B, seq_len, self.heads, self.head_dim).transpose(1, 2)
        v = v.reshape(B, seq_len, self.heads, self.head_dim).transpose(1, 2)

        past_len = 0
        if kv_cache is not None:
            past_k = kv_cache["k"]
            past_v = kv_cache["v"]
            past_mask = kv_cache["attn_mask"]
            past_len = int(kv_cache.get("seen_tokens", past_k.shape[2]))
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
            attn_mask = torch.cat([past_mask, attn_mask], dim=-1)
        if k.shape[2] > config.INFERENCE_WINDOW_SIZE:
            k = k[:, :, -config.INFERENCE_WINDOW_SIZE:, :]
            v = v[:, :, -config.INFERENCE_WINDOW_SIZE:, :]
            attn_mask = attn_mask[:, -config.INFERENCE_WINDOW_SIZE:]
        seen_tokens = past_len + seq_len
        new_kv = {"k": k, "v": v, "attn_mask": attn_mask, "seen_tokens": seen_tokens}
        
        k_len = k.shape[2]

        attn_mask = attn_mask.unsqueeze(1).unsqueeze(2).to(torch.bool)
        query_pos = torch.arange(seen_tokens - seq_len, seen_tokens, device=q.device).unsqueeze(1)
        key_pos = torch.arange(seen_tokens - k_len, seen_tokens, device=q.device).unsqueeze(0)
        causal_mask = (key_pos <= query_pos).unsqueeze(0).unsqueeze(0)
        attn_mask = attn_mask & causal_mask

        if self.use_rope:
            q = self.rope(q, position_offset=seen_tokens - seq_len)
            k = self.rope(k, position_offset=seen_tokens - k_len)

        attn = nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, is_causal=False
        )
        attn = attn.transpose(1, 2).reshape(B, seq_len, self.dim)
        attn = self.wo(attn)

        return attn, new_kv

    def forward(self, x, attn_mask=None):
        B, seq_len = x.shape[:2]

        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)
        q = q.reshape(B, seq_len, self.heads, self.head_dim).transpose(1, 2)
        k = k.reshape(B, seq_len, self.heads, self.head_dim).transpose(1, 2)
        v = v.reshape(B, seq_len, self.heads, self.head_dim).transpose(1, 2)
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).unsqueeze(2).to(torch.bool)
        else:
            attn_mask = None
        
        if self.use_rope:
            q = self.rope(q)
            k = self.rope(k)

        attn = nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, is_causal=True
        )
        attn = attn.transpose(1, 2).reshape(B, seq_len, self.dim)
        attn = self.wo(attn)
        return attn

class SiGLU(nn.Module):
    def __init__(self, dim, ffn_dim):
        super().__init__()
        self.f1 = nn.Linear(dim, ffn_dim)
        self.f2 = nn.Linear(dim, ffn_dim)
        self.f3 = nn.Linear(ffn_dim, dim)
        self.activator = nn.SiLU()
    def forward(self, x):
        return self.f3(self.activator(self.f1(x)) * self.f2(x))

class TransformerLayer(nn.Module):
    def __init__(self, dim, ffn_dim, heads, use_rope=True, rope_base=10000, dropout=0.00):
        super().__init__()
        self.attn = AttentionWithRope(dim, heads, use_rope, rope_base)
        self.ln1 = nn.RMSNorm(dim)
        self.ln2 = nn.RMSNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.ffn = SiGLU(dim, ffn_dim)
        self.activator = nn.SiLU()

    def forward_with_cache(self, x, attention_mask, kv_cache):
        # x: [B, seq_len, dim]
        residual = x
        attn_out, new_kv = self.attn.forward_with_cache(
            self.ln1(x), attention_mask, kv_cache
        )
        x = residual + self.dropout(attn_out)

        residual = x
        x = self.ffn(self.ln2(x))
        x = residual + self.dropout(x)
        return x, new_kv

    def forward(self, x, attn_mask=None):
        residual = x
        x = self.attn(self.ln1(x), attn_mask)
        x = residual + self.dropout(x)
        residual = x
        x = self.ffn(self.ln2(x))
        x = residual + self.dropout(x)
        return x

class ConvLayer(nn.Module):
    def __init__(self, dim, ffn_dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, ffn_dim, kernel_size=3, padding=1)
        self.activator = nn.SiLU()
    def forward(self, x):
        return self.activator(self.conv(x))

class GateLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Linear(dim, dim)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x, y):
        coef = self.sigmoid(self.gate(x))
        return x * coef + y * (1 - coef)

class SmallLlm(nn.Module):
    def __init__(self, max_token, dim, ffn_dim, heads, layers):
        super().__init__()
        self.layers = layers
        self.embed = nn.Embedding(max_token, dim, padding_idx=config.PAD_ID)
        self.transformer_layers = nn.ModuleList([
            TransformerLayer(dim, ffn_dim, heads, True, 10000) for _ in range(layers)
        ])
        # self.gate = GateLayer(dim)
        self.out = nn.Linear(dim, max_token)
    
    def forward(self, x, attention_mask=None):
        x = self.embed(x)
        x = self.transformer_layers[0](x, attention_mask)
        y = x
        for i in range(1, self.layers - 1):
            x = self.transformer_layers[i](x, attention_mask)
        x = self.transformer_layers[-1](x, attention_mask)
        # x = self.gate(x, y)
        x = self.out(x)
        return x
    
    def forward_with_kv_cache(self, x, attention_mask, kv_cache):
        # x: [B, seq_len]
        # kv_cache: [layer_idx] -> {"k": [B, heads, seq_len, head_dim], "v": [B, heads, seq_len, head_dim], "seen_tokens": int}
        if kv_cache is None:
            kv_cache = [None] * self.layers
        x = self.embed(x)

        x, kv_cache[0] = self.transformer_layers[0].forward_with_cache(
            x, attention_mask=attention_mask, kv_cache=kv_cache[0]
        )
        y = x
        for i in range(1, self.layers - 1):
            x, kv_cache[i] = self.transformer_layers[i].forward_with_cache(
                x, attention_mask=attention_mask, kv_cache=kv_cache[i]
            )
        x, kv_cache[-1] = self.transformer_layers[-1].forward_with_cache(
            x, attention_mask=attention_mask, kv_cache=kv_cache[-1]
        )
        # x = self.gate(x, y)
        x = self.out(x)
        return x, kv_cache
    
    def sample_next_id(self, logits, temperature, top_p, top_k, penalty_cache):
        probs = torch.softmax(logits.reshape(-1) / temperature, dim=-1) / penalty_cache.reshape(-1)
        top_k = max(1, min(int(top_k), probs.shape[0]))
        top_p = float(min(max(top_p, 1e-8), 1.0))

        topk_vals, topk_idxs = torch.topk(probs, k=top_k)
        sorted_probs, order = torch.sort(topk_vals, descending=True, dim=0)
        sorted_token_ids = topk_idxs[order]

        cumulative_probs = torch.cumsum(sorted_probs, dim=0)
        mask = cumulative_probs <= top_p
        mask[0] = True

        candidate_probs = sorted_probs[mask]
        candidate_token_ids = sorted_token_ids[mask]
        candidate_probs = candidate_probs / candidate_probs.sum()

        sampled_pos = torch.multinomial(candidate_probs, num_samples=1).item()
        return int(candidate_token_ids[sampled_pos].item())


    def generate(self, x, temperature=0.5, top_p=0.3, top_k=10, max_tokens=20, repetition_penalty=1.5):
        x = torch.as_tensor(x, dtype=torch.long, device=self.embed.weight.device)
        penalty_cache = torch.ones((self.embed.weight.shape[0]), device=self.embed.weight.device)
        kv_cache = [None] * self.layers
        result = []
        for _ in range(max_tokens):
            out, kv_cache = self.forward_with_kv_cache(
                x.unsqueeze(0), torch.ones((1, x.shape[0]), dtype=torch.bool, device=x.device), kv_cache
            )
            out = nn.functional.log_softmax(out, dim=-1)
            logits = out[:, -1, :].squeeze(0)
            next_id = self.sample_next_id(logits, temperature, top_p, top_k, penalty_cache)
            result.append(next_id)
            penalty_cache[next_id] = (penalty_cache[next_id] * repetition_penalty + 1) / 2
            x = torch.tensor([next_id], dtype=torch.long, device=x.device)
            if next_id == config.EOS_ID:
                break
        return result
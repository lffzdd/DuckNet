import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# 模型定义
class GPT(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers, num_heads, seq_length, dropout=0.1):
        super(GPT, self).__init__()
        self.hidden_size = hidden_size
        self.seq_length = seq_length
        self.num_heads = num_heads
        self.num_layers = num_layers

        # 词嵌入和位置嵌入
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.position_embedding = nn.Embedding(seq_length, hidden_size)

        # Transformer块
        self.layers = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, hidden_size * 4, dropout)
            for _ in range(num_layers)
        ])

        # 输出层
        self.ln_f = nn.LayerNorm(hidden_size)
        self.head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, x):
        b, t = x.size()

        # 创建位置张量
        position_ids = torch.arange(t, dtype=torch.long, device=x.device)
        position_ids = position_ids.unsqueeze(0).expand_as(x)  # 扩展以匹配输入的形状

        # 嵌入阶段
        token_embeddings = self.token_embedding(x)
        position_embeddings = self.position_embedding(position_ids)
        h = token_embeddings + position_embeddings

        # 通过所有Transformer层
        for layer in self.layers:
            h = layer(h)

        # LayerNorm和输出层
        h = self.ln_f(h)
        logits = self.head(h)

        return logits


class TransformerBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, feedforward_size, dropout):
        super(TransformerBlock, self).__init__()
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout)
        self.ln_1 = nn.LayerNorm(hidden_size)
        self.ff = nn.Sequential(
            nn.Linear(hidden_size, feedforward_size),
            nn.ReLU(),
            nn.Linear(feedforward_size, hidden_size),
            nn.Dropout(dropout),
        )
        self.ln_2 = nn.LayerNorm(hidden_size)

    def forward(self, x):
        # Self-attention
        attn_output, _ = self.attn(x, x, x)
        x = self.ln_1(x + attn_output)

        # Feedforward
        ff_output = self.ff(x)
        x = self.ln_2(x + ff_output)

        return x


# 模型使用
# 超参数
vocab_size = 50257  # GPT-2的词汇表大小
hidden_size = 768
num_layers = 12
num_heads = 12
seq_length = 128

# 创建模型
model = GPT(vocab_size, hidden_size, num_layers, num_heads, seq_length)


# 创建输入数据（随机生成）
input_ids = torch.randint(0, vocab_size, (1, seq_length))

# 前向传播
logits = model(input_ids)

# 输出
print(logits.shape)  # (1, seq_length, vocab_size)

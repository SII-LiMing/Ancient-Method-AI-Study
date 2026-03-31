import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        """
        初始化多头注意力机制
        :param d_model: 输入向量的维度 (例如 512)
        :param num_heads: 头的数量 (例如 8)
        """
        super(MultiHeadAttention, self).__init__()
        
        # 确保 d_model 可以被 num_heads 整除
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # 每个头的维度 (例如 512 / 8 = 64)
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        
        self.W_o = nn.Linear(d_model, d_model)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """
        计算缩放点积注意力
        """
        # Q 和 K 的点积，再除以 sqrt(d_k)
        # Q shape: (batch_size, num_heads, seq_len, d_k)
        # K.transpose shape: (batch_size, num_heads, d_k, seq_len)
        # scores shape: (batch_size, num_heads, seq_len, seq_len)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # 如果有 mask (例如在 Decoder 中防止看到未来的词，或者 padding mask)
        if mask is not None:
            # 将 mask 中为 0 的位置替换为一个极小的数 (如 -1e9)，这样 softmax 后权重趋近于 0
            scores = scores.masked_fill(mask == 0, -1e9)
            
        # 计算注意力权重
        attention_weights = torch.softmax(scores, dim=-1)
        
        # 权重与 V 相乘
        # output shape: (batch_size, num_heads, seq_len, d_k)
        output = torch.matmul(attention_weights, V)
        
        return output, attention_weights

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # 1. 线性映射并划分成 num_heads 个头
        # view 的操作将 d_model 拆分成 (num_heads, d_k)
        # transpose(1, 2) 是为了把 num_heads 移到前面，方便后续的矩阵乘法
        # 最终 shape: (batch_size, num_heads, seq_len, d_k)
        Q = self.W_q(query).reshape(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).reshape(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).reshape(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # 2. 计算 Scaled Dot-Product Attention
        # x shape: (batch_size, num_heads, seq_len, d_k)
        x, attn_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # 3. 拼接所有的头 (Concat)
        # 把 num_heads 移回原来的位置，并将 num_heads 和 d_k 重新合并成 d_model
        # contiguous() 是因为 transpose 破坏了内存连续性，view 需要连续的内存
        # shape 变化: (batch, heads, seq, d_k) -> (batch, seq, heads, d_k) -> (batch, seq, d_model)
        x = x.transpose(1, 2).reshape(batch_size, -1, self.d_model)
        
        # 4. 最后的线性映射层
        # shape: (batch_size, seq_len, d_model)
        output = self.W_o(x)
        
        return output
    

if __name__ == "__main__":
    # --- 超参数设置 ---
    batch_size = 2      # 一次处理 2 个句子
    seq_len = 5         # 每个句子有 5 个词 (Token)
    d_model = 512       # 词向量的维度是 512
    num_heads = 8       # 拆分成 8 个头
    
    # 打印一些提示信息
    print(f"初始化参数: batch_size={batch_size}, seq_len={seq_len}, d_model={d_model}, num_heads={num_heads}")
    
    # --- 1. 构造输入数据 ---
    # 在自注意力机制中，Q, K, V 通常来自于同一个输入 X
    # X 的 shape 是 (batch_size, seq_len, d_model)
    X = torch.randn(batch_size, seq_len, d_model)
    print(f"输入张量 X 的形状: {X.shape}")
    
    # --- 2. 实例化 MHA 模块 ---
    mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
    
    # --- 3. 前向传播计算 ---
    # 将 X 作为 Q, K, V 同时传入，这就是 "Self-Attention (自注意力)" 的名字由来
    output = mha(query=X, key=X, value=X, mask=None)
    
    # --- 4. 检查输出 ---
    print(f"输出张量的形状: {output.shape}")
    
    # 验证输入和输出的形状是否一致 (MHA 保持序列长度和维度不变)
    assert output.shape == X.shape, "错误：输出形状与输入形状不一致！"
    print("测试通过！输入和输出的形状完全一致。")
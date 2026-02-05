import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module): #多頭注意力機制
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        # 確保維度可以被頭數整除
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads # 每個頭的維度
        
        # 定義 W_q, W_k, W_v 的線性層
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        
        # 最後輸出的線性層  
        self.w_o = nn.Linear(d_model, d_model)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None): #縮放點積注意力機制
        # 1. 計算 Q * K^T
        # K 的形狀轉換為 [batch, heads, d_k, seq_len] 以便矩陣相乘
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # 2. 如果有 mask (例如在 Decoder 中)，將不需要關注的地方設為負無窮大
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        
        # 3. Softmax 歸一化取得機率分佈
        attn_probs = torch.softmax(attn_scores, dim=-1)
        
        # 4. 乘上 V
        output = torch.matmul(attn_probs, V)
        return output, attn_probs

    def forward(self, q, k, v, mask=None): #前向傳播
        batch_size = q.size(0)
        
        # 1. 線性投影並分頭 (Split Heads)
        # 變換形狀: [batch, seq_len, d_model] -> [batch, seq_len, heads, d_k] -> [batch, heads, seq_len, d_k]
        Q = self.w_q(q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(k).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(v).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # 2. 進入核心注意力機制
        output, self.attn_probs = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # 3. 串接所有頭 (Concatenate)
        # 變換形狀: [batch, heads, seq_len, d_k] -> [batch, seq_len, heads, d_k] -> [batch, seq_len, d_model]
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        # 4. 最後通過線性層
        return self.w_o(output)

# --- 測試區塊 (驗收) ---
if __name__ == "__main__":
    # 設定參數
    d_model = 512  # 論文中的標準維度
    num_heads = 8  # 論文中的頭數
    seq_len = 10   # 假設一句話有 10 個字
    batch_size = 2 # 一次處理 2 句話

    # 建立模型
    mha = MultiHeadAttention(d_model, num_heads)
    
    # 模擬輸入資料 (隨機生成)
    x = torch.randn(batch_size, seq_len, d_model)
    
    # 執行模型 (Self-Attention: Q=K=V=x)
    output = mha(x, x, x)
    
    print("輸入形狀:", x.shape)
    print("輸出形狀:", output.shape)
    
    if x.shape == output.shape:
        print(">>> 恭喜！Multi-Head Attention 本地部署並執行成功！")
    else:
        print(">>> 發生錯誤，維度不匹配。")
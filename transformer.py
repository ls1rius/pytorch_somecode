# %%
# code by Tae Hwan Jung(Jeff Jung) @graykode, Derek Miller @dmmiller612
# Reference : https://github.com/jadore801120/attention-is-all-you-need-pytorch
#           https://github.com/JayParks/transformer
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
# S: Symbol that shows starting of decoding input
# E: Symbol that shows starting of decoding output
# P: Symbol that will fill in blank sequence if current batch data size is short than time steps


class Config(object):

    """配置参数"""
    def __init__(self):
        self.train_path = 'data/ag_news/ag_news.train'
        self.val_path = None
        self.test_path = 'data/ag_news/ag_news.test'
        self.model_name = 'transformer'
        self.save_path =  'save_models/' + self.model_name + '.ckpt'        # 模型训练结果
        self.log_path = "save_models/log/" + self.model_name
        self.save_train_metrics_file = 'save_models/Transformer_train_log.csv'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备
        self.dropout = 0.5                                              # 随机失活
        self.require_improvement = 1000                                 # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = 4                                          # 类别数
        self.class_list = ['world','sports','business','technology']
        self.num_epochs = 20                                            # epoch数
        self.batch_size = 64                                           # mini-batch大小
        self.max_len = 128                                              # 每句话处理成的长度(短填长切)
        self.learning_rate = 5e-4                                       # 学习率
        self.d_model = 256
        self.d_head = 128
        self.d_ff = 256
        #self.hidden = 512
        #self.last_hidden = 512
        self.n_heads = 6
        self.n_encoder = 6
        self.n_vocab = 30522                                               # 词表大小，在运行时赋值

"""
def get_sinusoid_encoding_table(n_position, d_model):
    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_model)
    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_model)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
    return torch.FloatTensor(sinusoid_table)
"""

def get_attn_pad_mask(seq_q, seq_k,pad_idx = 0):
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    # eq(zero) is PAD token
    pad_attn_mask = seq_k.data.eq(pad_idx).unsqueeze(1)  # batch_size x 1 x len_k(=len_q), one is masking
    return pad_attn_mask.expand(batch_size, len_q, len_k)  # batch_size x len_q x len_k


class ScaledDotProductAttention(nn.Module):
    def __init__(self,d_model):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_model
    def forward(self, Q, K, V, attn_mask):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k) # scores : [batch_size x n_heads x len_q(=len_k) x len_k(=len_q)]
        scores.masked_fill_(attn_mask, -1e9) # Fills elements of self tensor with value where mask is one.
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context, attn

class MultiHeadAttention(nn.Module):
    def __init__(self,d_model,d_head,n_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.d_head = d_head
        self.n_heads = n_heads
        #d_k,d_v = d_head,d_head
        self.W_Q = nn.Linear(d_model, self.d_head * n_heads) #d_k
        self.W_K = nn.Linear(d_model, self.d_head * n_heads) #d_k
        self.W_V = nn.Linear(d_model, self.d_head * n_heads) #d_v
        self.linear = nn.Linear(n_heads * self.d_head, d_model) #d_v
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, Q, K, V, attn_mask):
        # q: [batch_size x len_q x d_model], k: [batch_size x len_k x d_model], v: [batch_size x len_k x d_model]
        residual, batch_size = Q, Q.size(0)
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        q_s = self.W_Q(Q).view(batch_size, -1, self.n_heads, self.d_head).transpose(1,2)  # q_s: [batch_size x n_heads x len_q x d_k]
        k_s = self.W_K(K).view(batch_size, -1, self.n_heads, self.d_head).transpose(1,2)  # k_s: [batch_size x n_heads x len_k x d_k]
        v_s = self.W_V(V).view(batch_size, -1, self.n_heads, self.d_head).transpose(1,2)  # v_s: [batch_size x n_heads x len_k x d_v]

        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1) # attn_mask : [batch_size x n_heads x len_q x len_k]

        # context: [batch_size x n_heads x len_q x d_v], attn: [batch_size x n_heads x len_q(=len_k) x len_k(=len_q)]
        context, attn = ScaledDotProductAttention(self.d_model)(q_s, k_s, v_s, attn_mask)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_head) # context: [batch_size x len_q x n_heads * d_v]
        output = self.linear(context)
        return self.layer_norm(output + residual), attn # output: [batch_size x len_q x d_model]

class PoswiseFeedForwardNet(nn.Module):
    def __init__(self,d_model,d_ff):
        super(PoswiseFeedForwardNet, self).__init__()
        """
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        """
        self.fc1 = nn.Linear(in_features=d_model,out_features=d_ff)
        self.fc2 = nn.Linear(in_features=d_ff,out_features=d_model)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, inputs):
        residual = inputs # inputs : [batch_size, len_q, d_model]
        hidden = nn.ReLU()(self.fc1(inputs))
        output = self.fc2(hidden)
        #output = nn.ReLU()(self.conv1(inputs.transpose(1, 2)))
        #output = self.conv2(output).transpose(1, 2)
        return self.layer_norm(output + residual)

class EncoderLayer(nn.Module):
    def __init__(self,config):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention(config.d_model,config.d_head,config.n_heads) #d_model,d_head,n_heads
        self.pos_ffn = PoswiseFeedForwardNet(config.d_model,config.d_ff)

    def forward(self, enc_inputs, enc_self_attn_mask):
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask) # enc_inputs to same Q,K,V
        enc_outputs = self.pos_ffn(enc_outputs) # enc_outputs: [batch_size x len_q x d_model]
        return enc_outputs, attn

class Positional_Encoding(nn.Module):
    def __init__(self, embed, pad_size, dropout, device):
        super(Positional_Encoding, self).__init__()
        self.device = device
        self.pe = torch.tensor([[pos / (10000.0 ** (i // 2 * 2.0 / embed)) for i in range(embed)] for pos in range(pad_size)])
        self.pe[:, 0::2] = np.sin(self.pe[:, 0::2])
        self.pe[:, 1::2] = np.cos(self.pe[:, 1::2])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = x + nn.Parameter(self.pe, requires_grad=False).to(self.device)
        out = self.dropout(out)
        return out

class Encoder(nn.Module):
    def __init__(self,config):
        super(Encoder, self).__init__()
        self.src_emb = nn.Embedding(config.n_vocab, config.d_model)
        self.postion_embedding = Positional_Encoding(config.d_model, config.max_len, config.dropout, config.device)
        #self.pos_emb = nn.Embedding.from_pretrained(get_sinusoid_encoding_table(config.max_len+1, config.d_model),freeze=True)
        self.layers = nn.ModuleList([EncoderLayer(config) for _ in range(config.n_encoder)])

    def forward(self, enc_inputs): # enc_inputs : [batch_size x source_len]
        #pos = torch.arange(max_len, dtype=torch.long, device=device)
        enc_outputs = self.src_emb(enc_inputs) #torch.LongTensor([[1,2,3,4,0]]
        enc_outputs = self.postion_embedding(enc_outputs)
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs)
        enc_self_attns = []
        for layer in self.layers:
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)
        return enc_outputs, enc_self_attns


class Transformer(nn.Module):
    def __init__(self,config):
        super(Transformer, self).__init__()
        self.max_len = config.max_len
        self.device  = config.device
        self.encoder = Encoder(config)
        self.projection = nn.Linear(config.d_model*config.max_len, config.num_classes, bias=False)
    def forward(self, enc_inputs):
        enc_outputs, enc_self_attns = self.encoder(enc_inputs)
        enc_outputs = enc_outputs.view(enc_outputs.size(0),-1)
        enc_logits = self.projection(enc_outputs) # dec_logits : [batch_size x src_vocab_size x tgt_vocab_size]
        return None,enc_logits #, enc_self_attns, dec_self_attns, dec_enc_attns

if __name__ == '__main__':
    config = Config()
    config.n_vocab = 100
    model = Transformer(config)
    model = model.to(config.device)
    import numpy as np
    x = torch.LongTensor(np.random.choice(100,(20,config.max_len)))
    x = x.to(config.device)
    out = model(x)
    print(out)


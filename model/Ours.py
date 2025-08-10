import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, attention_dropout_rate, num_heads, flow, remark=False):
        super(MultiHeadAttention, self).__init__()

        self.hidden_size = hidden_size
        self.attention_dropout_rate = attention_dropout_rate
        self.num_heads = num_heads
        self.remark = remark

        self.att_size = att_size = hidden_size // num_heads
        self.scale = att_size ** -0.5

        self.linear_q = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_k = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_v = nn.Linear(hidden_size, num_heads * att_size)
        self.att_dropout = nn.Dropout(attention_dropout_rate)

        self.output_layer = nn.Linear(num_heads * att_size, hidden_size)

        self.out_att = nn.Linear(flow * 2, flow)

        self.self_attention_norm = nn.LayerNorm(hidden_size)

    def forward(
            self,
            q,
            k,
            v,
            entropy,  # [b,n]
            adp,  # [n,n]
            mask=None,  # [b,n]
    ):
        orig_q_size = q.size()  # [b, n, d_model]

        d_k = self.att_size
        d_v = self.att_size
        batch_size = q.size(0)

        q = self.linear_q(q).view(batch_size, -1, self.num_heads, d_k)
        k = self.linear_k(k).view(batch_size, -1, self.num_heads, d_k)
        v = self.linear_v(v).view(batch_size, -1, self.num_heads, d_v)

        q = q.transpose(1, 2)
        v = v.transpose(1, 2)
        k = k.transpose(1, 2)

        x = torch.matmul(q, k.transpose(2, 3))
        x = x * self.scale
        adp = adp.unsqueeze(0).expand(batch_size, adp.shape[1], adp.shape[-1])
        entropy = entropy.unsqueeze(-1)
        deletion = adp * entropy
        deletion = deletion.unsqueeze(1).expand(batch_size, x.shape[1], x.shape[-1], x.shape[-1])

        x = torch.softmax(x, dim=3)
        deletion = torch.softmax(deletion, dim=3)

        x = torch.cat([x, deletion], dim=-1)
        x = self.out_att(x)

        x = torch.softmax(x, dim=3)
        x = self.att_dropout(x)

        x = torch.matmul(x, v)
        x = x.transpose(1, 2).contiguous()
        x = x.view(batch_size, -1, self.num_heads * d_v)

        x = self.output_layer(x)
        assert x.size() == orig_q_size
        return x

class Ours(nn.Module):

    def __init__(
            self,
            num_layer=1,
            d_model=512,
            nhead=8,
            dim_feedforward=2048,
            dropout=0.1,
            attention_dropout=0.1,
            in_seq=24,
            residual_channels=128,
            patch_size=4,
            patch_size_1=3,
            patch_size_2=6,
            patch_size_3=12,
            device="cuda:1",
            hide_dim=512,
            flow=144,
            my_adp=[],
            remark=False
    ):
        super().__init__()

        self.layers = nn.ModuleList(
            [
                EncoderLayer(
                    hidden_size=d_model,
                    ffn_size=dim_feedforward,
                    dropout_rate=dropout,
                    attention_dropout_rate=attention_dropout,
                    num_heads=nhead,
                    flow=flow,
                )
                for _ in range(num_layer)
            ]
        )
        self.final_ln = nn.LayerNorm(d_model)

        self.remark = remark
        self.in_seq = in_seq

        self.patch_size = patch_size
        self.patch_size_1 = patch_size_1
        self.patch_size_2 = patch_size_2
        self.patch_size_3 = patch_size_3
        self.device = device
        self.patch_num = int(in_seq / patch_size)
        self.patch_num_1 = int(in_seq / patch_size_1)
        self.patch_num_2 = int(in_seq / patch_size_2)
        self.patch_num_3 = int(in_seq / patch_size_3)
        self.patch_emb = nn.Linear(self.patch_size, d_model)
        self.last_mlp = nn.Linear(d_model, self.patch_size)
        self.dropout = nn.Dropout(dropout)
        self.my_adp = my_adp

        self.patch_lstm_multiscale = Multiscale(in_seq, hide_dim, residual_channels, self.patch_size_1,
                                                self.patch_size_2, self.patch_size_3, self.patch_num_1, self.patch_num_2,
                                                self.patch_num_3)

    def forward(self, data, mask):
        x = data
        means = torch.sum(x, dim=1) / mask.size(1)
        means = means.unsqueeze(1).detach()
        x = x - means
        stdev = torch.sqrt(torch.sum(x * x, dim=1) /
                           mask.size(1))
        stdev[stdev == 0] = 1e-5
        stdev = stdev.unsqueeze(1).detach()
        x /= stdev
        x_tr = x.transpose(1, 2)
        mask_tr = mask.transpose(1, 2)
        multiscale = self.patch_lstm_multiscale(x_tr, mask_tr)

        entropy = self.computational_entropy_matrix(mask, self.patch_num, self.patch_size)
        x_hat = torch.empty([x_tr.size(0), x_tr.size(1), 0])
        x_hat = x_hat.to(self.device)
        for i in range(self.patch_num):
            patch = multiscale[:, :, i * self.patch_size:i * self.patch_size + self.patch_size]
            patch = self.patch_emb(patch)
            for k, enc_layer in enumerate(self.layers):
                patch = enc_layer(patch, entropy=entropy[i, :, :], adp=self.my_adp[0])
            output = self.final_ln(patch)
            output = self.last_mlp(output)
            x_hat = torch.cat((x_hat, output), 2)

        x_hat = x_hat.transpose(1, 2)

        x_hat = x_hat * \
                (stdev[:, 0, :].unsqueeze(1).repeat(1, self.in_seq, 1))
        x_hat = x_hat + \
                (means[:, 0, :].unsqueeze(1).repeat(1, self.in_seq, 1))

        x_res = x_hat * mask + data * (1 - mask)
        return x_res

    def computational_entropy_matrix(self, mask, patch_num, patch_size):
        mask = 1 - mask
        entropy_list = []
        for i in range(patch_num):
            mask_patch = mask[:, i * patch_size:i * patch_size + patch_size, :]
            real_data = mask_patch.sum(dim=1)
            real_data_r = real_data / patch_size
            p = real_data_r
            entropy = -p * torch.log2(1 - p / 2)
            entropy_list = entropy_list + [entropy]
        entropy = torch.stack(entropy_list, dim=0)
        return entropy


class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate):
        super(FeedForwardNetwork, self).__init__()

        self.layer1 = nn.Linear(hidden_size, ffn_size)
        self.gelu = nn.GELU()
        self.layer2 = nn.Linear(ffn_size, hidden_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.gelu(x)
        x = self.layer2(x)
        return x

class EncoderLayer(nn.Module):
    def __init__(
            self, hidden_size, ffn_size, dropout_rate, attention_dropout_rate, num_heads, flow
    ):
        super(EncoderLayer, self).__init__()

        self.self_attention_norm = nn.LayerNorm(hidden_size)
        self.self_attention = MultiHeadAttention(
            hidden_size,
            attention_dropout_rate,
            num_heads,
            flow,
        )
        self.self_attention_dropout = nn.Dropout(dropout_rate)

        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = FeedForwardNetwork(hidden_size, ffn_size, dropout_rate)
        self.ffn_dropout = nn.Dropout(dropout_rate)

    def forward(self, x, entropy, adp, mask=None):
        y = self.self_attention_norm(x)
        y = self.self_attention(y, y, y, entropy=entropy, adp=adp, mask=mask)
        y = self.self_attention_dropout(y)
        x = x + y

        y = self.ffn_norm(x)
        y = self.ffn(y)
        y = self.ffn_dropout(y)
        x = x + y
        return x




import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class RBF(nn.Module):
    def __init__(self, centers, gamma):
        super(RBF, self).__init__()
        self.centers = torch.tensor(centers, dtype=torch.float32)
        self.gamma = gamma

    def forward(self, x):
        x = x.unsqueeze(-1)
        return torch.exp(-self.gamma * torch.square(x - self.centers.to(x.device)))


class ReRBF(nn.Module):
    def __init__(self, centers, gamma):
        super(ReRBF, self).__init__()
        self.centers = torch.tensor(centers, dtype=torch.float32)
        self.gamma = gamma

    def forward(self, x):
        x = torch.sqrt(torch.log(x) / (-self.gamma)) + self.centers.to(x.device)
        return x


class DistEncoder(nn.Module):
    def __init__(self, emb_dim, add_time=False):
        super(DistEncoder, self).__init__()
        self.rbf_param = {'bond_length': (np.arange(0, 10, 0.2), 10.0)}
        center = np.arange(0, 10, 0.2)
        gamma = 10.0
        self.rbf = RBF(center, gamma)
        self.linear = nn.Linear(len(center), emb_dim) if not add_time else nn.Linear(len(center) + 1, emb_dim)

    def forward(self, x, time=None):
        rbf_x = self.rbf(x)
        out = self.linear(rbf_x) if time is None else self.linear(torch.cat([rbf_x, time], dim=-1))
        return out


class DistDecoder(nn.Module):
    def __init__(self, emb_dim):
        super(DistDecoder, self).__init__()
        self.rbf_param = {'bond_length': (np.arange(0, 10, 0.2), 10.0)}
        center = np.arange(0, 10, 0.2)
        gamma = 10.0
        self.re_rbf = ReRBF(center, gamma)
        self.linear = nn.Linear(emb_dim, len(center))
        self.act = nn.Softmax(dim=3)

    def forward(self, edge_emb):
        edge_emb = self.act(self.linear(edge_emb))
        re_rbf = self.re_rbf(edge_emb)
        return re_rbf


class NodeTransformerLayer(nn.Module):
    def __init__(self, emb_dim, ffn_dim, num_heads, dropout, act_dropout=None, attn_dropout=None):
        super(NodeTransformerLayer, self).__init__()
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.head_dim = int(self.emb_dim / self.num_heads)
        self.dropout = dropout
        self.act_dropout = act_dropout
        self.attn_dropout = attn_dropout
        self.ffn_emb_dim = 2 * emb_dim if ffn_dim is None else ffn_dim

        self.node_ln = nn.LayerNorm(self.emb_dim)
        self.pair_ln = nn.LayerNorm(self.emb_dim)

        self.q_proj = nn.Linear(self.emb_dim, self.emb_dim)
        self.k_proj = nn.Linear(self.emb_dim, self.emb_dim)
        self.v_proj = nn.Linear(self.emb_dim, self.emb_dim)

        self.k_e_proj = nn.Linear(self.emb_dim, self.emb_dim)
        self.v_e_proj = nn.Linear(self.emb_dim, self.emb_dim)
        self.v_add = nn.Linear(self.head_dim, self.head_dim)

        self.dropout = nn.Dropout(dropout)

    def attention(self, atom_emb, pair_emb, pair_mask, batch_size, n_nodes):
        q = self.q_proj(atom_emb).reshape(batch_size, n_nodes, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        q *= (1 / self.head_dim ** 0.5)
        k_n = self.k_proj(atom_emb).reshape(batch_size, n_nodes, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k_e = self.k_e_proj(pair_emb).reshape(batch_size, n_nodes, n_nodes, self.num_heads,
                                              self.head_dim).permute(0, 3, 1, 2, 4)

        q = q.unsqueeze(3)
        k_n = k_n.unsqueeze(2)
        k = k_n + k_e

        attn_weight = torch.matmul(q, k.permute([0, 1, 2, 4, 3])).reshape(batch_size, self.num_heads, n_nodes, n_nodes)
        attn_weight += (1 - pair_mask.permute(0, 3, 1, 2)) * (-1e6)
        attn_prob = nn.functional.softmax(attn_weight, dim=3).unsqueeze(3)
        return attn_prob

    def attn_update(self, atom_emb, pair_emb, attn_prob, node_mask, batch_size, n_nodes):
        v = self.v_proj(atom_emb).reshape(batch_size, n_nodes, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v_n = v.unsqueeze(3)
        v_r = v.unsqueeze(2)
        v_e = self.v_e_proj(pair_emb).reshape(batch_size, n_nodes, n_nodes, self.num_heads,
                                              self.head_dim).permute(0, 3, 1, 2, 4)
        v_final = self.v_add(v_n + v_r + v_e)
        output = torch.matmul(attn_prob, v_final)
        output = output.squeeze(-2)
        output = output.transpose(1, 2).reshape(batch_size, n_nodes, self.emb_dim) * node_mask
        return output

    def forward(self, atom_emb, pair_emb, node_mask, pair_mask, batch_size, n_nodes):
        atom_emb = self.node_ln(atom_emb)
        pair_emb = self.pair_ln(pair_emb)

        attn_prob = self.attention(atom_emb, pair_emb, pair_mask, batch_size, n_nodes)
        attn_prob = self.dropout(attn_prob)
        output = self.attn_update(atom_emb, pair_emb, attn_prob, node_mask, batch_size, n_nodes)
        return output


class PairTransformerLayer(nn.Module):
    def __init__(self, emb_dim, ffn_dim, num_heads, dropout, act_dropout=None, attn_dropout=None):
        super(PairTransformerLayer, self).__init__()
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.head_dim = int(self.emb_dim / self.num_heads)
        self.dropout = dropout
        self.act_dropout = act_dropout
        self.attn_dropout = attn_dropout
        self.ffn_emb_dim = 2 * emb_dim if ffn_dim is None else ffn_dim

        self.ln = nn.LayerNorm(emb_dim)
        self.q_proj = nn.Linear(emb_dim, emb_dim)
        self.k_proj = nn.Linear(emb_dim, emb_dim)
        self.v_proj = nn.Linear(emb_dim, emb_dim)

        self.attn_dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(emb_dim, emb_dim)

    def forward(self, pair_emb, pair_mask, batch_size, n_nodes):
        pair_emb = self.ln(pair_emb)

        q = self.q_proj(pair_emb).reshape(batch_size, n_nodes, n_nodes, self.num_heads,
                                          self.head_dim).permute(0, 1, 3, 2, 4)
        q *= (1 / self.head_dim ** 0.5)
        q = q.unsqueeze(-2)     # (B, N, h, N, 1, d)

        k = self.k_proj(pair_emb).reshape(batch_size, n_nodes, n_nodes, self.num_heads,
                                          self.head_dim).permute(0, 1, 3, 4, 2)
        k = k.unsqueeze(-3)     # (B, N, h, 1, d, N)

        attn_prob = torch.matmul(q, k).squeeze(-2)   # (B, N, h, N, 1, N)
        attn_prob = nn.functional.softmax(attn_prob, dim=4)
        attn_prob = self.attn_dropout(attn_prob)

        v = self.v_proj(pair_emb).reshape(batch_size, n_nodes, n_nodes, self.num_heads,
                                          self.head_dim).permute(0, 1, 3, 2, 4)
        output = torch.matmul(attn_prob, v).permute(0, 1, 3, 2, 4).reshape(batch_size, n_nodes, n_nodes, self.emb_dim)
        output = self.out_proj(output) * pair_mask
        return output


class Low2High(nn.Module):
    def __init__(self, emb_dim):
        super(Low2High, self).__init__()
        self.emb_dim = emb_dim
        self.ln = nn.LayerNorm(emb_dim)
        self.linear1 = nn.Linear(emb_dim, emb_dim)
        self.linear2 = nn.Linear(emb_dim, emb_dim)
        self.lin_cat = nn.Linear(emb_dim * emb_dim, emb_dim)

    def forward(self, node_emb, node_mask):
        node_emb = self.ln(node_emb)
        left_act = (node_emb * self.linear1(node_emb)).unsqueeze(1).permute(0, 2, 3, 1)
        right_act = (node_emb * self.linear2(node_emb)).unsqueeze(1).permute(0, 1, 3, 2)
        b, c, d, r = left_act.size()
        left_act = left_act.reshape([b, c*d, r])
        right_act = right_act.reshape([b, r, c * d])
        act = torch.matmul(left_act, right_act).reshape([b, c, d, d, c]).permute(0, 1, 4, 2, 3).reshape([b, c, c, d*d])
        act = self.lin_cat(act)
        return act


class SimLow2High(nn.Module):
    def __init__(self, emb_dim):
        super(SimLow2High, self).__init__()
        self.emb_dim = emb_dim
        self.ln = nn.LayerNorm(emb_dim)
        self.linear1 = nn.Linear(emb_dim, emb_dim)
        self.linear2 = nn.Linear(emb_dim, emb_dim)
        self.lin_cat = nn.Linear(emb_dim * 2, emb_dim)

    def forward(self, node_emb, pair_mask):
        node_emb = self.ln(node_emb)
        left_act = (node_emb * self.linear1(node_emb)).unsqueeze(1) * pair_mask
        right_act = (node_emb * self.linear2(node_emb)).unsqueeze(2) * pair_mask
        act = self.lin_cat(torch.cat([left_act, right_act], dim=-1))
        return act


class TriTransLayer(nn.Module):
    def __init__(self, emb_dim, ffn_dim, num_heads, dropout, act_dropout=None, attn_dropout=None):
        super(TriTransLayer, self).__init__()
        self.emb_dim = emb_dim
        self.tri_dim = emb_dim
        self.num_heads = num_heads
        self.head_dim = int(self.emb_dim / self.num_heads)
        self.dropout = dropout
        self.act_dropout = act_dropout
        self.attn_dropout = attn_dropout
        self.ffn_emb_dim = 2 * emb_dim if ffn_dim is None else ffn_dim

        self.ln = nn.LayerNorm(emb_dim)
        self.q_proj = nn.Linear(emb_dim, emb_dim)
        self.k_e_proj = nn.Linear(emb_dim, emb_dim)
        self.k_a_proj = nn.Linear(int(emb_dim/2), emb_dim)
        self.v_proj = nn.Linear(emb_dim, emb_dim)
        self.v_a_proj = nn.Linear(int(emb_dim/2), emb_dim)

        self.attn_dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(emb_dim, emb_dim)
        self.out_gate = nn.Linear(emb_dim, emb_dim)

    def forward(self, pair_emb, tri_emb, pair_mask, batch_size, n_nodes):
        bias = (1 - pair_mask.permute(0, 3, 2, 1)) * 1e-9
        pair_emb = self.ln(pair_emb)
        q = self.q_proj(pair_emb).reshape(batch_size, n_nodes, n_nodes, self.num_heads,
                                          self.head_dim).permute(0, 1, 3, 2, 4)
        q *= (1 / self.head_dim ** 0.5)
        q = q.unsqueeze(-2)

        k_e = self.k_e_proj(pair_emb).reshape(batch_size, n_nodes, n_nodes, self.num_heads,
                                              self.head_dim).permute(0, 1, 3, 4, 2)
        k_e = k_e.unsqueeze(-3)
        k_a = self.k_a_proj(tri_emb).reshape(batch_size, n_nodes, n_nodes, n_nodes, self.num_heads,
                                             self.head_dim).permute(0, 1, 4, 2, 5, 3)
        k = k_e + k_a

        v_e = self.v_proj(pair_emb).reshape(batch_size, n_nodes, n_nodes, self.num_heads, self.head_dim)
        v_a = self.v_a_proj(tri_emb).reshape(batch_size, n_nodes, n_nodes, n_nodes, self.num_heads, self.head_dim)
        v_e = v_e.unsqueeze(2)
        v = v_e + v_a
        v = v.permute(0, 1, 4, 2, 3, 5)

        attn_prob = torch.matmul(q, k) + bias.unsqueeze(2).unsqueeze(4)
        attn_prob = nn.functional.softmax(attn_prob, dim=5)
        attn_prob = self.attn_dropout(attn_prob)
        out = torch.matmul(attn_prob, v).squeeze(4).permute(0, 1, 3, 2, 4).reshape(batch_size, n_nodes, n_nodes,
                                                                                   self.num_heads * self.head_dim)
        out = self.out_proj(out)
        gate = nn.functional.sigmoid(self.out_gate(pair_emb))
        out = out * gate
        return out


class FeedForwardNetwork(nn.Module):
    def __init__(self, emb_dim, hidden_dim, dropout=0.):
        super(FeedForwardNetwork, self).__init__()
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        # self.ln = nn.LayerNorm(emb_dim)
        self.lin1 = nn.Linear(emb_dim, hidden_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.lin2 = nn.Linear(hidden_dim, emb_dim)

    def forward(self, x):
        # x = self.ln(x)
        x = self.act(self.lin1(x))
        x = self.dropout(x)
        x = self.lin2(x)
        return x


class PosUpdate(nn.Module):
    def __init__(self, hidden_dim, act=nn.SiLU(), tanh=False, coord_range=15.0):
        super(PosUpdate, self).__init__()
        self.tanh = tanh
        self.act = act
        self.coord_range = coord_range
        layer = nn.Linear(hidden_dim, 1, bias=False)
        nn.init.xavier_uniform_(layer.weight, gain=0.001)
        self.coord_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + 3, hidden_dim),
            act,
            nn.Linear(hidden_dim, hidden_dim),
            act,
            layer
        )
        self.dist_decoder = nn.Linear(hidden_dim, 3)

    def forward(self, x_emb, pair_emb, pos, coord_diff, node_mask, pair_mask):
        edge_emb = self.dist_decoder(pair_emb)
        row, col = x_emb.unsqueeze(2).expand(pair_emb.shape), x_emb.unsqueeze(1).expand(pair_emb.shape)
        input_tensor = torch.cat([row, col, edge_emb], dim=3)
        if self.tanh:
            trans = coord_diff * torch.tanh(self.coord_mlp(input_tensor)) * self.coord_range
        else:
            trans = coord_diff * self.coord_mlp(input_tensor)
        trans = trans * pair_mask
        trans = trans.sum(2)
        pos = pos + trans
        return pos


class InterBlock(nn.Module):
    def __init__(self, emb_dim, hidden_dim, num_heads, dropout, node_dropout=None,
                 pair_dropout=None, dataset_name='qm9'):
        super(InterBlock, self).__init__()
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.head_dim = int(self.emb_dim / self.num_heads)
        self.dropout = dropout
        self.node_dropout = node_dropout if node_dropout is not None else dropout
        self.pair_dropout = pair_dropout if pair_dropout is not None else dropout
        self.ffn_dim = hidden_dim
        self.data_name = dataset_name

        self.node_attn = NodeTransformerLayer(
                emb_dim=self.emb_dim,
                ffn_dim=self.ffn_dim,
                num_heads=self.num_heads,
                dropout=self.dropout
            )
        self.node_attn_dropout = nn.Dropout(self.node_dropout)
        self.node_ffn = FeedForwardNetwork(emb_dim, hidden_dim, self.node_dropout)
        self.node_ffn_dropout = nn.Dropout(self.node_dropout)

        if self.data_name == 'qm9':
            self.low2high = Low2High(emb_dim)
        else:
            self.low2high = SimLow2High(emb_dim)
        self.low2high_dropout = nn.Dropout(self.pair_dropout)
        self.pair_ln = nn.LayerNorm(emb_dim)

        self.pair_attn = TriTransLayer(
            emb_dim=self.emb_dim,
            ffn_dim=self.ffn_dim,
            num_heads=self.num_heads,
            dropout=self.dropout
        )
        self.pair_attn_dropout = nn.Dropout(self.pair_dropout)
        self.pair_ffn = FeedForwardNetwork(emb_dim, hidden_dim, self.pair_dropout)
        self.pair_ffn_dropout = nn.Dropout(self.pair_dropout)

        self.pos_update = PosUpdate(emb_dim, act=nn.SiLU(), tanh=True, coord_range=20.0)

    def forward(self, atom_emb, pair_emb, pos, coord_diff, node_mask, pair_mask, batch_size, n_nodes, tri_emb=None):
        atom_residual = self.node_attn(atom_emb, pair_emb, node_mask, pair_mask, batch_size, n_nodes)
        atom_emb = atom_emb + self.node_attn_dropout(atom_residual)

        atom_residual = self.node_ffn(atom_emb)
        atom_emb = atom_emb + self.node_ffn_dropout(atom_residual)
        atom_emb = atom_emb * node_mask

        if self.data_name == 'qm9':
            outer = self.low2high(atom_emb, node_mask)
        elif self.data_name == 'drugs':
            outer = self.low2high(atom_emb, pair_mask)
        pair_emb = pair_emb + self.low2high_dropout(outer)
        pair_emb = self.pair_ln(pair_emb)

        if tri_emb is not None:
            pair_residual = self.pair_attn(pair_emb, tri_emb, pair_mask, batch_size, n_nodes)
            pair_emb = pair_emb + self.pair_attn_dropout(pair_residual)
            pair_residual = self.pair_ffn(pair_emb)
            pair_emb = pair_emb + self.pair_ffn_dropout(pair_residual)
            pair_emb = pair_emb * pair_mask
        else:
            pair_residual = self.pair_attn(pair_emb, pair_mask, batch_size, n_nodes)
            pair_emb = pair_emb + self.pair_attn_dropout(pair_residual)
            pair_residual = self.pair_ffn(pair_emb)
            pair_emb = pair_emb + self.pair_ffn_dropout(pair_residual)
            pair_emb = pair_emb * pair_mask

        pos = self.pos_update(atom_emb, pair_emb, pos, coord_diff, node_mask, pair_mask)
        return atom_emb, pair_emb, pos

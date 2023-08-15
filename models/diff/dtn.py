import torch
import torch.nn as nn
import numpy as np
from torch_geometric.nn.acts import swish
from .utils import atom_pos_to_pair_dist, remove_mean_with_mask, local_geometry_calc, RBF_Emb
from .layers import InterBlock, DistEncoder


class EquiGNN(nn.Module):
    def __init__(self, model_config, data_config):
        super(EquiGNN, self).__init__()
        self.num_layers = model_config['num_layers']
        self.emb_dim = model_config['emb_dim']
        self.hidden_dim = model_config['hidden_dim']
        self.num_heads = model_config['num_heads']
        self.dropout = model_config['dropout']
        self.pair_scale = model_config['pair_loss_scale']
        self.context_dim = len(model_config['context_col']) * int(model_config['context'])
        self.x_class, self.c_class = data_config['x_class'], data_config['c_class']
        self.include_an = model_config['include_an']
        self.include_de = model_config['include_de']
        self.node_nf = data_config['x_class'] + self.include_an + self.include_de
        self.act = nn.ReLU()
        self.activate = swish
        self.add_time = model_config['add_time']
        self.block_calc = model_config['block_calc']
        self.edge_emb = DistEncoder(self.emb_dim, add_time=self.add_time)
        self.angle_emb = RBF_Emb(int(self.emb_dim/2), (np.arange(0, np.pi, 0.1), 10), self.add_time)

        self.convs = nn.ModuleList()
        for i in range(self.num_layers):
            self.convs.append(InterBlock(
                emb_dim=self.emb_dim,
                hidden_dim=self.hidden_dim,
                num_heads=self.num_heads,
                dropout=self.dropout,
                dataset_name=data_config['name']
            ))

        atom_feat = self.node_nf + self.context_dim + 1
        self.xh_embedding = nn.Linear(atom_feat, self.emb_dim)
        self.node_pair_fuse = nn.Linear(3 * self.emb_dim, self.emb_dim)

        if self.block_calc:
            self.edge_emb_fuse = nn.Linear(2 * self.emb_dim, self.emb_dim)
            self.trip_emb_fuse = nn.Linear(self.emb_dim, int(self.emb_dim/2))
        self.xh_embedding_out = nn.Sequential(
            nn.Linear(self.emb_dim, self.hidden_dim),
            self.act,
            nn.Linear(self.hidden_dim, self.emb_dim),
            self.act,
            nn.Linear(self.emb_dim, atom_feat, bias=False)
        )

    def forward(self, z_t, t, node_mask, pair_mask, context=None):
        raise NotImplementedError

    def wrap_forward(self, node_mask, edge_mask, context):
        def fwd(state, time):
            return self._forward(state, time, node_mask, edge_mask, context)
        return fwd

    def unwrap_forward(self):
        return self._forward

    def _forward(self, z_t, t, node_mask, pair_mask, context):
        pos, xh = z_t[:, :, :3], z_t[:, :, 3:]
        batch_size, max_node = xh.shape[0], xh.shape[1]
        if np.prod(t.size()) == 1:
            xh_time = torch.empty_like(xh[:, :, 0:1]).fill_(t.item())
        else:
            xh_time = t.view(batch_size, 1).repeat(1, max_node).unsqueeze(-1)
        xh = torch.cat([xh, xh_time], dim=2) * node_mask
        if context is not None:
            xh = torch.cat([xh, context], dim=2)
        xh_emb = self.act(self.xh_embedding(xh))
        xh_emb = xh_emb * node_mask

        _, radial, coord_diff = atom_pos_to_pair_dist(pos)
        radial = (radial.unsqueeze(-1) * pair_mask).squeeze(-1)
        coord_diff = coord_diff * pair_mask
        if not self.add_time:
            pair_emb = self.edge_emb(radial)
        else:
            pair_emb = self.edge_emb(radial, xh_time.unsqueeze(-1) * pair_mask)
        pair_emb = self.node_pair_fuse(torch.cat([xh_emb.unsqueeze(1) * pair_mask,
                                                  xh_emb.unsqueeze(2) * pair_mask, pair_emb], dim=3))

        angle = local_geometry_calc(pos, pair_mask)
        if not self.add_time:
            tri_emb = self.act(self.angle_emb(angle))
        else:
            triplet_mask = pair_mask.permute(0, 1, 3, 2) * node_mask.unsqueeze(1)
            diag = (torch.ones((node_mask.shape[1], node_mask.shape[1], node_mask.shape[1]), device=node_mask.device) -
                    torch.eye(node_mask.shape[1], device=node_mask.device))
            triplet_mask = triplet_mask * diag
            tri_emb = self.act(self.angle_emb(angle, xh_time.unsqueeze(-1) * triplet_mask))
        for layer in self.convs:
            pair_emb = pair_emb * pair_mask
            xh_emb, pair_emb, pos = layer(xh_emb, pair_emb, pos, coord_diff, node_mask, pair_mask, batch_size,
                                          max_node, tri_emb=tri_emb)
            _, radial, coord_diff = atom_pos_to_pair_dist(pos)
            if self.block_calc:
                angle = local_geometry_calc(pos, pair_mask)
                if self.add_time:
                    tri_emb = self.act(self.trip_emb_fuse(torch.cat(
                        [self.angle_emb(angle, xh_time.unsqueeze(-1) * triplet_mask), tri_emb], dim=4)))
                    pair_emb = self.act(self.edge_emb_fuse(torch.cat(
                        [self.edge_emb(radial, xh_time.unsqueeze(-1)*pair_mask), pair_emb], dim=3)))
                else:
                    tri_emb = self.act(self.trip_emb_fuse(torch.cat([self.angle_emb(angle), tri_emb], dim=4)))
                    pair_emb = self.act(self.edge_emb_fuse(torch.cat([self.edge_emb(radial), pair_emb], dim=3)))
            pair_emb = self.node_pair_fuse(torch.cat(
                [xh_emb.unsqueeze(1) * pair_mask,  xh_emb.unsqueeze(2) * pair_mask, pair_emb], dim=3))

        xh_emb = self.xh_embedding_out(xh_emb) * node_mask
        xh_final = xh_emb[:, :, :self.node_nf]

        vel = (pos - z_t[:, :, :3])
        vel = remove_mean_with_mask(vel, node_mask)

        return torch.cat([vel, xh_final], dim=2)

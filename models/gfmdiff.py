
import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import one_hot, relu, softmax

from .diff import *
from torch_geometric.utils import to_dense_batch
from utils import DistributionNodes


class GFMDiff(nn.Module):
    def __init__(self, model_config, dataset_config, dynamics=EquiGNN):
        super(GFMDiff, self).__init__()
        self.dynamics = dynamics(model_config, dataset_config)
        self.loss_type = model_config['loss_type']
        self.T = model_config['diff_step']
        self.noise_schedule = model_config['noise_schedule']
        self.precision = model_config['noise_precision']
        self.n_dim = model_config['n_dim']
        self.emb_dim = model_config['emb_dim']
        self.hidden_dim = model_config['hidden_dim']
        self.context = model_config['context']
        self.context_col = model_config['context_col']
        self.context_dim = len(model_config['context_col']) * int(model_config['context'])
        self.context_prop = {'mean': [], 'std': []}
        if self.context:
            for col in self.context_col:
                self.context_prop['mean'].append(dataset_config['y_mean'][col])
                self.context_prop['std'].append(dataset_config['y_std'][col])
        self.bond, self.bond_mask = get_bond(dataset_config['name'])

        self.include_an = model_config['include_an']
        self.include_de = model_config['include_de']
        self.calc_dl = model_config['calc_dl']
        self.loss_weight = model_config['loss_weight']
        self.node_distribute = DistributionNodes(dataset_config['stat_nodes'])
        self.x_class = dataset_config['x_class']
        self.node_nf = self.x_class + self.include_an + self.include_de

        self.parameterization = 'eps'
        self.norm_values = dataset_config['norm_values']
        self.norm_bias = dataset_config['norm_bias']
        self.gamma = NoiseSchedule(self.noise_schedule, self.T, self.precision)

    def forward(self, batch_data, device):
        n_nodes = batch_data.n_nodes
        pos, onehot_x, atom_num, degree, node_mask, pair_mask, context = self.read_batch(batch_data, device)
        onehot_x, atom_num, degree = self.normalize(onehot_x, atom_num, degree)
        if self.context:
            context = node_mask * context.unsqueeze(1)
        pos = remove_mean_with_mask(pos, node_mask)
        delta_log_px = -torch.zeros_like(n_nodes, device=device) if self.training and self.loss_type == 'l2' \
            else torch.zeros_like(n_nodes, device=device)

        loss, loss_dict = self.compute_loss(pos, onehot_x, atom_num, degree, node_mask, pair_mask, n_nodes, context)

        neg_log_pxh = loss - delta_log_px

        log_pN = self.node_distribute.log_prob(n_nodes)

        neg_log_pxh = (neg_log_pxh - log_pN).mean(0)
        reg_term = torch.tensor([0.], device=neg_log_pxh.device)
        mean_abs_z = 0.
        return neg_log_pxh, reg_term, mean_abs_z

    def compute_loss(self, pos, onehot_x, atom_num, degree, node_mask, pair_mask, n_nodes, context=None):
        # Get timestep t and s
        t_lowerbound = 0 if self.training else 1
        t_int = torch.randint(t_lowerbound, self.T + 1, size=(pos.size(0), 1), device=pos.device).float()
        s_int = t_int - 1
        t_iszero = (t_int == 0).float()
        s = s_int / self.T
        t = t_int / self.T
        # Compute gamma_t/s, alpha_t and sigma_t
        gamma_s, gamma_t = self.gamma(s), self.gamma(t)
        alpha_t, sigma_t = alpha(gamma_t), sigma(gamma_t)
        # Sample zt ~ N(alpha_t x, sigma_t)

        eps = self.sample_pos_feat_noise(pos.shape[0], pos.shape[1], node_mask)
        xh = torch.cat([pos, onehot_x, atom_num, degree], dim=2) if self.include_de else torch.cat([pos, onehot_x, atom_num], dim=2)
        z_t = alpha_t.unsqueeze(-1) * xh + sigma_t.unsqueeze(-1) * eps

        # GNN prdiction
        gnn_out = self.dynamics._forward(z_t, t, node_mask, pair_mask, context)
        error = self.calc_error(gnn_out, eps, calc_dl=self.calc_dl, pair_mask=pair_mask, alpha=alpha_t)
        if self.training and self.loss_type == 'l2':
            SNR_weight = torch.ones_like(error)
        else:
            SNR_weight = (snr(gamma_s - gamma_t) - 1).squeeze(1)
        assert error.size() == SNR_weight.size()
        loss_t_larger_than_zero = 0.5 * SNR_weight * error

        if self.training and self.loss_type == 'l2':
            neg_log_constants = torch.zeros((pos.shape[0],), device=pos.device)
        else:
            neg_log_constants = -self.log_constants_pos_z0(pos, n_nodes)

        kl_prior = self.kl_prior(xh, n_nodes, node_mask)

        if self.training:
            loss_term_0 = -self.log_pxh_z0_wo_constant(onehot_x, atom_num, degree, z_t, gamma_t, eps, gnn_out, node_mask)
            t_isnotzero = 1 - t_iszero
            loss_t = loss_term_0 * t_iszero.squeeze() + loss_t_larger_than_zero * t_isnotzero.squeeze()
            if self.loss_type == 'l2':
                est_loss = loss_t
            else:
                num_terms = self.T + 1
                est_loss = num_terms * loss_t
            loss = kl_prior + est_loss + neg_log_constants
        else:
            loss_t = loss_t_larger_than_zero
            num_terms = self.T
            est_loss = num_terms * loss_t

            t_zeros = torch.zeros_like(s)
            gamma_0 = self.gamma(t_zeros)
            alpha_0 = alpha(gamma_0).unsqueeze(-1)
            sigma_0 = sigma(gamma_0).unsqueeze(-1)

            eps_0 = self.sample_pos_feat_noise(pos.shape[0], pos.shape[1], node_mask)
            z_0 = alpha_0 * xh + sigma_0 * eps_0
            gnn_out = self.dynamics._forward(z_0, t_zeros, node_mask, pair_mask, context)

            loss_term_0 = -self.log_pxh_z0_wo_constant(onehot_x, atom_num, degree, z_0, gamma_0, eps_0, gnn_out, node_mask)
            loss = kl_prior + est_loss + neg_log_constants + loss_term_0
        return loss, {'t': t_int.squeeze(), 'loss_t': loss.squeeze(), 'error': error.squeeze()}

    def read_batch(self, batch_data, device):
        x = batch_data.x
        pos = batch_data.pos
        batch = batch_data.batch

        if self.context:
            context = batch_data.y[:, self.context_col]
            context = (context - torch.tensor(self.context_prop['mean'], device=device)) / \
                      torch.tensor(self.context_prop['std'], device=device)
        else:
            context = None

        x, node_mask = to_dense_batch(x=x, batch=batch)
        pos, _ = to_dense_batch(x=pos, batch=batch)
        pos = pos.float()
        atom_num = x[:, :, 0].unsqueeze(-1).float()
        onehot_x = x[:, :, 1]
        if self.include_de:
            degree = x[:, :, 2].unsqueeze(-1).float()
        else:
            degree = None
        node_mask, pair_mask = create_mask(x)
        onehot_x = one_hot(onehot_x, num_classes=self.x_class) * node_mask
        return pos, onehot_x, atom_num, degree, node_mask, pair_mask, context

    def normalize(self, onehot_x, atom_num, degree):
        onehot_x = onehot_x / self.norm_values[0]
        atom_num = atom_num / self.norm_values[1]
        if degree is not None:
            degree = degree / self.norm_values[2]
        return onehot_x, atom_num, degree

    def unnormalize(self, onehot_x, atom_num, degree):
        onehot_x = onehot_x * self.norm_values[0]
        atom_num = atom_num * self.norm_values[1]
        if degree is not None:
            degree = degree * self.norm_values[2]
        return onehot_x, atom_num, degree

    def unnormalize_z(self, z, node_mask):
        x, h_x = z[:, :, :self.n_dim], z[:, :, self.n_dim:self.n_dim + self.x_class]
        h_a, h_d = z[:, :, self.n_dim + self.x_class].unsqueeze(-1), z[:, :, -1].unsqueeze(-1)
        h_x, h_a, h_d = self.unnormalize(h_x, h_a, h_d)
        output = torch.cat([x, h_x, h_a, h_d], dim=2)
        return output

    def sample_pos_feat_noise(self, n_samples, n_nodes, node_mask):
        z_pos = sample_center_gravity_zero_gaussian(size=(n_samples, n_nodes, self.n_dim), node_mask=node_mask,
                                                    device=node_mask.device)
        z_x = sample_gaussian(size=(n_samples, n_nodes, self.node_nf), node_mask=node_mask, device=node_mask.device)
        z = torch.cat([z_pos, z_x], dim=2)
        return z

    def calc_error(self, eps_t, eps, calc_dl=False, pair_mask=None, alpha=None):
        if self.training and self.loss_type == 'l2':
            error = ((eps - eps_t) ** 2).sum(2).sum(1) / (eps.shape[1] * eps.shape[2])
        else:
            error = ((eps - eps_t) ** 2).sum(2).sum(1)
        if calc_dl:
            pos_t, x_t, a_t, d_t = eps_t[:, :, :3], eps_t[:, :, 3:3+self.x_class], eps_t[:, :, -2], eps_t[:, :, -1]
            pos, x, a, d = eps[:, :, :3], eps[:, :, 3:3+self.x_class], eps[:, :, -2], eps[:, :, -1]
            x_prob = softmax(x_t.detach(), dim=-1)
            dist = (pos_t.unsqueeze(1) - pos_t.unsqueeze(2)).norm(dim=-1)
            pair_prob = x_prob.unsqueeze(1).unsqueeze(-2) * x_prob.unsqueeze(2).unsqueeze(-1) * self.norm_values[0]
            dist = dist.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(x.shape[0], x.shape[1], x.shape[1],
                                                                         x.shape[2], x.shape[2], 3)
            dist_mar = (dist - self.bond.to(dist.device)) * self.bond_mask.to(dist.device)
            is_bond = (dist_mar.min(-1)[0] < 0).long() * pair_mask.unsqueeze(-1)
            bond_prob = pair_prob * is_bond
            # degree = bond_prob.sum(-1).sum(-2)
            # degree = (degree * x_t).sum(dim=-1) / self.norm_values[2]
            degree = bond_prob.sum(-1).sum(-2).sum(-1) / self.norm_values[2]
            error_d = ((((degree + d) / 2 - d_t) ** 2) * alpha).sum(-1) / eps.shape[1]
            error = error + self.loss_weight['degree'] * error_d
        return error

    def log_constants_pos_z0(self, pos, n_nodes):
        pos_freedom_degree = (n_nodes - 1) * self.n_dim
        zeros = torch.zeros((pos.shape[0],), device=pos.device)
        gamma_0 = self.gamma(zeros)
        log_sigma_pos = 0.5 * gamma_0
        return pos_freedom_degree * (-log_sigma_pos - 0.5 * np.log(2 * np.pi))

    def kl_prior(self, xh, n_nodes, node_mask):
        """Computes the KL between q(z1 | x) and the prior p(z1) = Normal(0, 1).
        This is essentially a lot of work for something that is in practice negligible in the loss.
        However, you compute it so that you see it when you've made a mistake in your noise schedule.
        """
        ones = torch.ones((xh.size(0), 1), device=xh.device)
        gamma_T = self.gamma(ones).unsqueeze(-1)
        alpha_T = alpha(gamma_T)
        mu_T = alpha_T * xh
        mu_T_pos, mu_T_xh = mu_T[:, :, :self.n_dim], mu_T[:, :, self.n_dim:]
        sigma_T_pos, sigma_T_x = sigma(gamma_T).view(xh.size(0), ), sigma(gamma_T)

        zeros, ones = torch.zeros_like(mu_T_xh), torch.ones_like(sigma_T_x)
        kl_dist_xh = gaussian_kl(mu_T_xh, sigma_T_x, zeros, ones, node_mask)

        zeros, ones = torch.zeros_like(mu_T_pos), torch.ones_like(sigma_T_pos)
        pos_freedom_degree = (n_nodes - 1) * self.n_dim
        kl_dist_pos = gaussian_kl(mu_T_pos, sigma_T_pos, zeros, ones, node_mask, d=pos_freedom_degree)

        return kl_dist_xh + kl_dist_pos

    def log_pxh_z0_wo_constant(self, onehot_x, atom_num, degree, z_t, gamma_0, eps, gnn_out, node_mask, epsilon=1e-10):
        z_x = z_t[:, :, self.n_dim:self.n_dim+self.x_class]
        z_a = z_t[:, :, self.n_dim+self.x_class].unsqueeze(-1)

        eps_pos = eps[:, :, :self.n_dim]
        gnn_out_pos = gnn_out[:, :, :self.n_dim]

        sigma_0 = sigma(gamma_0).unsqueeze(-1)
        sigma_0_x = sigma_0 * self.norm_values[0]
        sigma_0_a = sigma_0 * self.norm_values[1]

        log_p_pos_z_wo_constant = -0.5 * self.calc_error(gnn_out_pos, eps_pos)

        xh_x = onehot_x * self.norm_values[0]
        xh_a = atom_num * self.norm_values[1]
        est_xh_x = z_x * self.norm_values[0]
        est_xh_a = z_a * self.norm_values[1]

        xh_x_centered = est_xh_x - 1
        log_ph_x_proportional = torch.log(cdf_standard_gaussian((xh_x_centered + 0.5) / sigma_0_x) -
                                          cdf_standard_gaussian((xh_x_centered - 0.5) / sigma_0_x) +
                                          epsilon)
        log_Z_x = torch.logsumexp(log_ph_x_proportional, dim=2, keepdim=True)
        log_probabilities_x = (log_ph_x_proportional - log_Z_x) * xh_x * node_mask
        log_ph_x = log_probabilities_x.sum(1).sum(1)

        xh_a_centered = xh_a - est_xh_a
        log_ph_a = torch.log(cdf_standard_gaussian((xh_a_centered + 0.5) / sigma_0_a) -
                              cdf_standard_gaussian((xh_a_centered - 0.5) / sigma_0_a) +
                              epsilon)
        log_ph_a = (log_ph_a * node_mask).sum(1).sum(1)

        log_ph = log_ph_x + log_ph_a

        if self.include_de:
            sigma_0_d = sigma_0 * self.norm_values[2]
            z_d = z_t[:, :, -1].unsqueeze(-1)
            xh_d = degree * self.norm_values[2]
            est_xh_d = z_d * self.norm_values[2]
            xh_d_centered = xh_d - est_xh_d
            log_ph_d = torch.log(cdf_standard_gaussian((xh_d_centered + 0.5) / sigma_0_d) -
                                  cdf_standard_gaussian((xh_d_centered - 0.5) / sigma_0_d) +
                                  epsilon)
            log_ph_d = (log_ph_d * node_mask).sum(1).sum(1)
            log_ph = log_ph + log_ph_d

        return log_p_pos_z_wo_constant + log_ph

    @torch.no_grad()
    def sample(self, n_samples, max_n_nodes, device, fix_noise=False, prop_dist=None):
        node_dist = self.node_distribute.sample(n_samples)
        node_mask = torch.zeros(n_samples, max_n_nodes, device=device)
        for i in range(n_samples):
            node_mask[i, 0: node_dist[i]] = 1
        pair_mask = node_mask.unsqueeze(-1) * node_mask.unsqueeze(-2)
        diag_mask = ~torch.eye(pair_mask.size(1), dtype=torch.bool, device=node_mask.device).unsqueeze(0)
        pair_mask *= diag_mask
        node_mask = node_mask.unsqueeze(-1)
        pair_mask = pair_mask.unsqueeze(-1)

        if self.context:
            assert prop_dist is not None
            context = prop_dist.sample_batch(node_dist)
            context = context.unsqueeze(1).repeat(1, max_n_nodes, 1).to(device) * node_mask
        else:
            context = None

        if fix_noise:
            z = self.sample_pos_feat_noise(1, max_n_nodes, node_mask)
        else:
            z = self.sample_pos_feat_noise(n_samples, max_n_nodes, node_mask)

        for s in reversed(range(0, self.T)):
            s_array = torch.full((n_samples, 1), fill_value=s, device=z.device)
            t_array = s_array + 1
            s_array = (s_array / self.T).unsqueeze(-1)
            t_array = (t_array / self.T).unsqueeze(-1)
            z = self.sample_p_zs_given_zt(s_array, t_array, z, node_mask, pair_mask, context)

        pos, onehot_x, atom_num, degree = self.sample_p_posx_given_z0(z, node_mask, pair_mask, context, fix_noise)

        max_cog = torch.sum(pos, dim=1, keepdim=True).abs().max().item()
        if max_cog > 5e-2:
            pos = remove_mean_with_mask(pos, node_mask)
        return pos, onehot_x, atom_num, degree, node_mask

    def sample_p_zs_given_zt(self, s, t, zt, node_mask, pair_mask, context, fix_noise=False):
        gamma_s = self.gamma(s)
        gamma_t = self.gamma(t)

        sigma2_t_given_s, sigma_t_given_s, alpha_t_given_s = sigma_and_alpha_t_given_s(gamma_t, gamma_s)
        sigma_s, sigma_t = sigma(gamma_s), sigma(gamma_t)

        eps_t = self.dynamics._forward(zt, t, node_mask, pair_mask, context)

        mu = zt / alpha_t_given_s - (sigma2_t_given_s / alpha_t_given_s / sigma_t) * eps_t
        sigma_st = sigma_t_given_s * sigma_s / sigma_t

        bs = 1 if fix_noise else mu.size(0)
        eps = self.sample_pos_feat_noise(bs, mu.shape[1], node_mask)
        zs = mu + sigma_st * eps

        zs = torch.cat([remove_mean_with_mask(zs[:, :, :self.n_dim], node_mask),
                        zs[:, :, self.n_dim:]], dim=2)
        return zs

    def sample_p_posx_given_z0(self, z0, node_mask, pair_mask, context, fix_noise=False):
        zeros = torch.zeros(size=(z0.shape[0], 1, 1), device=z0.device)
        gamma_0 = self.gamma(zeros)
        sigma_pos = snr(-0.5 * gamma_0)
        gnn_out = self.dynamics._forward(z0, zeros, node_mask, pair_mask, context)

        if self.parameterization == 'eps':
            sigma_0, alpha_0 = sigma(gamma_0), alpha(gamma_0)
            mu_pos = 1. / alpha_0 * (z0 - sigma_0 * gnn_out)
        else:
            mu_pos = gnn_out
        xh = mu_pos + sigma_pos * self.sample_pos_feat_noise(mu_pos.shape[0], mu_pos.shape[1], node_mask)
        pos = xh[:, :, :self.n_dim] * node_mask
        onehot_x = z0[:, :, self.n_dim:self.n_dim+self.x_class] * node_mask
        atom_num = z0[:, :, self.n_dim+self.x_class].unsqueeze(-1) * node_mask
        degree = z0[:, :, -1].unsqueeze(-1) * node_mask if self.include_de else None
        onehot_x, atom_num, degree = self.unnormalize(onehot_x, atom_num, degree)
        onehot_x = one_hot(torch.argmax(onehot_x, dim=2), onehot_x.shape[2]) * node_mask
        atom_num = torch.round(atom_num).long() * node_mask
        degree = torch.round(degree).long() * node_mask if self.include_de else None
        return pos, onehot_x, atom_num, degree

    @torch.no_grad()
    def sample_chain(self, n_samples, n_nodes, device, context, fix_noise=False, keep_frames=None):
        node_mask = torch.ones((n_samples, n_nodes, 1)).to(device)
        edge_mask = (1 - torch.eye(n_nodes)).unsqueeze(0)
        edge_mask = edge_mask.repeat(n_samples, 1, 1).unsqueeze(-1).to(device)

        z = self.sample_pos_feat_noise(n_samples, n_nodes, node_mask)
        if keep_frames is None:
            keep_frames = self.T
        else:
            assert keep_frames <= self.T
        chain = torch.zeros((keep_frames,) + z.size(), device=device)
        for s in reversed(range(0, self.T)):
            s_array = torch.full((n_samples, 1), fill_value=s, device=device)
            t_array = s_array + 1
            s_array = s_array / self.T
            t_array = t_array / self.T

            z = self.sample_p_zs_given_zt(s_array, t_array, z, node_mask, edge_mask, context)
            write_index = (s * keep_frames) // self.T
            chain[write_index] = self.unnormalize_z(z, node_mask)

        pos, onehot, atom_num, degree = self.sample_p_posx_given_z0(z, node_mask, edge_mask, context)
        xh = torch.cat([pos, onehot, atom_num, degree], dim=2)
        chain[0] = xh
        chain_flat = chain.view(n_samples * keep_frames, *z.size()[1:])
        return chain_flat

import torch
from related_works.targetdiff.models.molopt_score_model import center_pos, index_to_log_onehot, log_sample_categorical
from tqdm import tqdm
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from d3po_pytorch.diffusers_patch.ddim_with_logprob import ddim_step_with_logprob
from torch.nn import functional as F
import time
import numpy as np
import einops
from torch_scatter import scatter_sum, scatter_mean
from torch_geometric.data import Batch
from related_works.targetdiff.datasets.pl_data import FOLLOW_BATCH
from related_works.targetdiff.utils.evaluation import atom_num
from related_works.targetdiff.scripts.sample_diffusion import unbatch_v_traj

def sample_diffusion_ligand_with_logprob(model, data, num_samples, batch_size=16, device='cuda:0',
                            num_steps=None, pos_only=False, center_pos_mode='protein',
                            sample_num_atoms='prior',
                            # ↓↓↓↓↓↓↓↓↓↓ edited ↓↓↓↓↓↓↓↓↓↓ #
                            callback_on_step_end=None,
                            enable_grad=False,
                            log_probs_given_trajectory=None,
                            init_pos=None,
                            generator=None,
                            # ↑↑↑↑↑↑↑↑↑↑ edited ↑↑↑↑↑↑↑↑↑↑ #
    ):

    all_pred_pos, all_pred_v = [], []
    all_log_prob_traj = []

    all_pred_pos_traj, all_pred_v_traj = [], []
    all_pred_v0_traj, all_pred_vt_traj = [], []
    time_list = []
    num_batch = int(np.ceil(num_samples / batch_size))
    current_i = 0
    for i in tqdm(range(num_batch)):
        n_data = batch_size if i < num_batch - 1 else num_samples - batch_size * (num_batch - 1)
        batch = Batch.from_data_list([data.clone() for _ in range(n_data)], follow_batch=FOLLOW_BATCH).to(device)

        t1 = time.time()
        with torch.no_grad():
            batch_protein = batch.protein_element_batch
            if sample_num_atoms == 'prior':
                pocket_size = atom_num.get_space_size(data.protein_pos.detach().cpu().numpy())
                ligand_num_atoms = [atom_num.sample_atom_num(pocket_size).astype(int) for _ in range(n_data)]
                batch_ligand = torch.repeat_interleave(torch.arange(n_data), torch.tensor(ligand_num_atoms)).to(device)
            elif sample_num_atoms == 'range':
                ligand_num_atoms = list(range(current_i + 1, current_i + n_data + 1))
                batch_ligand = torch.repeat_interleave(torch.arange(n_data), torch.tensor(ligand_num_atoms)).to(device)
            elif sample_num_atoms == 'ref':
                batch_ligand = batch.ligand_element_batch
                ligand_num_atoms = scatter_sum(torch.ones_like(batch_ligand), batch_ligand, dim=0).tolist()
            else:
                raise ValueError

            # init ligand pos
            center_pos = scatter_mean(batch.protein_pos, batch_protein, dim=0)
            batch_center_pos = center_pos[batch_ligand]

            # ↓↓↓↓↓↓↓↓↓↓ edited ↓↓↓↓↓↓↓↓↓↓ #
            if init_pos is None:
                init_ligand_pos = batch_center_pos + torch.randn(*batch_center_pos.shape, generator=generator, device=device)
            else:
                init_ligand_pos = batch_center_pos + einops.rearrange(init_pos, "B N D -> (B N) D").type(batch_center_pos.dtype)
            # ↑↑↑↑↑↑↑↑↑↑ edited ↑↑↑↑↑↑↑↑↑↑ #
            
            # init ligand v
            if pos_only:
                init_ligand_v = batch.ligand_atom_feature_full
            else:
                uniform_logits = torch.zeros(len(batch_ligand), model.num_classes).to(device)
                init_ligand_v = log_sample_categorical(uniform_logits)
            
            # ↓↓↓↓↓↓↓↓↓↓ edited ↓↓↓↓↓↓↓↓↓↓ #
            if log_probs_given_trajectory is not None:
                log_probs_given_trajectory = einops.rearrange(log_probs_given_trajectory, "B T N D -> (B N) T D")
                log_probs_given_trajectory = log_probs_given_trajectory - batch_center_pos[:,None,:]
            # ↑↑↑↑↑↑↑↑↑↑ edited ↑↑↑↑↑↑↑↑↑↑ #

            r = sample_diffusion_with_logprob(
                model,
                protein_pos=batch.protein_pos,
                protein_v=batch.protein_atom_feature.float(),
                batch_protein=batch_protein,

                init_ligand_pos=init_ligand_pos,
                init_ligand_v=init_ligand_v,
                batch_ligand=batch_ligand,
                num_steps=num_steps,
                pos_only=pos_only,
                center_pos_mode=center_pos_mode,
                callback_on_step_end=callback_on_step_end,
                log_probs_given_trajectory=log_probs_given_trajectory,
                generator=generator,
                enable_grad=enable_grad,
            )

            
            ligand_pos, ligand_v, ligand_pos_traj, ligand_v_traj = r['pos'], r['v'], r['pos_traj'], r['v_traj']
            ligand_v0_traj, ligand_vt_traj = r['v0_traj'], r['vt_traj']
            # unbatch pos
            ligand_cum_atoms = np.cumsum([0] + ligand_num_atoms)
            ligand_pos_array = ligand_pos.cpu().numpy().astype(np.float64)
            all_pred_pos += [ligand_pos_array[ligand_cum_atoms[k]:ligand_cum_atoms[k + 1]] for k in
                             range(n_data)]  # num_samples * [num_atoms_i, 3]

            all_step_pos = [[] for _ in range(n_data)]
            for p in ligand_pos_traj:  # step_i
                p_array = p.cpu().numpy().astype(np.float64)
                for k in range(n_data):
                    all_step_pos[k].append(p_array[ligand_cum_atoms[k]:ligand_cum_atoms[k + 1]])
            all_step_pos = [np.stack(step_pos) for step_pos in
                            all_step_pos]  # num_samples * [num_steps, num_atoms_i, 3]
            all_pred_pos_traj += [p for p in all_step_pos]
            
            # ↓↓↓↓↓↓↓↓↓↓ edited ↓↓↓↓↓↓↓↓↓↓ #
            log_prob_traj = r['log_prob_traj']
            all_step_log_prob = [[] for _ in range(n_data)]
            for p in log_prob_traj:  # step_i
                p_array = p.cpu().numpy().astype(np.float64)
                for k in range(n_data):
                    all_step_log_prob[k].append(p_array[ligand_cum_atoms[k]:ligand_cum_atoms[k + 1]])
            all_step_log_prob = [np.stack(step_log_prob) for step_log_prob in
                            all_step_log_prob]  # num_samples * [num_steps, num_atoms_i]
            all_step_log_prob = [p.mean(1) for p in all_step_log_prob] # num_samples * [num_steps]
            all_log_prob_traj += [p for p in all_step_log_prob]
            # ↑↑↑↑↑↑↑↑↑↑ edited ↑↑↑↑↑↑↑↑↑↑ #

            # unbatch v
            ligand_v_array = ligand_v.cpu().numpy()
            all_pred_v += [ligand_v_array[ligand_cum_atoms[k]:ligand_cum_atoms[k + 1]] for k in range(n_data)]

            all_step_v = unbatch_v_traj(ligand_v_traj, n_data, ligand_cum_atoms)
            all_pred_v_traj += [v for v in all_step_v]

            if not pos_only:
                all_step_v0 = unbatch_v_traj(ligand_v0_traj, n_data, ligand_cum_atoms)
                all_pred_v0_traj += [v for v in all_step_v0]
                all_step_vt = unbatch_v_traj(ligand_vt_traj, n_data, ligand_cum_atoms)
                all_pred_vt_traj += [v for v in all_step_vt]
        t2 = time.time()
        time_list.append(t2 - t1)
        current_i += n_data
    return all_pred_pos, all_pred_v, all_log_prob_traj, all_pred_pos_traj, all_pred_v_traj, all_pred_v0_traj, all_pred_vt_traj, time_list


@torch.no_grad()
def sample_diffusion_with_logprob(self, protein_pos, protein_v, batch_protein,
                        init_ligand_pos, init_ligand_v, batch_ligand,
                        num_steps=None, center_pos_mode=None, pos_only=False,
                        noise=None, start_from_i=0,
                        # ↓↓↓↓↓↓↓↓↓↓ edited ↓↓↓↓↓↓↓↓↓↓ #
                        callback_on_step_end=None,
                        log_probs_given_trajectory=None,
                        generator=None,
                        enable_grad=False,
                        # ↑↑↑↑↑↑↑↑↑↑ edited ↑↑↑↑↑↑↑↑↑↑ #
                    ):

    # ↓↓↓↓↓↓↓↓↓↓ edited ↓↓↓↓↓↓↓↓↓↓ #
    assert isinstance(self.sampling_scheduler, DDIMScheduler)
    # ↑↑↑↑↑↑↑↑↑↑ edited ↑↑↑↑↑↑↑↑↑↑ #

    if noise is None:
        noise = torch.randn((num_steps-start_from_i, *init_ligand_pos.shape), device=init_ligand_pos.device)
    else:
        assert noise.size() == (num_steps-start_from_i, *init_ligand_pos.shape)

    if num_steps is None:
        num_steps = self.num_timesteps
    
    self.sampling_scheduler.set_timesteps(num_steps)

    num_graphs = batch_protein.max().item() + 1

    protein_pos, init_ligand_pos, offset = center_pos(
        protein_pos, init_ligand_pos, batch_protein, batch_ligand, mode=center_pos_mode)

    # ↓↓↓↓↓↓↓↓↓↓ edited ↓↓↓↓↓↓↓↓↓↓ #
    pos_traj, v_traj = [init_ligand_pos], [init_ligand_v]
    # ↑↑↑↑↑↑↑↑↑↑ edited ↑↑↑↑↑↑↑↑↑↑ #
    v0_pred_traj, vt_pred_traj = [], []
    pos0_traj = []
    log_prob_traj = []
    ligand_pos, ligand_v = init_ligand_pos, init_ligand_v
    # time sequence
    time_seq = self.sampling_scheduler.timesteps[start_from_i:]

    with torch.set_grad_enabled(enable_grad):
        for i, t_ in enumerate(tqdm(time_seq, desc='sampling', total=len(time_seq))):
            t = torch.full(size=(num_graphs,), fill_value=t_, dtype=torch.long, device=protein_pos.device)
            
            preds = self(
                protein_pos=protein_pos,
                protein_v=protein_v,
                batch_protein=batch_protein,

                init_ligand_pos=ligand_pos,
                init_ligand_v=ligand_v,
                batch_ligand=batch_ligand,
                time_step=t
            )
            # Compute posterior mean and variance
            if self.model_mean_type == 'noise':
                pred_pos_noise = preds['pred_ligand_pos'] - ligand_pos
                pos0_from_e = self._predict_x0_from_eps(xt=ligand_pos, eps=pred_pos_noise, t=t, batch=batch_ligand)
                v0_from_e = preds['pred_ligand_v']
            elif self.model_mean_type == 'C0':
                pos0_from_e = preds['pred_ligand_pos']
                v0_from_e = preds['pred_ligand_v']
            else:
                raise ValueError

            original_pos0 = pos0_from_e + offset[batch_ligand]
            pos0_traj.append(original_pos0.clone().cpu())
            # ↓↓↓↓↓↓↓↓↓↓ edited ↓↓↓↓↓↓↓↓↓↓ #
            log_prob_given_prev_sample = log_probs_given_trajectory[:,i] if log_probs_given_trajectory is not None else None
            ligand_pos, ligand_pos_meam, log_prob  = ddim_step_with_logprob(self.sampling_scheduler, pos0_from_e, t[0], ligand_pos, eta=1.0, return_dict=False, log_prob_given_prev_sample=log_prob_given_prev_sample, generator=generator)
            
            if callback_on_step_end is not None:
                callback_kwargs = {
                    "ligand_pos": ligand_pos,
                    "log_prob": log_prob,
                    "offset": offset[batch_ligand],
                }
                callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
                ligand_pos = callback_outputs.pop("ligand_pos", ligand_pos)

            log_prob_traj.append(log_prob)
            # ↑↑↑↑↑↑↑↑↑↑ edited ↑↑↑↑↑↑↑↑↑↑ #
            
            # (replace by above diffusers scheduler)
            # pos_model_mean = self.q_pos_posterior(x0=pos0_from_e, xt=ligand_pos, t=t, batch=batch_ligand)
            # pos_log_variance = extract(self.posterior_logvar, t, batch_ligand)
            # nonzero_mask = (1 - (t == 0).float())[batch_ligand].unsqueeze(-1) # no noise when t == 0
            # ligand_pos_next = pos_model_mean + nonzero_mask * (0.5 * pos_log_variance).exp() * noise[i]
            # ligand_pos = ligand_pos_next

            if not pos_only:
                log_ligand_v_recon = F.log_softmax(v0_from_e, dim=-1)
                log_ligand_v = index_to_log_onehot(ligand_v, self.num_classes)
                log_model_prob = self.q_v_posterior(log_ligand_v_recon, log_ligand_v, t, batch_ligand)
                ligand_v_next = log_sample_categorical(log_model_prob)

                v0_pred_traj.append(log_ligand_v_recon.clone().cpu())
                vt_pred_traj.append(log_model_prob.clone().cpu())
                ligand_v = ligand_v_next

            ori_ligand_pos = ligand_pos + offset[batch_ligand]
            pos_traj.append(ori_ligand_pos.clone().cpu())
            v_traj.append(ligand_v.clone().cpu())

    ligand_pos = ligand_pos + offset[batch_ligand]
    return {
        'pos': ligand_pos,
        'v': ligand_v,
        'pos_traj': pos_traj,
        'v_traj': v_traj,
        'v0_traj': v0_pred_traj,
        'vt_traj': vt_pred_traj,
        # ↓↓↓↓↓↓↓↓↓↓ edited ↓↓↓↓↓↓↓↓↓↓ #
        'log_prob_traj': log_prob_traj,
        # ↑↑↑↑↑↑↑↑↑↑ edited ↑↑↑↑↑↑↑↑↑↑ #
    }

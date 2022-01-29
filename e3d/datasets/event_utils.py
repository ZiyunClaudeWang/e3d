import torch
import numpy as np
import pdb

def exp_filter(frames_pos, frames_neg, alphas, down_samp=0.003, samp_time=0.0002):
    from scipy.sparse import lil_matrix
    out_size = int(len(frames_pos) * samp_time / down_samp)
    lin_ind = np.linspace(0, len(frames_pos) - 1, out_size).astype(np.int)
    act_ind = {val: i for i, val in enumerate(lin_ind)}
    filt_frames_pos = []
    filt_frames_neg = []
    for _ in range(lin_ind.shape[0]):
        tmp = []
        tmp2 = []
        for _ in range(len(alphas)):
            tmp.append(lil_matrix((480, 640), dtype=np.float32))
            tmp2.append(lil_matrix((480, 640), dtype=np.float32))
        filt_frames_pos.append(tmp)
        filt_frames_neg.append(tmp2)
    pos_acc = np.zeros((len(alphas), 480, 640), dtype=np.float32)
    neg_acc = np.zeros((len(alphas), 480, 640), dtype=np.float32)
    for fidx in range(0, len(frames_pos)):
        for tidx in range(len(alphas)):
            pos_acc[tidx] = pos_acc[tidx] + (alphas[tidx]) * (
                frames_pos[fidx].astype(np.float32) - pos_acc[tidx]
            )
            neg_acc[tidx] = neg_acc[tidx] + (alphas[tidx]) * (
                frames_neg[fidx].astype(np.float32) - neg_acc[tidx]
            )
            if fidx in act_ind:
                idx = act_ind[fidx]
                filt_frames_pos[idx][tidx] = lil_matrix(pos_acc[tidx])
                filt_frames_neg[idx][tidx] = lil_matrix(neg_acc[tidx])
    return filt_frames_pos, filt_frames_neg, act_ind


def calc_floor_ceil_delta(x): 
    x_fl = torch.floor(x + 1e-8)
    x_ce = torch.ceil(x - 1e-8)
    x_ce_fake = torch.floor(x) + 1

    dx_ce = x - x_fl
    dx_fl = x_ce_fake - x
    return [x_fl.long(), dx_fl], [x_ce.long(), dx_ce]

def create_update(x, y, t, dt, p, vol_size, device="cpu"):
    assert (x>=0).byte().all() 
    assert (x<vol_size[2]).byte().all()
    assert (y>=0).byte().all()
    assert (y<vol_size[1]).byte().all()
    assert (t>=0).byte().all() 
    #assert (t<vol_size[0] // 2).byte().all()

    if not (t < vol_size[0] // 2).byte().all():
        print(t[t >= vol_size[0] // 2])
        print(vol_size)
        pdb.set_trace()
        raise AssertionError()

    vol_mul = torch.where(p < 0,
                          torch.ones(p.shape, dtype=torch.long).to(device) * (vol_size[0] // 2),
                          torch.zeros(p.shape, dtype=torch.long).to(device))

    inds = (vol_size[1]*vol_size[2]) * (t + vol_mul)\
         + (vol_size[2])*y\
         + x

    vals = dt

    return inds, vals

def gen_discretized_event_volume(events, vol_size, device="cpu"):
    # volume is [t, x, y]
    # events are Nx4
    npts = events.shape[0]
    volume = events.new_zeros(vol_size)

    x = events[:, 0].long()
    y = events[:, 1].long()
    t = events[:, 2]

    t_min = t.min()
    t_max = t.max()
    t_scaled = (t-t_min) * ((vol_size[0] // 2-1) / (t_max-t_min + 1e-6))

    ts_fl, ts_ce = calc_floor_ceil_delta(t_scaled.squeeze())

    if (ts_fl[0] > 9).sum() > 0:
        pdb.set_trace()
    
    inds_fl, vals_fl = create_update(x, y,
                                     ts_fl[0], ts_fl[1],
                                     events[:, 3],
                                     vol_size,
                                     device=device)
    volume.view(-1).put_(inds_fl, vals_fl, accumulate=True)

    if (ts_ce[0] > 9).sum() > 0:
        pdb.set_trace()

    inds_ce, vals_ce = create_update(x, y,
                                     ts_ce[0], ts_ce[1],
                                     events[:, 3],
                                     vol_size,
                                     device=device)
    volume.view(-1).put_(inds_ce, vals_ce, accumulate=True)
    return volume
 
def normalize_event_volume(event_volume):
    event_volume_flat = event_volume.view(-1)
    nonzero = torch.nonzero(event_volume_flat)
    nonzero_values = event_volume_flat[nonzero]
    if nonzero_values.shape[0]:
        lower = torch.kthvalue(nonzero_values,
                               max(int(0.02 * nonzero_values.shape[0]), 1),
                               dim=0)[0][0]
        upper = torch.kthvalue(nonzero_values,
                               max(int(0.98 * nonzero_values.shape[0]), 1),
                               dim=0)[0][0]
        max_val = max(abs(lower), upper)
        event_volume = torch.clamp(event_volume, -max_val, max_val)
        event_volume /= max_val
    return event_volume

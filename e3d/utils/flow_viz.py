import torch
import numpy as np
import matplotlib.colors as colors


def single_flow2rgb(flow_x, flow_y, hsv_buffer=None, scale_factor=None):
    if hsv_buffer is None:
        hsv_buffer = np.empty((flow_x.shape[0], flow_x.shape[1], 3))
    hsv_buffer[:, :, 1] = 1.0
    hsv_buffer[:, :, 0] = (np.arctan2(flow_y, flow_x) + np.pi) / (2.0 * np.pi)

    hsv_buffer[:, :, 2] = np.linalg.norm(np.stack((flow_x, flow_y), axis=0), axis=0)

    flat = hsv_buffer[:, :, 2].reshape((-1))
    finite_set = np.isfinite(flat)
    if np.sum(finite_set) > 0:
        m = np.nanmax(flat[finite_set])
        if np.isclose(m, 0.0):
            m = 1.0

    if scale_factor is None:
        hsv_buffer[:, :, 2] /= m
    else:
        hsv_buffer[:, :, 2] /= scale_factor * 1.5


    return colors.hsv_to_rgb(hsv_buffer)

def color_wheel():
    x = np.linspace(-1.,1.,100)
    y = np.linspace(-1.,1.,100)

    xs, ys = np.meshgrid(x, y)

    rgb_flow = single_flow2rgb(xs, ys)
    return torch.from_numpy(rgb_flow.transpose(2,0,1))

def flow2rgb(flow, squeeze=False, scale_factor=None):
    flow_x, flow_y = (
        flow[:, 0, :, :].cpu().detach().numpy(),
        flow[:, 1, :, :].cpu().detach().numpy(),
    )

    if squeeze:
        flow_x = flow_x.squeeze()
        flow_y = flow_y.squeeze()

    hsv_buffer = np.empty((flow_x.shape[0], flow_x.shape[1], flow_x.shape[2], 3))

    for i in range(flow_x.shape[0]):
        single_flow2rgb(flow_x[i, ...], flow_y[i, ...], hsv_buffer[i, ...], scale_factor)

    hsv_buffer[np.logical_not(np.isfinite(hsv_buffer))] = 0.0

    return torch.from_numpy(colors.hsv_to_rgb(hsv_buffer).transpose((0, 3, 1, 2)))

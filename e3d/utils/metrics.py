import torch


def AEE(flow,ground_truth):
    gt_flow = ground_truth["gt_flow"]

    mask = ground_truth["full_mask"].float()

    flow = mask * flow["flow"]
    gt_flow = mask * gt_flow

    dx = flow.squeeze(0)[0,:,:] - gt_flow.squeeze(0)[0,:,:]
    dy = flow.squeeze(0)[1,:,:] - gt_flow.squeeze(0)[1,:,:]
    return torch.sum(torch.sqrt(dx**2 + dy**2))/float(mask.sum())

def percentage_outliers(flow, ground_truth):
    gt_flow = ground_truth["gt_flow"]

    mask = ground_truth["full_mask"].float()

    flow = mask * flow["scaled_flow"]
    gt_flow = mask * gt_flow

    dx = flow.squeeze(0)[0,:,:] - gt_flow.squeeze(0)[0,:,:]
    dy = flow.squeeze(0)[1,:,:] - gt_flow.squeeze(0)[1,:,:]
    gt_norms = torch.sqrt(gt_flow.squeeze(0)[0,:,:]**2 + gt_flow.squeeze(0)[1,:,:]**2)
    ee = torch.sqrt(dx**2 + dy**2)

    outlier_mask = (ee > 3.) * (ee > 0.05*gt_norms)

    outliers = outlier_mask.sum()

    return outliers, mask.sum()



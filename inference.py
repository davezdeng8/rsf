import torch
import utils

def flow_inference(pc1, ego_transform, boxes, box_transform, config, cc = True):
    """
    :param pc1: nx3 tensor
    :param R_ego: 3x3 tensor
    :param t_ego: 3 tensor
    :param boxes: kx8 tensor
    :param R: kx3x3 tensor
    :param t: kx3 tensor
    :return: predicted sf: nx3 tensor
    """
    expansion = torch.tensor([0,0,0,0,1,1,1,0], dtype=torch.float32, device='cuda')*.5
    expansion = torch.stack([expansion]*boxes.shape[0], dim=0)
    expanded_boxes = boxes+expansion
    bprt = torch.cat([expanded_boxes, box_transform.get_matrix()[:,:3,:3].reshape(-1,9), box_transform.get_matrix()[:,3,:3]], axis=-1)
    bprt = utils.prune_empty(pc1, bprt, threshold = config['prune_threshold'])
    bprt = utils.nms(bprt, confidence_threshold=config['confidence_threshold'])
    if bprt == None:
        motion_parameters = None
    else:
        motion_parameters = {'ego_transform': ego_transform, 'boxes':bprt[:,:8],
                         'box_transform': utils.get_rigid_transform(bprt[:, 8:17].reshape(-1, 3, 3), bprt[:, 17:20])}
    if bprt is not None:
        if cc:
            segmentation = utils.cc_in_box(pc1, bprt, seg_threshold=config['seg_threshold'])
        else:
            segmentation = utils.box_segment(pc1, bprt)
        R_apply, t_apply = bprt[:, 8:17].reshape(-1, 3, 3), bprt[:, 17:20]
        R_ego, t_ego = ego_transform.get_matrix()[:,:3,:3], ego_transform.get_matrix()[:,3,:3]
        R_combined, t_combined = torch.cat([R_ego, R_apply], dim = 0), torch.cat([t_ego, t_apply], dim = 0)
        final_transform = utils.get_rigid_transform(R_combined, t_combined)
        transformed_pts = final_transform[segmentation].transform_points(pc1.unsqueeze(1)).squeeze(1)

    else:
        transformed_pts = ego_transform.transform_points(pc1)
        segmentation = torch.zeros_like(pc1[:,0])
        # print('no detected objects')
    return transformed_pts - pc1, segmentation, motion_parameters
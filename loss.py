import torch
import torch.nn.functional as F
import numpy as np
from pytorch3d.ops.knn import knn_points
from pytorch3d import transforms
import utils
# from visualization import vis_flow

class ObjectLoss:
    def __init__(self, anchors):
        self.anchors = anchors
        self.anchor_rots = utils.angle2rot_2d(anchors[:, 6])
        self.num_boxes = len(anchors)

    def __call__(self, pc1, pc2, pc1_normals, pc2_normals, global_params, perbox_params, config):
        """
        :param pc1: Pointcloud object
        :param pc2: Pointcloud object
        :param global_params: bx12 tensor for ego motion (9 rotation + 3 translation)
        :param perbox_params: bxkx15 tensor (1 confidence + 8 box params + 4 rotation + 2 translation)
        :return: loss
        """

        pc1_padded, pc2_padded = pc1.points_padded(), pc2.points_padded()
        ego_transform = utils.global_params2Rt(global_params)
        boxes, box_transform = utils.perbox_params2boxesRt(perbox_params, self.anchors)
        box_transform_comp = transforms.Transform3d(matrix=ego_transform.get_matrix().detach().repeat_interleave(self.num_boxes, dim=0)).compose(box_transform)

        pc1_ego = ego_transform.transform_points(pc1_padded)
        pc1_normals_ego = transforms.Rotate(R = ego_transform.get_matrix()[:,:3,:3], device='cuda').transform_points(pc1_normals)
        bg_nnd_1 = knn_points(torch.cat((pc1_ego, pc1_normals_ego), dim=-1), torch.cat((pc2_padded, pc2_normals), dim=-1), pc1.num_points_per_cloud(), pc2.num_points_per_cloud())[0].squeeze(-1)
        bg_nnd_1 = torch.repeat_interleave(bg_nnd_1, self.num_boxes, dim=0)

        box_pc1, box_weights_1, weights_1, not_empty_1, box_pc1_normals = utils.box_weights(pc1, boxes, slope=config['sigmoid_slope'], normals = pc1_normals)
        box_pc1_t = box_transform_comp[not_empty_1].transform_points(box_pc1.points_padded()[not_empty_1])
        box_pc1_normals = transforms.Rotate(R = box_transform_comp[not_empty_1].get_matrix()[:,:3,:3], device='cuda').transform_points(box_pc1_normals[not_empty_1])
        fg_nnd_1 = knn_points(torch.cat((box_pc1_t, box_pc1_normals), dim=-1), torch.repeat_interleave(torch.cat((pc2_padded, pc2_normals), dim=-1), self.num_boxes, dim=0)[not_empty_1],
                            box_pc1.num_points_per_cloud()[not_empty_1], pc2.num_points_per_cloud().repeat_interleave(self.num_boxes)[not_empty_1])[0].squeeze(-1)
        fg_nnd_1 = fg_nnd_1+config['epsilon']#.005
        bg_nnd_1 = bg_nnd_1[not_empty_1]

        normalized_box_weights_1, normalized_weights_1 = utils.normalize(box_weights_1[not_empty_1], dim=-1), utils.normalize(weights_1[not_empty_1], dim=-1)
        fg_mean_1 = torch.sum(normalized_box_weights_1*fg_nnd_1, dim = 1, keepdim=True)
        bg_mean_1 = torch.sum(normalized_weights_1*bg_nnd_1, dim = 1, keepdim=True)

        confidence = boxes[:, :1][not_empty_1]
        foreground_loss = torch.mean(confidence*fg_mean_1)
        background_loss = torch.mean((1-confidence)*bg_mean_1)

        #heading loss
        avg_sf = ego_transform.inverse().transform_points(box_transform_comp.transform_points(boxes[..., 1:4].unsqueeze(1)).view(-1, self.num_boxes, 3))\
                 -boxes[..., 1:4].view(-1, self.num_boxes, 3)
        aligned_heading = torch.einsum('bij,abjk->abik', self.anchor_rots, perbox_params[:,:,7:9].unsqueeze(-1)).squeeze(-1)
        heading_loss = F.mse_loss(aligned_heading.reshape(-1, 2), avg_sf[...,:2].detach().reshape(-1, 2))*config['heading_loss_coeff']

        #angle loss
        R_angle_loss = torch.mean(torch.square(transforms.matrix_to_euler_angles(box_transform.get_matrix()[:,:3,:3], 'ZYX')[...,0]))*config['angle_loss_coeff']

        #box regularization
        box_mass_1 = torch.sum(weights_1, dim = 1)
        mass_loss = -torch.mean(box_mass_1)*config['mass_loss_coeff'] #.02
        dim_regularization2 = torch.mean(perbox_params[:,:,4:7]*perbox_params[:,:,4:7])*config['dim_loss_coeff'] #500

        loss = foreground_loss+background_loss+mass_loss+dim_regularization2+heading_loss+R_angle_loss
        return loss







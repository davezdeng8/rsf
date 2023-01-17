import numpy as np
import torch
import torch.optim as optim
import yaml
from lidarkitti import make_data_loader
import matplotlib.pyplot as plt
from visualization import vis_boxes_from_params, vis_flow
import rsf_utils
from rsf_loss import RSFLossv2, RSFLossCycle
from inference import flow_inference
from pytorch3d.structures import Pointclouds, list_to_padded
from pytorch3d.ops import estimate_pointcloud_normals, iterative_closest_point
from pytorch3d.ops.knn import knn_points
from pytorch3d import transforms
import sys
import argparse
import pickle
from collections import defaultdict


class SF_Optimizer:
    def __init__(self, anchors, config, pc1, pc2, pc1_normals, pc2_normals, mask1, mask2, seg1, seg2, R_ego, t_ego, sf, init_perbox=None, init_global=None, use_gt_ego=False, icp_init=False):
        self.anchors = anchors
        self.num_boxes = anchors.shape[0]
        self.config = config
        self.batch_size = len(pc1)

        pc1_opt, pc2_opt = [torch.clone(p) for p in pc1], [torch.clone(p) for p in pc2]
        pc1_normals_opt, pc2_normals_opt = [torch.clone(p) for p in pc1_normals], [torch.clone(p) for p in pc2_normals]

        self.pc1, self.pc2 = Pointclouds(pc1).to(device='cuda'), Pointclouds(pc2).to(device='cuda')
        self.pc1_normals, self.pc2_normals = list_to_padded(pc1_normals).to(device='cuda'), list_to_padded(pc2_normals).to(device='cuda')
        self.pc1_opt, self.pc2_opt = Pointclouds(pc1_opt).to(device='cuda'), Pointclouds(pc2_opt).to(device='cuda')
        self.pc1_normals_opt, self.pc2_normals_opt = list_to_padded(pc1_normals_opt).to(device='cuda'), list_to_padded(pc2_normals_opt).to(device='cuda')

        self.mask1, self.mask2 = mask1, mask2
        self.seg1, self.seg2 = seg1, seg2
        self.gt_R_ego, self.gt_t_ego = torch.stack(R_ego).transpose(-1, -2).to('cuda'), torch.stack(t_ego).to('cuda')
        self.gt_ego_transform = rsf_utils.get_rigid_transform(self.gt_R_ego, self.gt_t_ego)
        self.sf = sf
        self.predicted_flow, self.segmentation, self.motion_parameters = None, None, None
        self.updated = True

        if init_perbox is None:
            if config['cycle']:
                perbox_params = np.tile(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1.1, 0, 0, .9, 0, 0, 0, 1.1, 0, 0, .9, 0, 0]), (self.batch_size, self.num_boxes, 1))
            else:
                perbox_params = np.tile(np.array([0,0,0,0,0,0,0,0,0,1.1,0,0,.9,0,0]), (self.batch_size, self.num_boxes, 1))
            self.perbox_params = torch.tensor(perbox_params, requires_grad=True, device='cuda', dtype=torch.float32)
        else:
            self.perbox_params = init_perbox
        if init_global is None:
            if use_gt_ego:
                self.global_params = torch.cat([torch.stack(R_ego).transpose(-1, -2).to('cuda').reshape(len(R_ego), -1), torch.stack(t_ego).to('cuda')], dim=-1)
            elif icp_init:
                icp_output = iterative_closest_point(self.pc1_opt, self.pc2_opt)
                R_icp, t_icp, scale_icp = icp_output[3]
                self.global_params = torch.tensor(np.concatenate([R_icp.detach().cpu().numpy().reshape(R_icp.shape[0], -1),
                                        t_icp.detach().cpu().numpy()], axis=-1), requires_grad=True, device='cuda', dtype=torch.float32)
            else:
                self.global_params = torch.tensor([[1.1,0,0,0,1,0,0,0,.9,0,0,0]]*self.batch_size, requires_grad=True, device='cuda', dtype=torch.float32)
        else:
            self.global_params = init_global
        if use_gt_ego:
            self.opt = optim.Adam([self.perbox_params], lr=config['lr'])
        else:
            self.opt = optim.Adam([self.global_params, self.perbox_params], lr = config['lr'])
        if config['cycle']:
            self.loss_function = RSFLossCycle(anchors, config)
        else:
            self.loss_function = RSFLossv2(anchors, config)

    def optimize(self, epochs):
        for j in range(epochs):
            self.opt.zero_grad()
            loss = self.loss_function(self.pc1_opt, self.pc2_opt, self.pc1_normals_opt, self.pc2_normals_opt, self.global_params, self.perbox_params)
            if self.config['print_loss']:
                print(loss['total_loss'].item())
            loss['total_loss'].backward()
            self.opt.step()
        self.updated = True

    def predict(self):
        if self.updated:
            output_flow, output_seg, output_params = [], [], []
            with torch.no_grad():
                for vis_idx in range(self.batch_size):
                    predicted_flow, segmentation, motion_parameters = flow_inference(self.pc1.points_list()[vis_idx], self.global_params[vis_idx],
                                                                                    self.perbox_params[vis_idx], self.anchors, self.config, cc=False, cycle=self.config['cycle'])
                    output_flow.append(predicted_flow)
                    output_seg.append(segmentation)
                    output_params.append(motion_parameters)
            self.predicted_flow, self.segmentation, self.motion_parameters = output_flow, output_seg, output_params
            self.updated = False
        return self.predicted_flow, self.segmentation, self.motion_parameters

    def evaluate_flow(self):
        errors = defaultdict(list)
        predicted_flow_batch, segmentation_batch, motion_parameters_batch = self.predict()
        with torch.no_grad():
            for vis_idx, predicted_flow in enumerate(predicted_flow_batch):
                gt_sf = self.sf[vis_idx].to('cuda')
                metrics = rsf_utils.compute_epe(predicted_flow[self.mask1[vis_idx]], gt_sf, eval_stats=True)
                for k, v in metrics.items():
                    errors[k].append(v)
        return errors

    def evaluate_segmentation(self):
        errors = defaultdict(list)
        predicted_flow_batch, segmentation_batch, motion_parameters_batch = self.predict()
        with torch.no_grad():
            for vis_idx, segmentation in enumerate(segmentation_batch):
                gt_seg1 = self.seg1[vis_idx].to('cuda')
                precision_f, precision_b, recall_f, recall_b, accuracy, tp, fp, fn, tn = rsf_utils.precision_at_one(segmentation > 0, gt_seg1)
                errors['precision_f'].append(precision_f.item())
                errors['precision_b'].append(precision_b.item())
                errors['recall_f'].append(recall_f.item())
                errors['recall_b'].append(recall_b.item())
                errors['accuracy'].append(accuracy.item())
                errors['contains_moving'].append(torch.sum(gt_seg1).item()>0)
                errors['tp'].append(tp.item())
                errors['fp'].append(fp.item())
                errors['fn'].append(fn.item())
                errors['tn'].append(tn.item())
        return errors

    def evaluate_ego(self):
        ego_transform = rsf_utils.global_params2Rt(self.global_params)
        R_ego, t_ego = ego_transform.get_matrix()[:,:3,:3], ego_transform.get_matrix()[:,3,:3]
        rot_error = torch.abs(torch.rad2deg(rsf_utils.so3_relative_angle(R_ego, self.gt_R_ego)))
        trans_error = torch.linalg.norm(t_ego - self.gt_t_ego, dim=-1)
        return {'R_ego_error':rot_error.tolist(), 't_ego_error':trans_error.tolist(), 'contains_moving':[torch.sum(s).item()>0 for s in self.seg1]}

    def evaluate_chamfer(self):
        warped_pc_batch = []
        predicted_flow_batch, segmentation_batch, motion_parameters_batch = self.predict()
        with torch.no_grad():
            for vis_idx, predicted_flow in enumerate(predicted_flow_batch):
                pc1_eval = self.pc1.points_list()[vis_idx]
                warped_pc = pc1_eval + predicted_flow
                warped_pc_batch.append(warped_pc)
            warped_pc_batch = Pointclouds(warped_pc_batch)
            warped_normals = estimate_pointcloud_normals(warped_pc_batch, neighborhood_size=self.config['k_normals'])
            cat1 = torch.cat((warped_pc_batch.points_padded(), warped_normals), dim=-1)
            cat2 = torch.cat((self.pc2.points_padded(), self.pc2_normals), dim=-1)
            knn1 = knn_points(cat1, cat2, warped_pc_batch.num_points_per_cloud(), self.pc2.num_points_per_cloud())[0].squeeze(-1)
            knn2 = knn_points(cat2, cat1, self.pc2.num_points_per_cloud(), warped_pc_batch.num_points_per_cloud())[0].squeeze(-1)
            knn1 = [k[torch.nonzero(self.mask1[i])] for i, k in enumerate(knn1)]
            knn2 = [k[torch.nonzero(self.mask2[i])] for i, k in enumerate(knn2)]
            cd = [torch.mean(torch.cat(k, dim=0)).item() for k in zip(knn1, knn2)]
        return cd

    def visualize(self):
        with torch.no_grad():
            batch_prediction = self.predict()
            for vis_idx, prediction in enumerate(zip(*batch_prediction)):
                predicted_flow, segmentation, motion_parameters = prediction
                gt_sf = self.sf[vis_idx].to('cuda')
                pc1_eval, pc2_eval = self.pc1.points_list()[vis_idx], self.pc2.points_list()[vis_idx]
                ego_transform = rsf_utils.global_params2Rt(self.global_params)
                boxes, box_transform = rsf_utils.perbox_params2boxesRt(self.perbox_params, self.anchors)
                box_transform = transforms.Transform3d(
                    matrix=ego_transform.get_matrix().repeat_interleave(self.num_boxes, dim=0)).compose(box_transform)
                ego_transform = ego_transform[vis_idx]
                boxes = boxes[vis_idx * self.num_boxes:(vis_idx + 1) * self.num_boxes]
                box_transform = box_transform[vis_idx * self.num_boxes:(vis_idx + 1) * self.num_boxes]

                print('Optimized boxes')
                vis_boxes_from_params(pc1_eval, pc2_eval, ego_transform, boxes, box_transform)
                _, _, pred_motion_params = self.predict()
                print('Inferred boxes')
                vis_boxes_from_params(pc1_eval, pc2_eval, pred_motion_params[vis_idx]['ego_transform'], pred_motion_params[vis_idx]['boxes'],
                                      pred_motion_params[vis_idx]['box_transform'])
                print('Inferred flow')
                vis_flow(pc1_eval.detach().cpu().numpy(), pc2=pc2_eval.detach().cpu().numpy(), sf=predicted_flow.detach().cpu().numpy())
                if not torch.all(gt_sf == torch.zeros_like(gt_sf)):
                    print('Ground truth flow')
                    vis_flow(pc1_eval[self.mask1[vis_idx]].detach().cpu().numpy(),
                             pc2=pc2_eval[self.mask2[vis_idx]].detach().cpu().numpy(), sf=gt_sf.detach().cpu().numpy())
                    epe = torch.linalg.norm(gt_sf-predicted_flow[self.mask1[vis_idx]], dim=1)
                    print('EPE')
                    vis_flow(pc1_eval[self.mask1[vis_idx]].detach().cpu().numpy(), color=epe.detach().cpu().numpy())


def optimize(cfg):
    dataset_map = {'StereoKITTI_ME': 'stereo', 'SemanticKITTI_ME': 'semantic', 'LidarKITTI_ME': 'lidar', 'NuScenes_ME': 'nuscenes'}
    dataset = dataset_map[cfg['data']['dataset']]
    hyperparameters = cfg['hyperparameters']

    ##### GENERATE ANCHORS #####
    max_depth = 33
    min_depth = 2
    box_depth = hyperparameters['box_depth'] #6
    box_width = box_depth
    z_center = -1
    box_scale = hyperparameters['box_scale'] #1.25
    anchor_width = 1.6*box_scale
    anchor_length = 3.9*box_scale
    anchor_height = 1.5*box_scale

    anchors = []

    if dataset == 'stereo':
        for i, depth in enumerate(np.arange(min_depth, max_depth, box_depth)):
            row = torch.cat([torch.tensor([[x_coord, depth+box_depth/2, z_center, anchor_width, anchor_length, anchor_height, 0]])#, [x_coord, depth+box_depth/2, z_center, anchor_width, anchor_length, anchor_height, np.pi/2]])
                             for x_coord in np.arange(-1*i*box_width, (i+1)*box_width, 2*box_width)], dim = 0)
            anchors.append(row)

        anchors = torch.cat(anchors, dim=0)

    elif dataset == 'lidar':
        anchor_x = torch.arange(-34, 34, 4, dtype=torch.float32)
        anchor_y = torch.arange(-34, 34, 6, dtype=torch.float32)
        anchors_xy = torch.stack(torch.meshgrid(anchor_x, anchor_y), dim=-1)
        offsets = torch.tensor([[0, 3], [0, 0]]).repeat(anchors_xy.shape[0] // 2, 1)
        if anchors_xy.shape[0] % 2 != 0:
            offsets = torch.cat((offsets, torch.tensor([[0, 3]])), dim=0)
        anchors_xy += offsets.unsqueeze(1)
        anchors_xy = anchors_xy.view(-1, 2)
        anchors_xy -= torch.mean(anchors_xy, dim=0, keepdim=True)
        anchors = torch.cat((anchors_xy, torch.stack([torch.tensor([z_center, anchor_width, anchor_length, anchor_height, 0])] * anchors_xy.shape[0], dim=0)), dim=1)

    elif dataset == 'semantic':
        anchor_x = torch.arange(-34, 34, 4, dtype=torch.float32)
        anchor_y = torch.arange(-34, 34, 6, dtype=torch.float32)
        anchors_xy = torch.stack(torch.meshgrid(anchor_x, anchor_y), dim=-1)
        offsets = torch.tensor([[0,3],[0,0]]).repeat(anchors_xy.shape[0]//2, 1)
        if anchors_xy.shape[0] % 2 != 0:
            offsets = torch.cat((offsets, torch.tensor([[0,3]])), dim=0)
        anchors_xy += offsets.unsqueeze(1)
        anchors_xy = anchors_xy.view(-1,2)
        anchors_xy -= torch.mean(anchors_xy, dim=0, keepdim=True)
        anchors = torch.cat((anchors_xy, torch.stack([torch.tensor([z_center, anchor_width, anchor_length, anchor_height, 0])]*anchors_xy.shape[0], dim=0)), dim=1)

    elif dataset == 'nuscenes':
        anchor_x = torch.arange(-34, 34, 3, dtype=torch.float32)
        anchor_y = torch.arange(-34, 34, 6, dtype=torch.float32)
        anchors_xy = torch.stack(torch.meshgrid(anchor_x, anchor_y), dim=-1)
        offsets = torch.tensor([[0,3],[0,0]]).repeat(anchors_xy.shape[0]//2, 1)
        if anchors_xy.shape[0] % 2 != 0:
            offsets = torch.cat((offsets, torch.tensor([[0,3]])), dim=0)
        anchors_xy += offsets.unsqueeze(1)
        anchors_xy = anchors_xy.view(-1,2)
        anchors_xy -= torch.mean(anchors_xy, dim=0, keepdim=True)
        anchors = torch.cat((anchors_xy, torch.stack([torch.tensor([z_center, anchor_width, anchor_length, anchor_height, 0])]*anchors_xy.shape[0], dim=0)), dim=1)

    anchors = anchors.float().to(device='cuda')

    data = make_data_loader(cfg, phase='test')

    errors = defaultdict(list)

    for i, batch in enumerate(data):
        pc1, pc2, pc1_normals, pc2_normals, mask1, mask2, seg1, seg2, R_ego, t_ego, sf, filename = batch
        optimizer = SF_Optimizer(anchors, hyperparameters, pc1, pc2, pc1_normals, pc2_normals, mask1, mask2, seg1, seg2, R_ego, t_ego, sf)
        optimizer.optimize(hyperparameters['epochs'])

        if dataset == 'stereo' or dataset == 'lidar' or dataset == 'nuscenes':
            metrics = optimizer.evaluate_flow()
            print(str(i) + ' EPE: ' + str(metrics['epe']))
            for k, v in metrics.items():
                errors[k]+=v
        elif dataset == 'semantic':
            metrics = optimizer.evaluate_ego()
            print(str(i) + ' ego:' + str(metrics))
            for k, v in metrics.items():
                errors[k]+=v
        metrics = optimizer.evaluate_segmentation()
        print(str(i) + ' segmentation:' + str(metrics))
        for k, v in metrics.items():
            errors[k]+=v

        if cfg['misc']['visualize']:
            optimizer.visualize()
            break
        else:
            file = open(args.error_filename + '.pkl', 'wb')
            pickle.dump(errors, file)

    weights = np.array(errors['n'])
    total = np.sum(weights)
    for k, v in errors.items():
        output = np.sum(np.array(v) * weights) / total
        print(k + ' : ' + str(output))
    piou = np.sum(errors['tp']) / (np.sum(errors['tp']) + np.sum(errors['fp']) + np.sum(errors['fn']))
    print('IOU : ' + str(piou))
    niou = np.sum(errors['tn']) / (np.sum(errors['tn']) + np.sum(errors['fp']) + np.sum(errors['fn']))
    miou = .5*(piou+niou)
    print('mIOU : ' + str(miou))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--error_filename', type=str, default='errors_file')
    parser.add_argument('--cfg', type=str, default='configs/stereo_cfg.yaml')
    args = parser.parse_args()

    with open(args.cfg) as file:
        cfg = yaml.safe_load(file)

    optimize(cfg)

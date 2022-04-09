import numpy as np
import torch
import torch.optim as optim
import yaml
from lidarkitti import make_data_loader
import matplotlib.pyplot as plt
from visualization import vis_boxes_from_params, vis_flow, vis_boxes
import utils
from loss import ObjectLoss
from inference import flow_inference
from pytorch3d.structures import Pointclouds
from pytorch3d.ops import estimate_pointcloud_normals
from pytorch3d.ops.knn import knn_points
from pytorch3d import transforms
import argparse
from collections import defaultdict

class SF_Optimizer:
    def __init__(self, anchors, config, pc1, pc2, pc1_normals, pc2_normals, mask1, mask2, seg1, seg2, R_ego, t_ego, sf, filename):
        self.anchors = anchors
        self.num_boxes = anchors.shape[0]
        self.config = config
        self.batch_size = len(pc1)
        perbox_params = np.tile(np.array([0,0,0,0,0,0,0,0,0,1.1,0,0,.9,0,0]), (self.batch_size, self.num_boxes, 1))
        self.perbox_params = torch.tensor(perbox_params, requires_grad=True, device='cuda', dtype=torch.float32)
        self.global_params = torch.tensor([[1.1,0,0,0,1,0,0,0,.9,0,0,0]]*self.batch_size, requires_grad=True, device='cuda', dtype=torch.float32)
        self.opt = optim.Adam([self.global_params, self.perbox_params], lr = config['lr'])
        self.loss_function = ObjectLoss(anchors)
        self.pc1, self.pc2 = pc1, pc2
        self.pc1_normals, self.pc2_normals = pc1_normals, pc2_normals
        self.mask1, self.mask2 = mask1, mask2
        self.seg1, self.seg2 = seg1, seg2
        self.gt_R_ego, self.gt_t_ego = torch.stack(R_ego).transpose(-1, -2).to('cuda'), torch.stack(t_ego).to('cuda')
        self.gt_ego_transform = utils.get_rigid_transform(self.gt_R_ego, self.gt_t_ego)
        self.sf = sf
        self.filename = filename
        self.predicted_flow, self.segmentation, self.motion_parameters = None, None, None
        self.updated = True

    def optimize(self, epochs):
        for j in range(epochs):
            self.opt.zero_grad()
            loss = self.loss_function(self.pc1, self.pc2, self.pc1_normals, self.pc2_normals, self.global_params, self.perbox_params, self.config)
            if self.config['print_loss']:
                print(loss.item())
            loss.backward()
            self.opt.step()
        self.updated = True

    def predict(self):
        if self.updated:
            output_flow, output_seg, output_params = [], [], []
            with torch.no_grad():
                for vis_idx in range(self.batch_size):
                    pc1_eval, pc2_eval = self.pc1.points_list()[vis_idx], self.pc2.points_list()[vis_idx]
                    ego_transform = utils.global_params2Rt(self.global_params)
                    boxes, box_transform = utils.perbox_params2boxesRt(self.perbox_params, self.anchors)
                    box_transform = transforms.Transform3d(
                        matrix=ego_transform.get_matrix().repeat_interleave(self.num_boxes, dim=0)).compose(box_transform)
                    ego_transform = ego_transform[vis_idx]
                    boxes = boxes[vis_idx * self.num_boxes:(vis_idx + 1) * self.num_boxes]
                    box_transform = box_transform[vis_idx * self.num_boxes:(vis_idx + 1) * self.num_boxes]
                    predicted_flow, segmentation, motion_parameters = flow_inference(pc1_eval, ego_transform, boxes, box_transform, self.config, cc=False)
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
                metrics = utils.compute_epe(predicted_flow[self.mask1[vis_idx]], gt_sf, eval_stats=True)
                for k, v in metrics.items():
                    errors[k].append(v)
        return errors

    def evaluate_segmentation(self):
        errors = defaultdict(list)
        predicted_flow_batch, segmentation_batch, motion_parameters_batch = self.predict()
        with torch.no_grad():
            for vis_idx, segmentation in enumerate(segmentation_batch):
                gt_seg1 = self.seg1[vis_idx].to('cuda')
                precision_f, precision_b, recall_f, recall_b, accuracy = utils.precision_at_one(segmentation> 0, gt_seg1)
                errors['precision_f'].append(precision_f.item())
                errors['precision_b'].append(precision_b.item())
                errors['recall_f'].append(recall_f.item())
                errors['recall_b'].append(recall_b.item())
                errors['accuracy'].append(accuracy.item())
                errors['contains_moving'].append(torch.sum(gt_seg1).item()>0)
        return errors

    def evaluate_ego(self):
        ego_transform = utils.global_params2Rt(self.global_params)
        R_ego, t_ego = ego_transform.get_matrix()[:,:3,:3], ego_transform.get_matrix()[:,3,:3]
        rot_error = torch.abs(torch.rad2deg(utils.so3_relative_angle(R_ego, self.gt_R_ego)))
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
            for vis_idx in range(self.batch_size):
                gt_sf = self.sf[vis_idx].to('cuda')
                pc1_eval, pc2_eval = self.pc1.points_list()[vis_idx], self.pc2.points_list()[vis_idx]
                ego_transform = utils.global_params2Rt(self.global_params)
                boxes, box_transform = utils.perbox_params2boxesRt(self.perbox_params, self.anchors)
                box_transform = transforms.Transform3d(
                    matrix=ego_transform.get_matrix().repeat_interleave(self.num_boxes, dim=0)).compose(box_transform)
                ego_transform = ego_transform[vis_idx]
                boxes = boxes[vis_idx * self.num_boxes:(vis_idx + 1) * self.num_boxes]
                box_transform = box_transform[vis_idx * self.num_boxes:(vis_idx + 1) * self.num_boxes]
                predicted_flow, segmentation, motion_parameters = flow_inference(pc1_eval, ego_transform, boxes, box_transform, self.config, cc=False)

                vis_flow(pc1_eval.detach().cpu().numpy(), pc2_eval.detach().cpu().numpy(), sf=predicted_flow.detach().cpu().numpy())
                if not torch.all(gt_sf == torch.zeros_like(gt_sf)):
                    epe = torch.linalg.norm(gt_sf-predicted_flow[self.mask1[vis_idx]], dim=1)
                    vis_flow(pc1_eval.detach().cpu().numpy()[self.mask1[vis_idx]],
                             pc2_eval.detach().cpu().numpy()[self.mask2[vis_idx]],
                             sf=predicted_flow.detach().cpu().numpy()[self.mask1[vis_idx]],
                             sf_color=epe.detach().cpu().numpy())

def optimize(config):
    ##### GENERATE ANCHORS #####
    max_depth = 33
    min_depth = 2
    z_center = -1
    box_scale = 1.25
    anchor_width = 1.6*box_scale
    anchor_length = 3.9*box_scale
    anchor_height = 1.5*box_scale

    anchors = []

    if dataset == 'stereo':
        box_depth = 4
        box_width = 4
        for i, depth in enumerate(np.arange(min_depth, max_depth, box_depth)):
            row = torch.cat([torch.tensor([[x_coord, depth+box_depth/2, z_center, anchor_width, anchor_length, anchor_height, 0]])#, [x_coord, depth+box_depth/2, z_center, anchor_width, anchor_length, anchor_height, np.pi/2]])
                             for x_coord in np.arange(-1*i*box_width, (i+1)*box_width, 2*box_width)], dim = 0)
            anchors.append(row)
        anchors = torch.cat(anchors, dim=0)

    elif dataset == 'lidar' or dataset=='semantic':
        anchor_x = torch.arange(-34, 34, 4, dtype=torch.float32)
        anchor_x = anchor_x-torch.mean(anchor_x)
        anchor_y = torch.arange(-34, 34, 6, dtype=torch.float32)
        anchor_y = anchor_y-torch.mean(anchor_y)
        anchors_xy = torch.stack(torch.meshgrid(anchor_x, anchor_y), dim=-1).view(-1, 2)
        anchors = torch.cat((anchors_xy, torch.stack([torch.tensor([z_center, anchor_width, anchor_length, anchor_height, 0])]*anchors_xy.shape[0], dim=0)), dim=1)

    anchors = anchors.float().to(device='cuda')

    data = make_data_loader(cfg, phase='test')

    errors = defaultdict(list)

    for i, batch in enumerate(data):

        pc1, pc2, mask1, mask2, seg1, seg2, R_ego, t_ego, sf, filename = batch

        pc1, pc2 = Pointclouds(pc1).to(device='cuda'), Pointclouds(pc2).to(device='cuda')
        pc1_normals, pc2_normals = estimate_pointcloud_normals(pc1, neighborhood_size=config['k_normals']), estimate_pointcloud_normals(pc2, neighborhood_size=config['k_normals'])

        optimizer = SF_Optimizer(anchors, config, pc1, pc2, pc1_normals, pc2_normals, mask1, mask2, seg1, seg2, R_ego, t_ego, sf, filename)
        optimizer.optimize(config['epochs'])
        if dataset == 'stereo' or dataset == 'lidar':
            metrics = optimizer.evaluate_flow()
            print(str(i) + ': ' + str(metrics['epe']))
            for k, v in metrics.items():
                errors[k]+=v
        elif dataset == 'semantic':
            metrics = optimizer.evaluate_segmentation()
            print(str(i) + ':' + str(metrics))
            for k, v in metrics.items():
                errors[k]+=v
            metrics = optimizer.evaluate_ego()
            print(str(i) + ':' + str(metrics))
            for k, v in metrics.items():
                errors[k]+=v

        if visualize:
            optimizer.visualize()

    for k, v in errors.items():
        print(k + ' : ' + str(np.mean(v)))


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type= str, default = 'stereo')
parser.add_argument('--visualize', type = bool, default = False)
args = parser.parse_args()

visualize = args.visualize
dataset = args.dataset

if dataset == 'stereo':
    with open(r'stereo_cfg.yaml') as file:
        cfg = yaml.safe_load(file)
elif dataset == 'lidar':
    with open(r'lidar_cfg.yaml') as file:
        cfg = yaml.safe_load(file)

config = cfg['hyperparameters']
optimize(config)

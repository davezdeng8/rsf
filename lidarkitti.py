import os
import torch
import logging

import numpy as np
import torch.utils.data as data

def normal_frame(points):
    x = -points[...,0:1]
    y = points[...,2:3]
    z = points[...,1:2]
    return np.concatenate((x,y,z), axis = -1)

def rot_normal_frame(R):
    """
    Converts a rotation matrix R into normal coordinates
    :param R: 3x3 numpy array
    :return: 3x3 numpy array
    """
    return np.array([[R[0,0], -R[0,2], -R[0,1]], [-R[2,0], R[2,2], R[2,1]], [-R[1,0], R[1,2], R[1,1]]])

def collate_fn(data):
    output = [[to_tensor(data[j][i]) for j in range(len(data))] for i in range(len(data[0]))]
    return tuple(output)

def to_tensor(x):
    if isinstance(x, torch.Tensor):
        return x
    elif isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    elif isinstance(x, str):
        return x
    else:
        raise ValueError("Can not convert to torch tensor {}".format(x))

class MELidarDataset(data.Dataset):
    def __init__(self, phase, config):

        self.files = []
        self.root = config['data']['root']
        self.config = config
        self.num_points = config['misc']['num_points']
        self.remove_ground = True if (
                    config['data']['remove_ground'] and config['data']['dataset'] in ['StereoKITTI_ME', 'LidarKITTI_ME',
                                                                                      'SemanticKITTI_ME',
                                                                                      'WaymoOpen_ME']) else False
        self.only_front_points = ('only_front_points' in config['data']) and (config['data']['only_front_points'])
        self.dataset = config['data']['dataset']
        self.only_near_points = config['data']['only_near_points']
        self.phase = phase

        self.randng = np.random.RandomState()
        self.device = torch.device('cuda' if (torch.cuda.is_available() and config['misc']['use_gpu']) else 'cpu')

        self.augment_data = config['data']['augment_data']

        logging.info("Loading the subset {} from {}".format(phase, self.root))

        subset_names = open(self.DATA_FILES[phase]).read().split()

        for name in subset_names:
            self.files.append(name)


    def __getitem__(self, idx):
        file = os.path.join(self.root, self.files[idx])
        file_name = file.replace(os.sep, '/').split('/')[-1]

        # Load the data
        data = np.load(file)
        pc_1 = data['pc1']
        pc_2 = data['pc2']

        if 'pc1_cam_mask' in data:
            pc1_cam_mask = data['pc1_cam_mask']
        elif 'front_mask_s' in data:
            pc1_cam_mask = data['front_mask_s']
        else:
            pc1_cam_mask = np.ones(pc_1[:,0].shape, dtype=np.bool)

        if 'pc2_cam_mask' in data:
            pc2_cam_mask = data['pc2_cam_mask']
        elif 'front_mask_t' in data:
            pc2_cam_mask = data['front_mask_t']
        else:
            pc2_cam_mask = np.ones(pc_2[:,0].shape, dtype=np.bool)

        if 'pose_s' in data:
            pose_1 = data['pose_s']
        else:
            pose_1 = np.eye(4)

        if 'pose_t' in data:
            pose_2 = data['pose_t']
        else:
            pose_2 = np.eye(4)

        if 'mot_label_s' in data:
            labels_1 = data['mot_label_s']
        else:
            labels_1 = np.zeros(pc_1.shape[0])

        if 'mot_label_t' in data:
            labels_2 = data['mot_label_t']
        else:
            labels_2 = np.zeros(pc_2.shape[0])

        if 'flow' in data:
            flow = data['flow']
        else:
            flow = np.zeros((np.sum(pc1_cam_mask), 3), dtype=pc_1.dtype)

        # Remove the ground and far away points
        # In stereoKITTI the direct correspondences are provided therefore we remove,
        # if either of the points fullfills the condition (as in hplflownet, flot, ...)

        if self.dataset in ["SemanticKITTI_ME", 'LidarKITTI_ME', "WaymoOpen_ME"]:
            if self.remove_ground:
                if self.phase == 'test':
                    is_not_ground_s = (pc_1[:, 1] > -1.4)
                    is_not_ground_t = (pc_2[:, 1] > -1.4)

                    pc_1 = pc_1[is_not_ground_s, :]
                    labels_1 = labels_1[is_not_ground_s]
                    flow = flow[is_not_ground_s[pc1_cam_mask], :]
                    pc1_cam_mask = pc1_cam_mask[is_not_ground_s]

                    pc_2 = pc_2[is_not_ground_t, :]
                    labels_2 = labels_2[is_not_ground_t]
                    pc2_cam_mask = pc2_cam_mask[is_not_ground_t]

                # In the training phase we randomly select if the ground should be removed or not
                elif np.random.rand() > 1 / 4:
                    is_not_ground_s = (pc_1[:, 1] > -1.4)
                    is_not_ground_t = (pc_2[:, 1] > -1.4)

                    pc_1 = pc_1[is_not_ground_s, :]
                    labels_1 = labels_1[is_not_ground_s]
                    flow = flow[is_not_ground_s[pc1_cam_mask], :]
                    pc1_cam_mask = pc1_cam_mask[is_not_ground_s]

                    pc_2 = pc_2[is_not_ground_t, :]
                    labels_2 = labels_2[is_not_ground_t]
                    pc2_cam_mask = pc2_cam_mask[is_not_ground_t]

            if self.only_near_points:
                is_near_s = (np.amax(np.abs(pc_1), axis=1) < 35)
                is_near_t = (np.amax(np.abs(pc_2), axis=1) < 35)

                pc_1 = pc_1[is_near_s, :]
                labels_1 = labels_1[is_near_s]
                flow = flow[is_near_s[pc1_cam_mask], :]
                pc1_cam_mask = pc1_cam_mask[is_near_s]

                pc_2 = pc_2[is_near_t, :]
                labels_2 = labels_2[is_near_t]
                pc2_cam_mask = pc2_cam_mask[is_near_t]

            if self.only_front_points:
                is_front_s = pc_1[:, 2]>=(np.abs(pc_1[:, 0])-8)
                is_front_t = pc_2[:, 2]>=(np.abs(pc_2[:, 0])-8)

                pc_1 = pc_1[is_front_s, :]
                labels_1 = labels_1[is_front_s]
                flow = flow[is_front_s[pc1_cam_mask], :]
                pc1_cam_mask = pc1_cam_mask[is_front_s]

                pc_2 = pc_2[is_front_t, :]
                labels_2 = labels_2[is_front_t]
                pc2_cam_mask = pc2_cam_mask[is_front_t]

        else:
            if self.remove_ground:
                is_not_ground = np.logical_not(np.logical_and(pc_1[:, 1] < -1.4, pc_2[:, 1] < -1.4))
                pc_1 = pc_1[is_not_ground, :]
                pc_2 = pc_2[is_not_ground, :]
                flow = flow[is_not_ground, :]

            if self.only_near_points:
                is_near = np.logical_and(pc_1[:, 2] < 35, pc_1[:, 2] < 35)
                pc_1 = pc_1[is_near, :]
                pc_2 = pc_2[is_near, :]
                flow = flow[is_near, :]

        # Augment the point cloud by randomly rotating and translating them (recompute the ego-motion if augmention is applied!)
        if self.augment_data and self.phase != 'test':
            T_1 = np.eye(4)
            T_2 = np.eye(4)

            T_1[0:3, 3] = (np.random.rand(3) - 0.5) * 0.5
            T_2[0:3, 3] = (np.random.rand(3) - 0.5) * 0.5

            T_1[1, 3] = (np.random.rand(1) - 0.5) * 0.1
            T_2[1, 3] = (np.random.rand(1) - 0.5) * 0.1

            pc_1 = (np.matmul(T_1[0:3, 0:3], pc_1.transpose()) + T_1[0:3, 3:4]).transpose()
            pc_2 = (np.matmul(T_2[0:3, 0:3], pc_2.transpose()) + T_2[0:3, 3:4]).transpose()

            pose_1 = np.matmul(pose_1, np.linalg.inv(T_1))
            pose_2 = np.matmul(pose_2, np.linalg.inv(T_2))

            rel_trans = np.linalg.inv(pose_2) @ pose_1

            R_ego = rel_trans[0:3, 0:3]
            t_ego = rel_trans[0:3, 3:4]
        else:
            # Compute relative pose that transform the point from the source point cloud to the target
            rel_trans = np.linalg.inv(pose_2) @ pose_1
            R_ego = rel_trans[0:3, 0:3]
            t_ego = rel_trans[0:3, 3:4]

        # Sample n points for evaluation before the voxelization
        # If less than desired points are available just consider the maximum
        if pc_1.shape[0] > self.num_points:
            idx_1 = np.random.choice(pc_1.shape[0], self.num_points, replace=False)
        else:
            idx_1 = np.random.choice(pc_1.shape[0], pc_1.shape[0], replace=False)

        if pc_2.shape[0] > self.num_points:
            idx_2 = np.random.choice(pc_2.shape[0], self.num_points, replace=False)
        else:
            idx_2 = np.random.choice(pc_2.shape[0], pc_2.shape[0], replace=False)

        pc_1_eval = pc_1[idx_1, :]
        flow_idx = np.cumsum(pc1_cam_mask)-1
        flow_idx = flow_idx[idx_1[pc1_cam_mask[idx_1]]]
        assert np.all(flow_idx>=0)
        flow_eval = flow[flow_idx]
        labels_1_eval = labels_1[idx_1]
        pc1_cam_mask = pc1_cam_mask[idx_1]

        pc_2_eval = pc_2[idx_2, :]
        labels_2_eval = labels_2[idx_2]
        pc2_cam_mask = pc2_cam_mask[idx_2]

        pc_1_eval = normal_frame(pc_1_eval).astype(np.float32)
        pc_2_eval = normal_frame(pc_2_eval).astype(np.float32)
        flow_eval = normal_frame(flow_eval).astype(np.float32)
        labels_1_eval = labels_1_eval.astype(np.float32)
        labels_2_eval = labels_2_eval.astype(np.float32)

        R_ego = np.transpose(rot_normal_frame(R_ego)).astype(np.float32)
        t_ego = normal_frame(t_ego.reshape(3)).astype(np.float32)

        return (pc_1_eval, pc_2_eval, pc1_cam_mask, pc2_cam_mask, labels_1_eval, labels_2_eval, R_ego, t_ego, flow_eval, file)

    def __len__(self):
        return len(self.files)

    def reset_seed(self, seed=41):
        logging.info('Resetting the data loader seed to {}'.format(seed))
        self.randng.seed(seed)


class StereoKITTI_ME(MELidarDataset):
    # 3D Match dataset all files
    DATA_FILES = {
        'test': 'kittisf_file_list.txt'
    }

class LidarKITTI_ME(MELidarDataset):
    # 3D Match dataset all files
    DATA_FILES = {
        'test': 'kittisf_file_list.txt'
    }

# Map the datasets to string names
ALL_DATASETS = [StereoKITTI_ME, LidarKITTI_ME]

dataset_str_mapping = {d.__name__: d for d in ALL_DATASETS}


def make_data_loader(config, phase, neighborhood_limits=None, shuffle_dataset=None):
    """
    Defines the data loader based on the parameters specified in the config file
    Args:
        config (dict): dictionary of the arguments
        phase (str): phase for which the data loader should be initialized in [train,val,test]
        shuffle_dataset (bool): shuffle the dataset or not
    Returns:
        loader (torch data loader): data loader that handles loading the data to the model
    """

    assert config['misc']['run_mode'] in ['train', 'val', 'test']

    if shuffle_dataset is None:
        shuffle_dataset = config['misc']['run_mode'] != 'test'

    # Select the defined dataset
    Dataset = dataset_str_mapping[config['data']['dataset']]

    dset = Dataset(phase, config=config)

    drop_last = False if config['misc']['run_mode'] == 'test' else True

    loader = torch.utils.data.DataLoader(
        dset,
        batch_size=config[phase]['batch_size'],
        shuffle=shuffle_dataset,
        num_workers=config[phase]['num_workers'],
        collate_fn=collate_fn,
        pin_memory=False,
        drop_last=drop_last
    )

    return loader
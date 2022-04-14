import torch
import torch.utils.data as data
import trimesh
import json
import numpy as np
import os
import tqdm
import utils.general as utils


class CRXDataSet(data.Dataset):

    def __init__(self, dataset_path, split, points_batch=1024, d_in=3, with_gt=False, with_normals=False):

        self.dataset_path = dataset_path
        self.split = split
        self.points_batch = points_batch
        self.d_in = d_in
        self.with_gt = with_gt
        self.with_normals = with_normals

        self.load(dataset_path)

    def load_points_normals(self, index):
        return np.memmap(self.samples[index]['mesh_preproc_cached'], dtype='float32', mode='r').reshape(-1,6).astype(np.float32)

    def __getitem__(self, index):

        point_set_mnlfld = torch.from_numpy(self.load_points_normals(index)).float()

        random_idx = torch.randperm(point_set_mnlfld.shape[0])[:self.points_batch]
        point_set_mnlfld = torch.index_select(point_set_mnlfld, 0, random_idx)

        if self.with_normals:
            normals = point_set_mnlfld[:, -self.d_in:]  # todo adjust to case when we get no sigmas

        else:
            normals = torch.empty(0)

        return point_set_mnlfld[:, :self.d_in], normals, index

    def __len__(self):
        return len(self.samples)

    def get_info(self, index):
        return [self.samples[index]['case_identifier']]

    def load(self, dataset_path):
        with open(dataset_path) as f:
            self.dataset = json.load(f)
        self.samples = [s for s in self.dataset['database']['samples'] if s['case_identifier'] in self.split]
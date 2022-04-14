import os
import sys
import argparse
import json

import trimesh
import numpy as np
from tqdm import tqdm

code_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, code_dir)

import utils.general as utils

def symmetrize_mesh(mesh):
    vertices_sym = mesh.vertices.copy()
    vertices_sym[:,0] = -vertices_sym[:,0]
    faces_sym = np.concatenate((mesh.faces[:,0,None], mesh.faces[:,2,None], mesh.faces[:,1,None]), axis=-1)
    return trimesh.Trimesh(vertices_sym, faces_sym)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-path', required=True, type=str, help='Database path')
    parser.add_argument('--output-dataset-path', required=True, type=str, help='Output dataset path')

    args = parser.parse_args()

    with open(args.dataset_path) as f:
        dataset = json.load(f)

    # Symmetrize DB
    samples_dir = os.path.join(os.path.dirname(args.dataset_path), 'cases')
    symmetric_samples = []
    print('Symmetrizing DB')
    for sample in tqdm(dataset['database']['samples']):
        symmetric_sample = {}

        # Create symmetric sample directory
        symmetric_sample['case_identifier'] = sample['case_identifier'] + '_sym'
        sample_dir = os.path.join(samples_dir, symmetric_sample['case_identifier'])
        utils.mkdir_ifnotexists(sample_dir)

        # Symmetrize mesh
        mesh = trimesh.load_mesh(sample['mesh'])
        mesh_sym = symmetrize_mesh(mesh)
        symmetric_sample['mesh'] = os.path.join(sample_dir, 'mesh.obj')
        symmetric_sample['mesh_bounds'] = mesh_sym.bounds.tolist()
        mesh_sym.export(symmetric_sample['mesh'])

        # Symmetrize yaw angles
        symmetric_sample['yaw_angles'] = (-np.array(sample['yaw_angles'])).tolist()

        symmetric_samples.append(symmetric_sample)

    dataset['database']['samples'] += symmetric_samples

    # Compute stats for normalization
    min_corner = [-1,-1,-1]
    max_corner = [ 1, 1, 1]
    mean_bounds = np.zeros((2,3))
    print('Computing stats for normalization')
    for sample in tqdm(dataset['database']['samples']):
        # Compute mean center
        mean_bounds += np.array(sample['mesh_bounds']) / len(dataset['database']['samples'])
    mean_center = np.sum(mean_bounds, axis=0) / 2
    max_mean_edge = np.max(mean_bounds[1]-mean_bounds[0])
    target_center = (np.array(min_corner) + np.array(max_corner)) / 2
    target_max_edge = np.max(np.array(max_corner) - np.array(min_corner))

    displacement = target_center - mean_center
    scaling = target_max_edge / max_mean_edge

    # Normlaize samples
    print('Normalizing and caching samples')
    for sample in tqdm(dataset['database']['samples']):
        mesh_file = sample['mesh']

        # Load vertices and normals
        mesh = trimesh.load_mesh(mesh_file)

        # Normalize vertices
        point_set_mnlfld = mesh.vertices.astype(np.float32)
        point_set_mnlfld = (point_set_mnlfld + displacement) * scaling

        # Export preprocessed mesh
        mesh_preproc_file = os.path.splitext(mesh_file)[0]+'_preproc.obj'
        trimesh.Trimesh(point_set_mnlfld, mesh.faces).export(mesh_preproc_file)
        sample['mesh_preproc'] = mesh_preproc_file

        # Cache mesh for faster loading
        sample['mesh_preproc_cached'] = f'{mesh_preproc_file}.memmap'
        normals = mesh.vertex_normals.astype(np.float32)
        point_set_mnlfld = np.concatenate((point_set_mnlfld, normals), axis=-1)
        # Save memmap
        fp = np.memmap(sample['mesh_preproc_cached'], dtype='float32', mode='w+', shape=point_set_mnlfld.shape)
        fp[:] = point_set_mnlfld[:]
        del fp

    with open(args.output_dataset_path, 'w') as f:
        json.dump(dataset, f, indent=4)

    sys.exit(0)
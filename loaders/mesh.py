import numpy as np
import os
import torch
import torch.utils.data as data
import pymeshlab


def aug_mesh(face, jitter_sigma=0.01, jitter_clip=0.05):
    # jitter
    jittered_data = np.clip(jitter_sigma * np.random.randn(*face[:, :3].shape), -1 * jitter_clip, jitter_clip)
    face = np.concatenate((face[:, :3] + jittered_data, face[:, 3:]), 1)
    return face


def load_mesh(root, augement=False, name='mesh_500face', max_faces=500):
    filename = root/f'{name}.obj'
    face, neighbor_index = process_mesh(str(filename), max_faces)
    # data augmentation
    if augement:
        face = aug_mesh(face)
    # to tensor
    face = torch.from_numpy(face).float()
    neighbor_index = torch.from_numpy(neighbor_index).long()
    # reorganize
    face = face.permute(1, 0).contiguous()
    centers, corners, normals = face[:3], face[3:12], face[12:]
    corners = corners - torch.cat([centers, centers, centers], 0)

    mesh = torch.cat((centers, corners, normals), dim=0)

    return (mesh, neighbor_index)


def find_neighbor(faces, faces_contain_this_vertex, vf1, vf2, except_face):
    for i in (faces_contain_this_vertex[vf1] & faces_contain_this_vertex[vf2]):
        if i != except_face:
            face = faces[i].tolist()
            face.remove(vf1)
            face.remove(vf2)
            return i

    return except_face


def process_mesh(path, max_faces):
    ms = pymeshlab.MeshSet()
    ms.clear()

    # load mesh
    ms.load_new_mesh(path)
    mesh = ms.current_mesh()

    # # clean up
    # mesh, _ = pymesh.remove_isolated_vertices(mesh)
    # mesh, _ = pymesh.remove_duplicated_vertices(mesh)

    # get elements
    vertices = mesh.vertex_matrix()
    faces = mesh.face_matrix()

    if faces.shape[0] != max_faces:     # only occur once in train set of Manifold40
        print("Model with more than {} faces ({}): {}".format(max_faces, faces.shape[0], path))
        return None, None

    # move to center
    center = (np.max(vertices, 0) + np.min(vertices, 0)) / 2
    vertices -= center

    # normalize
    max_len = np.max(vertices[:, 0]**2 + vertices[:, 1]**2 + vertices[:, 2]**2)
    vertices /= np.sqrt(max_len)

    # get normal vector
    ms.clear()
    mesh = pymeshlab.Mesh(vertices, faces)
    ms.add_mesh(mesh)
    face_normal = ms.current_mesh().face_normal_matrix()

    # get neighbors
    faces_contain_this_vertex = []
    for i in range(len(vertices)):
        faces_contain_this_vertex.append(set([]))
    centers = []
    corners = []
    for i in range(len(faces)):
        [v1, v2, v3] = faces[i]
        x1, y1, z1 = vertices[v1]
        x2, y2, z2 = vertices[v2]
        x3, y3, z3 = vertices[v3]
        centers.append([(x1 + x2 + x3) / 3, (y1 + y2 + y3) / 3, (z1 + z2 + z3) / 3])
        corners.append([x1, y1, z1, x2, y2, z2, x3, y3, z3])
        faces_contain_this_vertex[v1].add(i)
        faces_contain_this_vertex[v2].add(i)
        faces_contain_this_vertex[v3].add(i)

    neighbors = []
    for i in range(len(faces)):
        [v1, v2, v3] = faces[i]
        n1 = find_neighbor(faces, faces_contain_this_vertex, v1, v2, i)
        n2 = find_neighbor(faces, faces_contain_this_vertex, v2, v3, i)
        n3 = find_neighbor(faces, faces_contain_this_vertex, v3, v1, i)
        neighbors.append([n1, n2, n3])

    centers = np.array(centers)
    corners = np.array(corners)
    faces = np.concatenate([centers, corners, face_normal], axis=1)
    neighbors = np.array(neighbors)

    # fill for n < max_faces with randomly picked faces
    num_point = len(faces)
    if num_point < max_faces:
        fill_face = []
        fill_neighbor_index = []
        for i in range(max_faces - num_point):
            index = np.random.randint(0, num_point)
            fill_face.append(faces[index])
            fill_neighbor_index.append(neighbors[index])
        faces = np.concatenate((faces, np.array(fill_face)))
        neighbors = np.concatenate((neighbors, np.array(fill_neighbor_index)))

    return faces, neighbors

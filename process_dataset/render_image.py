import math
import os
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np
import pyrender
import trimesh
from PIL import Image
from tqdm import tqdm

_pdir = Path(__file__).resolve().parent
if str(_pdir) not in sys.path:
    sys.path.insert(0, str(_pdir))
from obj_io import readobj, save_modified_obj


def _rotation_x_mat(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def _rotation_y_mat(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


def save(pid, data_id, vid, save_path, extr, intr, depth, img, mask,
         img_hr=None):
    img_save_path = os.path.join(save_path, 'img', data_id + '_' + '%03d' % pid)
    depth_save_path = os.path.join(save_path, 'depth',
                                   data_id + '_' + '%03d' % pid)
    mask_save_path = os.path.join(save_path, 'mask',
                                  data_id + '_' + '%03d' % pid)
    parm_save_path = os.path.join(save_path, 'parm',
                                  data_id + '_' + '%03d' % pid)
    Path(img_save_path).mkdir(exist_ok=True, parents=True)
    Path(parm_save_path).mkdir(exist_ok=True, parents=True)
    Path(mask_save_path).mkdir(exist_ok=True, parents=True)
    Path(depth_save_path).mkdir(exist_ok=True, parents=True)

    depth = depth * 2.0 ** 15
    cv2.imwrite(os.path.join(depth_save_path, '{}.png'.format(vid)),
                depth.astype(np.uint16))
    img = (np.clip(img, 0, 1) * 255.0 + 0.5).astype(np.uint8)[:, :, ::-1]
    mask = (np.clip(mask, 0, 1) * 255.0 + 0.5).astype(np.uint8)
    cv2.imwrite(os.path.join(img_save_path, '{}.jpg'.format(vid)), img)
    if img_hr is not None:
        img_hr = (np.clip(img_hr, 0, 1) * 255.0 + 0.5).astype(np.uint8)[:, :,
                 ::-1]
        cv2.imwrite(os.path.join(img_save_path, '{}_hr.jpg'.format(vid)),
                    img_hr)
    cv2.imwrite(os.path.join(mask_save_path, '{}.png'.format(vid)), mask)
    np.save(os.path.join(parm_save_path, '{}_intrinsic.npy'.format(vid)), intr)
    np.save(os.path.join(parm_save_path, '{}_extrinsic.npy'.format(vid)), extr)


def _rotation_matrix_to_pose_light_dir(forward_world):
    """DirectionalLight in pyrender shines along local -Z; align -Z with forward_world."""
    f = np.asarray(forward_world, dtype=np.float64).reshape(3)
    f = f / (np.linalg.norm(f) + 1e-12)
    z_col = -f
    up_hint = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    if abs(np.dot(up_hint, z_col)) > 0.95:
        up_hint = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    x_axis = np.cross(up_hint, z_col)
    x_axis /= np.linalg.norm(x_axis) + 1e-12
    y_axis = np.cross(z_col, x_axis)
    R = np.stack([x_axis, y_axis, z_col], axis=1)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    return T


def _taichi_camera_extrinsic(cam_pos, target, up=(0.0, -1.0, 0.0)):
    """Match taichi_three.Camera.export_extrinsic / set() convention."""
    pos = np.asarray(cam_pos, dtype=np.float64).reshape(3)
    target = np.asarray(target, dtype=np.float64).reshape(3)
    upv = np.asarray(up, dtype=np.float64).reshape(3)
    fwd = target - pos
    fwd = fwd / (np.linalg.norm(fwd) + 1e-12)
    right = np.cross(fwd, upv)
    right = right / (np.linalg.norm(right) + 1e-12)
    cam_up = np.cross(right, fwd)
    cam_up = cam_up / (np.linalg.norm(cam_up) + 1e-12)
    M = np.stack([right, cam_up, fwd], axis=1)
    R_ext = M.T
    t_ext = -R_ext @ pos
    extrinsic = np.zeros((3, 4), dtype=np.float64)
    extrinsic[:, :3] = R_ext
    extrinsic[:, 3] = t_ext
    return extrinsic, right, cam_up, fwd


def _gl_cam_pose_from_taichi(cam_pos, target, up=(0.0, -1.0, 0.0)):
    """OpenGL camera node pose (camera to world): +X right, +Y up, -Z view axis."""
    pos = np.asarray(cam_pos, dtype=np.float64).reshape(3)
    target = np.asarray(target, dtype=np.float64).reshape(3)
    upv = np.asarray(up, dtype=np.float64).reshape(3)
    fwd = target - pos
    fwd = fwd / (np.linalg.norm(fwd) + 1e-12)
    right = np.cross(fwd, upv)
    right = right / (np.linalg.norm(right) + 1e-12)
    cam_up = np.cross(right, fwd)
    cam_up = cam_up / (np.linalg.norm(cam_up) + 1e-12)
    R_cw = np.stack([right, cam_up, -fwd], axis=1)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R_cw
    T[:3, 3] = pos
    return T


def _obj_to_trimesh(obj, texture_wh_rgb_u8):
    """Build a textured trimesh from readobj() dict (unwelds corners for UVs)."""
    faces = obj['f']
    verts = obj['vi']
    if faces is None or verts is None:
        raise ValueError('OBJ must contain faces and vertices')

    if obj['vt'] is not None and texture_wh_rgb_u8 is not None:
        vt = obj['vt']
        corner_xyz = []
        corner_uv = []
        for tri in faces:
            for k in range(3):
                vi_i = int(tri[k, 0])
                vt_i = int(tri[k, 1])
                corner_xyz.append(verts[vi_i, :3])
                corner_uv.append(vt[vt_i, :2])
        corner_xyz = np.asarray(corner_xyz, dtype=np.float64)
        corner_uv = np.asarray(corner_uv, dtype=np.float64)
        corner_uv[:, 1] = 1.0 - corner_uv[:, 1]
        new_faces = np.arange(corner_xyz.shape[0], dtype=np.int64).reshape(-1, 3)
        tex_hw = np.ascontiguousarray(
            np.transpose(texture_wh_rgb_u8.astype(np.uint8), (1, 0, 2)))
        pil_image = Image.fromarray(tex_hw, mode='RGB')
        vis = trimesh.visual.TextureVisuals(uv=corner_uv, image=pil_image)
        mesh = trimesh.Trimesh(
            vertices=corner_xyz,
            faces=new_faces,
            visual=vis,
            process=False,
        )
        mat = getattr(mesh.visual, 'material', None)
        if mat is not None and hasattr(mat, 'doubleSided'):
            mat.doubleSided = True
        return mesh

    fvi = faces[:, :, 0].astype(np.int64)
    mesh = trimesh.Trimesh(
        vertices=np.asarray(verts[:, :3], dtype=np.float64),
        faces=fvi,
        vertex_colors=np.tile(np.array([180, 180, 180, 255], dtype=np.uint8),
                              (verts.shape[0], 1)),
        process=False,
    )
    return mesh


class StaticRenderer:
    def __init__(self):
        self.scene = pyrender.Scene(
            ambient_light=np.array([0.1, 0.1, 0.1], dtype=np.float64),
            bg_color=np.array([0.0, 0.0, 0.0, 0.0]),
        )
        self.mesh_node = None
        self.light_nodes = []
        self.N = 10
        self._offscreen = {}
        self.camera_light()

    def _offscreen_renderer(self, w, h):
        key = (int(w), int(h))
        if key not in self._offscreen:
            self._offscreen[key] = pyrender.OffscreenRenderer(
                viewport_width=key[0],
                viewport_height=key[1],
            )
        return self._offscreen[key]

    def change_all(self):
        self.camera_light()

    def check_update(self, obj):
        return

    def camera_light(self):
        for n in self.light_nodes:
            try:
                self.scene.remove_node(n)
            except ValueError:
                pass
        self.light_nodes = []
        light_dir = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        for l in range(6):
            rot = np.matmul(
                _rotation_x_mat(math.radians(np.random.uniform(-30, 30))),
                _rotation_y_mat(math.radians(360 // 6 * l)),
            )
            d = rot @ light_dir
            d = d / (np.linalg.norm(d) + 1e-12)
            light = pyrender.DirectionalLight(
                color=np.ones(3, dtype=np.float64),
                intensity=2.5,
            )
            pose = _rotation_matrix_to_pose_light_dir(d)
            node = self.scene.add(light, pose=pose)
            self.light_nodes.append(node)

    def _set_mesh_trimesh(self, tm):
        if self.mesh_node is not None:
            self.scene.remove_node(self.mesh_node)
            self.mesh_node = None
        mat = getattr(tm.visual, 'material', None)
        if mat is not None and hasattr(mat, 'doubleSided'):
            mat.doubleSided = True
        mesh_pr = pyrender.Mesh.from_trimesh(tm, smooth=False)
        self.mesh_node = self.scene.add(mesh_pr, pose=np.eye(4))

    def add_model(self, obj, tex=None):
        tm = _obj_to_trimesh(obj, tex)
        self._set_mesh_trimesh(tm)

    def modify_model(self, index, obj, tex=None):
        _ = index
        tm = _obj_to_trimesh(obj, tex)
        self._set_mesh_trimesh(tm)

    def render_frame(self, width, height, fx, fy, cx, cy, cam_pos, look_at_center):
        """Render RGB, depth proxy, mask; intrinsics/extrinsics match taichi_three."""
        extrinsic, _, _, _ = _taichi_camera_extrinsic(cam_pos, look_at_center)
        intrinsic = np.zeros((3, 3), dtype=np.float64)
        intrinsic[0, 0] = fx
        intrinsic[1, 1] = fy
        intrinsic[0, 2] = cx
        intrinsic[1, 2] = cy
        intrinsic[2, 2] = 1.0

        if sys.platform == 'darwin':
            pfx, pfy, pcx, pcy = fx * 0.5, fy * 0.5, cx * 0.5, cy * 0.5
        else:
            pfx, pfy, pcx, pcy = fx, fy, cx, cy

        cam = pyrender.IntrinsicsCamera(
            pfx, pfy, pcx, pcy, znear=0.01, zfar=100.0,
        )
        pose = _gl_cam_pose_from_taichi(cam_pos, look_at_center)
        cam_node = self.scene.add(cam, pose=pose)
        r = self._offscreen_renderer(width, height)
        try:
            color_rgba, depth = r.render(self.scene)
        finally:
            self.scene.remove_node(cam_node)

        img = np.clip(color_rgba[..., :3].astype(np.float64) / 255.0, 0.0, 1.0)
        valid = np.isfinite(depth) & (depth > 0.0)
        zbuf = np.zeros_like(depth, dtype=np.float64)
        zbuf[valid] = 1.0 / (depth[valid] + 1e-8)
        mask = np.zeros((depth.shape[0], depth.shape[1], 3), dtype=np.float64)
        mask[valid] = 1.0

        img = np.swapaxes(img, 0, 1)
        zbuf = np.swapaxes(zbuf, 0, 1)
        mask = np.swapaxes(mask, 0, 1)

        return (
            extrinsic.astype(np.float32),
            intrinsic.astype(np.float32),
            zbuf.astype(np.float32),
            img.astype(np.float32),
            mask.astype(np.float32),
        )


def render_data(renderer, smplx_path, data_path, phase, data_id, save_path, cam_nums, res,
                dis=1.0, is_thuman=False):

    obj_path = os.path.join(data_path, data_id, '%s.obj' % data_id)
    smpl_obj_path = os.path.join(smplx_path, data_id, 'mesh_smplx.obj')
    texture_path = data_path
    img_path = os.path.join(texture_path, data_id, 'material0.jpeg')
    texture_bgr = cv2.imread(img_path)
    if texture_bgr is None:
        raise FileNotFoundError(f'Could not read texture: {img_path}')
    texture = texture_bgr[:, :, ::-1]
    texture = np.ascontiguousarray(texture)
    texture = texture.swapaxes(0, 1)[:, ::-1, :]
    obj = readobj(obj_path, scale=1)
    smpl_obj = readobj(smpl_obj_path, scale=1)
    original_smpl_obj = readobj(smpl_obj_path, scale=1)

    transform_dict = {}

    vy_max = np.max(obj['vi'][:, 1])
    vy_min = np.min(obj['vi'][:, 1])
    height_delta = np.random.uniform(-0.05, 0.05, 1)
    print(height_delta)
    human_height = 1.80 + height_delta
    obj['vi'][:, :3] = obj['vi'][:, :3] / (vy_max - vy_min) * human_height
    offset = np.min(obj['vi'][:, 1])
    obj['vi'][:, 1] -= offset

    transform_dict['vy_max'] = vy_max
    transform_dict['vy_min'] = vy_min
    transform_dict['height_delta'] = height_delta
    transform_dict['human_height'] = human_height
    transform_dict['offset'] = offset

    smpl_obj['vi'][:, :3] = smpl_obj['vi'][:, :3] / (vy_max - vy_min) * human_height
    smpl_obj['vi'][:, 1] -= offset

    look_at_center = np.array([0, 0.85, 0])
    base_cam_pitch = -8

    move_range = 0.1 if human_height < 1.80 else 0.05
    delta_x = np.max(obj['vi'][:, 0]) - np.min(obj['vi'][:, 0])
    delta_z = np.max(obj['vi'][:, 2]) - np.min(obj['vi'][:, 2])
    if delta_x > 1.0 or delta_z > 1.0:
        move_range = 0.01

    move_delta_axis_0 = np.random.uniform(-move_range, move_range, 1)
    move_delta_axis_2 = np.random.uniform(-move_range, move_range, 1)
    print(move_delta_axis_0)
    obj['vi'][:, 0] += move_delta_axis_0
    obj['vi'][:, 2] += move_delta_axis_2
    output_obj_path = os.path.join(data_path, data_id,
                                   '%s_modified.obj' % data_id)
    save_modified_obj(obj_path, list(obj['vi']), output_obj_path)
    transform_dict['delta_x'] = delta_x
    transform_dict['delta_z'] = delta_z
    transform_dict['move_range'] = move_range
    transform_dict['move_delta_axis_0'] = move_delta_axis_0
    transform_dict['move_delta_axis_2'] = move_delta_axis_2

    smpl_obj['vi'][:, 0] += move_delta_axis_0
    smpl_obj['vi'][:, 2] += move_delta_axis_2

    output_transform_path = os.path.join(data_path, data_id, '%s_transform.npy' % data_id)
    output_smpl_obj_path = os.path.join(data_path, data_id, '%s_smplx_modified.obj' % data_id)
    output_original_smpl_obj_path = os.path.join(data_path, data_id, '%s_smplx.obj' % data_id)

    save_modified_obj(smpl_obj_path, list(smpl_obj['vi']), output_smpl_obj_path)
    save_modified_obj(smpl_obj_path, list(original_smpl_obj['vi']), output_original_smpl_obj_path)
    np.save(output_transform_path, transform_dict)

    if renderer.mesh_node is not None:
        renderer.modify_model(0, obj, texture)
    else:
        renderer.add_model(obj, texture)

    degree_interval = 360 / cam_nums
    angle_list1 = list(range(360 - int(degree_interval // 2), 360))
    angle_list2 = list(range(0, 0 + int(degree_interval // 2)))
    angle_list = angle_list1 + angle_list2
    angle_base = np.random.choice(angle_list, 1)[0]

    if is_thuman:
        smpl_path = os.path.join(smplx_path, data_id,
                                 'smplx_param.pkl')
        with open(smpl_path, 'rb') as f:
            smpl_para = pickle.load(f)

        y_orient = smpl_para['global_orient'][0][1]
        angle_base += (y_orient * 180.0 / np.pi)

    for pid in range(cam_nums):
        angle = angle_base + pid * degree_interval

        def render(dis, angle, look_at_center, p, render_2k=False):
            ori_vec = np.array([0, 0, dis], dtype=np.float64)
            rot = np.matmul(
                _rotation_y_mat(math.radians(angle)),
                _rotation_x_mat(math.radians(p)),
            )
            fwd = rot @ ori_vec
            cam_pos = look_at_center + fwd

            x_min = 0
            y_min = -25
            cx = res[0] * 0.5
            cy = res[1] * 0.5
            fx = res[0] * 0.8
            fy = res[1] * 0.8
            _cx = cx - x_min
            _cy = cy - y_min

            w0, h0 = int(res[0]), int(res[1])
            extr, intr, depth_map, img, mask = renderer.render_frame(
                w0, h0, fx, fy, _cx, _cy, cam_pos, look_at_center,
            )

            if render_2k:
                fx2 = res[0] * 0.8 * 2
                fy2 = res[1] * 0.8 * 2
                _cx2 = (res[0] * 0.5 - x_min) * 2
                _cy2 = (res[1] * 0.5 - y_min) * 2
                w1, h1 = int(res[0] * 2), int(res[1] * 2)
                _, _, _, img_hr, _ = renderer.render_frame(
                    w1, h1, fx2, fy2, _cx2, _cy2, cam_pos, look_at_center,
                )
                return extr, intr, depth_map, img, mask, img_hr

            return extr, intr, depth_map, img, mask

        extr, intr, depth, img, mask = render(dis, angle, look_at_center,
                                              base_cam_pitch, False)
        save(pid, data_id, 0, save_path, extr, intr, depth, img, mask)
        extr, intr, depth, img, mask = render(dis,
                                              (angle + degree_interval) % 360,
                                              look_at_center, base_cam_pitch,
                                              False)
        save(pid, data_id, 1, save_path, extr, intr, depth, img, mask)

        angle1 = (angle + (np.random.uniform() * degree_interval / 2)) % 360
        angle2 = (angle + degree_interval / 2) % 360
        angle3 = (angle + degree_interval - (
                    np.random.uniform() * degree_interval / 2)) % 360

        extr, intr, depth, img, mask, img_hr = render(dis, angle1,
                                                      look_at_center,
                                                      base_cam_pitch, True)
        save(pid, data_id, 2, save_path, extr, intr, depth, img, mask, img_hr)
        extr, intr, depth, img, mask, img_hr = render(dis, angle2,
                                                      look_at_center,
                                                      base_cam_pitch, True)
        save(pid, data_id, 3, save_path, extr, intr, depth, img, mask, img_hr)
        extr, intr, depth, img, mask, img_hr = render(dis, angle3,
                                                      look_at_center,
                                                      base_cam_pitch, True)
        save(pid, data_id, 4, save_path, extr, intr, depth, img, mask, img_hr)


if __name__ == '__main__':

    np.random.seed(42)

    cam_nums = 16
    scene_radius = 2.0
    res = (1024, 1024)
    smplx_root = 'datasets/THuman/THuman2.0_smplx/'
    thuman_root = 'datasets/THuman/THuman2.0_Release/'
    source_root = 'datasets/THuman'
    save_root = 'datasets/THuman/'
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    renderer = StaticRenderer()

    for phase in ['train', 'val']:

        split_file = os.path.join(source_root,'split_{}.txt'.format(phase))
        thuman_list = []

        with open(split_file, 'r') as f:
            for line in f:
                human_name = line.strip()
                thuman_list.append(human_name)
        thuman_list.sort()

        save_path = os.path.join(save_root, phase)

        for data_id in tqdm(thuman_list):

            render_data(renderer, smplx_root, thuman_root, phase, data_id, save_path,
                        cam_nums, res, dis=scene_radius, is_thuman=True)

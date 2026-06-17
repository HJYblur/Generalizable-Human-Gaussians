"""OBJ load/save helpers without importing taichi (used by render_image.py)."""
import numpy as np


def _append(faces, indices):
    if len(indices) == 4:
        faces.append([indices[0], indices[1], indices[2]])
        faces.append([indices[2], indices[3], indices[0]])
    elif len(indices) == 3:
        faces.append(indices)
    else:
        assert False, len(indices)


def readobj(path, scale=1):
    vi = []
    vt = []
    vn = []
    faces = []

    with open(path, 'r') as myfile:
        lines = myfile.readlines()

    for line in lines:
        try:
            typ, fields = line.split(maxsplit=1)
            fields = [float(_) for _ in fields.split()]
        except ValueError:
            continue

        if typ == 'v':
            vi.append(fields)
        elif typ == 'vt':
            vt.append(fields)
        elif typ == 'vn':
            vn.append(fields)

    for line in lines:
        try:
            typ, fields = line.split(maxsplit=1)
            fields = fields.split()
        except ValueError:
            continue

        if typ != 'f':
            continue

        indices = [[int(_) - 1 if _ != '' else 0 for _ in field.split('/')]
                   for field in fields]
        _append(faces, indices)

    ret = {}
    ret['vi'] = None if len(vi) == 0 else np.array(vi).astype(np.float32) * scale
    ret['vt'] = None if len(vt) == 0 else np.array(vt).astype(np.float32)
    ret['vn'] = None if len(vn) == 0 else np.array(vn).astype(np.float32)
    ret['f'] = None if len(faces) == 0 else np.array(faces).astype(np.int32)
    return ret


def save_modified_obj(original_obj_path, modified_vertices, output_obj_path):
    print('before:')
    print('original_obj_path: {}'.format(original_obj_path))
    print('output_obj_path: {}'.format(output_obj_path))
    with open(original_obj_path, 'r') as original_obj_file:
        lines = original_obj_file.readlines()

    modified_lines = []

    for line in lines:
        tokens = line.strip().split()

        if not tokens:
            modified_lines.append(line)
            continue

        if tokens[0] == 'v':
            vertex = modified_vertices.pop(0)
            modified_line = f"v {vertex[0]} {vertex[1]} {vertex[2]}\n"
            modified_lines.append(modified_line)
        else:
            modified_lines.append(line)

    with open(output_obj_path, 'w') as output_obj_file:
        output_obj_file.writelines(modified_lines)
    print('after')
    print('save to {}'.format(output_obj_path))
    print('###')

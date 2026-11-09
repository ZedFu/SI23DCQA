"""Convert OBJ vertices to PLY, preserving the original vertex-based sampling."""

import argparse
from pathlib import Path

from data_utils import read_annotations


def read_obj_rgb(input_name, save_ply):
    vertices = []
    with open(input_name, encoding='utf-8') as stream:
        for line in stream:
            tokens = line.split()
            if tokens and tokens[0] == 'v':
                xyz = [float(x) for x in tokens[1:4]]
                if len(xyz) != 3:
                    raise ValueError(f'Invalid vertex in {input_name}')
                rgb = [max(0, min(255, int(float(x) * 255))) for x in tokens[4:7]]
                vertices.append(xyz + (rgb if len(rgb) == 3 else [128, 128, 128]))
    if not vertices:
        raise ValueError(f'No vertices in {input_name}')
    target = Path(save_ply)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('w', encoding='utf-8') as stream:
        stream.write(f'ply\nformat ascii 1.0\nelement vertex {len(vertices)}\n')
        stream.write('property float x\nproperty float y\nproperty float z\n')
        stream.write('property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n')
        for vertex in vertices:
            stream.write(' '.join(map(str, vertex)) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', default='./data')
    parser.add_argument('--ann', default='all-result.txt')
    args = parser.parse_args()
    root = Path(args.data_root)
    for record in read_annotations(root / args.ann):
        video = Path(record['video_path'])
        relative = Path(*video.parts[1:]).with_suffix('.ply')
        target = root / 'pointcloud' / relative
        read_obj_rgb(root / record['mesh_path'], target)
        print(target)


if __name__ == '__main__':
    main()

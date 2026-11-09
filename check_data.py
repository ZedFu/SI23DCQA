"""Check annotations and prepared features before starting an experiment."""

import argparse
from pathlib import Path

from data_utils import feature_dir, read_annotations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', default='./data')
    parser.add_argument('--ann', default='all-result.txt')
    parser.add_argument('--features', action='store_true', help='Check all features required by MultiFeatureModel.')
    parser.add_argument('--check-shapes', action='store_true', help='Also inspect NumPy array dimensions (requires numpy).')
    parser.add_argument('--limit', type=int, help='Check only the first N records.')
    args = parser.parse_args()
    root = Path(args.data_root)
    records = read_annotations(root / args.ann)
    if args.limit is not None and args.limit < 1:
        parser.error('--limit must be positive')
    records = records[:args.limit]
    problems = []
    for record in records:
        paths = [root / record['image_path'], root / record['video_path']]
        if args.features or args.check_shapes:
            folder = feature_dir(root, record['video_path'])
            paths += [folder / name for name in ['3D.pt', 'compare2.pt', 'patch.npy']]
            paths += [folder / 'frames' / f'{i:03d}.png' for i in range(8)]
            paths += [folder / f'feature_{i}_{kind}_feature.npy' for i in range(8) for kind in ['slow', 'fast']]
        missing = [path for path in paths if not path.is_file()]
        problems += [f'Missing: {path}' for path in missing]
        if args.check_shapes and not missing:
            import numpy as np

            try:
                patches = np.load(folder / 'patch.npy', mmap_mode='r', allow_pickle=False)
                if patches.ndim != 3 or patches.shape[0] < 1 or patches.shape[1:] != (2048, 3):
                    problems.append(f'Invalid patch shape {patches.shape}: {folder}')
                for i in range(8):
                    for kind, size in [('slow', 2048), ('fast', 256)]:
                        path = folder / f'feature_{i}_{kind}_feature.npy'
                        array = np.load(path, mmap_mode='r', allow_pickle=False)
                        if array.squeeze().shape != (size,):
                            problems.append(f'Invalid feature shape {array.shape}: {path}')
            except (ValueError, OSError) as error:
                problems.append(f'{folder}: {error}')
    print(f'Checked {len(records)} samples; {len(problems)} problems.')
    for problem in problems[:20]:
        print(problem)
    if len(problems) > 20:
        print(f'... and {len(problems) - 20} more problems.')
    if problems:
        raise SystemExit(1)


if __name__ == '__main__':
    main()

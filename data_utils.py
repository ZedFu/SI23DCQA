"""Annotation parsing and feature paths, without deep learning dependencies."""

from pathlib import Path
import math
import os
import random


def read_annotations(path):
    records = []
    with open(path, encoding='utf-8') as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            fields = line.strip().split('|')
            if len(fields) != 6:
                raise ValueError(f'{path}:{line_number}: expected 6 pipe-separated columns')
            image, mesh, video = fields[:3]
            scores = [float(v) for v in fields[3:]]
            if not all(math.isfinite(v) for v in scores):
                raise ValueError(f'{path}:{line_number}: MOS values must be finite')
            records.append(dict(image_path=image, mesh_path=mesh, video_path=video,
                                overall=scores[0], color=scores[1], shape=scores[2]))
    if not records:
        raise ValueError(f'No samples in {path}')
    return records


def split_by_image(ann_file, ratio=0.8, seed=42):
    """Split unique input images, keeping every generated result with its image.

    ``ratio`` applies to the number of images, so sample counts can differ from
    this ratio when images have different numbers of generated results. Image
    keys are sorted before shuffling to make membership independent of row order.
    A local RNG keeps the split reproducible without reseeding training randomness.
    """
    if not 0 < ratio < 1:
        raise ValueError('ratio must be strictly between 0 and 1')
    groups = {}
    for record in read_annotations(ann_file):
        key = os.path.normpath(record['image_path'])
        groups.setdefault(key, []).append(record)

    images = sorted(groups)
    num_train = int(ratio * len(images))
    if num_train == 0 or num_train == len(images):
        raise ValueError('Image split requires at least one input image in each partition; '
                         'add more images or adjust ratio')
    rng = random.Random(seed)
    rng.shuffle(images)
    train = [record for image in images[:num_train] for record in groups[image]]
    test = [record for image in images[num_train:] for record in groups[image]]
    # Mix generated results within each partition, as in the original row shuffle.
    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


def feature_dir(root, video_path):
    video = Path(video_path)
    if video.is_absolute() or '..' in video.parts or not video.parts or video.parts[0] != 'video_single':
        raise ValueError(f'Expected video_single/<category>/<method>/<object>.mp4: {video_path}')
    return Path(root) / 'features' / Path(*video.parts[1:]).with_suffix('')

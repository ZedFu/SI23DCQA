#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
for subset in all aigi mri true; do
    python train_my.py -c "config/my-${subset}.yml" "$@"
done

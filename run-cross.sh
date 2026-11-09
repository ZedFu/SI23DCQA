#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python train_my_cross.py -c config-cross/my-ta.yml "$@"
python train_my_cross.py -c config-cross/my-tm.yml "$@"

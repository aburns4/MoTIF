# Copyright 2022 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/bin/bash

# Set SCC project
#$ -P ivc-ml

# Request 1 CPUs
#$ -pe omp 3

# Specify hard time limit
#$ -l h_rt=01:30:00

# get email when job begins
#$ -m beas

set -e
set -x

export OMP_NUM_THREADS=${NSLOTS}
export TF_NUM_INTEROP_THREADS=${NSLOTS}
export TF_NUM_INTRAOP_THREADS=1

python -m seq2act.bin.seq2act_decode --problem "motif" \
                                     --data_files "/projectnb/ivc-ml/aburns4/seq2act_clean_start/data/motif_ua_ut_final/*.tfrecord" \
                                     --checkpoint_path "/projectnb/ivc-ml/aburns4/MoTIF/task_automation/seq2act/ckpt_hparams/grounding" \
                                     --output_dir "/projectnb/ivc-ml/aburns4/MoTIF/task_automation/seq2act/decode/motif/check_old_split_7_13/"

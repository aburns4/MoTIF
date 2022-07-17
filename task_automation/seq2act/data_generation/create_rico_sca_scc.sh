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

# virtualenv -p python3.7 .
# source ./bin/activate

# pip install -r seq2act/data_generation/requirements.txt
# Set SCC project
#$ -P ivc-ml

# Request 1 CPUs
#$ -pe omp 1

# Specify hard time limit
#$ -l h_rt=01:30:00

# get email when job begins
#$ -m beas

set -e
set -x

mkdir -p ${PWD}"/seq2act/data/rico_sca/ua_ut_tfexample_final"

python -m seq2act.data_generation.create_android_synthetic_dataset \
--input_dir=${PWD}"/seq2act/data/rico_sca/raw" \
--output_dir=${PWD}"/seq2act/data/rico_sca/ua_ut_tfexample_final" \
--filter_file=${PWD}"/seq2act/data_generation/ricosca_ua_ut_filter.txt" \
--thread_num=10 \
--shard_num=10 \
--vocab_file=${PWD}"/seq2act/data_generation/commoncrawl_rico_vocab_subtoken_44462" \
--input_candiate_file=${PWD}"/seq2act/data_generation/google-10000-english-no-swears.txt" \
--logtostderr


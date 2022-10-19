
#!/bin/bash -l

set -e
set -x

dir=${PWD}
parentdir="$(dirname "$dir")"

python create_motif_dataset_no_thread.py --dataset "motif" \
                                         --input_dir ${parentdir}"/data/motif/raw/traces_02_14_21" \
                                         --output_dir ${parentdir}"/data/motif/seq2act_debug_type_su_all" \
                                         --num_shards 1 \
                                         --split su_all \
                                         --use_high_level_goal=true

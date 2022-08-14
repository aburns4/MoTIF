
#!/bin/bash -l

set -e
set -x 

dir=${PWD}
parentdir="$(dirname "$dir")"

python all_in_one_motif_preprocess.py --data_dir ${parentdir}"/data/motif/raw/traces_02_14_21" \
                                      --save_dir "processed_motif_deduped" \
                                      --dedup_cycles=true
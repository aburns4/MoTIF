#!/bin/bash -l

python -m seq2act.bin.seq2act_train_eval --exp_mode "train" \
                                         --experiment_dir ${PWD}"/ckpt_hparams/tuple_ua_ut_all_final2/" \
                                         --hparam_file ${PWD}"/ckpt_hparams/tuple_extract/" \
                                         --train_steps 1000000 \
                                         --train_file_list ${PWD}"/data/android_howto/tfexample/*.tfrecord,${PWD}/data/rico_sca/ua_ut_tfexample_final/*.tfrecord" \
                                         --train_batch_sizes "64,64" \
                                         --train_source_list "android_howto,rico_sca"

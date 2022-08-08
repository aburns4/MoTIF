# Task Automation Benchmarking
We provide code for benchmarking mobile app task automation with three models: Seq2Seq [1], MOCA [2], and Seq2Act [3]. The first two are vision-language navigation approaches, and the latter is a transformer encoder-decoder framework. 

There are additional folders for each model that provide data processing instructions of how to obtain the necessary features and data format to train and evaluate each. Note, some of the Seq2Act processing code has been re-used to generate data files for the VLN methods. See the README in each subfolder for more information.

[1] Shridhar et al. A Benchmark for Interpreting Grounded Instructions for Everyday Tasks. CVPR 2020.

[2] Singh et al. Factorizing Perception and Policy for Interactive Instruction Following ICCV 2021.

[3] Li et al. Mapping Natural Language Instructions to Mobile UI Action Sequences. ACL 2020.

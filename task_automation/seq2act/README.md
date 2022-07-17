# Seq2act: Mapping Natural Language Instructions to Mobile UI Action Sequences
This repository contains the modified code for the models and the experimental framework for "Mapping Natural Language Instructions to Mobile UI Action Sequences" by Yang Li, Jiacong He, Xin Zhou, Yuan Zhang, and Jason Baldridge, which is accepted in 2020 Annual Conference of the Association for Computational Linguistics (ACL 2020). This model is one of the baselines used to benchmark our mobile app task automation problem.

## Datasets
We released the AndroidHowTo tfrecords directly because it is time consuming and memory heavy to process that data from scratch. The PixelHelp tfrecords can be accessed here from the original Seq2Act GitHub. 

For RicoSCA and MoTIF, we provide code to process the tfrecords yourself because they end up being large files. Here is how you do it:

For RicoSCA, download the original raw data from here and place it under `data/rico_sca/raw`; this raw folder should be the "combined" folder you get with the .json and .jpg Rico files after unzipping the linked Rico data. Then, run the following. NOTE: The authors of Seq2Act never released the text file containing input candidate words for their synthetic typing events. As a result, I chose a list of the top 10k English non-swear words. Feel free to change this based on your needs or design choices.

```
create_rico_sca.sh
```

For MoTIF, if you want to start from scratch or modify the original data processing, first download the raw data here and place it in the `data/motif/raw` directory. Next, run the `all_in_one_motif_processing.py` file to generate json files that contain information on each interaction trace in MoTIF. In this file we clean the captured action sequences from duplicate events, technical failures, and cyclic behavior. You can skip this step and download our already processed jsons here.

Once you have the cleaned data, unzip it in the `motif_data_generation` folder and run the following command

```
make_motif_tfrecords.sh
```

## Setup

Install the packages that required by our codebase, and perform a test over the setup by running a minimal verion of the model and the experimental framework.

```
sh seq2act/run.sh
```

## Run Experiments.

* Train (and continuously evaluate) seq2act Phrase Tuple Extraction models.

```
sh seq2act/bin/train_seq2act.sh --experiment_dir=./your_parser_exp_dir --train=parse --hparam_file=./seq2act/ckpt_hparams/tuple_extract
```

Then copy your lastest checkpoint from your_parser_exp_dir to `./seq2act/ckpt_hparams/tuple_extract/`

* Train (and continuously evaluate) seq2act Grounding models.

```
sh seq2act/bin/train_seq2act.sh --experiment_dir=./your_grounding_exp_dir --train=ground --hparam_file=./seq2act/ckpt_hparams/grounding
```

Then copy your latest checkpoint from your_grounding_exp_dir to `./seq2act/ckpt_hparams/grounding/`

NOTE: You can also try out our pre-trained checkpoint for end-to-end grounding
by downloading the checkpoint [here](https://storage.googleapis.com/gresearch/seq2act/ccg3-transformer-6-dot_product_attention-lr_0.003_rd_0.1_ad_0.1_pd_0.2.tar.gz). 
Once downloaded, you can extract the checkpoint files from the zip file, which 
result in 1 file named 'checkpoint' and 3 files with "model.ckpt-250000*".
You can then move these files to  to `./seq2act/ckpt_hparams/grounding/`

* Test the grounding model or only the phrase extraction model by running the decoder.

```
sh seq2act/bin/decode_seq2act.sh --output_dir=./your_decode_dir
```

If you use any of the materials, please cite the following paper.

```
@inproceedings{seq2act,
  title = {Mapping Natural Language Instructions to Mobile UI Action Sequences},
  author = {Yang Li and Jiacong He and Xin Zhou and Yuan Zhang and Jason Baldridge},
  booktitle = {Annual Conference of the Association for Computational Linguistics (ACL 2020)},
  year = {2020},
  url = {https://www.aclweb.org/anthology/2020.acl-main.729.pdf},
}
```

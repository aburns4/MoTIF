# Seq2act: Mapping Natural Language Instructions to Mobile UI Action Sequences
This repository contains the modified code for the models and the experimental framework for "Mapping Natural Language Instructions to Mobile UI Action Sequences" by Yang Li, Jiacong He, Xin Zhou, Yuan Zhang, and Jason Baldridge, which is accepted in 2020 Annual Conference of the Association for Computational Linguistics (ACL 2020). This model is one of the baselines used to benchmark our mobile app task automation problem.

## Datasets
We release the AndroidHowTo [tfrecords]() directly because it is time consuming and memory heavy to process that data from scratch. The PixelHelp tfrecords can be accessed [here]() from the original Seq2Act GitHub. 

For RicoSCA and MoTIF, we provide code to process the tfrecords yourself because they end up being large files. Here is how you do it:

For RicoSCA, download the original [raw data]() and place it under `data/rico_sca/raw`; this raw folder should be the "combined" folder you get with the .json and .jpg Rico files after unzipping the linked Rico data. Then, run the following. NOTE: The authors of Seq2Act never released the text file containing input candidate words for their synthetic typing events. As a result, I chose a list of the top 10k English non-swear words. COnsequently, their exact experiments are not reproduceable from scratch. Feel free to change this file based on your needs or design choices.

```
sh create_rico_sca.sh
```

The bash script has different flags you can change as you need. The filter files we provide create different splits of RicoSCA data for evaluating MoTIF; e.g., the ua_ut filter creates a split of the training data for evaluating an unseen app and unseen task test split. See the `data_generation` folder for more info.

For MoTIF, if you want to start from scratch or modify the original data processing, first download the two raw data folders [here]() and [here]() and place it in the `data/motif/raw` directory. Next, run

```
sh dedup.sh
```

to generate json files that contain information on each interaction trace in MoTIF. 
In this file we clean the captured action sequences from duplicate events, technical failures, and cyclic behavior. Our default is to remove cycles, but our processing code is not perfect (there are many edge cases/it is challenging to cover all situations with the same code). If you have suggested improvements for how we handle these cases please raise an issue and let us know. You can skip this step and download our already processed jsons [here]().

Once you have the cleaned data, unzip it in the `motif_data_generation` folder and run the following command

```
sh make_motif_tfrecords.sh
```

For more information see the `motif_data_generation` README.

## Setup

Install the packages that required by the codebase using our provided environment yaml:

```
conda env create -f environment.yml
```

## Run Experiments.

* Train (and continuously evaluate) seq2act Phrase Tuple Extraction models. NOTE: the original Seq2Act model trained on 128 batch size for the tuple extraction model, but the largest I could fit into memory with my computational resources was 128.

```
sh train_seq2act.sh
```

* Train (and continuously evaluate) seq2act grounding models. Change reference checkpoint paths and others accordingly to reflect your saved tuple extraction models.

```
sh train_ground_script.sh
```

* Test the model end-to-end, run the decoder. Only the grounding outputs make sense when evaluating the high-level goal instruction, because we do not have ground truth step by step instruction spans to evaluate the tuple extraction model (i.e., when evaluating high-level goal MoTIF, ignore the .joint_refs decoder outputs). When we evaluate the step-by-step instructions of MoTIF, both outputs are sound.

```
sh decode_seq2act.sh
```

To obtain performance values, run the following with the appropriate .decode_act path. We allow for grounding prediction to be within 1 UI object index of the ground truth because often the model predicts the text correct option, while humans typically click on the visually correct option. See Figure 4 left of the main paper for an example of this. You can swap out the code for the comments in the file for an "exact match" metric. 

```
python decode.motif.grounding_acc.py
```

We release our two tuple extraction checkpoints for the [unseen app unseen task]() split and [seen app unseen task]() split. See more information on these splits in `motif_data_generation`. Similarly, here are grounding checkpoints for the [unseen app unseen task]() split and [seen app unseen task]() split. 

NOTE: You can also try out the original Seq2Act pre-trained checkpoint for end-to-end grounding
by downloading the checkpoint [here](https://storage.googleapis.com/gresearch/seq2act/ccg3-transformer-6-dot_product_attention-lr_0.003_rd_0.1_ad_0.1_pd_0.2.tar.gz). 
Once downloaded, you can extract the checkpoint files from the zip file, which 
result in 1 file named 'checkpoint' and 3 files with "model.ckpt-250000*".
You can then move these files to a folder under `ckpt_params` for decoding. If you use any of the materials, please cite the following paper.

```
@inproceedings{seq2act,
  title = {Mapping Natural Language Instructions to Mobile UI Action Sequences},
  author = {Yang Li and Jiacong He and Xin Zhou and Yuan Zhang and Jason Baldridge},
  booktitle = {Annual Conference of the Association for Computational Linguistics (ACL 2020)},
  year = {2020},
  url = {https://www.aclweb.org/anthology/2020.acl-main.729.pdf},
}
```
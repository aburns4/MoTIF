# Seq2act: Mapping Natural Language Instructions to Mobile UI Action Sequences
This repository contains the modified code for the models and the experimental framework for "Mapping Natural Language Instructions to Mobile UI Action Sequences" by Yang Li, Jiacong He, Xin Zhou, Yuan Zhang, and Jason Baldridge (ACL 2020). This model is one of the baselines used to benchmark our mobile app task automation problem.

## Datasets
We release the AndroidHowTo [tfrecords](https://drive.google.com/file/d/1pWUH6of95LNzc6E774Cyq6pK7yP96tCm/view?usp=sharing) directly because it is time consuming and memory heavy to process that data from scratch. The PixelHelp tfrecords can be accessed [here](https://github.com/google-research-datasets/seq2act/tree/master/data/pixel_help) from the original Seq2Act GitHub. 

For RicoSCA and MoTIF, we provide code to process the tfrecords yourself because they end up being large files. Here is how you do it:

For RicoSCA, download the original [raw data](https://storage.googleapis.com/crowdstf-rico-uiuc-4540/rico_dataset_v0.1/unique_uis.tar.gz) and place it under `data/rico_sca/raw`; this raw folder should be the "combined" folder you get with the .json and .jpg Rico files after unzipping the linked Rico data. Then, run the following.

```
sh create_rico_sca.sh
```
 NOTE: The authors of Seq2Act never released the text file containing input candidate words for their synthetic typing events. As a result, I chose a list of the top 10k English non-swear words. Consequently, their exact experiments are not reproduceable from scratch. Feel free to change this file based on your needs or design choices.

The bash script has different flags you can change as you need. The filter files we provide create different splits of RicoSCA data for evaluating MoTIF; e.g., the ua_ut filter creates a split of the training data for evaluating an unseen app and unseen task test split. See the `data_generation` folder for more info.

For MoTIF, if you want to start from scratch or modify the original data processing, first download the two raw data folders [here](https://drive.google.com/file/d/1XScaD4Pr3K9a9E013wQdh4qd-svdkeVe/view?usp=sharing) (rename this first folder from "raw" to "traces_02_14_21") and [here]() (should be already named "traces_03_17_21") and place it in the `data/motif/raw` directory. Next, run

```
sh dedup.sh
```

to generate json files that contain information on each interaction trace in MoTIF. 
In this file we clean the captured action sequences from duplicate events, technical failures, and cyclic behavior. Our default is to remove cycles, but our processing code is not perfect (there are many edge cases/it is challenging to cover all situations with the same code). If you have suggested improvements for how we handle these cases please raise an issue and let us know. You can skip this step and download our already processed jsons [here](https://drive.google.com/file/d/1sX6WJjuHAC4rTYERv4gyoT-kLZc5A4ey/view?usp=sharing).

Once you have the cleaned data, unzip it in the `motif_data_generation` folder and run the following command

```
sh make_motif_tfrecords.sh
```

For more information see the `motif_data_generation` README.

## Setup

Install the packages required by the codebase using our provided environment yaml:

```
conda env create -f environment.yml
```

## Run Experiments.

* Train (and continuously evaluate) seq2act Phrase Tuple Extraction models. NOTE: the original Seq2Act model trained on 128 batch size for the tuple extraction model, but the largest I could fit into memory with my computational resources was 64.

```
sh train_seq2act.sh
```

* Train (and continuously evaluate) seq2act grounding models. Change reference checkpoint paths and others accordingly to reflect your saved tuple extraction models.

```
sh train_ground_script.sh
```

* To test the model end-to-end, run the decoder. Only the grounding outputs make sense when evaluating the high-level goal instruction, because we do not have ground truth step by step instruction spans to evaluate the tuple extraction model (i.e., when evaluating high-level goal MoTIF, ignore the .joint_refs decoder outputs). When we evaluate the step-by-step instructions of MoTIF, both outputs are sound.

```
sh decode_seq2act.sh
```

To obtain performance values, run the following with the appropriate `.decode_act` path. We allow for grounding prediction to be within 1 UI object index of the ground truth because often the model predicts the textually correct option, while humans typically click on the visually correct option. See Figure 4 left of the main paper for an example of this. You can swap out the code for the comments in the file for an "exact match" metric. 

```
python decode.motif.grounding_acc.py
```

We release [checkpoints]() for the unseen app unseen task split and seen app unseen task split. See more information on these splits in `motif_data_generation`.

The above checkpoint folder also contains the orginal trained model checkpoints released from Seq2Act. NOTE: The key differences between this model and the model I trained on seen apps and unseen tasks is that their model trained on different typing commands with an unknown distribution (as previously mentioned, they did not release the file required to recreate those commands). Additionally, they were able to train with a larger batch size.

If you use any of the materials, please cite our paper and the following paper.

```
@inproceedings{seq2act,
  title = {Mapping Natural Language Instructions to Mobile UI Action Sequences},
  author = {Yang Li and Jiacong He and Xin Zhou and Yuan Zhang and Jason Baldridge},
  booktitle = {Annual Conference of the Association for Computational Linguistics (ACL 2020)},
  year = {2020},
  url = {https://www.aclweb.org/anthology/2020.acl-main.729.pdf},
}
```

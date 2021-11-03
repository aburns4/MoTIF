# MoTIF
Mobile App Tasks with Iterative Feedback (MoTIF): Addressing Task Feasibility in Interactive Visual Environments
NAACL ViGIL Workshop Paper 2021

# Data Extraction and Set Up
To start, download the raw data from https://drive.google.com/file/d/1bZ8CtPM1U8QDEg2zc6W9hnnwsL8ubSZ-/view?usp=sharing and unzip it.
```
unzip -q MoTIF_raw.zip
```

You should now have a folder named `raw_data`. Next, make an experiments directory where models and performance results will be written.

```
mkdir experiments
```

Follow along all instructions in the `feature_extraction` folder to obtain the text (FastText, Screen2Vec, CLIP) and image (Icon, ResNet, CLIP) features.

After, your repo should have the following structure.

# Running Experiments
We provide configuration files under the `configs` folder. All config files except the FastText ablations should be run with the input config name and path to CLIP text embeddings. For FastText ablations, the text embedding path is not neeeded. E.g.,

```
# run clip combination experiment, which results in highest performance
python main.py -c combo_clip_v20_cat.json -v clip_text_vectors.npy
```

To run a FastText ablation, you need not add the path to the vocabulary embeddings (it's the current input default, feel free to change).

```
# run language ablation with FastText instead of CLIP
python main.py -c ft_lang_et_id_cls_v20_avg.json
```

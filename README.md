# MoTIF
Mobile App Tasks with Iterative Feedback (MoTIF): Addressing Task Feasibility in Interactive Visual Environments
NAACL ViGIL Workshop Paper 2021

## Data Extraction and Set Up
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
```
MoTIF
  --configs/
    clip_lang_et_id_cls_v20_avg.json
    clip_lang_et_id_cls_v20_cat.json
    ...
  --data/
     --clip_image_features/
     --icon_crops/
     --resnet_features/
     --screen2vec_features/
     --view_hierarchy_features/
     clip_text_vectors.npy
     fasttext_vectors.npy
     feasibility_annotations.p
     icon_features.npy
     test_traces.txt
     train_traces.txt
     w2i_map.json
  --experiments/
  --feature_extraction/
    --CLIP/
    --Screen2Vec/
    --semantic-icon-classifier/
    get_fasttext_vocab.py
    get_icons.py
    get_resnet.py
    get_view_fts.py
    widget_exception_dims.json
    wiki.en.bin
  --raw_data/
    --app/
      --trace_id/
        --screens/
        --view_hierarchies/
        feasibility.txt
        metadata.json
        tast.txt
  data_loader.py
  main.py
  model.py
  trainer.py
  utils.py
```

## Running Experiments
We provide configuration files under the `configs` folder. All config files except the FastText ablations should be run with the input config name and path to CLIP text embeddings. For FastText ablations, the text embedding path is not neeeded. E.g.,

```
# run clip combination experiment, which results in highest performance
python main.py -c combo_clip_v20_cat.json -v clip_text_vectors.npy
```

To run a FastText ablation, you need not add the path to the vocabulary embeddings (it's the current text embedding default, feel free to change).

```
# run language ablation with FastText instead of CLIP
python main.py -c ft_lang_et_id_cls_v20_avg.json
```

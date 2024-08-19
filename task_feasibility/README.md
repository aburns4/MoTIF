# MoTIF
A Dataset for Interactive Vision-Language Navigation with Unknown Command Feasibility. *Andrea Burns, Deniz Arsan, Sanjna Agrawal, Ranjitha Kumar, Kate Saenko, Bryan A. Plummer*. ECCV 2022.

## Environment
You can build an anaconda environment using our `requirements.txt` file, which has a comprehensive list of the packages installed in the environment used to run experiments. Note, it may contain unnecessary packages. The most important packages are:
* pytorch 1.7.1
* torchvision 0.8.2
* numpy 
* scikit-learn

## Data Extraction and Set Up
To start, download the [raw data](https://drive.google.com/file/d/1Zxm-onsO5MURcKYrRVqJjov0Zb3U1lGf/view?usp=sharing) and [features](https://drive.google.com/file/d/1nr1O7uV_WqOSy9-nkW1ns-hKDpepTGEB/view?usp=sharing). Now extract the files.
```
tar -xf motif_raw_data.tar.gz
tar -xf data.tar.gz
```

You should now have folders named `raw_data` and `data`. Next, make an experiments directory where models and performance results will be written.

```
mkdir experiments
```

If you need the feature extraction code, you can check it out in the `feature_extraction` folder and follow along the instructions there to obtain the text (FastText, Screen2Vec, CLIP) and image (Icon, ResNet, CLIP) features.

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

### Citation
If you use our code or data, please consider citing our work :)
```
@inproceedings{burns2022motifvln,
      title={A Dataset for Interactive Vision Language Navigation with Unknown Command Feasibility}, 
      author={Andrea Burns and Deniz Arsan and Sanjna Agrawal and Ranjitha Kumar and Kate Saenko and Bryan A. Plummer},
      booktitle={European Conference on Computer Vision (ECCV)},
      year={2022}
}
```

# MoTIF
Mobile App Tasks with Iterative Feedback (MoTIF): Addressing Task Feasibility in Interactive Visual Environments
Andrea Burns, Deniz Arsan, Sanjna Agrawal, Ranjitha Kumar, Kate Saenko, Bryan A. Plummer
NAACL ViGIL Workshop Spotlight Paper 2021

COMING SOON: All language annotations (all 6.1k collected tasks), code for automating mobile app tasks, and additional data! Check back soon for updates. 

## Data Extraction and Set Up
To start, download the raw data and features from https://drive.google.com/file/d/1ZAV7Xi7SMWNubxf7MB8_fjPupg-o_RWC/view?usp=sharing and https://drive.google.com/file/d/1GSVA8sz-SKcioNhCq1lWH58x_u5FXfDQ/view?usp=sharing. Now extract the files.
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
@article{burns2021mobile,
      title={Mobile App Tasks with Iterative Feedback (MoTIF): Addressing Task Feasibility in Interactive Visual Environments}, 
      author={Andrea Burns and Deniz Arsan and Sanjna Agrawal and Ranjitha Kumar and Kate Saenko and Bryan A. Plummer},
      journal={Visually Grounded Interaction and Language Workshop at the North American Chapter of the Association for Computational Linguistics (NAACL)}
      year={2021},
      url={http://arxiv.org/abs/2104.08560},
      archivePrefix={arXiv},
      eprint={2104.08560}
}
```

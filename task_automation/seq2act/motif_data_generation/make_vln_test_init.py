import json
import glob
import os
from common import get_view_hierarchy_list
import numpy as np
import re
from PIL import Image
import tensorflow.compat.v1 as tf

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string(
    "env_dir", "../../alfred/data/motif/eval_envs_motif",
    "Full path to the directory where test time graph initialization is to be stored.")

def ui_obj_to_str(ui_obj):
    # creates id by concatenating
    # different UI element fields
    ui_str_id = []

    ui_str_id += str(ui_obj.obj_type)
    ui_str_id += str(ui_obj.text)
    ui_str_id += str(ui_obj.resource_id)
    ui_str_id += str(ui_obj.android_class)
    ui_str_id += str(ui_obj.android_package)
    ui_str_id += str(ui_obj.content_desc)
    ui_str_id += str(ui_obj.clickable)
    ui_str_id += str(ui_obj.visible)
    ui_str_id += str(ui_obj.enabled)
    ui_str_id += str(ui_obj.focusable)
    ui_str_id += str(ui_obj.focused)
    ui_str_id += str(ui_obj.scrollable)
    ui_str_id += str(ui_obj.long_clickable)
    ui_str_id += str(ui_obj.selected)

    return " ".join(ui_str_id)

def get_uid(view_path, vh, vh_w, vh_h):
    # get unique id for state in state-action graph
    # consists of concat pointer values for all elems
    # which are visible and selected
    # followed by concat word2index str of leaf node text
    view_hierarchy_leaf_nodes = get_view_hierarchy_list(view_path, vh_w, vh_h)
    ui_objs = [ele.uiobject for ele in view_hierarchy_leaf_nodes]
    ui_objs_str = [ui_obj_to_str(ui) for ui in ui_objs]
    return " ".join(ui_objs_str)

def make_test_map(app, trace_id):
    # make state-action space graph for a given app

    with open(os.path.join(FLAGS.env_dir, app + '_map.json')) as f:
        raw_id_to_pretty = json.load(f)

    with open('processed_motif_deduped/' + trace_id + '.json') as f:
        trace_info = json.load(f)
        views = trace_info['images']
        vh_w = trace_info['vh_w']
        vh_h = trace_info['vh_h']

    try:
        init_view_path = os.path.join('../data/motif/raw/traces_03_17_21/', app, trace_id, 'view_hierarchies', views[0]) + '.jpg'
        with open(init_view_path) as f:
            init_vh = json.load(f)
    except:
        init_view_path = os.path.join('../data/motif/raw/traces_02_14_21/', app, trace_id, 'view_hierarchies', views[0]) + '.jpg'
        with open(init_view_path) as f:
            init_vh = json.load(f)
    init_uid = get_uid(init_view_path, init_vh['activity']['root'], vh_w, vh_h)
    return raw_id_to_pretty[init_uid]


def main():
    test_uid_init = {}

    # make eval envs for all test splits at once
    with open('eccv_motif_ricosca_app_unseen_task_unseen.json') as f:
        test_split = json.load(f)['test']
    
    with open('eccv_motif_ricosca_app_unseen_task_seen.json') as f:
        test_split += json.load(f)['test']

    with open('eccv_motif_ricosca_app_seen_task_unseen_all.json') as f:
        test_split += json.load(f)['test']

    with open('eccv_motif_ricosca_app_seen_task_unseen_curr.json') as f:
        test_split += json.load(f)['test']

    test_split = list(set(test_split))
    print(len(test_split))

    for trace in test_split:
        app = glob.glob('../data/motif/raw/traces_03_17_21/*/' + trace)
        if len(app) == 0:
            app = glob.glob('../data/motif/raw/traces_02_14_21/*/' + trace)
        app = app[0]
        app = app.split('/')[-2]

        pretty_uid = make_test_map(app, trace)
        test_uid_init[trace] = pretty_uid

    with open(os.path.join(FLAGS.env_dir, 'test_uid_init_map.json'), 'w') as f:
        json.dump(test_uid_init, f)

main()
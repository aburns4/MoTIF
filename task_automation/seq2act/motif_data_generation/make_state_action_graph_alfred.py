import json
import glob
import common
import os
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


def get_uid(view_path, vh_w, vh_h):
    # get unique id for state in state-action graph
    # consists of concat str values from leaf nodes
    view_hierarchy_leaf_nodes = common.get_view_hierarchy_list(view_path, vh_w, vh_h)
    ui_objs = [ele.uiobject for ele in view_hierarchy_leaf_nodes]
    ui_objs_str = [ui_obj_to_str(ui) for ui in ui_objs]
    return " ".join(ui_objs_str)

def get_absolute_coords(bbox, width, height):
    return [bbox[0] * width, bbox[1] * height, bbox[2] * width, bbox[3] * height]

def update_sag(sag, trace_path, trace_info, raw_id_to_pretty, pretty_idx):
    for i in range(len(trace_info['images'])-1):
        curr_view = trace_info['images'][i] 
        curr_view_path = os.path.join(trace_path, 'view_hierarchies', curr_view) + '.jpg'

        curr_uid = get_uid(curr_view_path, trace_info['vh_w'], trace_info['vh_h'])
        if curr_uid in raw_id_to_pretty:
            curr_uid_pretty = raw_id_to_pretty[curr_uid]
        else:
            # if not in uid dict yet, add and update idx
            curr_uid_pretty = pretty_idx
            raw_id_to_pretty[curr_uid] = pretty_idx
            pretty_idx += 1

        # get action type in (type, swipe, click) and bbox of corresponding UI elem
        action_type = trace_info['actions'][i]
        raw_gest_abs = get_absolute_coords(trace_info['raw_gestures'][i][0] + trace_info['raw_gestures'][i][-1], trace_info['screen_w'], trace_info['screen_h'])
        chosen_bbox = (trace_info['screen_bboxes'][i] if trace_info['screen_bboxes'][i] else raw_gest_abs) # this is a bit weird, swipe case not sure how to deal w

        # get which view results from performing this action
        # (i.e. apply transition and get next state)
        if i == len(trace_info['images'])-2:
            # reached the end
            next_uid_pretty = -1
        else:
            next_view = trace_info['images'][i+1]
            next_view_path = os.path.join(trace_path, 'view_hierarchies', next_view) + '.jpg'

            next_uid = get_uid(next_view_path, trace_info['vh_w'], trace_info['vh_h'])
            if next_uid in raw_id_to_pretty:
                next_uid_pretty = raw_id_to_pretty[next_uid]
            else:
                next_uid_pretty = pretty_idx
                raw_id_to_pretty[next_uid] = pretty_idx
                pretty_idx += 1

        # add state-action pair to graph
        if curr_uid_pretty in sag:
            already_bbox = [p[0] for p in sag[curr_uid_pretty]['children'][action_type]]
            already_in = chosen_bbox in already_bbox
            if not already_in:
                sag[curr_uid_pretty]['children'][action_type].append([chosen_bbox, next_uid_pretty])
        else:
            sag[curr_uid_pretty] = {'view_id': curr_view,
                                    'children': {'click': [],
                                                 'swipe': [],
                                                 'type': []
                                                 }
                                    }
            sag[curr_uid_pretty]['children'][action_type].append([chosen_bbox, next_uid_pretty])

    return sag, raw_id_to_pretty, pretty_idx


def make_sag(app):
    # make state-action space graph for a given app
    root = '../data/motif/raw/traces_03_17_21'
    all_traces = glob.glob(os.path.join(root, app, '*'))
    root2 = '../data/motif/raw/traces_02_14_21'
    all_traces += glob.glob(os.path.join(root2, app, '*'))

    # state action graph that will be filled with
    # pretty uid : {view_id:
    #               children: {action_type: [bbox, next_resulting_view]}}
    # key : value structure
    sag = {}
    # the raw uid is the concat fields of UI objects
    # I don't want to look at those long str so I convert to ints and keep
    # the mapping
    raw_id_to_pretty = {}
    pretty_idx = 0

    for trace in all_traces:
        trace_id = trace.split('/')[5]
        try:
            with open(os.path.join('processed_motif_deduped', trace_id + '.json')) as f:
                trace_info = json.load(f)
        except:
            continue


        sag, raw_id_to_pretty, pretty_idx = update_sag(sag, trace, trace_info, raw_id_to_pretty, pretty_idx)

    return sag, raw_id_to_pretty


def main():
    # remember, these are diff format than those of MOCA
    if not os.path.isdir(FLAGS.env_dir):
        os.mkdir(FLAGS.env_dir)
   
    test_apps = [x.split('/')[-1] for x in glob.glob('../data/motif/raw/traces_03_17_21/*')]
    i = 0
    apps = test_apps
    print('%d many apps to create test time graphs for' % len(test_apps))
    for app in apps:
        print((app, i))
        i += 1
        
        graph, uid_map = make_sag(app)
        save_graph_path = os.path.join(FLAGS.env_dir, app + '_graph.json')
        save_map_path = os.path.join(FLAGS.env_dir, app + '_map.json')
        
        with open(save_graph_path, 'w') as f:
            json.dump(graph, f)
        with open(save_map_path, 'w') as f:
            json.dump(uid_map, f)

main()

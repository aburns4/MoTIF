import json
import os
import glob
import common
import numpy as np
from PIL import Image
import real_action_generator
import tensorflow.compat.v1 as tf

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string(
    "data_dir", "/projectnb/ivc-ml/aburns4/stage2/traces_02_14_21",
    "Full path to the directory containing the data files for a set of tasks.")
flags.DEFINE_string(
    "save_dir", "seq2act_debug",
    "Full path to the directory for saving the tf record file.")
flags.DEFINE_bool(
    "dedup_cycles", True,
    "Whether cycles should be removed from human demonstrations.")

# load task id to name mapping
TASK_I_TO_N = {}
with open(os.path.join(FLAGS.data_dir, 'tasknames.csv')) as csvfile:
    reader = csvfile.readlines()
    for row in reader:
        row = row.strip().split(' ')
        TASK_I_TO_N[row[0]] = ' '.join(row[1:])


def get_metadata(trace_path):
    metadata_path = trace_path + '/metadata.json'
    with open(metadata_path) as f:
        metadata = json.load(f)
        gestures = metadata['gestures']
        view_paths = [trace_path + '/view_hierarchies/' + x for x in metadata['gestures'].keys()]
    return view_paths, gestures


def get_screen_dims(views, trace_exceptions='widget_exception_dims.json'):
    with open(trace_exceptions) as f:
        widget_exceptions = json.load(f)

    trace = views[0].split('/')[-3]
    if trace in widget_exceptions:
        return [int(widget_exceptions[trace][0]), int(widget_exceptions[trace][1])]

    for view in views:
        try:
            with open(view, 'r') as f:
                data = json.load(f)
                bbox = data['activity']['root']['bounds']
                width = bbox[2] - bbox[0]
                height = bbox[3] - bbox[1]
            if bbox[0] == 0 and bbox[1] == 0:
                return width, height
        except:
            continue


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
    view_hierarchy_leaf_nodes = common.get_view_hierarchy_list(view_path, vh_w, vh_h)
    ui_objs = [ele.uiobject for ele in view_hierarchy_leaf_nodes]
    ui_objs_str = [ui_obj_to_str(ui) for ui in ui_objs]
    return " ".join(ui_objs_str)


def convert_view_to_screen_dims(ui_bboxs, scale_x, scale_y):
    # need to convert to screen localization
    transformed = []
    for bbox in ui_bboxs:
        bbox_width = bbox.x2 - bbox.x1
        bbox_height = bbox.y2 - bbox.y1
        new_x1 = bbox.x1 * scale_x
        new_y1 = (bbox.y1 * scale_y) - 65
        new_x2 = (bbox.x1 + bbox_width) * scale_x
        new_y2 = ((bbox.y1 + bbox_height) * scale_y) - 65
        transformed.append([new_x1, new_y1, new_x2, new_y2])
    return transformed


def convert_bbox_to_midpoint(ui_bboxs):
    # ui bboxes are already scaled between 0 and 1
    mp = []
    for box in ui_bboxs:
        mid_x = 0.5 * (box[0] + box[2])
        mid_y = 0.5 * (box[1] + box[3])
        mp.append([mid_x, mid_y])
    return mp


def gesture_to_ui(view_path, gests, ui_obj_list, scale_x, scale_y, img_w, img_h):
    view = view_path.split('/')[-1]
    if len(gests[view]) > 1:
        # swiping action
        dist = np.linalg.norm(np.array(gests[view][-1]) - np.array(gests[view][0]))
        if dist > 0.01:
            return None, -1

    action = [gests[view][0][0] * img_w, gests[view][0][1] * img_h]
    raw_ui_bboxs = [elem.bounding_box for elem in ui_obj_list]
    scaled_ui_bboxs = convert_view_to_screen_dims(raw_ui_bboxs, scale_x, scale_y)
    midpoints = convert_bbox_to_midpoint(scaled_ui_bboxs)
    min_dist = 10000000
    min_dist_obj = None
    min_dist_obj_idx = None
    idx = 0
    for i in range(len(midpoints)):
        pt = midpoints[i]

        # calculate distance between action screen location
        # and each ui leaf node (distance to its midpoint)
        curr_dist = np.linalg.norm(np.array(action) - np.array(pt))

        # also check that the gesture falls within the lowest distance bbox
        pt_in_box = (action[0] >= scaled_ui_bboxs[i][0]) and (action[0] <= scaled_ui_bboxs[i][2]) and (action[1] >= scaled_ui_bboxs[i][1]) and (action[1] <= scaled_ui_bboxs[i][3])
        if curr_dist < min_dist and pt_in_box:
            min_dist = curr_dist
            min_dist_obj = ui_obj_list[idx]
            min_dist_obj_idx = idx
        idx += 1

    # we do this again if there isn't a min dist obj yet
    # a weird edge case for some reason
    if not min_dist_obj:   
        idx = 0 
        for i in range(len(midpoints)):
            pt = midpoints[i]
            # calculate distance between action screen location
            # and each ui leaf node (distance to its midpoint)
            curr_dist = np.linalg.norm(np.array(action) - np.array(pt))
            if curr_dist < min_dist:
                min_dist = curr_dist
                min_dist_obj = ui_obj_list[idx]
                min_dist_obj_idx = idx
            idx += 1

    assert min_dist_obj is not None
    assert min_dist_obj_idx is not None
    return min_dist_obj, min_dist_obj_idx

def check_consecutive(uis):
    consec = False
    for i in range(len(uis) - 1):
        first = uis[i]
        second = uis[i + 1]
        if (not first) or (not second):
            # swiping action
            continue
        if first.obj_type.value == 4 and second.obj_type.value == 4:
            # back to back typing events that constitute the same typing input
            consec = True
    return consec


def remove_to_keep(idxs, total_obj):
    # convert slices of duplicates to the slices we want to keep
    # it's just easier this way
    keep_slices = []
    begin_slice = [0, idxs[0][0]]
    keep_slices.append(begin_slice)
    for i in range(len(idxs) - 1):
        curr_slice = idxs[i]
        next_slice = idxs[i + 1]
        keep_slices.append([curr_slice[1], next_slice[0]])
    end_slice = [idxs[-1][1], total_obj]
    keep_slices.append(end_slice)
    return keep_slices


def get_text_duplicates(ui_objs):
    # if there are consecutive typing events contributing to the same
    # final text input, find their indices
    slices = []
    for i in range(len(ui_objs) - 1):
        curr_obj = ui_objs[i]
        if (not curr_obj) or curr_obj.obj_name is None:
            continue
        if curr_obj.obj_type.value == 4:
            # typing object, start slice here
            begin = i
            end = i
            for j in range(i + 1, len(ui_objs)):
                next_obj = ui_objs[j]
                if (not next_obj) or next_obj.obj_name is None:
                    break
                if next_obj.obj_type.value == 4:
                    if curr_obj.obj_name in next_obj.obj_name:
                        # continued typing of same element
                        end = j
                    else:
                        break
                else:
                    break
            if begin == end:
                continue
            else:
                # there may be overlapping slices
                # don't add unless new and non-overlapping
                new_s = [begin, end]
                add = True
                for s in slices:
                    if new_s[0] >= s[0] and new_s[1] <= s[1]:
                        add = False
                if add:
                    slices.append(new_s)
    if len(slices) > 0:
        slices = remove_to_keep(slices, len(ui_objs))
    return slices


def remove_dups(keep_slices, chosen_uis, views_used, action_instrs, target_idxs, actions):
    clean_chosen_ui = []
    clean_views = []
    clean_action_instrs = []
    clean_target_idxs = []
    clean_actions = []
    for i in range(len(keep_slices)):
        begin = keep_slices[i][0]
        end = keep_slices[i][1]
        clean_chosen_ui += chosen_uis[begin:end]
        clean_views += views_used[begin:end]
        clean_action_instrs += action_instrs[begin:end]
        clean_target_idxs += target_idxs[begin:end]
        clean_actions += actions[begin:end]
    return clean_chosen_ui, clean_views, clean_action_instrs, clean_target_idxs, clean_actions


def get_actions(all_views, all_gestures, all_actions, all_chosen_objs, kept_view_idxs, target_obj_idxs, vh_w, vh_h, scale_x, scale_y):
    # get action classes for each t step
    # and additional metadata needed for annotations
    action_instrs = []
    chosen_uis = []
    views_used = []
    text_so_far = []
    kept_target_idxs = []
    kept_actions = []
    all_leaf_nodes = []
    added_back = False
    for view_idx in kept_view_idxs:
        view_path = all_views[view_idx]
        view_chosen = all_chosen_objs[view_idx]
        view_target_idx = target_obj_idxs[view_idx]
        view_action = all_actions[view_idx]
        if not view_chosen and view_target_idx == 0:
            # reasons could be: missing gesture, corrupted view hierarchy, missing min dist UI object
            continue

        view = view_path.split('/')[-1]
        missing = (len(all_gestures[view]) == 0)

        if missing and view_path == all_views[-1]:
            # case where final action state achieved
            # and no action is taken there 
            views_used.append(view_path)
            continue

        view_hierarchy_leaf_nodes = common.get_view_hierarchy_list(view_path, vh_w, vh_h)
        ui_obj_list = [ele.uiobject for ele in view_hierarchy_leaf_nodes]

        all_curr_text = [[ui.obj_name, ui.text]for ui in ui_obj_list]
        all_curr_text = [y for x in all_curr_text for y in x]
        if 'Google Play Store keeps stopping' in all_curr_text:
            # emulator or phone issue
            continue

        # first check if any typing occurred
        idx = 0
        for ele in view_hierarchy_leaf_nodes:
            if ele.uiobject.obj_type.value == 4 and ele.uiobject.obj_name is not None:
                # EDITTEXT Type
                if ele.uiobject.obj_name.lower() in text_so_far:
                    # don't confuse text filled in from previous t steps
                    # as a new typing event
                    continue

                # keep track of text inputs
                # to make sure we don't consider a single type input
                # as multiple smaller 
                text_so_far.append(ele.uiobject.obj_name.lower())
                
                chosen = ele.uiobject
                
                chosen_uis.append(chosen)
                views_used.append(view_path)
                # add in index of correct element associated with type event
                kept_target_idxs.append(idx)
                instr, action_type, input_content_str, verb_str, obj_desc_str, was_missing = real_action_generator.get_type_info(chosen, ui_obj_list)
                action_instrs.append(instr)
                action = common.Action(
                    instruction_str=instr,
                    verb_str=verb_str,
                    obj_desc_str=obj_desc_str,
                    input_content_str=input_content_str,
                    action_type=action_type,
                    action_rule=common.ActionRules.REAL,
                    target_obj_idx=idx) # correct text element idx
                kept_actions.append(action)
                all_leaf_nodes.append(view_hierarchy_leaf_nodes)
                added_back = True
            idx+=1

        if not missing:
            # get chosen object and its index in list of ui objects
            # this index becomes the class label for grounding
            # (if grounding is performed over UI objects and not screen)

            # only want to add to features from views we have with gestures
            # and don't want to have duplicate swipe events
            if view_chosen is None:
                if view_target_idx == -1:
                    # swiping event
                    next_view_idx = kept_view_idxs.index(view_idx) + 1
                    safe_to_idx = (next_view_idx < len(kept_view_idxs))
                    if safe_to_idx and all_chosen_objs[kept_view_idxs[next_view_idx]]:
                        # then we've reached final swipe action
                        chosen_uis.append(view_chosen)
                        views_used.append(view_path)
                        kept_target_idxs.append(view_target_idx)
                        action_instrs.append(view_action.instruction_str)
                        kept_actions.append(view_action)
            else:
                # not swiping, don't need to check
                # if there was a typing event prior to this, this action still needs to be captured
                chosen_uis.append(view_chosen)
                views_used.append(view_path)
                kept_target_idxs.append(view_target_idx)
                action_instrs.append(view_action.instruction_str)
                kept_actions.append(view_action)
            all_leaf_nodes.append(view_hierarchy_leaf_nodes)

    assert len(chosen_uis) == len(action_instrs) 

    consec_et = check_consecutive(chosen_uis)
    if consec_et:
        dup_slices = get_text_duplicates(chosen_uis)
        if len(dup_slices) > 0:
            (chosen_uis, views_used, action_instrs,
                kept_target_idxs, kept_actions) = remove_dups(dup_slices, chosen_uis, views_used, action_instrs, kept_target_idxs, kept_actions)

    assert len(chosen_uis) == len(action_instrs) == len(all_leaf_nodes)

    view_bboxes = []
    screen_bboxes = []
    action_class = []
    ui_types = []
    chosen_uis_text = []
    # made it this far, view was actually used and duplicates have been accounted for
    for entry in chosen_uis:
        if not entry:
            action_class.append('swipe')
            ui_types.append(0)
            chosen_uis_text.append("")
            view_bboxes.append(None)
            screen_bboxes.append(None)
        elif entry.obj_type.value == 4 and entry.obj_name is not None:
            action_class.append('type')
            b = entry.bounding_box
            view_bboxes.append(b)
            # currently in view hierarchy coordinates, not screen
            screen_b = convert_view_to_screen_dims([b], scale_x, scale_y)
            screen_bboxes.append(screen_b[0])
            ui_types.append(entry.obj_type.value)
            chosen_uis_text.append(entry.obj_name.lower())
        else:
            action_class.append('click')
            b = entry.bounding_box
            view_bboxes.append(b)
            # currently in view hierarchy coordinates, not screen
            screen_b = convert_view_to_screen_dims([b], scale_x, scale_y)
            screen_bboxes.append(screen_b[0])
            ui_types.append(entry.obj_type.value)
            x = entry.text
            y = entry.obj_name
            if y:
                chosen_uis_text.append(entry.obj_name.lower())
            else:
                chosen_uis_text.append(entry.text)
    return action_class, action_instrs, ui_types, chosen_uis_text, view_bboxes, screen_bboxes, views_used, kept_target_idxs, chosen_uis, kept_actions

def make_clean_trace(app, trace_id):
    trace_path = os.path.join(FLAGS.data_dir, app, trace_id)
    views, gestures = get_metadata(trace_path)
    img = Image.open(views[0].replace('view_hierarchies', 'screens'))
    
    try:
        vh_w, vh_h = get_screen_dims(views)
        scale_x = img.width / vh_w
        scale_y = (img.height + 65) / vh_h
    except:
        iss = 'Screen dimension issue: ' + trace_path
        print(iss)
        return list(range(len(views))), [iss]

    all_view_uids = []
    all_view_actions = []
    all_text = []
    all_chosen = []
    all_idxs = []
    kept_gestures = []
    login_keywords = ['agree','log in', 'login', 'sign in', 'signin', 'allow', 'ziptuser2@gmail.com', 'appcrawler3116']
    for view in views:
        try:
            with open(view) as f:
                vh = json.load(f)
        except:
            # saved view hierarchy json issue
            all_view_uids.append('') 
            all_view_actions.append(None)
            all_chosen.append(None)
            all_idxs.append(0)
            kept_gestures.append(None)
            all_text.append("no login issue")
            continue

        uid = get_uid(view, vh['activity']['root'], vh_w, vh_h)
        all_view_uids.append(uid)

        view_hierarchy_leaf_nodes = common.get_view_hierarchy_list(view, vh_w, vh_h)
        ui_obj_list = [ele.uiobject for ele in view_hierarchy_leaf_nodes]

        if len(gestures[view.split('/')[-1]]) > 0:
            try:
                chosen, idx = gesture_to_ui(view, gestures, ui_obj_list, scale_x, scale_y, img.width, img.height)
            except:
                # min dist object is None
                all_view_actions.append(None)
                all_chosen.append(None)
                all_idxs.append(0)
                kept_gestures.append(None)
                all_text.append("no login issue")
                continue

            all_chosen.append(chosen)
            all_idxs.append(idx)
            actions, _ = real_action_generator.load_all_actions([chosen], [idx], trace_path, [gestures[view.split('/')[-1]]])

            all_view_actions.append(actions[0])
            if idx != -1:
                if chosen.text.lower() in login_keywords:
                    all_text.append("login")
                elif chosen.obj_name and chosen.obj_name.lower() in login_keywords:
                    all_text.append("login")
                else:
                    all_text.append("no login issue")
            else:
                all_text.append("no login issue")
        else:
            # missing gesture
            all_view_actions.append(None)
            all_text.append("no login issue")
            all_chosen.append(None)
            all_idxs.append(0)
        kept_gestures.append(gestures[view.split('/')[-1]])

    all_actions, _ = real_action_generator.load_all_actions(all_chosen, all_idxs, trace_path, kept_gestures)
    assert len(all_text) == len(all_view_actions) == len(all_view_uids)
    to_save = clean_idxs(all_view_uids, all_text, all_view_actions, all_chosen)

    return views, gestures, to_save, all_view_uids, all_actions, all_chosen, all_idxs, img.width, img.height, vh_w, vh_h, scale_x, scale_y

def clean_idxs(all_view_uids, all_text, all_view_actions, all_chosen_uis):
    keep_idxs = []
    kept_uids = []

    # if the same state occurs back to back, take the last one
    # as this may be due to interface delays/failures
    for i in range(len(all_view_uids)-1):
        if all_view_uids[i] == '':
            # couldn't load view hierarchy json
            continue

        if all_view_uids[i] in kept_uids and FLAGS.dedup_cycles:
            # start over from cycle idx
            start_slice_idx = kept_uids.index(all_view_uids[i])
            end_slice_idx = i
            keep_idxs = keep_idxs[:start_slice_idx] 
            kept_uids = kept_uids[:start_slice_idx] 
        
        if ((all_chosen_uis[i] == all_chosen_uis[i+1]) or (all_view_uids[i] == all_view_uids[i+1])) and all_view_actions[i]:
            # consecutive views which have the same state
            if not all_view_actions[i+1]:
                # if the second doesn't have an action, then we want to keep the first
                keep_idxs.append(i)
                kept_uids.append(all_view_uids[i])
            else:
                # otherwise they both have acions
                if (all_view_actions[i].action_type == all_view_actions[i+1].action_type) and (all_view_actions[i].target_obj_idx == all_view_actions[i+1].target_obj_idx):
                    # duplicate event where state and action are same at consecutive tsteps
                    # this could happen if the UI was unresponsive for a moment
                    continue
        else:
            # different states
            keep_idxs.append(i)
            kept_uids.append(all_view_uids[i])

    if all_view_uids[-1] not in kept_uids: 
        # if final state wasn't added (which it wouldn't be if it didn't have a gesture)
        # add it back because we want a final state (all sequences end up being +1 the # of actions) 
        assert len(all_view_uids) > 0
        keep_idxs.append(len(all_view_uids)-1)

    return keep_idxs


def get_feat_dict(app, trace_id, views, target_idxs, instrs, actions, action_classes, action_types, ui_objs_text, ui_objs, screen_bboxes, view_bboxes, raw_gestures, im_w, im_h, vh_w, vh_h, scale_x, scale_y):
    feat_dict = {}

    feat_dict['trace_id'] = trace_id
    feat_dict['goal'] = TASK_I_TO_N[trace_id]
    feat_dict['instr'] = instrs
    feat_dict['app'] = app
    feat_dict['screen_w'] = im_w
    feat_dict['screen_h'] = im_h
    feat_dict['vh_w'] = vh_w
    feat_dict['vh_h'] = vh_h
    feat_dict['scale_x'] = scale_x
    feat_dict['scale_y'] = scale_y

    login_keywords = ['log in', 'login', 'sign in', 'signin', 'allow', 'ziptuser2@gmail.com', 'appcrawler3116', 'password']
    ui_text = [x.lower() for x in ui_objs_text]
    indices = [0]
    had_login = False
    for t in ui_text:
        if t in login_keywords:
            had_login = True
            idx = len(ui_text) - list(reversed(ui_text)).index(t)
            indices.append(idx)

    shortcut = max(indices)
    feat_dict['actions'] = action_classes
    feat_dict['verb_str'] = [action.verb_str for action in actions]
    feat_dict['obj_desc_str'] = [action.obj_desc_str for action in actions]
    feat_dict['input_str'] = [action.input_content_str for action in actions]
    feat_dict['ui_types'] = action_types
    feat_dict['screen_bboxes'] = screen_bboxes
    feat_dict['view_bboxes'] = [[b.x1, b.y1, b.x2, b.y2] if b else [] for b in view_bboxes]
    feat_dict['images'] = [view.split('/')[-1][:-4] for view in views]
    feat_dict['raw_gestures'] = [raw_gestures[view.split('/')[-1]] for view in views]
    feat_dict['ui_target_idxs'] = target_idxs

    if len(feat_dict['images']) == len(feat_dict['actions']):
        # add redundant last state for stop action at the end
        feat_dict['images'].append(feat_dict['images'][-1])
    if len(feat_dict['images']) > len(feat_dict['actions']) + 1:
        feat_dict['images'] = feat_dict['images'][:-1]
    if len(feat_dict['images']) != len(feat_dict['actions']) + 1:
        feat_dict['images'] = feat_dict['images'][:len(feat_dict['actions']) + 1]

    if shortcut < len(ui_text):
        feat_dict['actions'] = feat_dict['actions'][shortcut:]
        feat_dict['ui_types'] = feat_dict['ui_types'][shortcut:]
        feat_dict['screen_bboxes'] = feat_dict['screen_bboxes'][shortcut:]
        feat_dict['view_bboxes'] = feat_dict['view_bboxes'][shortcut:]
        feat_dict['images'] = feat_dict['images'][shortcut:]
        feat_dict['instr'] = feat_dict['instr'][shortcut:]
        feat_dict['raw_gestures'] = feat_dict['raw_gestures'][shortcut:]
        feat_dict['ui_target_idxs'] = feat_dict['ui_target_idxs'][shortcut:]
        feat_dict['verb_str'] = feat_dict['verb_str'][shortcut:]
        feat_dict['obj_desc_str'] = feat_dict['obj_desc_str'][shortcut:]
        feat_dict['input_str'] = feat_dict['input_str'][shortcut:]

        ui_objs = ui_objs[shortcut:]

    assert len(feat_dict['actions']) > 0
    assert (len(feat_dict['actions']) == len(feat_dict['ui_types']) == len(feat_dict['view_bboxes'])
        == len(feat_dict['screen_bboxes']) == len(feat_dict['instr']) == (len(feat_dict['images']) - 1))
    return feat_dict, ui_objs


def main():
    if not os.path.isdir(FLAGS.save_dir):
        os.mkdir(FLAGS.save_dir)

    samples_clean = {}
    with open('eccv_motif_app_seen_task_unseen_curr.json') as f:
        splits_1 = json.load(f)
        splits_1 = set(splits_1['train'] + splits_1['test'])
    
    with open('eccv_motif_app_seen_task_unseen_all.json') as f:
        splits_2 = json.load(f)
        splits_2 = set(splits_2['train'] + splits_2['test'])

    with open('eccv_motif_app_unseen_task_unseen.json') as f:
        splits_3 = json.load(f)
        splits_3 = set(splits_3['train'] + splits_3['test'])
    
    with open('eccv_motif_app_unseen_task_seen.json') as f:
        splits_4 = json.load(f)
        splits_4 = set(splits_4['train'] + splits_4['test'])


    traces = list(splits_1.union(splits_2).union(splits_3).union(splits_4))
    print(len(traces))
    print(FLAGS.dedup_cycles)
    for i in range(len(traces)):
        trace = traces[i]
        if i % 100 == 0:
            print(i)

        app = glob.glob(os.path.join(FLAGS.data_dir, '*', trace))[0]
        app = app.split('/')[-2]

        (all_views, all_gestures, saved_view_idxs, kept_view_uids, 
            kept_view_actions, kept_chosen_objs, target_idxs, im_w, im_h, vh_w, vh_h, scale_x, scale_y) = make_clean_trace(app, trace)
        (action_class, action_instrs, ui_types, chosen_uis_text, 
            view_bboxes, screen_bboxes, final_views_used, final_tidxs, ui_objs, actions) = get_actions(all_views, all_gestures, kept_view_actions, kept_chosen_objs, saved_view_idxs, target_idxs, vh_w, vh_h, scale_x, scale_y)
        
        if len(final_views_used) > 0:
            (fd, uis) = get_feat_dict(app, trace, final_views_used, final_tidxs, action_instrs, actions, action_class, ui_types, chosen_uis_text, ui_objs, screen_bboxes, view_bboxes, all_gestures, im_w, im_h, vh_w, vh_h, scale_x, scale_y)
        else:
            continue

        with open(os.path.join(FLAGS.save_dir, trace + '.json'), 'w') as f:
            json.dump(fd, f)
main()
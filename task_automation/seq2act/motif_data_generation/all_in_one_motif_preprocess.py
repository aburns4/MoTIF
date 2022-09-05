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
flags.DEFINE_list(
    "splits_to_process",
    ["eccv_motif_app_seen_task_unseen_curr.json", "eccv_motif_app_seen_task_unseen_all.json",
     "eccv_motif_app_unseen_task_unseen.json", "eccv_motif_app_unseen_task_seen.json"],
    "List of filenames storing dataset splits to load and process.")

def load_traces(split_list):
    # load all trace ids for a list of splits
    to_load = set()
    for split in split_list:
        with open(split) as f:
            data = json.load(f)
            to_load.update(data['train'] + data['test'])
    return list(to_load)


def remove_dups(keep_slices, to_dedup):
    clean = []
    for i in range(len(keep_slices)):
        begin = keep_slices[i][0]
        end = keep_slices[i][1]
        clean += to_dedup[begin:end]
    return clean


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


def convert_bbox_to_midpoint(ui_bboxs):
    # ui bboxes are already scaled between 0 and 1
    mp = []
    for box in ui_bboxs:
        mid_x = 0.5 * (box[0] + box[2])
        mid_y = 0.5 * (box[1] + box[3])
        mp.append([mid_x, mid_y])
    return mp


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


class MoTIFSample(object):

    def __init__(self, app, data_dir, trace_id, widget_exceptions):
        # load task id to name mapping
        TASK_I_TO_N = {}
        with open(os.path.join(data_dir, 'tasknames.csv')) as csvfile:
            reader = csvfile.readlines()
            for row in reader:
                row = row.strip().split(' ')
                TASK_I_TO_N[row[0]] = ' '.join(row[1:])
        
        with open(widget_exceptions) as f:
            self.widget_exceptions = json.load(f)

        self.app = app
        self.task_i_to_n = TASK_I_TO_N
        self.trace_id = trace_id
        self.trace_path = os.path.join(data_dir, app, trace_id)

        self.set_metadata()
        self.set_screen_dims()

        img = Image.open(self.og_view_paths[0].replace('view_hierarchies', 'screens'))
        self.im_w = img.width
        self.im_h = img.height
        self.scale_x = self.im_w / self.vh_w
        self.scale_y = (self.im_h + 65) / self.vh_h


    def set_metadata(self):
        metadata_path = os.path.join(self.trace_path, 'metadata.json')
        with open(metadata_path) as f:
            metadata = json.load(f)
            gestures = metadata['gestures']
            view_paths = [os.path.join(self.trace_path, 'view_hierarchies', x) for x in metadata['gestures'].keys()]
        self.og_view_paths = view_paths
        self.og_gestures = gestures

    def set_screen_dims(self):
        trace = self.og_view_paths[0].split('/')[-3]

        if trace in self.widget_exceptions:
            self.vh_w = int(self.widget_exceptions[trace][0])
            self.vh_h = int(self.widget_exceptions[trace][1])
        else:
            for view in self.og_view_paths:
                try:
                    with open(view, 'r') as f:
                        data = json.load(f)
                        bbox = data['activity']['root']['bounds']
                        width = bbox[2] - bbox[0]
                        height = bbox[3] - bbox[1]
                    if bbox[0] == 0 and bbox[1] == 0:
                        self.vh_w = width
                        self.vh_h = height
                        break
                except:
                    continue
        

    def get_uid(self, view_path):
        # get unique id for state in state-action graph
        # consists of concat pointer values for all elems
        # which are visible and selected
        # followed by concat word2index str of leaf node text
        view_hierarchy_leaf_nodes = common.get_view_hierarchy_list(view_path, self.vh_w, self.vh_h)
        ui_objs = [ele.uiobject for ele in view_hierarchy_leaf_nodes]
        ui_objs_str = [ui_obj_to_str(ui) for ui in ui_objs]
        return " ".join(ui_objs_str)

    def convert_view_to_screen_dims(self, ui_bboxs):
        # need to convert to screen localization
        transformed = []
        for bbox in ui_bboxs:
            bbox_width = bbox.x2 - bbox.x1
            bbox_height = bbox.y2 - bbox.y1
            new_x1 = bbox.x1 * self.scale_x
            new_y1 = (bbox.y1 * self.scale_y) - 65
            new_x2 = (bbox.x1 + bbox_width) * self.scale_x
            new_y2 = ((bbox.y1 + bbox_height) * self.scale_y) - 65
            transformed.append([new_x1, new_y1, new_x2, new_y2])
        return transformed

    def gesture_to_ui(self, view_path, ui_obj_list):
        view = view_path.split('/')[-1]
        if len(self.og_gestures[view]) > 1:
            # swiping action
            dist = np.linalg.norm(np.array(self.og_gestures[view][-1]) - np.array(self.og_gestures[view][0]))
            if dist > 0.01:
                return None, -1

        action = [self.og_gestures[view][0][0] * self.im_w,
                  self.og_gestures[view][0][1] * self.im_h]
        raw_ui_bboxs = [elem.bounding_box for elem in ui_obj_list]
        scaled_ui_bboxs = self.convert_view_to_screen_dims(raw_ui_bboxs)
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
            pt_in_box = ((action[0] >= scaled_ui_bboxs[i][0]) 
                         and (action[0] <= scaled_ui_bboxs[i][2])
                         and (action[1] >= scaled_ui_bboxs[i][1])
                         and (action[1] <= scaled_ui_bboxs[i][3]))
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

    def make_clean_trace(self):
        all_view_uids = []
        all_view_actions = []
        all_text = []
        all_chosen = []
        all_idxs = []
        kept_gestures = []
        login_keywords = ['agree','log in', 'login', 'sign in', 'signin', 'allow', 'ziptuser2@gmail.com', 'appcrawler3116']
        for view in self.og_view_paths:
            try:
                with open(view) as f:
                    vh = json.load(f)
            except:
                # saved view hierarchy json issue
                # print('View hierarchy issue')
                all_view_uids.append('') 
                all_view_actions.append(None)
                all_chosen.append(None)
                all_idxs.append(0)
                kept_gestures.append(None)
                all_text.append("no login issue")
                continue

            uid = self.get_uid(view)
            all_view_uids.append(uid)


            view_hierarchy_leaf_nodes = common.get_view_hierarchy_list(view, self.vh_w, self.vh_h)
            ui_obj_list = [ele.uiobject for ele in view_hierarchy_leaf_nodes]

            if len(self.og_gestures[view.split('/')[-1]]) > 0:
                try:
                    chosen, idx = self.gesture_to_ui(view, ui_obj_list)
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
                actions, _ = real_action_generator.load_all_actions([chosen], [idx], self.task_i_to_n[self.trace_id], [self.og_gestures[view.split('/')[-1]]])

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
            kept_gestures.append(self.og_gestures[view.split('/')[-1]])

        all_actions, _ = real_action_generator.load_all_actions(all_chosen, all_idxs, self.task_i_to_n[self.trace_id], kept_gestures)
        final_idxs = [x.target_obj_idx if x else None for x in all_actions]
        assert len(all_text) == len(all_view_actions) == len(all_view_uids)
        to_save = clean_idxs(all_view_uids, all_text, all_view_actions, all_chosen)

        self.to_save_view_idxs = to_save
        self.all_view_uids = all_view_uids
        self.all_actions = all_actions
        self.all_chosen_ui = all_chosen
        self.target_idxs_before = all_idxs
        self.target_idxs_after = final_idxs


    def set_actions(self):
        # get action classes for each t step
        # and additional metadata needed for annotations
        action_instrs = []
        chosen_uis = []
        views_used = []
        text_so_far = []
        kept_target_idxs = []
        kept_target_idxs_before = []
        kept_actions = []
        all_leaf_nodes = []
        added_back = False
        for view_idx in self.to_save_view_idxs:
            view_path = self.og_view_paths[view_idx]
            view_chosen = self.all_chosen_ui[view_idx]
            view_target_idx = self.target_idxs_after[view_idx]
            view_target_idx_before = self.target_idxs_before[view_idx]
            view_action = self.all_actions[view_idx]
            if not view_chosen and view_target_idx_before == 0:
                # reasons could be: missing gesture, corrupted view hierarchy, missing min dist UI object
                print('accounted for error')
                continue

            view = view_path.split('/')[-1]
            missing = (len(self.og_gestures[view]) == 0)

            if missing and view_path == self.og_view_paths[-1]:
                # case where final action state achieved
                # and no action is taken there 
                views_used.append(view_path)
                continue

            view_hierarchy_leaf_nodes = common.get_view_hierarchy_list(view_path, self.vh_w, self.vh_h)
            ui_obj_list = [ele.uiobject for ele in view_hierarchy_leaf_nodes]

            all_curr_text = [[ui.obj_name, ui.text] for ui in ui_obj_list]
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
                    kept_target_idxs_before.append(idx)
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
                    # swiping event
                    next_view_idx = self.to_save_view_idxs.index(view_idx) + 1
                    safe_to_idx = (next_view_idx < len(self.to_save_view_idxs))
                    if safe_to_idx and self.all_chosen_ui[self.to_save_view_idxs[next_view_idx]]:
                        # then we've reached final swipe action
                        chosen_uis.append(view_chosen)
                        views_used.append(view_path)
                        kept_target_idxs.append(view_target_idx)
                        kept_target_idxs_before.append(view_target_idx_before)
                        action_instrs.append(view_action.instruction_str)
                        kept_actions.append(view_action)
                        all_leaf_nodes.append(view_hierarchy_leaf_nodes)
                else:
                    # not swiping, don't need to check
                    # if there was a typing event prior to this, this action still needs to be captured
                    chosen_uis.append(view_chosen)
                    views_used.append(view_path)
                    kept_target_idxs.append(view_target_idx)
                    kept_target_idxs_before.append(view_target_idx_before)
                    action_instrs.append(view_action.instruction_str)
                    kept_actions.append(view_action)
                    all_leaf_nodes.append(view_hierarchy_leaf_nodes)

        assert len(chosen_uis) == len(action_instrs) == len(all_leaf_nodes)

        consec_et = check_consecutive(chosen_uis)
        if consec_et:
            dup_slices = get_text_duplicates(chosen_uis)
            if len(dup_slices) > 0:
                chosen_uis = remove_dups(dup_slices, chosen_uis)
                views_used = remove_dups(dup_slices, views_used)
                action_instrs = remove_dups(dup_slices, action_instrs)
                kept_target_idxs = remove_dups(dup_slices, kept_target_idxs)
                kept_target_idxs_before = remove_dups(dup_slices, kept_target_idxs_before)
                kept_actions = remove_dups(dup_slices, kept_actions)

        assert len(chosen_uis) == len(action_instrs) 

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
                screen_b = self.convert_view_to_screen_dims([b])
                screen_bboxes.append(screen_b[0])
                ui_types.append(entry.obj_type.value)
                chosen_uis_text.append(entry.obj_name.lower())
            else:
                action_class.append('click')
                b = entry.bounding_box
                view_bboxes.append(b)
                # currently in view hierarchy coordinates, not screen
                screen_b = self.convert_view_to_screen_dims([b])
                screen_bboxes.append(screen_b[0])
                ui_types.append(entry.obj_type.value)
                x = entry.text
                y = entry.obj_name
                if y:
                    chosen_uis_text.append(entry.obj_name.lower())
                else:
                    chosen_uis_text.append(entry.text)

        self.dedup_action_class = action_class
        self.dedup_instrs = action_instrs
        self.dedup_ui_types = ui_types
        self.dedup_ui_text = chosen_uis_text
        self.dedup_view_bbox = view_bboxes
        self.dedup_screen_bbox = screen_bboxes
        self.dedup_views = views_used
        self.dedup_target_idxs_after = kept_target_idxs
        self.dedup_target_idxs_before = kept_target_idxs_before
        self.dedup_ui_objs = chosen_uis
        self.dedup_actions = kept_actions


    def get_feat_dict(self):
        print(self.trace_path)
        self.make_clean_trace()
        self.set_actions()

        if len(self.dedup_views) == 0:
            return {}, None

        feat_dict = {}

        feat_dict['trace_id'] = self.trace_id
        feat_dict['goal'] = self.task_i_to_n[self.trace_id]
        feat_dict['instr'] = self.dedup_instrs
        feat_dict['app'] = self.app
        feat_dict['screen_w'] = self.im_w
        feat_dict['screen_h'] = self.im_h
        feat_dict['vh_w'] = self.vh_w
        feat_dict['vh_h'] = self.vh_h
        feat_dict['scale_x'] = self.scale_x
        feat_dict['scale_y'] = self.scale_y

        login_keywords = ['log in', 'login', 'sign in', 'signin', 'allow', 'ziptuser2@gmail.com', 'appcrawler3116', 'password']
        ui_text = [x.lower() for x in self.dedup_ui_text]
        indices = [0]
        had_login = False
        for t in ui_text:
            if t in login_keywords:
                had_login = True
                idx = len(ui_text) - list(reversed(ui_text)).index(t)
                indices.append(idx)

        shortcut = max(indices)
        feat_dict['actions'] = self.dedup_action_class
        feat_dict['verb_str'] = [action.verb_str for action in self.dedup_actions]
        feat_dict['obj_desc_str'] = [action.obj_desc_str for action in self.dedup_actions]
        feat_dict['input_str'] = [action.input_content_str for action in self.dedup_actions]
        feat_dict['ui_types'] = self.dedup_ui_types
        feat_dict['screen_bboxes'] = self.dedup_screen_bbox
        feat_dict['view_bboxes'] = [[b.x1, b.y1, b.x2, b.y2] if b else [] for b in self.dedup_view_bbox]
        feat_dict['images'] = [view.split('/')[-1][:-4] for view in self.dedup_views]
        feat_dict['raw_gestures'] = [self.og_gestures[view.split('/')[-1]] for view in self.dedup_views]
        feat_dict['ui_target_idxs'] = self.dedup_target_idxs_after # after swipe is processed
        feat_dict['ui_target_idxs_before'] = self.dedup_target_idxs_before # before swipe is processed

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
            feat_dict['ui_target_idxs_before'] = feat_dict['ui_target_idxs_before'][shortcut:]
            feat_dict['verb_str'] = feat_dict['verb_str'][shortcut:]
            feat_dict['obj_desc_str'] = feat_dict['obj_desc_str'][shortcut:]
            feat_dict['input_str'] = feat_dict['input_str'][shortcut:]

            self.dedup_ui_objs = self.dedup_ui_objs[shortcut:]

        assert len(feat_dict['actions']) > 0
        assert (len(feat_dict['actions']) == len(feat_dict['ui_types']) == len(feat_dict['view_bboxes'])
            == len(feat_dict['screen_bboxes']) == len(feat_dict['instr']) == (len(feat_dict['images']) - 1))
        return feat_dict, self.dedup_ui_objs


def main():
    if not os.path.isdir(FLAGS.save_dir):
        os.mkdir(FLAGS.save_dir)

    traces = load_traces(FLAGS.splits_to_process)
    print(len(traces))
    print(FLAGS.dedup_cycles)
    for i in range(len(traces)):
        trace = traces[i]
        if i % 100 == 0:
            print(i)

        app = glob.glob(os.path.join(FLAGS.data_dir, '*', trace))[0]
        app = app.split('/')[-2]

        mysample = MoTIFSample(app, FLAGS.data_dir, trace, 'widget_exception_dims.json')
        sample_dict, objs = mysample.get_feat_dict()

        if sample_dict:
            with open(os.path.join(FLAGS.save_dir, trace + '.json'), 'w') as f:
                json.dump(sample_dict, f)
main()
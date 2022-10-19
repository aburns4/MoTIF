# coding=utf-8
# Copyright 2021 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Creates screen dataset with tfExample proto in TFRecord format.

For all the valid xml or json files in the input directory, it parses their
view hierarchy attributes, extracts the feature data into tf.train.Example proto
and saves the results with TFRecord format in the output
directory as multiple sharded files. A file containing the dimension data for
padding purpose is also created.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import operator
import os
import threading

import concurrent.futures
import numpy as np
import tensorflow.compat.v1 as tf
import json
import glob
import json
from PIL import Image

import common
import config
import proto_utils
import real_action_generator
import view_hierarchy

gfile = tf.gfile
flags = tf.flags
FLAGS = flags.FLAGS

_INPUT_DIR = '/tmp/input'
_OUTPUT_DIR = '/tmp/'
_FILTER_FILE = '/tmp/'

_NUM_THREADS_DEFAULT = 10
_PADDING_DIMENSIONS_FILE_NAME = 'padding_dimensions.txt'
_TOKEN_TYPE = 'subtoken'
_NUM_SHARDS_DEFAULT = 10  # config.SHARD_NUM
_MAX_WORD_NUM_UPPER_DEFAULT = config.MAX_WORD_NUM_UPPER_BOUND
_MAX_WORD_LENGTH_UPPER_DEFAULT = config.MAX_WORD_LENGTH_UPPER_BOUND
_DATASET_TYPE_DEFAULT = 'motif'
_STATS_DIMENSIONS = True
_MAX_WORD_NUM = 200
_MAX_WORD_LENGTH = 50

_FREQ_OBJ_TYPE = [
    view_hierarchy.UIObjectType.UNKNOWN,
    view_hierarchy.UIObjectType.BUTTON,
    view_hierarchy.UIObjectType.IMAGEVIEW,
    view_hierarchy.UIObjectType.TEXTVIEW,
]

_INFREQUENT_OBJ_TYPE_MAX_RATIO = 0.9
_FILTER_ACTIONS_BY_NAME = True
_FILTER_ACTION_BY_TYPE = True

flags.DEFINE_enum(
    'split', 'all', ['uu', 'us', 'su_curr', 'su_all', 'all'],
    'Full path to the directory for saving filter file or RICO.')
flags.DEFINE_string(
    'input_dir', _INPUT_DIR,
    'Full path to the directory containing the data files for a set of tasks.')
flags.DEFINE_string(
    'output_dir', _OUTPUT_DIR,
    'Full path to the directory for saving the tf record file.')
flags.DEFINE_boolean(
    'use_high_level_goal', True,
    'Whether the instruction is defined as the high level goal or step by step instruction.')
flags.DEFINE_integer(
    'num_threads', _NUM_THREADS_DEFAULT,
    'The number of threads to process the data files concurrently.')
flags.DEFINE_integer(
    'num_shards', _NUM_SHARDS_DEFAULT,
    'The number of sharded files to save the created dataset.')
flags.DEFINE_integer('max_word_num_upper', _MAX_WORD_NUM_UPPER_DEFAULT,
                     'The max number of words for building model features.')
flags.DEFINE_integer('max_word_length_upper', _MAX_WORD_LENGTH_UPPER_DEFAULT,
                     'The max length of words for building model features.')
flags.DEFINE_enum('dataset', _DATASET_TYPE_DEFAULT,
                  ['android_settings', 'rico', 'motif'],
                  'The type of supported dataset.')
flags.DEFINE_enum('file_to_generate', 'tf_example',
                  ['tf_example', 'corpus'],
                  'Whether generate feature tfexample or corpus txt.')

longest_stats = collections.Counter()
distributions = collections.defaultdict(collections.Counter)

def get_kept_view_hierarchies(traces):
    vh_to_load = []
    screen_dims = []
    for tr in traces:
        json_path = os.path.join('seq2act_9_5', tr + '.json') # processed_motif_deduped
        if os.path.exists(json_path): 
            with open(json_path) as f:
                trace_info = json.load(f)
                views_kept = trace_info['images']
                view_paths = [os.path.join(FLAGS.input_dir, trace_info['app'], tr, 'view_hierarchies', v + '.jpg') for v in views_kept]
                vh_to_load += view_paths
                for i in range(len(view_paths)):
                    screen_dims.append([trace_info['vh_w'], trace_info['vh_h']])
        else:
            continue
    return vh_to_load, screen_dims

def _get_data_dimensions(input_dir, splits):
  """Processes the dimension data.

  The dimension data includes maximum word numbers and maximum word lengths
  from all the ui objects across all the .xml/.json files in input_dir.
  It will be used for padding purpose.

  The results are written in <output_dir>/padding_dimensions.txt.

  Args:
    input_dir: The directory that contains the input xml/json files.

  Returns:
    A tuple (max_word_num_all_files, max_word_length_all_files)
    max_word_num_all_files: The max number of words from all the ui objects
      across all the .xml/.json files.
    max_word_length_all_files: The max length of words from all the ui objects
      across all the .xml/.json files.
  """
  max_word_num_all_files = 0
  max_word_length_all_files = 0
  all_files, all_dims = get_kept_view_hierarchies(splits)
  print('%d view hierarchies to load' % len(all_files))
  # We can use ThreadPool since these are IO-bound operations.
  with concurrent.futures.ThreadPoolExecutor(FLAGS.num_threads) as executor:
    futures = []
    for i in range(len(all_files)):
      file_path = all_files[i]
      w, h = all_dims[i]
      futures.append(executor.submit(common.get_word_statistics, file_path, True, w, h))

    for future in concurrent.futures.as_completed(futures):
      _, max_word_num_one_file, max_word_length_one_file = future.result()
      max_word_num_all_files = max(max_word_num_all_files,
                                   max_word_num_one_file)
      max_word_length_all_files = max(max_word_length_all_files,
                                      max_word_length_one_file)
  tf.logging.info('max_word_num_all_files=%d, max_word_length_all_files=%d',
                  max_word_num_all_files, max_word_length_all_files)

  return max_word_num_all_files, max_word_length_all_files

def _process_dimensions(input_dir, output_dir, splits):
    """Processes the dimension data.

  The dimension data includes maximum word numbers and maximum word lengths
  from all the ui objects across all the .xml/.json files in input_dir.
  It will be used for padding purpose.

  The results are written in <output_dir>/padding_dimensions.txt.

  Args:
    input_dir: The directory that contains the input xml/json files.
    output_dir: The directory that saves output dimension data.

  Returns:
    A tuple (max_word_num, max_word_length)
    max_word_num: The max number of words for building model features.
    max_word_length: The max length of words for building model features.
  """
    tf.logging.info('Processing data dimensions...')
    max_word_num, max_word_length = _get_data_dimensions(input_dir, splits) # [200, 50]
    print((max_word_num, max_word_length))
    tf.logging.info('Done...max word num = %d, max word length = %d' % (max_word_num, max_word_length))

    # Apply pre-configured upper bound to clip possibly rare outlier values.
    max_word_num = min(max_word_num, FLAGS.max_word_num_upper)
    max_word_length = min(max_word_length, FLAGS.max_word_length_upper)
    tf.logging.info('After clipping...max word num = %d, max word length = %d' % (max_word_num, max_word_length))

    with gfile.GFile(
            os.path.join(output_dir, _PADDING_DIMENSIONS_FILE_NAME), 'w+') as f:
        f.write('max_word_num: %d\nmax_word_length: %d\n' %
                (max_word_num, max_word_length))
    return max_word_num, max_word_length

def add_back_action_info(action):
    input_str_pos_padding = [
        config.LABEL_DEFAULT_VALUE_INT, config.LABEL_DEFAULT_VALUE_INT
    ]
    input_prep_word = action.instruction_str[:-len(action.obj_desc_str)].strip(' ').split(' ')[-1]
    action.verb_str_pos = [0, _count_chars(action.verb_str)]
    swipe_prep_word = ('until ' if 'until' in action.instruction_str else 'to ')
    if action.action_type in [common.ActionTypes.CLICK]:       
        action.obj_str_pos = [
            _count_chars(action.verb_str) + 1,
            _count_chars(action.instruction_str)
        ]
        action.input_str_pos = input_str_pos_padding

    elif action.action_type in [common.ActionTypes.INPUT]:
        action.input_str_pos = [
            _count_chars(action.verb_str) + 1,
            _count_chars('%s %s' % (action.verb_str, action.input_content_str))
        ]
        action.obj_str_pos = [
            _count_chars(
                '%s %s %s' %
                (action.verb_str, action.input_content_str, input_prep_word)) + 1,
            _count_chars(action.instruction_str)
        ]
    # All the rests are swipe actions
    else:
        action.input_str_pos = input_str_pos_padding
        start = _count_chars(action.verb_str) + 1
        if 'to the left' in action.instruction_str:
            start += _count_chars('to the left ')
        elif 'to the right' in action.instruction_str:
            start += _count_chars('to the right ')
        elif 'down' in action.instruction_str:
            start += _count_chars('down ')
        else:
            start += _count_chars('up ')

        start += _count_chars(swipe_prep_word) 
        action.obj_str_pos = [start, _count_chars(action.instruction_str)]

def _get_full_feature_dict(file_path, max_word_num,
                           max_word_length, use_goal):
    """Gets full padded feature dictionary from xml/json file_path.

  Args:
    dataset_type: The supported dataset type.
    file_path: The full path of xml/json file.
    max_word_num: The max number of words in each ui object.
    max_word_length: The max length of words in each ui object.

  Returns:
    padded_feature_dict: A dictionary that contains ui_object features, view
    hierarchy leaf node adjacency matrix features and synthetic actionq
    features. Each value of the feature dicionary is padded. The padding shape
    is as follow:
    feature_dict = {
      'instruction_str': synthetic action strings, np array of string,
          shape = (phrase_count,)
      'instruction_word_id_seq': encoded action words, np int array, shape =
          (phrase_count, max_word_num)
      'instruction_rule_id': representing action rule(single_object_rule/
          grid_context_rule/neighbor_context_rule), np int array,
          shape = (phrase_count,)
      'ui_obj_str_seq': string of ui object name/content_description/resourceid,
          np string array, shape = (ui_object_num)
      'ui_obj_word_id_seq': encoded word sequence, np int array, shape =
          (ui_object_num, max_word_num)
      'ui_obj_type_id_seq': object type ids, np int array, shape =
          (ui_object_num,)
      'ui_obj_clickable_seq': clickable sequence, np int array, shape =
          (ui_object_num,)
      'ui_obj_cord_x_seq': x coordinate sequence, np float array, shape =
          (ui_object_num*2,)
      'ui_obj_cord_y_seq': y coordinate sequence, np float array, shape =
          (ui_object_num*2,)
      'ui_obj_v_distance': vertical relation matrix, np float array, shape =
          (ui_object_num, ui_object_num)
      'ui_obj_h_distance': horizontal relation matrix, np float array, shape =
          (ui_object_num, ui_object_num)
      'ui_obj_dom_distance': dom relation matrix, np int array, shape =
          (ui_object_num, ui_object_num)
      'ui_obj_dom_location_seq': index of pre-order/in-order/post-order in view
          hierarchy tree. np int array, shape = (ui_object_num*3,)
      'verb_id_seq': representing action verb id(click/input/swipe), np int
          array, shape = (phrase_count,)
      'verb_str_position_seq': index of verb string, np int array,
          shape = (phrase_count*2,)
      'ui_target_id_seq': index of ui object target, np int array,
          shape = (phrase_count,)
      'input_str_position_seq': input words' start end position in instruction,
          np int array, shape = (phrase_count*2,)
      'obj_desc_position_seq': target object words' start end position,
          np int array, shape = (phrase_count*2,)
    }
  """
    with open(file_path) as f:
        trace_info = json.load(f)

    full_feature = {}

    ui_object_nums = []
    vh_leaf_nodes = []
    for view in trace_info['images'][:-1]:
        view_path = os.path.join(FLAGS.input_dir, trace_info['app'], trace_info['trace_id'], 'view_hierarchies', view + '.jpg')
        view_hierarchy_leaf_nodes = common.get_view_hierarchy_list(view_path, trace_info['vh_w'], trace_info['vh_h'])
        ui_object_num = len(view_hierarchy_leaf_nodes)
        ui_object_nums.append(ui_object_num)
        ui_obj_list = [ele.uiobject for ele in view_hierarchy_leaf_nodes]
        vh_leaf_nodes.append(view_hierarchy_leaf_nodes)

    # initialize first set of view hierarchy features
    # will concat rest of time steps after
    # 140 was the max logged
    max_ui_object_num = 140 # max(ui_object_nums)
    # print('MAX UI NUM = %d' % max_ui_object_num)
    padded_obj_feature_dict = proto_utils.get_ui_objects_feature_dict(
        vh_leaf_nodes[0],
        padding_shape=(max_ui_object_num, max_word_num, max_word_length),
        lower_case=True)
    
    for key in padded_obj_feature_dict:
        padded_obj_feature_dict[key] = np.expand_dims(padded_obj_feature_dict[key], axis=0)
    full_feature.update(padded_obj_feature_dict)

    for vh in vh_leaf_nodes[1:]:
        padded_obj_feature_dict = proto_utils.get_ui_objects_feature_dict(
            vh,
            padding_shape=(max_ui_object_num, max_word_num, max_word_length),
            lower_case=True)
        for key in padded_obj_feature_dict:
            padded_obj_feature_dict[key] = np.expand_dims(padded_obj_feature_dict[key], axis=0)
            # concat along step_num dim
            full_feature[key] = np.concatenate((full_feature[key], padded_obj_feature_dict[key]), axis=0)

    # only need to load and pad actions once
    action_list = []
    for i in range(len(trace_info['instr'])):
        instr = trace_info['instr'][i]
        verb_str = trace_info['verb_str'][i]
        obj_desc_str = trace_info['obj_desc_str'][i]
        input_content_str = trace_info['input_str'][i] 
        if trace_info['actions'][i] == 'click':
            action_type = common.ActionTypes.CLICK
        elif trace_info['actions'][i] == 'type':
            action_type = common.ActionTypes.INPUT
        else:
            action_type = common.ActionTypes.SWIPE
        target_object_idx = trace_info['ui_target_idxs'][i]
        action = common.Action(instruction_str=instr.lower(), 
                                verb_str=verb_str.lower(), 
                                obj_desc_str=obj_desc_str.lower(),
                                input_content_str=input_content_str.lower(),
                                action_type=action_type,
                                action_rule=common.ActionRules.REAL,
                                target_obj_idx=target_object_idx)
        add_back_action_info(action)
        action_list.append(action) 

    padded_syn_feature_dict, missing_ref = real_action_generator.get_real_feature_dict(action_list, max_word_num, max_word_length, use_goal, trace_info['goal'])

    full_feature.update(padded_syn_feature_dict)
    full_feature['ui_obj_cord_x_seq'] = full_feature['ui_obj_cord_x_seq'] / float(trace_info['vh_w'])
    full_feature['ui_obj_cord_y_seq'] = full_feature['ui_obj_cord_y_seq'] / float(trace_info['vh_h'])

    return full_feature, missing_ref

def _assert_feature_value(feature):
    """Asserts feature value doesn't have -1, except for anchor features."""
    # ui target id seq can have -1 bc it includes swipe actions
    anchor_features = [
        'instruction_word_id_seq', 'ui_obj_type_id_seq', 'verb_id_seq', 'ui_target_id_seq'
    ]
    for key in feature:
        if key in anchor_features:
            continue
        if -1 in feature[key]:
            tf.logging.info('[FATAL]: Feature %s contains -1', key)
            return False
    return True


def _assert_feature_shape(feature, expected_shape):
    """Asserts feature shape is legal, same as expected_shape."""
    assert set(feature.keys()) == set(expected_shape.keys(
    )), '[FATAL] feature keys %s different from expected %s' % (
        sorted(feature.keys()), sorted(expected_shape.keys()))
    for key in feature:
        if feature[key].shape != expected_shape[key]:
            print(key)
            print(feature[key].shape)
            print(expected_shape[key])
            tf.logging.info('[FATAL] feature %s shape is different from expected',
                            key)
            return False
    return True

def _count_chars(char_string):
    return len(char_string)

def _process_features(tf_record_writer, writer_lock, dataset_type, file_path, max_word_num, max_word_length, use_goal):
    """Processes features from one xml/json file.

  Args:
    dataset_type: The supported dataset type.
    file_path: The full path of xml/json file.
    max_word_num: The max number of words in each ui object.
    max_word_length: The max length of words in each ui object. synthetic input
      actions.
  """
    feature_dict, missing_ref = _get_full_feature_dict(
        file_path,
        max_word_num,
        max_word_length,
        use_goal
    )

    ui_object_num = feature_dict['ui_obj_str_seq'].shape[1]
    step_num = feature_dict['ui_target_id_seq'].shape[0]
    expected_feature_shape = {
        'instruction_str': (step_num,),
        'instruction_word_id_seq': (step_num, max_word_num),
        'instruction_rule_id': (step_num,),
        'ui_obj_str_seq': (step_num, ui_object_num),
        'ui_obj_word_id_seq': (step_num, ui_object_num, max_word_num),
        'ui_obj_type_id_seq': (step_num, ui_object_num),
        'ui_obj_clickable_seq': (step_num, ui_object_num),
        'ui_obj_cord_x_seq': (step_num, ui_object_num * 2),
        'ui_obj_cord_y_seq': (step_num, ui_object_num * 2),
        'ui_obj_v_distance': (step_num, ui_object_num, ui_object_num),
        'ui_obj_h_distance': (step_num, ui_object_num, ui_object_num),
        'ui_obj_dom_distance': (step_num, ui_object_num, ui_object_num),
        'ui_obj_dom_location_seq': (step_num, ui_object_num * 3),
        'verb_id_seq': (step_num,),
        'ui_target_id_seq': (step_num,),
        'verb_str_position_seq': (step_num * 2,),
        'input_str_position_seq': (step_num * 2,),
        'obj_desc_position_seq': (step_num * 2,),
    }

    # When feature_dict['verb_id_seq'] is not always padded value, 
    # generate tfexample
    bool1 = _assert_feature_shape(feature_dict, expected_feature_shape)
    bool2 = _assert_feature_value(feature_dict)
    bool3 = (not np.array(feature_dict['verb_id_seq'] == config.LABEL_DEFAULT_INVALID_INT).all())

    if bool1 and bool2 and bool3: 
        tf_proto = proto_utils.features_to_tf_example(feature_dict)
        with writer_lock:
            tf_record_writer.write(tf_proto.SerializeToString())

def _write_dataset(dataset_type, input_dir, output_dir, traces, max_word_num,
                   max_word_length):
    """Processes features from xml/json files and writes results to dataset files.

  Args:
    dataset_type: The supported dataset type.
    input_dir: The directory that contains the input xml/json files.
    output_dir: The directory that saves output dimension data.
    max_word_num: The max number of words in each ui object.
    max_word_length: The max length of words in each ui object. synthetic input
      actions.
  """
    tf.logging.info('Processing data features...')
        
    tf_record_writers = []
    writer_locks = []
    for shard in range(FLAGS.num_shards):
        tf_record_writers.append(
            tf.python_io.TFRecordWriter(
                os.path.join(
                    output_dir, '%s_%d.tfrecord' %
                                (dataset_type, shard))))
        writer_locks.append(threading.Lock())

    num_processed_files = 0
    # We can use ThreadPool since these are IO-bound operations.
    with concurrent.futures.ThreadPoolExecutor(FLAGS.num_threads) as executor:
        futures = []

        for tr in sorted(traces):
            file_path = os.path.join('seq2act_9_5', tr + '.json') # processed_motif_deduped
            shard = num_processed_files % FLAGS.num_shards
            
            if os.path.exists(file_path):               
                futures.append(
                    executor.submit(_process_features, tf_record_writers[shard],
                                    writer_locks[shard], dataset_type, file_path,
                                    max_word_num, max_word_length, FLAGS.use_high_level_goal))
                num_processed_files += 1
            else:
                print('Missing %s' % tr)
                continue
        concurrent.futures.wait(futures)

    for shard in range(FLAGS.num_shards):
        tf_record_writers[shard].close()
    print('NUMBER OF PROCESSED FILES = %d' % num_processed_files)

def create_dataset(dataset_type, input_dir, output_dir, splits):
    """Converts xml/json files in input_dir to tfExample files in output_dir.

  tfExample file contains multiple tf.Example proto and the feature dictionary
  of tf.Example proto defined by _get_full_feature_dict()

  Args:
    dataset_type: The supported dataset type.
    input_dir: The directory that contains input xml/json file.
    output_dir: The directory that saves output files.
  """
    if _STATS_DIMENSIONS:
        max_word_num, max_word_length = _process_dimensions(input_dir, output_dir, splits)
    else:
        max_word_num, max_word_length = _MAX_WORD_NUM, _MAX_WORD_LENGTH
    _write_dataset(dataset_type, input_dir, output_dir, splits, max_word_num, max_word_length)

def main(_):
    if not os.path.isdir(FLAGS.output_dir):
        os.mkdir(FLAGS.output_dir)

    with open('eccv_motif_app_seen_task_unseen_all.json') as f:
        traces_to_process_su_all = json.load(f)['test']
    with open('eccv_motif_app_seen_task_unseen_curr.json') as f:
        traces_to_process_su_curr = json.load(f)['test']
    with open('eccv_motif_app_unseen_task_unseen.json') as f:
        traces_to_process_uu = json.load(f)['test']
    with open('eccv_motif_app_unseen_task_seen.json') as f:
        traces_to_process_us = json.load(f)['test']

    if FLAGS.split == 'all':
        traces_to_process = traces_to_process_su_all + traces_to_process_su_curr + traces_to_process_uu + traces_to_process_us
    elif FLAGS.split == 'uu':
        traces_to_process = traces_to_process_uu
    elif FLAGS.split == 'us':
        traces_to_process = traces_to_process_us
    elif FLAGS.split == 'su_all':
        traces_to_process = traces_to_process_su_all
    else:
        traces_to_process = traces_to_process_su_curr

    traces_to_process = list(set(traces_to_process))
    print('Test traces to process %d' % len(traces_to_process))
    create_dataset(FLAGS.dataset, FLAGS.input_dir, FLAGS.output_dir, traces_to_process)
    sums = collections.defaultdict(int)

    tf.logging.info('\n\n%s\n\n', longest_stats)
    with open(os.path.join(FLAGS.output_dir, 'stats.txt'), 'w+') as writer:

        for key, distribution in distributions.items():
            writer.write(
                '%s: %s\n' %
                (key, sorted(distribution.items(), key=operator.itemgetter(0))))

        for key, distribution in sums.items():
            writer.write('%s: %s\n' %
                         (key, sorted(sums.items(), key=operator.itemgetter(0))))


if __name__ == '__main__':
    FLAGS.set_default('logtostderr', True)
    tf.app.run(main)

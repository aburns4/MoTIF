import json
import glob
import re
import os

view_data_path = '../data/view_hierarchy_features'
if not os.path.isdir(view_data_path):
    os.mkdir(view_data_path)

view_paths = glob.glob('../raw_data/*/*/view_hierarchies/*')

data_dict = {'text': [], 'ids': [], 'classes': []}
def recurse(node):
    node_keys = list(node.keys())
    data_dict['classes'].append(node['class'].split('.')[-1].lower()[:50])
    if 'resource-id' in node_keys:
        data_dict['ids'].append(node['resource-id'].split('/')[-1].lower()[:50])
    if 'text' in node_keys:
        updated_text = re.sub("[^0-9a-zA-Z]+", " ", node['text'])
        split = updated_text.lower().split(' ')
        for word in split: 
            data_dict['text'].append(word[:50])
    if 'children' in node:
        for child in node['children']:  
            recurse(child)

def __main__():
    global data_dict
    global xid

    print(len(view_paths))
    i = 0
    for path in view_paths:
        if i % 5000 == 0:
            print(i)
        i += 1
        trace_id = path.split('/')[-1][:-4]
        with open(path) as f:
            try:
                data = json.load(f)
                recurse(data['activity']['root'])
            except:
                continue

        data_dict['text'] = list(set(data_dict['text']))
        data_dict['ids'] = list(set(data_dict['ids']))
        data_dict['classes'] = list(set(data_dict['classes']))

        save_to = os.path.join(view_data_path, trace_id)
        with open(save_to, 'w') as f:
            json.dump(data_dict, f)

        data_dict = {'text': [], 'ids': [], 'classes': []}

__main__()

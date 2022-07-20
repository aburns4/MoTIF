import json
import glob
import os
from PIL import Image

trace_paths = glob.glob('../raw_data/*/*')
grouped_view_paths = [glob.glob(x + '/view_hierarchies/*') for x in trace_paths]
view_icons = []

resnet_path = '../data/icon_crops'
if not os.path.isdir(resnet_path):
    os.mkdir(resnet_path)

def recurse(node, root_area, sx, sy):
    clickable = node['clickable']
    visible = node['visible-to-user']
    w = (node['bounds'][2] - node['bounds'][0])
    h = (node['bounds'][3] - node['bounds'][1])
    node_area = w * h
    if max(w, h) == 0:
        node_aspect = 0.0
    else:
        node_aspect = min(w, h) / max(w, h)

    if clickable and visible and (node_area < 0.05*root_area) and (node_aspect > 0.75):
        c = [node['bounds'][0] * sx, 
             (node['bounds'][1] * sy) - 65,
             (node['bounds'][0] + w) * sx,
             ((node['bounds'][1] + h) * sy) - 65]
        view_icons.append(c)

    if 'children' in node:
        for child in node['children']:  
            recurse(child, root_area, sx, sy)


def get_screen_dims(views, trace_exceptions='widget_exception_dims.json'):
    with open(trace_exceptions) as f:
        widget_exceptions = json.load(f)

    trace = views[0].split('/')[-3]
    if trace in widget_exceptions:
        return [int(widget_exceptions[trace][0]), int(widget_exceptions[trace][1])]

    for view in views:
        with open(view, 'r') as f:
            data = json.load(f)
            bbox = data['activity']['root']['bounds']
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
        if bbox[0] == 0 and bbox[1] == 0:
            return width, height


def __main__():
    global view_icons
    all_view_icons = {}
    num_icons_tot = 0
    i = 0

    for trace in grouped_view_paths:
        try:
            view_width, view_height = get_screen_dims(trace)
        except Exception:
            continue

        for path in trace:
            img_path = path.replace('view_hierarchies', 'screens')
            img = Image.open(img_path)

            if i % 2000 == 0:
                print(i)
            i += 1
            trace_id = path.split('/')[-1][:-4]
            with open(path) as f: 
                try:
                    data = json.load(f)
                    root_size = data['activity']['root']['bounds']
                    scaleX = img.width / view_width
                    scaleY = (img.height + 65) / view_height
                    root_a = (root_size[2]-root_size[0])*(root_size[3]-root_size[1])
                    recurse(data['activity']['root'], root_a, scaleX, scaleY)
                except Exception:
                    continue
            all_view_icons[trace_id] = view_icons
            num_icons_tot += len(view_icons)
            view_icons = []

    print('Total tokens = %d' % num_icons_tot)
    print('Writing crops to file now...')
    for view_id in all_view_icons:
        for b in range(len(all_view_icons[view_id])):
            crop = img.crop(all_view_icons[view_id][b]).resize((32, 32))
            save_to = '../data/icon_crops/' + view_id + '.' + str(b) + '.png'
            crop.save(save_to)
    print('Done!')

__main__()

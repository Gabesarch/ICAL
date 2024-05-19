import json
import glob
import os
import ipdb
import numpy as np
st = ipdb.set_trace


path_base = "/home/gsarch/repo/forecasting/ego4d_forecasting/models/prompts/examples"
path = os.path.join(path_base, 'recognition', 'actions')
examples = {}
files = os.listdir(path)
for file_ in files:
    file_path = os.path.join(path, file_)
    with open(file_path) as f:
        file_rec_actions = f.read()
    file_reasoning = file_path.replace('actions', 'reasoning')
    with open(file_reasoning) as f:
        file_rec_reasoning = f.read()
    image_paths = os.listdir(file_path.replace('actions', 'images').replace('.txt', ''))
    image_paths_idxs = np.argsort([int(im_path.replace('.png', '')) for im_path in image_paths])
    image_paths = [os.path.join(path.replace('actions', 'images'), file_.replace('.txt', ''), image_paths[im_path_idx]) for im_path_idx in list(image_paths_idxs)]
    example = [file_rec_reasoning, file_rec_actions, image_paths]
    examples[file_.replace('.txt', '')] = example

with open(os.path.join(path.replace('/actions', ''), 'examples.json'), "w") as outfile: 
    json.dump(examples, outfile, indent=4, sort_keys=True)


path_base = "/home/gsarch/repo/forecasting/ego4d_forecasting/models/prompts/examples"
path = os.path.join(path_base, 'forecasting', 'actions')
examples = {}
files = os.listdir(path)
for file_ in files:
    file_path = os.path.join(path, file_)
    with open(file_path) as f:
        file_rec_actions = f.read()
    file_reasoning = file_path.replace('actions', 'reasoning')
    with open(file_reasoning) as f:
        file_rec_reasoning = f.read()
    file_video_actions = file_path.replace('actions', 'video_actions')
    with open(file_video_actions) as f:
        file_rec_video = f.read()
    image_paths = os.listdir(file_path.replace('actions', 'images').replace('.txt', ''))
    image_paths_idxs = np.argsort([int(im_path.replace('.png', '')) for im_path in image_paths])
    image_paths = [os.path.join(path.replace('actions', 'images'), file_.replace('.txt', ''), image_paths[im_path_idx]) for im_path_idx in list(image_paths_idxs)]
    example = [file_rec_reasoning, file_rec_actions, file_rec_video, image_paths]
    examples[file_.replace('.txt', '')] = example
with open(os.path.join(path.replace('/actions', ''), 'examples.json'), "w") as outfile: 
    json.dump(examples, outfile, indent=4, sort_keys=True)
st()

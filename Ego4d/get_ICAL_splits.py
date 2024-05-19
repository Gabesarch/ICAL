import json
import ipdb
st = ipdb.set_trace
import numpy as np
import copy
np.random.seed(32)

with open("/PATh/TO/EGO4D/data/long_term_anticipation/annotations/fho_lta_val.json") as json_file:
    json_data = json.load(json_file)

clip_dict = {}
for clip in json_data['clips']:
    clip_uid = clip['clip_uid']
    if clip_uid not in clip_dict.keys():
        clip_dict[clip_uid] = []
    clip_dict[clip_uid].append(clip)
for k in list(clip_dict.keys()):
    if len(clip_dict[k]) - 20 - 1 < 1:
        del clip_dict[k]

idxs = set(np.random.choice(list(range(len(clip_dict.keys()))), replace=False, size=100))
clip_dict_keys = list(clip_dict.keys())
clips_train = []
clips_test = []
clips_test_seen = []
for idx in range(len(clip_dict_keys)):
    if idx in idxs:
        k = clip_dict_keys[idx]
        clips = clip_dict[k]
        if len(clips) - 20 - 1<1:
            continue
        idx_choose = np.random.choice(list(range(len(clips) - 20 - 1)))
        clips_train.extend(clips[idx_choose:])
        idx_choose = np.random.choice(list(range(len(clips) - 20 - 1)))
        clips_test_seen.extend(clips[idx_choose:])
    else:
        k = clip_dict_keys[idx]
        clips = clip_dict[k]
        if len(clips) - 20 - 1<1:
            continue
        idx_choose = np.random.choice(list(range(len(clips) - 20 - 1)))
        clips_test.extend(clips[idx_choose:])

json_data_train = copy.deepcopy(json_data)
json_data_train['clips'] = clips_train
json_data_test = copy.deepcopy(json_data)
json_data_test['clips'] = clips_test
json_data_test_seen = copy.deepcopy(json_data)
json_data_test_seen['clips'] = clips_test_seen

with open('/PATh/TO/EGO4D/data/long_term_anticipation/annotations/fho_lta_ICAL_train.json', 'w') as f:
    json.dump(json_data_train, f)

with open('/PATh/TO/EGO4D/data/long_term_anticipation/annotations/fho_lta_ICAL_test.json', 'w') as f:
    json.dump(json_data_test, f)

with open('/PATh/TO/EGO4D/data/long_term_anticipation/annotations/fho_lta_ICAL_test_seen.json', 'w') as f:
    json.dump(json_data_test_seen, f)

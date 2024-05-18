import numpy as np
import torch
import torch.utils.data
import torchvision.transforms as transforms
from PIL import Image
from arguments import args
from utils.aithor import get_masks_from_seg
import glob
import os
import ipdb
st = ipdb.set_trace
from tqdm import tqdm
import random
import time
import utils.geom
import utils.aithor
import pickle

import torch
import re
try:
    from torch._six import container_abcs, string_classes, int_classes
except:
    import collections.abc as container_abcs
    int_classes = int
    string_classes = str
import h5py

from teach.utils import (
    create_task_thor_from_state_diff,
    load_images,
    save_dict_as_json,
    with_retry,
    load_json
)
import torchvision.transforms as T

targets_to_output = ['boxes', 'masks', 'labels', 'obj_targets', 'expert_action']
history_targets = ['masks']

# fix the seed for reproducibility
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

class RICKDataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        split,
        W, H, 
        idx2actions,
        id_to_name,
        start_index=None, 
        end_index=None,
        shuffle=False,
        images=None,
        subsample=None,
        cache = True,
        ):
        """
        Args:
            root_dir (string): root directory with all the images, etc.
            W, H (int): width and height of frame
            max_objs (int): maximum objects per image allowable
            id_to_name (dict): converts object name idx to word
        """

        with open(f'./data/teach_idm_{split}.p', 'rb') as f:
            file_list = pickle.load(f)
        
        root_dir = os.path.join(args.teach_data_dir, 'images', split)

        st()

        self.idx2actions, self.id_to_name = idx2actions, id_to_name
        self.name_to_id = {v:k for (k,v) in id_to_name.items()}
        self.actions2idx = {v:k for (k,v) in idx2actions.items()}

        self.no_object_label = max(list(self.id_to_name.keys()))

        if cache and images is None:
            '''
            cache paths for faster loading
            '''
            file_cache = os.path.join('data', root_dir.replace('/', '_')+f'_{split}'+'_image_paths'+'.p')
            if not os.path.exists(file_cache):
                print(f"Getting image paths from {root_dir} ...")
                self.images = []
                for tfd_instance in tqdm(file_list):
                    files_to_add = glob.glob(os.path.join(root_dir, tfd_instance) + '/driver.frame.*.jpeg')
                    self.images.extend(files_to_add)
                self.images, idx_caches = self.remove_bad_actions()
                with open(file_cache, 'wb') as handle:
                    pickle.dump(self.images, handle, protocol=pickle.HIGHEST_PROTOCOL)
            else:
                with open(file_cache, 'rb') as handle:
                    self.images = pickle.load(handle)
                # images = []
                # for image in self.images:
                #     images.append(image.replace('/projects/katefgroup/embodied_llm/dataset', '/scratch'))
                # self.images = images
                # with open(file_cache, 'wb') as handle:
                #     pickle.dump(self.images, handle, protocol=pickle.HIGHEST_PROTOCOL)
        elif images is None:
            print(f"Getting image paths from {root_dir} ...")
            self.images = []
            for tfd_instance in tqdm(file_list):
                files_to_add = glob.glob(os.path.join(root_dir, tfd_instance) + '/driver.frame.*.jpeg')
                self.images.extend(files_to_add)
            self.images, idx_caches = self.remove_bad_actions()
        else:
            self.images = images

        # self.images, idx_caches = self.remove_bad_actions()
        # with open(file_cache, 'wb') as handle:
        #     pickle.dump(self.images, handle, protocol=pickle.HIGHEST_PROTOCOL)
                
        if shuffle:
            random.shuffle(self.images)
        if end_index is not None:
            if start_index is None:
                start_index = 0
            self.images = self.images[start_index:end_index]
        if subsample is not None:
            self.images = self.images[::subsample]

        # if split=="train":
        #     action_weights, label_weights = self.get_loss_weights()
        #     args.action_weights = np.float32(action_weights)
        #     args.label_weights = np.float32(label_weights)
        #     st()

        self.image_mean = np.array([0.485,0.456,0.406]).reshape(1,3,1,1)
        self.image_std = np.array([0.229,0.224,0.225]).reshape(1,3,1,1)
        
        self.W, self.H, self.fov = int(W), int(H), int(args.fov)

        self.resize_transform = T.Resize((self.W,self.H))

    def __len__(self):
        return len(self.images)

    def get_loss_weights(
        self,
    ):
        '''
        Gets proportion of each action for cross entropy loss
        This should just be called once for new data when the weights are needed
        '''
        print("Getting loss weights....")

        actions = {i:1 for i in range(len(self.idx2actions.keys()))}
        labels = {i:1 for i in list(self.id_to_name.keys())}
        for idx in tqdm(range(len(self.images))):
            image_t = self.images[idx]
            self.image_t = image_t
            
            edh_file = os.path.split(image_t.replace('images', 'tfd_instances'))[0] + '.tfd.json'

            # task_instance = os.path.join(instance_dir, file)
            instance = load_json(edh_file)
            frame_index = instance['driver_images_future'].index(os.path.split(self.image_t)[-1])

            action_name = instance['driver_actions_future'][frame_index]['action_name']
            object_name = instance['driver_actions_future'][frame_index]['oid']
            
            if action_name in ["Text", "Navigation", "SearchObject", "SelectOid", "OpenProgressCheck"]:
                continue

            actions[self.actions2idx[action_name]] += 1

            if object_name is None:
                labels[self.name_to_id["no_object"]] += 1
            else:
                object_name_ = object_name.split('|')[0]
                if "Sliced" in object_name and "Sliced" not in object_name_:
                    object_name_ += "Sliced"
                labels[self.name_to_id[object_name_]] += 1
            
        action_counts = np.array(list(actions.values()))
        action_weights = sum(action_counts) / (len(action_counts)*action_counts)
        action_weights = action_weights/np.median(action_weights) # normalize to be maximum value of 5
        print("Action counts:", action_counts)
        print("Action weights:", str(list(np.round(action_weights,6))))
        np.set_printoptions(suppress=True)
        with open('output.txt', 'a') as f:
            print("Label weights:", str(list(np.round(action_weights,6))), file=f)

        labels_counts = np.array(list(labels.values()))
        label_weights = sum(labels_counts) / (len(labels_counts)*labels_counts)
        label_weights = label_weights/np.median(label_weights) # normalize to be maximum value of 5
        print("Label counts:", labels_counts)
        print("Label weights:", str(list(np.round(label_weights,6))))
        np.set_printoptions(suppress=True)
        with open('output.txt', 'a') as f:
            print("Label weights:", str(list(np.round(label_weights,6))), file=f)
        return action_weights, label_weights

    def remove_bad_actions(self):
        print("Removing bad images...")
        images = []
        idx_keep = []
        for idx in tqdm(range(len(self.images))):
            image_t = self.images[idx]
            self.image_t = image_t
            
            edh_file = os.path.split(image_t.replace('images', 'tfd_instances'))[0] + '.tfd.json'

            instance = load_json(edh_file)
            if os.path.split(self.image_t)[-1] not in instance['driver_images_future']:
                continue
            frame_index = instance['driver_images_future'].index(os.path.split(self.image_t)[-1])
            if frame_index==len(instance['driver_images_future'])-1:
                # first frame is starting frame
                continue

            action_name = instance['driver_actions_future'][frame_index]['action_name']
            object_name = instance['driver_actions_future'][frame_index]['oid']
            
            if action_name in ["Text", "Navigation", "SearchObject", "SelectOid", "OpenProgressCheck"]:
                continue

            if object_name is not None and object_name.split('|')[0] not in self.name_to_id.keys():
                continue
            
            images.append(image_t)
            idx_keep.append(idx)
        print(f"Removed {len(self.images) - len(images)} images")

        return images, idx_keep

    def transform_images(self, images):
        return (images - self.image_mean) / self.image_std

    def __getitem__(self, idx, only_fetch_demos=False, subgoal_info=None):

        image_t = self.images[idx]
        self.image_t = image_t

        edh_file = os.path.split(image_t.replace('images', 'tfd_instances'))[0] + '.tfd.json'
        instance = load_json(edh_file)

        frame_index_t = instance['driver_images_future'].index(os.path.split(self.image_t)[-1])
        frame_index_tplus1 = frame_index_t + 1

        action_name = instance['driver_actions_future'][frame_index_t]['action_name']
        object_name = instance['driver_actions_future'][frame_index_t]['oid']
        image_tplus1 = os.path.join(os.path.split(image_t)[0], instance['driver_images_future'][frame_index_tplus1])

        targets = {}

        # ##### current observation is last index #####
        images = []
        images.append(np.asarray(self.resize_transform(Image.open(image_t))))
        images.append(np.asarray(self.resize_transform(Image.open(image_tplus1))))

        targets['expert_action'] = np.asarray([self.actions2idx[action_name]])

        if object_name is None:
            targets['labels'] = np.asarray([self.no_object_label])
        else:
            object_name_ = object_name.split('|')[0]
            if "Sliced" in object_name and "Sliced" not in object_name_:
                object_name_ += "Sliced"
            targets['labels'] = np.asarray([self.name_to_id[object_name_]])
        images = np.asarray(images) * 1./255
        images = images.transpose(0, 3, 1, 2)
        images = self.transform_images(images).astype(np.float32) # normalize for resnet
        
        samples = {}
        samples["images"] = images
        samples["targets"] = targets
        return samples

np_str_obj_array_pattern = re.compile(r'[SaUO]')

default_collate_err_msg_format = (
    "default_collate: batch must contain tensors, numpy arrays, numbers, "
    "dicts or lists; found {}")

def my_collate(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""

    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        # out_ = None
        # if torch.utils.data.get_worker_info() is not None:
        #     # If we're in a background process, concatenate directly into a
        #     # shared memory tensor to avoid an extra copy
        numel = sum([x.numel() for x in batch])
        storage = elem.storage()._new_shared(numel)
        out = elem.new(storage)

        # lens = [len(b) for b in batch]
        it = iter(batch)
        elem_size = len(next(it))

        if not all(len(elem) == elem_size for elem in batch):
            out_ = batch
        else:
            out_ = torch.stack(batch, 0, out=out)    
        
        return out_
                
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(my_collate_err_msg_format.format(elem.dtype))

            return my_collate([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int_classes):
        return torch.tensor(batch)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, container_abcs.Mapping):
        if all(e in targets_to_output for e in list(elem.keys())):
            for t in batch:
                for k in t.keys():
                    t[k] = torch.as_tensor(t[k])
            return batch # used for targets 
        return {key: my_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(my_collate(samples) for samples in zip(*batch)))
    elif isinstance(elem, container_abcs.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            pass
            # raise RuntimeError('each element in list of batch should be of equal size')
            # return batch
            transposed = batch
        else:
            transposed = zip(*batch)
        return [my_collate(samples) for samples in transposed]

    raise TypeError(my_collate_err_msg_format.format(elem_type))


from torch.utils.data import Sampler, Dataset
import math
from typing import TypeVar, Optional, Iterator
import torch.distributed as dist
class DistributedWeightedSampler(Sampler):
    def __init__(self, dataset, num_replicas=None, rank=None, replacement=True):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.replacement = replacement


    def calculate_weights(self, targets):
        class_sample_count = torch.tensor(
            [(targets == t).sum() for t in torch.unique(targets, sorted=True)])
        weight = 1. / class_sample_count.double()
        samples_weight = torch.tensor([weight[t] for t in targets])
        return samples_weight

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)
        if self.shuffle:
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        # get targets (you can alternatively pass them in __init__, if this op is expensive)
        targets = self.dataset.targets
        targets = targets[self.rank:self.total_size:self.num_replicas]
        assert len(targets) == self.num_samples
        weights = self.calculate_weights(targets)

        return iter(torch.multinomial(weights, self.num_samples, self.replacement).tollist())

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch

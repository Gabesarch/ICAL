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
# from torch._six import container_abcs, string_classes, int_classes
try:
    from torch._six import container_abcs, string_classes, int_classes
except:
    import collections.abc as container_abcs
    int_classes = int
    string_classes = str
# from SOLQ.util import box_ops
import h5py


ORIGINAL_IMAGES_FOLDER = "raw_images"
HISTORY_IMAGES_FOLDER = "raw_history_images"
HIGH_RES_IMAGES_FOLDER = "high_res_images"
DEPTH_IMAGES_FOLDER = "depth_images"
INSTANCE_MASKS_FOLDER = "instance_masks"
TARGETS_FOLDER = "targets"

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
        root_dir, root_dir2, root_dir3 = None, None, None
        if split == 'train':
            root_dir = args.alfred_load_train_root
            root_dir2 = args.alfred_load_train_root2
            root_dir3 = args.alfred_load_train_root3
        elif split == 'valid_seen':
            root_dir = args.alfred_load_valid_seen_root
        elif split == "valid_unseen":
            root_dir = args.alfred_load_valid_unseen_root
        else:
            assert False

        self.idx2actions, self.id_to_name = idx2actions, id_to_name

        self.no_object_label = max(list(self.id_to_name.keys()))

        st()

        if cache and images is None:
            '''
            cache paths for faster loading
            '''
            file_cache = os.path.join('data', root_dir.replace('/', '_')+f'_{split}'+'_image_paths'+'.p')
            if not os.path.exists(file_cache):
                print(f"Getting image paths from {root_dir} ...")
                self.images = [d for d in tqdm(glob.glob(root_dir + 2 * '/*' + f'/{ORIGINAL_IMAGES_FOLDER}/*'))]
                with open(file_cache, 'wb') as handle:
                    pickle.dump(self.images, handle, protocol=pickle.HIGHEST_PROTOCOL)
            else:
                with open(file_cache, 'rb') as handle:
                    self.images = pickle.load(handle)
        elif images is None:
            print(f"Getting image paths from {root_dir} ...")
            self.images = [d for d in tqdm(glob.glob(root_dir + 2 * '/*' + f'/{ORIGINAL_IMAGES_FOLDER}/*'))]
            if root_dir2 is not None:
                print(f"Getting image paths from {root_dir2} ...")
                images2 = [d for d in tqdm(glob.glob(root_dir2 + 2 * '/*' + f'/{ORIGINAL_IMAGES_FOLDER}/*'))]
                self.images += images2
            if root_dir3 is not None:
                print(f"Getting image paths from {root_dir3} ...")
                images3 = [d for d in tqdm(glob.glob(root_dir3 + 2 * '/*' + f'/{ORIGINAL_IMAGES_FOLDER}/*'))]
                self.images += images3
        else:
            self.images = images

        if cache:
            file_cache = os.path.join('data', root_dir.replace('/', '_')+f'_{split}'+'.npy')
            if not os.path.exists(file_cache):
                self.images, idx_caches = self.remove_done_images()
                np.save(file_cache, np.asarray(idx_caches))
            else:
                idx_caches = np.load(file_cache)
                self.images = [self.images[idx_] for idx_ in list(idx_caches)]
        else:
            self.images, idx_caches = self.remove_done_images()

        # self.get_label_weights()
                
        if shuffle:
            random.shuffle(self.images)
        if end_index is not None:
            if start_index is None:
                start_index = 0
            self.images = self.images[start_index:end_index]
        if subsample is not None:
            self.images = self.images[::subsample]

        self.image_mean = np.array([0.485,0.456,0.406]).reshape(1,3,1,1)
        self.image_std = np.array([0.229,0.224,0.225]).reshape(1,3,1,1)
        
        self.W, self.H, self.fov = int(W), int(H), int(args.fov)

    def __len__(self):
        return len(self.images)

    def number_unique_episodes(self):
        unique_eps = set()
        for im in self.images:
            key = im.split('/')[-4:-2]
            unique_eps.add(tuple(key))
        return len(unique_eps)

    def remove_without_hdf5(self):
        images2 = []
        for idx in tqdm(range(len(self.images))):
            image_t = self.images[idx]
            targets_path = image_t.replace(ORIGINAL_IMAGES_FOLDER, TARGETS_FOLDER).replace('.png', '')
            targets_path, tag = os.path.split(targets_path)
            if os.path.exists(targets_path+'/targets.hdf5'):
                images2.append(image_t)
        print(f"NOTE: REMOVING {len(self.images) - len(images2)} IMAGES WITHOUT HDF5 TARGETS!")
        self.images = images2

    def filter_no_path(self):
        images2 = []
        for idx in tqdm(range(len(self.images))):
            image_t = self.images[idx]
            targets_path = image_t.replace(ORIGINAL_IMAGES_FOLDER, TARGETS_FOLDER).replace('.png', '')
            targets_path, tag = os.path.split(targets_path)
            targets_episode = h5py.File(targets_path+'/targets.hdf5', 'r')
            data = targets_episode[tag+'_planner_paths'][()]
            targets_episode.close()
            if not np.all(np.isnan(data)):
                images2.append(image_t)
        print(f"NOTE: REMOVING {len(self.images) - len(images2)} IMAGES WITHOUT PLANNER PATH!")
        self.images = images2

    def get_action_weights(
        self,
    ):
        '''
        Gets proportion of each action for cross entropy loss
        This should just be called once for new data when the weights are needed
        '''
        print("Getting action weights....")

        actions = {i:0 for i in range(13)}
        for idx in tqdm(range(len(self.images))):
            image_t = self.images[idx]
            targets_path = image_t.replace(ORIGINAL_IMAGES_FOLDER, TARGETS_FOLDER).replace('.png', '')
            targets_path, tag = os.path.split(targets_path)
            targets_episode = h5py.File(targets_path+'/targets.hdf5', 'r')
            data = int(targets_episode[tag+'_expert_action'][()])
            actions[data] += 1
            targets_episode.close()
            
        action_counts = np.array(list(actions.values()))
        action_weights = sum(action_counts) / (len(action_counts)*action_counts)
        action_weights = action_weights/max(action_weights) * 5 # normalize to be maximum value of 5
        print("Action counts:", action_counts)
        print("Action weights:", action_weights)
        return action_weights

    def get_label_weights(
        self,
    ):
        '''
        Gets proportion of each action for cross entropy loss
        This should just be called once for new data when the weights are needed
        '''
        print("Getting action weights....")

        labels = {i:1 for i in list(self.id_to_name.keys())}
        for idx in tqdm(range(len(self.images))):
            image_t = self.images[idx]
            targets_path = image_t.replace(ORIGINAL_IMAGES_FOLDER, TARGETS_FOLDER).replace('.png', '')
            targets_path, tag = os.path.split(targets_path)
            targets_episode = h5py.File(targets_path+'/targets.hdf5', 'r')
            targets_to_output = ['expert_action', "obj_targets", "labels"]
            targets = {}
            for target in targets_to_output:
                if target in ['masks', 'subgoal_percent_complete', 'goal_pose']:
                    continue
                data = np.asarray(targets_episode[tag+f'_{target}'])
                if target in ['labels', 'obj_targets', 'obj_instance_ids', 'expert_action']:
                    data = data.astype(np.int64)
                if len(data.shape)==0:
                    data = np.asarray([data])
                targets[target] = data
            
            targets_episode.close()
            interm_idxs = targets['obj_targets']
            if int(targets["expert_action"]) in [0,1,2,3,4] or len(targets['labels'][interm_idxs])==0 or int(targets['labels'][interm_idxs])>=self.no_object_label: #[5,6,7,8,9,10,11]:
                targets['labels'] = np.asarray([self.no_object_label])
            else:
                targets['labels'] = np.asarray(targets['labels'][interm_idxs])
            labels[int(targets['labels'][0])] += 1
            targets_episode.close()
            
        labels_counts = np.array(list(labels.values()))
        label_weights = sum(labels_counts) / (len(labels_counts)*labels_counts)
        label_weights = label_weights/np.median(label_weights) * 5 # normalize to be maximum value of 5
        print("Label counts:", labels_counts)
        print("Label weights:", label_weights)
        return label_weights

    def get_images_X_steps_away(self, X):
        
        print(f"Filtering images {X} steps away...")
        images = []
        idxs = []
        for idx in tqdm(range(len(self.images))):
            image_t = self.images[idx]
            image_episode_idx = int(os.path.split(image_t)[-1].split('_')[0])
            image_folder = os.path.split(image_t)[0]
            history_folder = image_folder.replace(ORIGINAL_IMAGES_FOLDER, HISTORY_IMAGES_FOLDER)
            current_subgoal_index = int(image_t[-6:-4]) # get subgoal index
            subgoal_images_path = [d for d in glob.glob(image_folder+'/*')+glob.glob(history_folder+'/*')
                if (
                    int(d[-6:-4])==current_subgoal_index 
                    )]
            subgoal_images_arg_idxs = np.argsort(np.array([int(os.path.split(d)[-1].split('_')[0]) for d in subgoal_images_path]))
            subgoal_images_path = [subgoal_images_path[d_i] for d_i in list(subgoal_images_arg_idxs)]
            goal_subgoal_image_path = subgoal_images_path[-1]
            goal_subgoal_targets_path = goal_subgoal_image_path.replace(ORIGINAL_IMAGES_FOLDER, TARGETS_FOLDER).replace('.png', '')
            image_subgoal_index = [image_t==im for im in subgoal_images_path]
            image_subgoal_index = np.where(image_subgoal_index)[0]
            if (len(subgoal_images_path)-1) - int(image_subgoal_index) <= X: # add one to X since we have "done" action
                images.append(image_t)
                idxs.append(idx)

        return images, idxs

    def remove_done_images(self):
        print("Removing done images...")
        images = []
        idx_keep = []
        for idx in tqdm(range(len(self.images))):
            image_t = self.images[idx]
            targets_path = image_t.replace(ORIGINAL_IMAGES_FOLDER, TARGETS_FOLDER).replace('.png', '')
            targets_path, tag = os.path.split(targets_path)
            targets_episode = h5py.File(targets_path+'/targets.hdf5', 'r')
            data = int(targets_episode[tag+'_expert_action'][()])
            targets_episode.close()
            if data!=12:
                # TODO: last subgoal does not have done? fix this in datagen
                images.append(image_t)
                idx_keep.append(idx)
        print(f"Removed {len(self.images) - len(images)} done images")

        return images, idx_keep

    def format_subgoal(
        self,
        subgoal
        ):
        # format goal instruction
        subgoal = f'{self.idx2actions[subgoal[0]].replace("Object", "")} the {self.id_to_name[subgoal[1]]}.'
        subgoal = subgoal.replace('ToggleOn', 'Turn On')
        subgoal = subgoal.replace('ToggleOff', 'Turn Off')
        subgoal = subgoal.replace('Pickup', 'Pick Up')
        if self.name_to_mapped_name is not None and self.map_names:
            name_to_mapped_name_keys = list(self.name_to_mapped_name.keys())
            subgoal = subgoal.lower().replace('.','').split()
            for word_i in range(len(subgoal)):
                if subgoal[word_i] in name_to_mapped_name_keys:
                    subgoal[word_i] = self.name_to_mapped_name[subgoal[word_i]]
            subgoal = ' '.join(word for word in subgoal) + '.'
        return subgoal

    def transform_images(self, images):
        return (images - self.image_mean) / self.image_std

    def __getitem__(self, idx, only_fetch_demos=False, subgoal_info=None):

        image_t = self.images[idx]
        self.image_t = image_t
        image_episode_idx = int(os.path.split(image_t)[-1].split('_')[0])
        image_folder = os.path.split(image_t)[0]
        history_folder = image_folder.replace(ORIGINAL_IMAGES_FOLDER, HISTORY_IMAGES_FOLDER)
        history_images_path = [d for d in glob.glob(image_folder+'/*')+glob.glob(history_folder+'/*')
            if (
                int(os.path.split(d)[-1].split('_')[0])==image_episode_idx+1 
                )]
        image_tplus1 = history_images_path[0]
        
        targets_path = image_t.replace(ORIGINAL_IMAGES_FOLDER, TARGETS_FOLDER).replace('.png', '')
        history_idxs = np.array([int(os.path.split(d)[-1].split('_')[0]) for d in history_images_path])
        history_arg_idxs = np.argsort(history_idxs)

        targets_path, tag = os.path.split(targets_path)
        targets_episode = h5py.File(targets_path+'/targets.hdf5', 'r')

        targets = {}

        # ##### current observation is last index #####
        images = []
        images.append(np.asarray(Image.open(image_t)))
        images.append(np.asarray(Image.open(image_tplus1)))

        targets_to_output = ["expert_action", "obj_targets", "labels"]
        for target in targets_to_output:
            if target in ['masks', 'subgoal_percent_complete', 'goal_pose']:
                continue
            data = np.asarray(targets_episode[tag+f'_{target}'])
            if target in ['labels', 'obj_targets', 'obj_instance_ids', 'expert_action']:
                data = data.astype(np.int64)
            if len(data.shape)==0:
                data = np.asarray([data])
            targets[target] = data
        
        targets_episode.close()

        interm_idxs = targets['obj_targets']
        if int(targets["expert_action"]) in [0,1,2,3,4] or len(targets['labels'][interm_idxs])==0 or int(targets['labels'][interm_idxs])>=self.no_object_label: #[5,6,7,8,9,10,11]:
            targets['labels'] = np.asarray([self.no_object_label])
        else:
            targets['labels'] = np.asarray(targets['labels'][interm_idxs])

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

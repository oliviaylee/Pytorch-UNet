# Dataloaders for training

# remember: padding sequence to 20 if shorter, cap at 100

# (1) train input: images

# (2) train input: language annotation (as text)

# (3) train label: trajectories as a list of centroids
# normalize x, y of centroids? by image h, w

# (4) train label: contact points [x_left, y_top, x_right, y_bottom] -> segmentation mask
# one hot image

# save out contact points as a mask so we don't need to recompute each time

import logging
import numpy as np
import torch
from PIL import Image
from functools import lru_cache
from functools import partial
from itertools import repeat
from multiprocessing import Pool
from os import listdir
from os.path import splitext, isfile, join
from pathlib import Path
from torch.utils.data import Dataset
from tqdm import tqdm


def load_image(filename):
    ext = splitext(filename)[1]
    if ext == '.npy':
        return Image.fromarray(np.load(filename))
    elif ext in ['.pt', '.pth']:
        return Image.fromarray(torch.load(filename).numpy())
    else:
        return Image.open(filename)


def unique_mask_values(idx, mask_dir, mask_suffix):
    mask_file = list(mask_dir.glob(idx + mask_suffix + '.*'))[0]
    mask = np.asarray(load_image(mask_file))
    if mask.ndim == 2:
        return np.unique(mask)
    elif mask.ndim == 3:
        mask = mask.reshape(-1, mask.shape[-1])
        return np.unique(mask, axis=0)
    else:
        raise ValueError(f'Loaded masks should have 2 or 3 dimensions, found {mask.ndim}')

def generate_mask(im, contact_points):
    mask = np.zeros((im.shape[0], im.shape[1]))
    x1, y1, x2, y2 = np.round(contact_points[0]), np.round(contact_points[1]), np.round(contact_points[2]), np.round(contact_points[3])
    mask[x1:x2, y1:y2] = 1.0
    return mask

class BasicDataset(Dataset):
    def __init__(self, frame_data_dir: str = '/iris/u/oliviayl/repos/affordance-learning/epic_kitchens/results/video_data', scale: float = 1.0, mask_suffix: str = ''):
        self.frame_data_dir = Path(frame_data_dir)
        self.ids = []
        self.image_paths = []
        self.lang_annotations = []
        self.trajectories = []
        self.contact_points = []

        for vid_data_path in glob.glob(self.frame_data_dir):
            v = open(os.path.join(self.frame_data_dir, vid_data_path))
            data_json = json.load(v)
            for f in data_json['frame_data']:
                self.ids.append(os.path.join(vid_data_path, f))
                self.image_paths.append(data_json['frame_data'][f]['img_path'])
                self.lang_annotations.append(data_json['frame_data'][f]['lang_annotation'])
                trajectory = data_json['frame_data'][f]['trajectory']
                hand = trajectory[list(trajectory.keys())[-1]].keys()[0]
                centroids = [trajectory[x][hand]['centroid'] for x in trajectory.keys()]
                self.trajectories.append(centroids)
                self.contact_points.append(data_json['frame_data'][f]['contact_points'])

        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix

        # self.ids = [splitext(file)[0] for file in listdir(images_dir) if isfile(join(images_dir, file)) and not file.startswith('.')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')

        logging.info(f'Creating dataset with {len(self.ids)} examples')
        # logging.info('Scanning mask files to determine unique values')
        # with Pool() as p:
        #     unique = list(tqdm(
        #         p.imap(partial(unique_mask_values, mask_dir=self.mask_dir, mask_suffix=self.mask_suffix), self.ids),
        #         total=len(self.ids)
        #     ))

        self.mask_values = [0.0, 1.0] # list(sorted(np.unique(np.concatenate(unique), axis=0).tolist()))
        logging.info(f'Unique mask values: {self.mask_values}')

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def preprocess(mask_values, pil_img, scale, is_mask):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img = np.asarray(pil_img)

        if is_mask:
            mask = np.zeros((newH, newW), dtype=np.int64)
            for i, v in enumerate(mask_values):
                if img.ndim == 2:
                    mask[img == v] = i
                else:
                    mask[(img == v).all(-1)] = i

            return mask

        else:
            if img.ndim == 2:
                img = img[np.newaxis, ...]
            else:
                img = img.transpose((2, 0, 1))

            if (img > 1).any():
                img = img / 255.0

            return img

    @staticmethod
    def preprocess_trajectory(traj, img_size, min_len=20, max_len=100):
        w, h = img_size
        traj = [[x[0] / w, x[1] / h] for x in traj]
        if len(traj) < min_len:
            traj.extend([traj[-1] for _ in range(min_len - len(traj))]) # pad up to length 20
        elif len(traj) > max_len:
            traj = traj[:max_len]
        return traj
    
    @staticmethod
    def preprocess_lang(lang_annot):
        return ' '.join(lang_annot)

    def __getitem__(self, idx):
        name = self.ids[idx]
        video, frame = name[:name.find('/')], name[name.find('/')+1:]

        img_path, lang_annot, traj, contact_pts = self.image_paths[idx], self.lang_annotations[idx], self.trajectories[idx], self.contact_points[idx]

        # mask_file = list(self.mask_dir.glob(name + self.mask_suffix + '.*'))
        # img_file =  list(self.images_dir.glob(name + '.*'))

        # assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        # assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        # mask = load_image(mask_file[0])
        img = load_image(img_path)
        mask = generate_mask(img, contact_pts)

        assert img.size == mask.size, \
            f'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(self.mask_values, img, self.scale, is_mask=False)
        # mask = self.preprocess(self.mask_values, mask, self.scale, is_mask=True)
        traj = self.preprocess_trajectory(traj, img.size)
        lang_annot = self.preprocess_lang(lang_annot)

        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous(),
            'traj': torch.as_tensor(traj.copy()).float().contiguous()
            'lang_annot': lang_annot.copy()
        }
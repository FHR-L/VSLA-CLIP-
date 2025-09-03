# encoding: utf-8

import os
import os.path as osp
import glob



__all__ = ['CARGO', ]

from datasets.image.bases import BaseImageDataset


class CARGO(BaseImageDataset):
    dataset_dir = "CARGO"
    dataset_name = 'cargo'

    def __init__(self, root='datasets', **kwargs):
        super(CARGO, self).__init__()
        self.root = root
        self.data_dir = self.root

        self.train_dir = osp.join(self.data_dir, 'train')
        self.query_dir = osp.join(self.data_dir, 'query')
        self.gallery_dir = osp.join(self.data_dir, 'gallery')

        train = self.process_dir(self.train_dir, is_train=True)
        query = self.process_dir(self.query_dir, is_train=False)
        gallery = self.process_dir(self.gallery_dir, is_train=False)

        self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(
            self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(
            self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(
            self.gallery)


    def process_dir(self, dir_path, is_train=True):
        img_paths = []
        for cam_index in range(13):
            img_paths = img_paths + glob.glob(osp.join(dir_path, f'Cam{cam_index + 1}', '*.jpg'))

        data = []
        for img_path in img_paths:
            pid = int(img_path.split('/')[-1].split('_')[2])
            camid = int(img_path.split('/')[-1].split('_')[0][3:])
            viewid = 1 if camid <= 5 else 2
            camid -= 1  # index starts from 0

            if is_train:
                pid = pid-1
            data.append((img_path, pid, camid, viewid))
        return data


class CARGO_AA(BaseImageDataset):
    dataset_dir = "CARGO"
    dataset_name = 'cargo_aa'

    def __init__(self, root='datasets', **kwargs):
        super(CARGO_AA, self).__init__()
        self.root = root
        self.data_dir = self.root

        self.train_dir = osp.join(self.data_dir, 'train')
        self.query_dir = osp.join(self.data_dir, 'query')
        self.gallery_dir = osp.join(self.data_dir, 'gallery')

        # train = self.process_dir(self.train_dir, is_train=True)
        # query = self.process_dir(self.query_dir, is_train=False)
        # gallery = self.process_dir(self.gallery_dir, is_train=False)

        train = self.process_dir(self.train_dir, is_train=True)
        query = self.process_dir(self.query_dir, is_train=False)
        gallery = self.process_dir(self.gallery_dir, is_train=False)

        self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(
            self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(
            self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(
            self.gallery)

    def process_dir(self, dir_path, is_train=True):
        img_paths = []
        for cam_index in range(13):
            img_paths = img_paths + glob.glob(osp.join(dir_path, f'Cam{cam_index + 1}', '*.jpg'))

        data = []
        for img_path in img_paths:
            pid = int(img_path.split('/')[-1].split('_')[2])
            camid = int(img_path.split('/')[-1].split('_')[0][3:])
            viewid = 1 if camid <= 5 else 2
            camid -= 1  # index starts from 0
            if viewid == 2:
                continue
            data.append((img_path, pid, camid, viewid))
        return data

class CARGO_GG(BaseImageDataset):
    dataset_dir = "CARGO"
    dataset_name = 'cargo_gg'

    def __init__(self, root='datasets', **kwargs):
        super(CARGO_GG, self).__init__()
        self.root = root
        self.data_dir = self.root

        self.train_dir = osp.join(self.data_dir, 'train')
        self.query_dir = osp.join(self.data_dir, 'query')
        self.gallery_dir = osp.join(self.data_dir, 'gallery')

        train = self.process_dir(self.train_dir, is_train=True)
        query = self.process_dir(self.query_dir, is_train=False)
        gallery = self.process_dir(self.gallery_dir, is_train=False)

        self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(
            self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(
            self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(
            self.gallery)


    def process_dir(self, dir_path, is_train=True):
        img_paths = []
        for cam_index in range(13):
            img_paths = img_paths + glob.glob(osp.join(dir_path, f'Cam{cam_index + 1}', '*.jpg'))

        data = []
        for img_path in img_paths:
            pid = int(img_path.split('/')[-1].split('_')[2])
            camid = int(img_path.split('/')[-1].split('_')[0][3:])
            viewid = 1 if camid <= 5 else 2
            if viewid == 1:
                continue
            camid -= 1  # index starts from 0

            if is_train:
                pid = pid - 1
            #     camid = self.dataset_name + "_" + str(camid)
            data.append((img_path, pid, camid, viewid))
        return data


class CARGO_AG(BaseImageDataset):
    dataset_dir = "CARGO"
    dataset_name = 'cargo_ag'

    def __init__(self, root='datasets', **kwargs):
        super(CARGO_AG, self).__init__()
        self.root = root
        self.data_dir = self.root

        self.train_dir = osp.join(self.data_dir, 'train')
        self.query_dir = osp.join(self.data_dir, 'query')
        self.gallery_dir = osp.join(self.data_dir, 'gallery')

        train = self.process_dir(self.train_dir, is_train=True)
        query = self.process_dir(self.query_dir, is_train=False)
        gallery = self.process_dir(self.gallery_dir, is_train=False)

        self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(
            self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(
            self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(
            self.gallery)

    def process_dir(self, dir_path, is_train=True):
        img_paths = []
        for cam_index in range(13):
            img_paths = img_paths + glob.glob(osp.join(dir_path, f'Cam{cam_index + 1}', '*.jpg'))

        data = []
        for img_path in img_paths:
            pid = int(img_path.split('/')[-1].split('_')[2])
            camid = int(img_path.split('/')[-1].split('_')[0][3:])
            viewid = 1 if camid <= 5 else 2
            camid = 1 if camid <= 5 else 2
            camid -= 1  # index starts from 0

            if is_train:
                pid = pid - 1
            data.append((img_path, pid, camid, viewid))
        return data

import os
import shutil
import numpy as np
import cv2
from utils import get_config, get_data_shape


class BaseHandler:
    def __init__(self, cfg_path):
        self.cfg = get_config(cfg_path)
        self.data_shape = get_data_shape()
        self.packs = self.get_data_length_information()
        self.file_length = self.get_file_length()
        self.in_folders = os.listdir(self.cfg['in_root'])
        self.vis_path = None

    def run(self):
        self.make_output_root()
        if self.cfg['packaged_data']:
            self.handle_packaged_data()
        else:
            self.handle_normal_data()

    def get_data_length_information(self):
        return (
            np.product(self.data_shape['features_buffer']),
            np.product(self.data_shape['nav_features']),
            np.product(self.data_shape['traffic_convention']),
            np.product(self.data_shape['desire']),
            np.product(self.data_shape['output'])
        )

    def get_file_length(self):
        file_length = 0
        for i, item in enumerate(self.packs):
            if i == 1 and not self.cfg['have_nav']:
                continue
            file_length += item
        return file_length

    def make_output_root(self):
        if not os.path.exists(self.cfg['out_root']):
            os.mkdir(self.cfg['out_root'])
        if self.cfg['visualize']['active'] and self.cfg['visualize']['save_image']:
            self.vis_path = f"{self.cfg['out_root']}/{self.cfg['visualize']['save_path']}"
            if not os.path.exists(self.vis_path):
                os.mkdir(self.vis_path)

    def make_output_data_folder(self, folder):
        out_folder = f"{self.cfg['out_root']}/{folder}"
        if not os.path.exists(out_folder):
            os.mkdir(out_folder)
        return out_folder

    def process_image(self, image):
        return image

    def handle_normal_data(self):
        data_name = self.cfg['handle']['normal_name']
        for folder in self.in_folders:
            data_path = f"{self.cfg['in_root']}/{folder}"
            filename1 = f"{data_path}/{data_name['image']}"
            filename2 = f"{data_path}/{data_name['big_image']}"
            frame = np.fromfile(filename1, np.float32).reshape(12, 128, 256)
            frame_big = np.fromfile(filename2, np.float32).reshape(12, 128, 256)
            out_img = self.extract_images(frame_big, frame)

            if self.cfg['visualize']['active']:
                self.visualize(folder, out_img)

            if not self.cfg['handle']['active']:
                continue
            out_path = self.make_output_data_folder(folder)
            out_names = self.cfg['handle']['normal_name']
            processed_image = self.process_image(out_img)
            processed_input_imgs, processed_big_input_imgs = self.transform_back(processed_image)
            processed_input_imgs.astype(np.float32).tofile(f"{out_path}/{out_names['image']}")
            processed_big_input_imgs.astype(np.float32).tofile(f"{out_path}/{out_names['big_image']}")
            self.copy_files(data_name, data_path, out_path)

    def handle_packaged_data(self):
        data_name = self.cfg['handle']['packaged_name']
        for folder in self.in_folders:
            data_path = f"{self.cfg['in_root']}/{folder}"
            imgs_path = f"{data_path}/{data_name['image']}"
            files_path = f"{data_path}/{data_name['file']}"
            frames = np.fromfile(imgs_path, np.float32)
            frames = frames.reshape(-1, 24, 128, 256)
            files = np.fromfile(files_path, np.float32)
            files = files.reshape(-1, self.file_length)
            assert frames.shape[0] == files.shape[0], "file and image hold different sequence."
            for i in range(frames.shape[0]):
                merged_folder = f"{folder}{i:04d}"
                frame_big = frames[i, :12, :, :]
                frame = frames[i, 12:, :, :]
                file_dict = self.get_files(files[i, :])
                out_path = self.make_output_data_folder(merged_folder)
                out_img = self.extract_images(frame_big, frame)

                if self.cfg['handle']['unpack_only']:
                    processed_input_imgs = frame
                    processed_big_input_imgs = frame_big
                    if self.cfg['visualize']['active']:
                        self.visualize(merged_folder, out_img)
                else:
                    processed_image = self.process_image(out_img)
                    processed_input_imgs, processed_big_input_imgs = self.transform_back(processed_image)
                    if self.cfg['visualize']['active']:
                        self.visualize(merged_folder, processed_image)

                if not self.cfg['handle']['active']:
                    continue

                file_dict['image'] = processed_input_imgs.astype(np.float32)
                file_dict['big_image'] = processed_big_input_imgs.astype(np.float32)
                self.save_files(out_path, file_dict)

    def get_files(self, file):
        fb, nf, tc, de, ou = self.packs
        file_dict = {}
        if self.cfg['have_nav']:
            file_dict['features_buffer'] = file[:fb]
            file_dict['nav_features'] = file[fb:fb + nf]
            file_dict['traffic_convention'] = file[fb + nf:fb + nf + tc]
            file_dict['desire'] = file[fb + nf + tc: fb + nf + tc + de]
            file_dict['output'] = file[fb + nf + tc + de:]
        else:
            file_dict['features_buffer'] = file[:fb]
            file_dict['traffic_convention'] = file[fb:fb + tc]
            file_dict['desire'] = file[fb + tc: fb + tc + de]
            file_dict['output'] = file[fb + tc + de:]
        for key in file_dict:
            file_dict[key] = file_dict[key].reshape(self.data_shape[key])
        return file_dict

    def save_files(self, out_folder, file_dict):
        for key in file_dict:
            file_dict[key].tofile(f"{out_folder}/{self.cfg['handle']['normal_name'][key]}")

    def copy_files(self, data_name, src_path, tgt_path):
        for key in data_name:
            if 'image' in key:
                continue
            if os.path.exists(f"{src_path}/{data_name[key]}"):
                shutil.copy2(f"{src_path}/{data_name[key]}", f"{tgt_path}/{data_name[key]}")

    def extract_images(self, frame_big, frame):
        f1, f2 = np.split(frame, 2, axis=0)
        bgr1 = self.handle_one_image(f1)
        bgr2 = self.handle_one_image(f2)
        bgr = np.concatenate([bgr1, bgr2], axis=1)

        f_big1, f_big2 = np.split(frame_big, 2, axis=0)
        bgr_big1 = self.handle_one_image(f_big1)
        bgr_big2 = self.handle_one_image(f_big2)
        bgr_big = np.concatenate([bgr_big1, bgr_big2], axis=1)

        output = np.concatenate([bgr, bgr_big], axis=0)
        return output

    def handle_one_image(self, frame):
        y = np.zeros((128 * 2, 256 * 2), np.uint8)
        y[::2, ::2] = frame[0, :, :].astype(np.uint8)
        y[::2, 1::2] = frame[1, :, :].astype(np.uint8)
        y[1::2, ::2] = frame[2, :, :].astype(np.uint8)
        y[1::2, 1::2] = frame[3, :, :].astype(np.uint8)
        v = frame[4, :, :].astype(np.uint8)
        u = frame[5, :, :].astype(np.uint8)
        # # Resize u and v color channels to be the same size as y
        u = cv2.resize(u, (y.shape[1], y.shape[0]))
        v = cv2.resize(v, (y.shape[1], y.shape[0]))

        yvu = cv2.merge((y, v, u))  # Stack planes to 3D matrix (use Y,V,U ordering)
        return cv2.cvtColor(yvu, cv2.COLOR_YUV2BGR)

    def transform_back(self, image):
        yuv_image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        images, big_images = np.split(yuv_image, 2, axis=0)
        input_imgs = self.stack_back_images(images)
        big_input_imgs = self.stack_back_images(big_images)
        return input_imgs, big_input_imgs

    def stack_back_images(self, images):
        img1, img2 = np.split(images, 2, axis=1)
        stack = []
        for img in [img1, img2]:
            stack.append(img[::2, ::2, 0])
            stack.append(img[::2, 1::2, 0])
            stack.append(img[1::2, ::2, 0])
            stack.append(img[1::2, 1::2, 0])
            stack.append(img[::2, ::2, 1])
            stack.append(img[::2, ::2, 2])
        return np.stack(stack, axis=0).reshape(1, 12, 128, 256)

    def visualize(self, folder, image):
        if self.cfg['visualize']['show_image']:
            cv2.imshow(f"image", image)
            cv2.waitKey(1000 // self.cfg['visualize']['show_fps'])
        if self.cfg['visualize']['save_image']:
            save_path = self.vis_path + '/' + folder + '.jpg'
            cv2.imwrite(save_path, image)

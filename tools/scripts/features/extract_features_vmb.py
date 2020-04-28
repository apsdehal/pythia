# Requires vqa-maskrcnn-benchmark to be built and installed
# Category mapping for visual genome can be downloaded from
# https://dl.fbaipublicfiles.com/pythia/data/visual_genome_categories.json
# When the --background flag is set, the index saved with key "objects" in
# info_list will be +1 of the Visual Genome category mapping above and 0
# is the background class. When the --background flag is not set, the
# index saved with key "objects" in info list will match the Visual Genome
# category mapping.
import argparse
import glob
import os

import cv2
import numpy as np
import torch

# from maskrcnn_benchmark.config import cfg
# from maskrcnn_benchmark.layers import nms
# from maskrcnn_benchmark.modeling.detector import build_detection_model
# from maskrcnn_benchmark.structures.image_list import to_image_list
# from maskrcnn_benchmark.utils.model_serialization import load_state_dict
from PIL import Image
from torchvision.models.detection.image_list import ImageList

from mmf.utils.configuration import get_mmf_cache_dir
from mmf.utils.download import download
from mmf.utils.file_io import PathManager


def modify_state_dict(state):
    from copy import deepcopy

    state = deepcopy(state)
    model = state.pop("model", state)
    new_state = {}
    anchors = []
    for key, value in model.items():
        original_key = key
        key = key.replace("stem.", "")
        if "fpn_inner" in key:
            layer_number = key.split(".")[-2][-1]
            key = key.replace(
                "fpn_inner" + layer_number, "inner_blocks." + str(int(layer_number) - 1)
            )
        if "fpn_layer" in key:
            layer_number = key.split(".")[-2][-1]
            key = key.replace(
                "fpn_layer" + layer_number, "layer_blocks." + str(int(layer_number) - 1)
            )
        if "box.feature_extractor" in key:
            key = key.replace("box.feature_extractor", "box_head")
        if "box.predictor" in key:
            key = key.replace("box.predictor", "box_predictor")
        new_state[key] = value
    return {"model": new_state}


class FeatureExtractor:
    MODEL_URL = (
        "https://dl.fbaipublicfiles.com/pythia/detectron_model/detectron_model.pth"
    )
    CONFIG_URL = (
        "https://dl.fbaipublicfiles.com/pythia/detectron_model/detectron_model.yaml"
    )
    MAX_SIZE = 1333
    MIN_SIZE = 800

    def __init__(self):
        self.args = self.get_parser().parse_args()
        self.detection_model = self._build_detection_model()

        PathManager.mkdirs(self.args.output_folder)

    def _try_downloading_necessities(self):
        if self.args.model_file is None:
            print("Downloading model and configuration")
            cache_dir = get_mmf_cache_dir()
            self.args.model_file = os.path.join(
                cache_dir, self.MODEL_URL.split("/")[-1]
            )
            self.args.config_file = os.path.join(
                cache_dir, self.CONFIG_URL.split("/")[-1]
            )
            download_file(self.MODEL_URL, cache_dir, self.args.model_file)
            download_file(self.CONFIG_URL, cache_dir, self.args.model_file)

    def get_parser(self):
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--model_file", default=None, type=str, help="Detectron model file"
        )
        parser.add_argument(
            "--config_file", default=None, type=str, help="Detectron config file"
        )
        parser.add_argument(
            "--start_index", default=0, type=int, help="Index to start from "
        )
        parser.add_argument("--end_index", default=None, type=int, help="")
        parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
        parser.add_argument(
            "--num_features",
            type=int,
            default=100,
            help="Number of features to extract.",
        )
        parser.add_argument(
            "--output_folder", type=str, default="./output", help="Output folder"
        )
        parser.add_argument("--image_dir", type=str, help="Image directory or file")
        parser.add_argument(
            "--feature_name",
            type=str,
            help="The name of the feature to extract",
            default="fc6",
        )
        parser.add_argument(
            "--exclude_list",
            type=str,
            help="List of images to be excluded from feature conversion. Each image on a new line",
            default="./list",
        )
        parser.add_argument(
            "--confidence_threshold",
            type=float,
            default=0,
            help="Threshold of detection confidence above which boxes will be selected",
        )
        parser.add_argument(
            "--background",
            action="store_true",
            help="The model will output predictions for the background class when set",
        )
        parser.add_argument(
            "--maskrcnn",
            action="store_true",
            help="The model will output predictions for the background class when set",
        )
        return parser

    def _build_detection_model(self):
        checkpoint = torch.load(self.args.model_file, map_location=torch.device("cpu"))
        model_dict = checkpoint.pop("model")

        if self.args.maskrcnn:
            from maskrcnn_benchmark.config import cfg
            from maskrcnn_benchmark.modeling.detector import build_detection_model
            from maskrcnn_benchmark.utils.model_serialization import load_state_dict

            cfg.merge_from_file(self.args.config_file)
            cfg.freeze()

            model = build_detection_model(cfg)
            load_state_dict(model, model_dict)
        else:
            from mmf.models.detection import build_custom_mb_faster_rcnn

            model = build_custom_mb_faster_rcnn()
            updated_dict = modify_state_dict(model_dict)
            model.load_state_dict(updated_dict["model"])

        model.to("cuda")
        model.eval()

        return model

    def _image_transform(self, path):
        img = Image.open(path)
        im = np.array(img).astype(np.float32)

        if im.shape[-1] > 3:
            im = np.array(img.convert("RGB")).astype(np.float32)

        # IndexError: too many indices for array, grayscale images
        if len(im.shape) < 3:
            im = np.repeat(im[:, :, np.newaxis], 3, axis=2)

        im = im[:, :, ::-1]
        im -= np.array([102.9801, 115.9465, 122.7717])
        im_shape = im.shape
        im_height = im_shape[0]
        im_width = im_shape[1]
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])

        # Scale based on minimum size
        im_scale = self.MIN_SIZE / im_size_min

        # Prevent the biggest axis from being more than max_size
        # If bigger, scale it down
        if np.round(im_scale * im_size_max) > self.MAX_SIZE:
            im_scale = self.MAX_SIZE / im_size_max

        im = cv2.resize(
            im, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR
        )
        img = torch.from_numpy(im).permute(2, 0, 1)

        im_info = {"width": im_width, "height": im_height}

        return img, im_scale, im_info

    def _process_feature_extraction(
        self, output, im_scales, im_infos, feature_name="fc6", conf_thresh=0
    ):
        import pdb; pdb.set_trace()
        if self.args.maskrcnn:
            batch_size = len(output[0]["proposals"])
            n_boxes_per_image = [len(boxes) for boxes in output[0]["proposals"]]
            score_list = output[0]["scores"].split(n_boxes_per_image)
            score_list = [torch.nn.functional.softmax(x, -1) for x in score_list]
            feats = output[0][feature_name].split(n_boxes_per_image)
            cur_device = score_list[0].device
            import maskrcnn_benchmark

            nms = maskrcnn_benchmark.layers.nms
        else:
            batch_size = len(output)
            n_boxes_per_image = [len(o["proposals"]) for o in output]
            score_list = [o["scores"] for o in output]
            score_list = [torch.nn.functional.softmax(x, -1) for x in score_list]
            feats = [o[feature_name] for o in output]
            cur_device = score_list[0].device
            import torchvision

            nms = torchvision.ops.boxes.nms

        feat_list = []
        info_list = []

        for i in range(batch_size):
            if self.args.maskrcnn:
                proposals = output[0]["proposals"][i]
                bbox = proposals.bbox
            else:
                proposals = output[i]["proposals"]
                bbox = proposals
            dets = bbox / im_scales[i]
            scores = score_list[i]
            max_conf = torch.zeros((scores.shape[0])).to(cur_device)
            conf_thresh_tensor = torch.full_like(max_conf, conf_thresh)
            start_index = 1
            # Column 0 of the scores matrix is for the background class
            if self.args.background:
                start_index = 0
            for cls_ind in range(start_index, scores.shape[1]):
                cls_scores = scores[:, cls_ind]
                keep = nms(dets, cls_scores, 0.5)
                max_conf[keep] = torch.where(
                    # Better than max one till now and minimally greater than conf_thresh
                    (cls_scores[keep] > max_conf[keep])
                    & (cls_scores[keep] > conf_thresh_tensor[keep]),
                    cls_scores[keep],
                    max_conf[keep],
                )
            best_scores, best_score_indices = scores.max(dim=-1)
            ke = nms(dets, best_scores, 0.5)
            to_keep = torch.sort(best_scores[ke], descending=True)[1][:100]
            ke = ke[to_keep]

            sorted_scores, sorted_indices = torch.sort(max_conf, descending=True)
            num_boxes = (sorted_scores[: self.args.num_features] != 0).sum()
            keep_boxes = sorted_indices[: self.args.num_features]
            feat_list.append(feats[i][keep_boxes])
            bbox = (
                getattr(proposals[keep_boxes], "bbox", proposals[keep_boxes])
                / im_scales[i]
            )
            # Predict the class label using the scores
            objects = torch.argmax(scores[keep_boxes][:, start_index:], dim=1)

            info_list.append(
                {
                    "bbox": bbox.cpu().numpy(),
                    "num_boxes": num_boxes.item(),
                    "objects": objects.cpu().numpy(),
                    "cls_prob": scores[keep_boxes][:, start_index:].cpu().numpy(),
                    "image_width": im_infos[i]["width"],
                    "image_height": im_infos[i]["height"],
                }
            )

        return feat_list, info_list

    def get_detectron_features(self, image_paths):
        img_tensor, im_scales, im_infos = [], [], []

        for image_path in image_paths:
            im, im_scale, im_info = self._image_transform(image_path)
            img_tensor.append(im)
            im_scales.append(im_scale)
            im_infos.append(im_info)

        # Image dimensions should be divisible by 32, to allow convolutions
        # in detector to work
        if self.args.maskrcnn:
            from maskrcnn_benchmark.structures.image_list import to_image_list

            current_img_list = to_image_list(img_tensor, size_divisible=32)
        else:
            current_img_list = self.to_image_list(img_tensor, size_divisible=32)
        current_img_list = current_img_list.to("cuda")

        with torch.no_grad():
            output = self.detection_model(current_img_list)

        feat_list = self._process_feature_extraction(
            output,
            im_scales,
            im_infos,
            self.args.feature_name,
            self.args.confidence_threshold,
        )

        return feat_list

    def to_image_list(self, tensors, size_divisible=0):
        """
        tensors can be an ImageList, a torch.Tensor or
        an iterable of Tensors. It can't be a numpy array.
        When tensors is an iterable of Tensors, it pads
        the Tensors with zeros so that they have the same
        shape
        """
        if isinstance(tensors, torch.Tensor) and size_divisible > 0:
            tensors = [tensors]

        if isinstance(tensors, ImageList):
            return tensors
        elif isinstance(tensors, torch.Tensor):
            # single tensor shape can be inferred
            if tensors.dim() == 3:
                tensors = tensors[None]
            assert tensors.dim() == 4
            image_sizes = [tensor.shape[-2:] for tensor in tensors]
            return ImageList(tensors, image_sizes)
        elif isinstance(tensors, (tuple, list)):
            max_size = tuple(max(s) for s in zip(*[img.shape for img in tensors]))

            # TODO Ideally, just remove this and let me model handle arbitrary
            # input sizs
            if size_divisible > 0:
                import math

                stride = size_divisible
                max_size = list(max_size)
                max_size[1] = int(math.ceil(max_size[1] / stride) * stride)
                max_size[2] = int(math.ceil(max_size[2] / stride) * stride)
                max_size = tuple(max_size)

            batch_shape = (len(tensors),) + max_size
            batched_imgs = tensors[0].new(*batch_shape).zero_()
            for img, pad_img in zip(tensors, batched_imgs):
                pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)

            image_sizes = [im.shape[-2:] for im in tensors]

            return ImageList(batched_imgs, image_sizes)
        else:
            raise TypeError(
                "Unsupported type for to_image_list: {}".format(type(tensors))
            )

    def _chunks(self, array, chunk_size):
        for i in range(0, len(array), chunk_size):
            yield array[i : i + chunk_size]

    def _save_feature(self, file_name, feature, info):
        file_base_name = os.path.basename(file_name)
        file_base_name = file_base_name.split(".")[0]
        info_file_base_name = file_base_name + "_info.npy"
        file_base_name = file_base_name + ".npy"

        np.save(
            PathManager.open(
                os.path.join(self.args.output_folder, file_base_name), mode="wb"
            ),
            feature.cpu().numpy(),
        )
        np.save(
            PathManager.open(
                os.path.join(self.args.output_folder, info_file_base_name), mode="wb"
            ),
            info,
        )

    def files_based_on_extension(self, path, extensions):
        files = []
        all_files = PathManager.ls(path)
        for f in all_files:
            for ext in extensions:
                if f.endswith(ext):
                    files.append(os.path.join(path, f))
                    break
        return files

    def extract_features(self):
        image_dir = self.args.image_dir

        if os.path.isfile(image_dir):
            features, infos = self.get_detectron_features([image_dir])
            self._save_feature(image_dir, features[0], infos[0])
        else:
            extensions = [".png", ".jpg", ".jpeg"]
            files = self.files_based_on_extension(image_dir, extensions)
            files = {f: 1 for f in files}
            exclude = {}

            if os.path.exists(self.args.exclude_list):
                with open(self.args.exclude_list, "r") as f:
                    lines = f.readlines()
                    for line in lines:
                        exclude[
                            line.strip("\n").split(os.path.sep)[-1].split(".")[0]
                        ] = 1
            output_files = self.files_based_on_extension(
                self.args.output_folder, [".npy"]
            )
            output_dict = {}
            for f in output_files:
                file_name = f.split(os.path.sep)[-1].split(".")[0]
                output_dict[file_name] = 1

            for f in list(files.keys()):
                file_name = f.split(os.path.sep)[-1].split(".")[0]
                if file_name in output_dict or file_name in exclude:
                    files.pop(f)

            files = list(files.keys())

            finished = 0

            end_index = self.args.end_index
            if end_index is None:
                end_index = len(files)
            start_index = self.args.start_index

            total = len(files[start_index:end_index])

            for chunk in self._chunks(
                files[start_index:end_index], self.args.batch_size
            ):
                features, infos = self.get_detectron_features(chunk)
                for idx, file_name in enumerate(chunk):
                    self._save_feature(file_name, features[idx], infos[idx])
                finished += len(chunk)

                if finished % 200 == 0:
                    print("Processed {}/{}".format(finished, total))


if __name__ == "__main__":
    feature_extractor = FeatureExtractor()
    feature_extractor.extract_features()

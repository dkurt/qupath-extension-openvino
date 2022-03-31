# This script creates quantized INT8 version of StarDist model
import os
import cv2 as cv
import numpy as np
import argparse
import random
from math import pi

from tifffile import imread

from addict import Dict
from compression.graph import load_model, save_model
from compression.api.data_loader import DataLoader
from compression.engines.ie_engine import IEEngine
from compression.api.metric import Metric
from compression.pipeline.initializer import create_pipeline

parser = argparse.ArgumentParser(description="Quantizes OpenVino model to int8.",
                                 add_help=True, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--xml", default="dsb2018_heavy_augment.xml", type=str,
                    help="XML file for OpenVINO to quantize")
parser.add_argument("--data", default="./dsb2018/train", type=str,
                    help="Data directory root")
parser.add_argument("--int8_dir", default="./optimized", type=str,
                    help="INT8 directory for calibrated OpenVINO model")
args = parser.parse_args()


def normalize_percentile(inp, percentiles=[1.0, 99.0]):
    num_channels = inp.shape[2]
    total = inp.shape[0] * inp.shape[1]
    num_colors = 256
    scale = []
    offset = []
    for ch in range(num_channels):
        # Compute a histogram for a channel
        hist_item = cv.calcHist([inp], [ch], None, [num_colors], [0, num_colors])

        # Find two percentiles from colors distribution
        counter = 0
        i = 0
        ind = int(percentiles[i] / 100.0 * total)
        rng = [0, 0]
        for color in range(num_colors):
            counter += hist_item[color]
            if counter >= ind:
                rng[i] = color
                if i == 1:
                    break
                else:
                    i += 1
                ind = int(percentiles[i] / 100.0 * total)
        scale.append(1.0 / (rng[1] - rng[0]))
        offset.append(-rng[0])

    return (inp + offset) * scale


class DatasetsDataLoader(DataLoader):

    def __init__(self, config):
        if not isinstance(config, Dict):
            config = Dict(config)
        super().__init__(config)

        self.images = []
        self.masks = []
        for name in os.listdir(config['images']):
            img = imread(config['images'] + "/" + name)
            if img.shape[0] == 256 and img.shape[1] == 256:
                self.images.append(config['images'] + "/" + name)
                self.masks.append(config['masks'] + "/" + name)

    @property
    def size(self):
        return len(self.images)

    def __len__(self):
        return self.size

    def __getitem__(self, item):
        img = imread(self.images[item])
        mask = imread(self.masks[item])

        img = np.expand_dims(img, axis=-1)
        img = normalize_percentile(img)
        img = np.expand_dims(img.transpose(2, 0, 1), axis=0)

        return (item, mask), img

# Dictionary with the FP32 model info
model_config = Dict({
    'model_name': args.xml.split()[0],
    "model": args.xml,
    "weights": args.xml.replace('.xml', '.bin')
})

# Dictionary with the engine parameters
engine_config = Dict({
    'device': 'CPU',
    'stat_requests_number': 4,
    'eval_requests_number': 4
})

# Dictionary witn input dataset info
dataset_config = Dict({
    'images': args.data + "/images",
    'masks': args.data + "/masks",
})

# Quantization algorithm settings
algorithms = [
    {
        'name': 'DefaultQuantization', # Optimization algorithm name
        'params': {
            'target_device': 'CPU',
            'preset': 'performance', # Preset [performance (default), accuracy] which controls the quantization mode
                                     # (symmetric and asymmetric respectively)
            'stat_subset_size': 300  # Size of subset to calculate activations statistics that can be used
                                     # for quantization parameters calculation.
        }
    }
]

# Load the model.
model = load_model(model_config)

# Initialize the data loader and metric.
data_loader = DatasetsDataLoader(dataset_config)
# metric = AccuracyMetric()

# Initialize the engine for metric calculation and statistics collection.
engine = IEEngine(engine_config, data_loader, None)

# Initialize the engine for metric calculation and statistics collection.
pipeline = create_pipeline(algorithms, engine)

compressed_model = pipeline.run(model)

# Save the compressed model.
save_model(compressed_model, args.int8_dir)

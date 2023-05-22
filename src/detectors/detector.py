from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from progress.bar import Bar
import time
import torch

from models.model import create_model, load_model
# from utils.debugger import Debugger

class SA2T_Detector(object):
    def __init__(self, args):
        self.args = args
        print('Creating model...')
        self.model = create_model(args)
        self.model = load_model(self.model, args.load_model, resume=True)
        self.model = self.model.to(args.device)
        self.model.eval()

    def run(self, graph):
        net_time, post_time = 0, 0
        tot_time = 0

        # Inference 
        graph = graph.to(self.args.device)
        start_time = time.time()
        output = self.process(graph)
        forward_time = time.time()
        net_time += forward_time - start_time

        return {'results': output, 'net_time': net_time}

    def process(self, graph):
        with torch.no_grad():
            output = self.model(graph)
        return output

    def pre_process(self, graph, meta=None):
        raise NotImplementedError

    def post_process(self, output, graph):
       
        output = torch.clamp(output, min=0., max=1.)
        return output

    def merge_outputs(self, detections):
        raise NotImplementedError

    def debug(self, debugger, graph, dets, output, scale=1):
        raise NotImplementedError

    
from lib.test.tracker.basetracker import BaseTracker
import torch
from lib.test.tracker.seqtrack_utils import sample_target, transform_image_to_crop
import cv2
from lib.utils.box_ops import box_xywh_to_xyxy, box_xyxy_to_cxcywh
from lib.models.diffusiontrack import build_diffusiontrack
from lib.test.tracker.seqtrack_utils import Preprocessor
from lib.utils.box_ops import clip_box, clip_box_batch
import numpy as np


class DIFFUSIONTRACK(BaseTracker):
    def __init__(self, params, dataset_name):
        super(DIFFUSIONTRACK, self).__init__(params)
        network = build_diffusiontrack(params.cfg)
        print('load checkpoint from: ' + self.params.checkpoint)
        pth = torch.load(self.params.checkpoint, map_location='cpu')['net']
        network.load_state_dict(pth, strict=True)
        self.cfg = params.cfg
        self.seq_format = self.cfg.DATA.SEQ_FORMAT
        self.num_template = self.cfg.TEST.NUM_TEMPLATES
        self.bins = self.cfg.MODEL.BINS
        if self.cfg.TEST.WINDOW == True: # for window penalty
            self.hanning = torch.tensor(np.hanning(self.bins)).unsqueeze(0).cuda()
            self.hanning = self.hanning
        else:
            self.hanning = None
        self.start = self.bins + 1 # start token
        self.network = network.cuda()
        self.network.eval()
        self.preprocessor = Preprocessor()
        self.state = None
        self.debug = params.debug
        self.frame_id = 0
        self.video_index = 0

        # online update settings
        DATASET_NAME = dataset_name.upper()
        if hasattr(self.cfg.TEST.UPDATE_INTERVALS, DATASET_NAME):
            self.update_intervals = self.cfg.TEST.UPDATE_INTERVALS[DATASET_NAME]
        else:
            self.update_intervals = self.cfg.TEST.UPDATE_INTERVALS.DEFAULT
        print("Update interval is: ", self.update_intervals)
        if hasattr(self.cfg.TEST.UPDATE_THRESHOLD, DATASET_NAME):
            self.update_threshold = self.cfg.TEST.UPDATE_THRESHOLD[DATASET_NAME]
        else:
            self.update_threshold = self.cfg.TEST.UPDATE_THRESHOLD.DEFAULT
        print("Update threshold is: ", self.update_threshold)



    def initialize(self, image, info: dict):

        # get the initial templates
        z_patch_arr, _ = sample_target(image, info['init_bbox'], self.params.template_factor,
                                       output_sz=self.params.template_size)

        template = self.preprocessor.process(z_patch_arr)
        self.template_list = [template] * self.num_template

        # get the initial sequence i.e., [start]
        batch = template.shape[0]
        self.init_seq = (torch.ones([batch, 1]).to(template) * self.start).type(dtype=torch.int64)

        self.state = info['init_bbox']
        self.frame_id = 0

    def track(self, image, info: dict = None):
        H, W, _ = image.shape
        self.frame_id += 1
        x_patch_arr, resize_factor = sample_target(image, self.state, self.params.search_factor,
                                                   output_sz=self.params.search_size)  # (x1, y1, w, h)
        search = self.preprocessor.process(x_patch_arr)
        images_list = self.template_list + [search]

        # run the encoder
        with torch.no_grad():
            xz = self.network.forward_encoder(images_list)

        # run the decoder
        with torch.no_grad():
            if isinstance(xz, list):
                xz = xz[0]
            out_dict = self.network.inference_decoder(xz=xz,
                                                      sequence=self.init_seq,
                                                      window=self.hanning,
                                                      seq_format=self.seq_format)

        pred_boxes = out_dict['pred_boxes'].view(-1, 4).cpu().numpy()
        pred_logits = out_dict['pred_logits'].view(-1, 1).cpu().numpy()
        best_idx = np.argmax(pred_logits)
        confidence = np.max(pred_logits)
        pred_box = pred_boxes[best_idx, :]

        if 'ensemble_logits' in out_dict:
            pred_boxes_em = out_dict['ensemble_boxes'].view(-1, 4).cpu().numpy()
            pred_logits_em = out_dict['ensemble_logits'].view(-1, 1).cpu().numpy()
            best_idx_em = np.argmax(pred_logits_em)
            pred_box_em = pred_boxes_em[best_idx_em, :]

        # 'ensemble_logits':ensemble_logits[-1], 'ensemble_boxes': ensemble_boxes[-1]}

        # get the final box result
        self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)

        # update the template
        if self.num_template > 1:
            conf_score = confidence   # the confidence score
            if (self.frame_id % self.update_intervals == 0) and (conf_score > self.update_threshold):
                z_patch_arr, _ = sample_target(image, self.state, self.params.template_factor,
                                               output_sz=self.params.template_size)
                template = self.preprocessor.process(z_patch_arr)
                self.template_list.append(template)
                if len(self.template_list) > self.num_template:
                    self.template_list.pop(1)

        # for debug
        if self.debug == 1:
            x1, y1, w, h = self.state
            image_BGR = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.rectangle(image_BGR, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color=(0, 0, 255), thickness=2)
            cv2.imshow('vis', image_BGR)
            cv2.waitKey(1)

        return {"target_bbox": self.state}

    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        x1, y1, x2, y2 = pred_box
        w, h = (x2-x1), (y2-y1)
        cx = (x1 + x2)*0.5
        cy = (y1 + y2)*0.5

        long_side = self.params.search_size / resize_factor
        cx_abs = cx / resize_factor
        cy_abs = cy / resize_factor

        cx_current = cx_abs + cx_prev - long_side/2
        cy_current = cy_abs + cy_prev - long_side/2


        w_abs = w / resize_factor
        h_abs = h / resize_factor
        x1_current = cx_current - 0.5*w_abs
        y1_current = cy_current - 0.5*h_abs

        return [x1_current, y1_current, w_abs, h_abs]

    def map_box_back_batch(self, pred_box: torch.Tensor, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
       # cx, cy, w, h = pred_box.unbind(-1) # (N,4) --> (N,)
        x1, y1, x2, y2 = pred_box.unbind(-1)
        w, h = (x2 - x1), (y2 - y1)
        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5

        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return torch.stack([cx_real - 0.5 * w, cy_real - 0.5 * h, w, h], dim=-1)

def random_2points(x1, y1, x2, y2):
    import random
    a = random.random()

    if 0< a <0.25:
        point1 = (x1, y1)
        point2 = (x2, y2)

    elif 0.25<a<0.5:
        point1 = (x1, y2)
        point2 = (x2, y1)

    elif 0.5 < a < 0.75:
        point1 = (x2, y1)
        point2 = (x1, y2)

    elif 0.75 < a < 1:
        point1 = (x2, y2)
        point2 = (x1, y1)
    else:
        point1 = (x1, y1)
        point2 = (x2, y2)
    return point1, point2


def get_tracker_class():
    return DIFFUSIONTRACK

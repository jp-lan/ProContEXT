import math

from lib.models.procontext import build_procontext
from lib.test.tracker.basetracker import BaseTracker
import torch

from lib.test.tracker.vis_utils import gen_visualization
from lib.test.utils.hann import hann2d
from lib.train.data.processing_utils import sample_target
# for debug
import cv2
import os
from copy import deepcopy

from lib.test.tracker.data_utils import Preprocessor
from lib.utils.box_ops import clip_box
from lib.utils.ce_utils import generate_mask_cond


class ProContEXT(BaseTracker):
    def __init__(self, params, dataset_name):
        super(ProContEXT, self).__init__(params)
        network = build_procontext(params.cfg, training=False)
        network.load_state_dict(torch.load(self.params.checkpoint, map_location='cpu')['net'], strict=True)
        self.cfg = params.cfg
        self.network = network.cuda()
        self.network.eval()
        self.preprocessor = Preprocessor()
        self.state = None

        self.feat_sz = self.cfg.TEST.SEARCH_SIZE // self.cfg.MODEL.BACKBONE.STRIDE
        # motion constrain
        self.output_window = hann2d(torch.tensor([self.feat_sz, self.feat_sz]).long(), centered=True).cuda()

        # for debug
        self.debug = params.debug
        self.use_visdom = params.debug
        self.frame_id = 0
        if self.debug:
            if not self.use_visdom:
                self.save_dir = "debug"
                if not os.path.exists(self.save_dir):
                    os.makedirs(self.save_dir)
            else:
                # self.add_hook()
                self._init_visdom(None, 1)
        # for save boxes from all queries
        self.z_dict1 = {}
        self.z_dict_list = []
        self.update_intervals = [200]

    def initialize(self, image, info: dict):
        # crop templates
        crop_resize_patches = [sample_target(image, info['init_bbox'], factor, output_sz=self.params.template_size)
                                                    for factor in self.params.template_factor]
        z_patch_arr, resize_factor, z_amask_arr = zip(*crop_resize_patches)
        for idx in range(len(z_patch_arr)):
            template = self.preprocessor.process(z_patch_arr[idx], z_amask_arr[idx])
            with torch.no_grad():
                self.z_dict1 = template
            self.z_dict_list.append(self.z_dict1)
        self.box_mask_z = []
        if self.cfg.MODEL.BACKBONE.CE_LOC:
            for i in range(len(self.params.template_factor) * 2):
                template_bbox = self.transform_bbox_to_crop(info['init_bbox'], resize_factor[0],
                                                            template.tensors.device).squeeze(1)
                self.box_mask_z.append(generate_mask_cond(self.cfg, 1, template.tensors.device, template_bbox))

        #init dynamic templates with static templates
        for idx in range(len(self.params.template_factor)):
            self.z_dict_list.append(deepcopy(self.z_dict_list[idx]))

        # save states
        self.state = info['init_bbox']
        self.frame_id = 0

    def track(self, image, info: dict = None):
        H, W, _ = image.shape
        self.frame_id += 1
        x_patch_arr, resize_factor, x_amask_arr = sample_target(image, self.state, self.params.search_factor,
                                                                output_sz=self.params.search_size)  # (x1, y1, w, h)
        search = self.preprocessor.process(x_patch_arr, x_amask_arr)

        with torch.no_grad():
            x_dict = search
            # merge the template and the search
            # run the transformer
            if isinstance(self.z_dict_list, (list, tuple)):
                self.z_dict = []
                for i in range(len(self.params.template_factor) * 2):
                    self.z_dict.append(self.z_dict_list[i].tensors)
            out_dict = self.network.forward(template=self.z_dict, search=x_dict.tensors, ce_template_mask=self.box_mask_z)

        # add hann windows
        pred_score_map = out_dict['score_map']
        conf_score = out_dict['score']
        response = self.output_window * pred_score_map
        pred_boxes = self.network.box_head.cal_bbox(response, out_dict['size_map'], out_dict['offset_map'])
        pred_boxes = pred_boxes.view(-1, 4)
        # Baseline: Take the mean of all pred boxes as the final result
        pred_box = (pred_boxes.mean(
            dim=0) * self.params.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]
        # get the final box result
        self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)

        for idx, update_i in enumerate([100]):
            if self.frame_id % update_i == 0 and conf_score > 0.7:
                crop_resize_patches2 = [sample_target(image, self.state, factor, output_sz=self.params.template_size)
                                                    for factor in self.params.template_factor]
                z_patch_arr2, _, z_amask_arr2 = zip(*crop_resize_patches2)
                for idx_s in range(len(z_patch_arr2)):
                    template_t = self.preprocessor.process(z_patch_arr2[idx_s], z_amask_arr2[idx_s])
                    self.z_dict_list[idx_s+len(self.params.template_factor)] = template_t

        # for debug
        if self.debug:
            if not self.use_visdom:
                x1, y1, w, h = self.state
                image_BGR = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                cv2.rectangle(image_BGR, (int(x1),int(y1)), (int(x1+w),int(y1+h)), color=(0,0,255), thickness=2)
                save_path = os.path.join(self.save_dir, "%04d.jpg" % self.frame_id)
                cv2.imwrite(save_path, image_BGR)
            else:
                self.visdom.register((image, info['gt_bbox'].tolist(), self.state), 'Tracking', 1, 'Tracking')

                self.visdom.register(torch.from_numpy(x_patch_arr).permute(2, 0, 1), 'image', 1, 'search_region')
                self.visdom.register(torch.from_numpy(self.z_patch_arr).permute(2, 0, 1), 'image', 1, 'template')
                self.visdom.register(pred_score_map.view(self.feat_sz, self.feat_sz), 'heatmap', 1, 'score_map')
                self.visdom.register((pred_score_map * self.output_window).view(self.feat_sz, self.feat_sz), 'heatmap', 1, 'score_map_hann')

                if 'removed_indexes_s' in out_dict and out_dict['removed_indexes_s']:
                    removed_indexes_s = out_dict['removed_indexes_s']
                    removed_indexes_s = [removed_indexes_s_i.cpu().numpy() for removed_indexes_s_i in removed_indexes_s]
                    masked_search = gen_visualization(x_patch_arr, removed_indexes_s)
                    self.visdom.register(torch.from_numpy(masked_search).permute(2, 0, 1), 'image', 1, 'masked_search')

                while self.pause_mode:
                    if self.step:
                        self.step = False
                        break

        return {"target_bbox": self.state}

    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def map_box_back_batch(self, pred_box: torch.Tensor, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box.unbind(-1) # (N,4) --> (N,)
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return torch.stack([cx_real - 0.5 * w, cy_real - 0.5 * h, w, h], dim=-1)

    def add_hook(self):
        conv_features, enc_attn_weights, dec_attn_weights = [], [], []

        for i in range(12):
            self.network.backbone.blocks[i].attn.register_forward_hook(
                # lambda self, input, output: enc_attn_weights.append(output[1])
                lambda self, input, output: enc_attn_weights.append(output[1])
            )

        self.enc_attn_weights = enc_attn_weights


def get_tracker_class():
    return ProContEXT

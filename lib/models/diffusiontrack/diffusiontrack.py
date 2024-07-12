"""
SeqTrack Model
"""
import torch
import math
from torch import nn
import torch.nn.functional as F

from lib.utils.misc import NestedTensor

from lib.models.diffusiontrack.encoder import build_encoder
from .decoder import build_decoder
from .head import DynamicHead
from .loss import HungarianMatcherDynamicK, SetCriterionDynamicK, SetCriterionAllProposal
from lib.utils.box_ops import box_xyxy_to_cxcywh, box_cxcywh_to_xyxy
from lib.utils.pos_embed import get_sinusoid_encoding_table, get_2d_sincos_pos_embed
import random
from detectron2.layers import batched_nms
from lib.utils.box_ops import giou_loss
from torch.nn.functional import l1_loss
import copy

def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def extract(a, t, x_shape):
    """extract the appropriate  t  index for a batch of indices"""
    batch_size = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class DIFFUSIONTRACK(nn.Module):
    """ This is the base class for DIFFUSIONTRACK """
    def __init__(self, cfg, encoder, hidden_dim,
                 bins=1000, feature_type='x', num_frames=1, num_template=1, stride = 16, ):
        """ Initializes the model.
        Parameters:
            encoder: torch module of the encoder to be used. See encoder.py
            decoder: torch module of the decoder architecture. See decoder.py
        """
        super().__init__()
        self.encoder = encoder
        self.encoder_feature =  cfg.MODEL.HEAD.ENCODER_FEATURE
        self.num_patch_x = self.encoder.body.num_patches_search
        self.num_patch_z = self.encoder.body.num_patches_template
        self.side_fx = int(math.sqrt(self.num_patch_x))
        self.side_fz = int(math.sqrt(self.num_patch_z))
        self.stride = stride
        self.x_h = self.side_fx * stride
        self.z_h = self.side_fz * stride

        self.hidden_dim = hidden_dim
        self.bottleneck = nn.Linear(encoder.num_channels, hidden_dim)  # the bottleneck layer
        self.head = DynamicHead(cfg, [stride, hidden_dim])

        self.num_frames = num_frames
        self.num_template = num_template
        self.feature_type = feature_type

        # Different type of visual features for decoder.
        # Since we only use one search image for now, the 'x' is same with 'x_last' here.
        if self.feature_type == 'x':
            num_patches = self.num_patch_x * self.num_frames
        elif self.feature_type == 'xz':
            num_patches = self.num_patch_x * self.num_frames + self.num_patch_z * self.num_template
        elif self.feature_type == 'token':
            num_patches = 1
        else:
            raise ValueError('illegal feature type')

        # position embeding for the decocder
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_dim))
        pos_embed = get_sinusoid_encoding_table(num_patches, self.pos_embed.shape[-1], cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        """for diffusion settings"""
        self.num_proposals = cfg.MODEL.HEAD.NUM_PROPOSALS
        self.hidden_dim = cfg.MODEL.HEAD.HIDDEN_DIM
        self.num_heads = cfg.MODEL.HEAD.NUM_HEADS

        # build diffusion
        timesteps = 1000
        sampling_timesteps = 1  # cfg.MODEL.DiffusionDet.SAMPLE_STEP
        self.objective = 'pred_x0'
        betas = cosine_beta_schedule(timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)
        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        self.sampling_timesteps = default(sampling_timesteps, timesteps)
        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = 1.
        self.self_condition = False
        self.scale = 2.  # cfg.MODEL.DiffusionDet.SNR_SCALE
        self.box_renewal = True
        self.use_ensemble = True

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        self.register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        self.register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))
        self.register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # Build Criterion.
        class_weight = cfg.MODEL.LOSS.CLASS_WEIGHT
        giou_weight = cfg.MODEL.LOSS.GIOU_WEIGHT
        l1_weight = cfg.MODEL.LOSS.L1_WEIGHT
        no_object_weight = cfg.MODEL.LOSS.NO_OBJECT_WEIGHT
        self.use_focal = cfg.MODEL.LOSS.USE_FOCAL
        self.deep_supervision = cfg.MODEL.LOSS.DEEP_SUPERVISION
        self.num_classes = 1

        matcher = HungarianMatcherDynamicK(
            cfg=cfg, cost_class=class_weight, cost_bbox=l1_weight, cost_giou=giou_weight, use_focal=self.use_focal
        )
        weight_dict = {"loss_ce": class_weight, "loss_bbox": l1_weight, "loss_giou": giou_weight}

        if self.deep_supervision:
            aux_weight_dict = {}
            for i in range(self.num_heads - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        losses = ["labels", "boxes"]

        self.criterion = SetCriterionDynamicK(
            cfg=cfg, num_classes=self.num_classes, matcher=matcher, weight_dict=weight_dict, eos_coef=no_object_weight,
            losses=losses, use_focal=self.use_focal, )


    def forward(self, images_list=None, xz=None, boxes=None,  mode="encoder", t = None,
                 targets = None,
                outputs= None, new_targets= None ):
        """
        image_list: list of template and search images, template images should precede search images
        xz: feature from encoder
        seq: input sequence of the decoder
        mode: encoder or decoder.
        """
        if mode == "encoder":
            return self.forward_encoder(images_list)
        elif mode == "decoder":
            return self.forward_decoder(xz, boxes, t)
        elif mode == "prepare_target":
            return self.prepare_targets(targets)
        elif mode == "compute_loss":
            return self.criterion(outputs, new_targets), self.criterion.weight_dict
        else:
            raise ValueError

    def forward_encoder(self, images_list):
        # Forward the encoder
        xz = self.encoder(images_list)
        return xz


    def forward_decoder(self, xz, boxes, t):
        if isinstance(xz, dict):
            xz = [xz['xz']]
        if isinstance(xz, list):
            xz = [xz[0]['xz']]
        xz_mem = xz[-1]

        B, _, _ = xz_mem.shape
        dec_mem = xz_mem[:, 0:self.num_patch_x * self.num_frames]

        # align the dimensions of the encoder and decoder
        if dec_mem.shape[-1] != self.hidden_dim:
            dec_mem = self.bottleneck(dec_mem)  # [B, NL, C]
        # dec_mem = dec_mem.permute(1,0,2)  #[NL,B, C]

        outputs_class, outputs_coord, box_history = self.head(dec_mem, boxes, t, None)
        output = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}

        if self.deep_supervision:
            output['aux_outputs'] = [{'pred_logits': a, 'pred_boxes': b}
                                     for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

        return output


    def inference_decoder(self, xz, sequence=None, window=None, seq_format='xywh'):
        if isinstance(xz, list):
            xz = [xz[0]['xz']]
        if isinstance(xz, dict):
            xz = [xz['xz']]
        # the decoder for testing
        xz_feat = xz[0]
        x_feat = xz_feat[:, 0:self.num_patch_x * self.num_frames]
        if x_feat.shape[-1] != self.hidden_dim:
            x_feat_bottleneck = self.bottleneck(x_feat)  # [B, NL, C]
        output = self.ddim_sample(backbone_x_feats = x_feat_bottleneck)
        return output

    def prepare_diffusion_repeat(self, gt_boxes):
        """
        :param gt_boxes: (cx, cy, w, h), normalized
        :param num_proposals:
        """
        t = torch.randint(0, self.num_timesteps, (1,), device=self.device).long()
        noise = torch.randn(self.num_proposals, 4, device=self.device)

        num_gt = gt_boxes.shape[0]
        if not num_gt:  # generate fake gt boxes if empty gt boxes
            gt_boxes = torch.as_tensor([[0.5, 0.5, 1., 1.]], dtype=torch.float, device=self.device)
            num_gt = 1

        num_repeat = self.num_proposals // num_gt  # number of repeat except the last gt box in one image
        repeat_tensor = [num_repeat] * (num_gt - self.num_proposals % num_gt) + [num_repeat + 1] * (
                self.num_proposals % num_gt)
        assert sum(repeat_tensor) == self.num_proposals
        random.shuffle(repeat_tensor)
        repeat_tensor = torch.tensor(repeat_tensor, device=self.device)

        gt_boxes = (gt_boxes * 2. - 1.) * self.scale
        x_start = torch.repeat_interleave(gt_boxes, repeat_tensor, dim=0)

        # noise sample
        x = self.q_sample(x_start=x_start, t=t, noise=noise)

        x = torch.clamp(x, min=-1 * self.scale, max=self.scale)
        x = ((x / self.scale) + 1) / 2.

        diff_boxes = box_cxcywh_to_xyxy(x)

        return diff_boxes, noise, t

    def prepare_diffusion_concat(self, gt_boxes):
        """
        :param gt_boxes: (cx, cy, w, h), normalized
        :param num_proposals:
        """
        t = torch.randint(0, self.num_timesteps, (1,), device=gt_boxes.device).long()
        noise = torch.randn(self.num_proposals, 4, device=gt_boxes.device)

        num_gt = gt_boxes.shape[0]
        if not num_gt:  # generate fake gt boxes if empty gt boxes
            gt_boxes = torch.as_tensor([[0.5, 0.5, 1., 1.]], dtype=torch.float, device=gt_boxes.device)
            num_gt = 1

        if num_gt < self.num_proposals:
            box_placeholder = torch.randn(self.num_proposals - num_gt, 4,
                                          device=gt_boxes.device) / 6. + 0.5  # 3sigma = 1/2 --> sigma: 1/6
            box_placeholder[:, 2:] = torch.clip(box_placeholder[:, 2:], min=1e-4)
            x_start = torch.cat((gt_boxes, box_placeholder), dim=0)
        elif num_gt > self.num_proposals:
            select_mask = [True] * self.num_proposals + [False] * (num_gt - self.num_proposals)
            random.shuffle(select_mask)
            x_start = gt_boxes[select_mask]
        else:
            x_start = gt_boxes

        x_start = (x_start * 2. - 1.) * self.scale

        # noise sample
        x = self.q_sample(x_start=x_start, t=t, noise=noise)

        x = torch.clamp(x, min=-1 * self.scale, max=self.scale)
        x = ((x / self.scale) + 1) / 2.

        diff_boxes = box_cxcywh_to_xyxy(x)

        return diff_boxes, noise, t

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def prepare_targets(self, targets_input):
        new_targets = []
        diffused_boxes = []
        noises = []
        ts = []
        bs = targets_input.size(0)
        for i in range(bs):
            target={}
            gt_boxes = targets_input[i:i+1,:] # / image_size_xyxy
            d_boxes, d_noise, d_t = self.prepare_diffusion_concat(gt_boxes)
            diffused_boxes.append(d_boxes)
            noises.append(d_noise)
            ts.append(d_t)
            #target = gt_boxes.squeeze(0).to(targets_input.device)

            target["labels"] = torch.tensor([0], dtype=torch.int64).to(targets_input.device)
            target["boxes"] = gt_boxes.to(targets_input.device)

            boxes_xyxy = box_cxcywh_to_xyxy(gt_boxes)
            h = w = self.side_fx * self.stride
            boxes_xyxy_abs = boxes_xyxy * h
            target["boxes_xyxy"] = boxes_xyxy_abs.to(targets_input.device)

            image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float, device=targets_input.device)
            target["image_size_xyxy"] = image_size_xyxy.to(targets_input.device)
            image_size_xyxy_tgt = image_size_xyxy.unsqueeze(0).repeat(len(gt_boxes), 1)
            target["image_size_xyxy_tgt"] = image_size_xyxy_tgt.to(targets_input.device)
            area = (boxes_xyxy_abs[0, 2] - boxes_xyxy_abs[0, 0]) * (boxes_xyxy_abs[0, 3]-boxes_xyxy_abs[0, 1])
            area = area.to(targets_input.device)
            target["area"] = area.repeat(len(gt_boxes))

            new_targets.append(target)

        t_s_target = torch.stack(ts)
        t_s_target = t_s_target.squeeze(-1)

        return new_targets, torch.stack(diffused_boxes), torch.stack(noises), t_s_target

    def predict_noise_from_start(self, x_t, t, x0):
        return (
                (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) /
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def model_predictions(self, backbone_x_feats, images_whwh, x, t, clip_x_start=True):
        x_boxes = torch.clamp(x, min=-1 * self.scale, max=self.scale)
        x_boxes = ((x_boxes / self.scale) + 1) / 2
        x_boxes = box_cxcywh_to_xyxy(x_boxes)
        x_boxes = x_boxes * images_whwh[:, None, :]
        outputs_class, outputs_coord, bbox_history = self.head(backbone_x_feats, x_boxes, t, None)

        x_start = outputs_coord[-1]  # (batch, num_proposals, 4) predict boxes: absolute coordinates (x1, y1, x2, y2)
        x_start = x_start / images_whwh[:, None, :]
        x_start = box_xyxy_to_cxcywh(x_start)
        x_start = (x_start * 2 - 1.) * self.scale
        x_start = torch.clamp(x_start, min=-1 * self.scale, max=self.scale)
        pred_noise = self.predict_noise_from_start(x, t, x_start)

        return [pred_noise, x_start], outputs_class, outputs_coord, bbox_history

    @torch.no_grad()
    def ddim_sample(self, backbone_x_feats,  clip_denoised=True):
        try:
           device = backbone_x_feats[0].device
        except:
            device = backbone_x_feats.device

        batch = 1  # images_whwh.shape[0]
        shape = (batch, self.num_proposals, 4)
        total_timesteps, sampling_timesteps, eta, objective = self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        img = torch.randn(shape, device=device)
        images_whwh = torch.tensor([[self.x_h, self.x_h, self.x_h, self.x_h ]], device=device)

        ensemble_score, ensemble_label, ensemble_coord = [], [], []
        x_start = None
        for time, time_next in time_pairs:
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            self_cond = x_start if self.self_condition else None

            preds, outputs_class, outputs_coord, bbox_history = self.model_predictions(backbone_x_feats, images_whwh, img, time_cond, clip_x_start=clip_denoised)
          #  bbox_history = outputs_coord.clone().squeeze(1)
            cls_history_no = outputs_class.clone().squeeze(1)
            cls_history = torch.sigmoid(cls_history_no)

            pred_noise, x_start = preds[0], preds[1]

            if self.box_renewal:  # filter
                score_per_image, box_per_image = outputs_class[-1][0], outputs_coord[-1][0]
                threshold = 0.5
                score_per_image = torch.sigmoid(score_per_image)
                value, _ = torch.max(score_per_image, -1, keepdim=False)
                keep_idx = value > threshold
                num_remain = torch.sum(keep_idx)

                pred_noise = pred_noise[:, keep_idx, :]
                x_start = x_start[:, keep_idx, :]
                img = img[:, keep_idx, :]

            if self.sampling_timesteps > 1:
                outputs_class = torch.sigmoid( outputs_class[-1].squeeze())
                outputs_coord = outputs_coord[-1].squeeze()
                outputs_label = outputs_class.clone()
                one = torch.ones_like(outputs_class, device=outputs_class.device)
                zero = torch.zeros_like(outputs_class, device=outputs_class.device)

                outputs_label = torch.where(outputs_label>0.6, one, zero)
                ensemble_score.append(outputs_class)
                ensemble_coord.append(outputs_coord)
                ensemble_label.append(outputs_label)

            if time_next < 0:
                img = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(img)

            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

            if self.box_renewal:  # filter
                # replenish with randn boxes
             #  if num_remain.item() < 1:
               # add = int(10 - num_remain.item())
              #  if add > 0:
               #    img = torch.cat((img, torch.randn(1, add, 4, device=img.device)), dim=1)

                 img = torch.cat((img, torch.randn(1, self.num_proposals - num_remain, 4, device=img.device)), dim=1)
                  # print(str(img.size()))


        if self.sampling_timesteps > 1:
            box_pred_per_image = torch.cat(ensemble_coord, dim=0)
            scores_per_image = torch.cat(ensemble_score, dim=0)
            labels_per_image = torch.cat(ensemble_label, dim=0)

            keep = batched_nms(box_pred_per_image, scores_per_image, labels_per_image, 0.5)
            box_pred_per_image = box_pred_per_image[keep]
            scores_per_image = scores_per_image[keep]
            labels_per_image = labels_per_image[keep]

            output = {'pred_logits': scores_per_image, 'pred_boxes': box_pred_per_image, 'bbox_history': bbox_history, 'cls_history':cls_history}
                      #'ensemble_logits':ensemble_logits[-1], 'ensemble_boxes': ensemble_boxes[-1]}
        else:
            pred_logits = torch.sigmoid(outputs_class[-1])
            pred_boxes = outputs_coord
            output = {'pred_logits': pred_logits[-1], 'pred_boxes': pred_boxes[-1], 'bbox_history': bbox_history, 'cls_history':cls_history}

        return output







class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build_diffusiontrack(cfg):
    encoder = build_encoder(cfg)
    model = DIFFUSIONTRACK(
        cfg = cfg,
        encoder = encoder,
        hidden_dim=cfg.MODEL.HIDDEN_DIM,
        bins = cfg.MODEL.BINS,
        feature_type = cfg.MODEL.FEATURE_TYPE,
        num_frames = cfg.DATA.SEARCH.NUMBER,
        num_template = cfg.DATA.TEMPLATE.NUMBER,
        stride = cfg.MODEL.STRIDE,
    )

    return model


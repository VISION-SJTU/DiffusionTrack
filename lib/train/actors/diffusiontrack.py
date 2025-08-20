from . import BaseActor
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy, box_xyxy_to_cxcywh, box_cxcywh_to_xyxy, box_iou
import torch


class DiffusionTrackActor(BaseActor):
    """ Actor for training the SeqTrack"""
    def __init__(self, net, settings, cfg):
        super().__init__(net)
        self.settings = settings
        self.bs = self.settings.batchsize  # batch size
        self.seq_format = cfg.DATA.SEQ_FORMAT

    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'template', 'search', 'search_anno'.
            template_images: (N_t, batch, 3, H, W)
            search_images: (N_s, batch, 3, H, W)
        returns:
            loss    - the training loss
            status  -  dict containing detailed losses
        """
        # forward pass
        outputs, new_targets = self.forward_pass(data)

        # compute losses
        loss, status = self.compute_losses(outputs, new_targets)

        return loss, status

    def forward_pass(self, data):
        n, b, _, img_h, img_w = data['search_images'].shape   # n,b,c,h,w
        search_img = data['search_images'].view(-1, *data['search_images'].shape[2:])  # (n*b, c, h, w)
        search_list = search_img.split(b,dim=0)
        template_img = data['template_images'].view(-1, *data['template_images'].shape[2:])
        template_list = template_img.split(b,dim=0)
        feature_xz = self.net(images_list=template_list+search_list, mode='encoder') # forward the encoder

        # box of search region
        targets = data['search_anno'].permute(1,0,2).reshape(-1, data['search_anno'].shape[2])   # x0y0wh
        targets = box_xywh_to_xyxy(targets)   # x0y0wh --> x0y0x1y1
        targets = torch.max(targets, torch.tensor([0.]).to(targets)) # Truncate out-of-range values
        targets = torch.min(targets, torch.tensor([1.]).to(targets))
        targets = box_xyxy_to_cxcywh(targets) # cxcywh, (b, 4)


        # target sequence
        new_targets, diffused_boxes, noises, ts = self.net(targets=targets, mode = 'prepare_target')
        hw = data['search_images'].size()
        images_whwh = torch.tensor([[hw[-1],hw[-1], hw[-1], hw[-1] ]], device=diffused_boxes.device)
        diffused_boxes = diffused_boxes * images_whwh[:, None, :]

        outputs = self.net(xz=feature_xz, boxes=diffused_boxes, t =ts, mode="decoder")

        return outputs, new_targets

    def compute_losses(self, outputs, new_targets, return_status=True):
        loss_dict, weight_dict = self.net(outputs = outputs, new_targets = new_targets, mode = 'compute_loss')
        new_loss_dict= {}
        for k in loss_dict.keys():
            if k in weight_dict:
                new_loss_dict[k] = loss_dict[k] * weight_dict[k]

        losses = sum(new_loss_dict.values())
        status = loss_dict.copy()
        status['total_loss'] = losses.clone().detach().item()
        if return_status:
            # status for log
            return losses, status
        else:
            return losses

    def to(self, device):
        """ Move the network to device
        args:
            device - device to use. 'cpu' or 'cuda'
        """
        self.net.to(device)
       # self.objective['ce'].to(device)


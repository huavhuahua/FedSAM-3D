import numpy as np
import torch
import torch.nn as nn
import copy
from torch.utils.data import SubsetRandomSampler, DataLoader
import torchio as tio
import torch.nn.functional as F
from utils.click_method import get_next_click3D_torch_2
from monai.losses import DiceCELoss

click_methods = {
    'random': get_next_click3D_torch_2,
}

class ALA:
    def __init__(self,
                args,
                cid: int,
                loss: nn.Module,
                batch_size: int, 
                rand_percent: int, 
                layer_idx: int = 0,
                eta: float = 1.0,
                device: str = 'cpu', 
                threshold: float = 0.1,
                num_pre_loss: int = 8) -> None:
        """
        Initialize ALA module

        Args:
            cid: Client ID. 
            loss: The loss function. 
            train_data: The reference of the local training data.
            batch_size: Weight learning batch size.
            rand_percent: The percent of the local training data to sample.
            layer_idx: Control the weight range. By default, all the layers are selected. Default: 0
            eta: Weight learning rate. Default: 1.0
            device: Using cuda or cpu. Default: 'cpu'
            threshold: Train the weight until the standard deviation of the recorded losses is less than a given threshold. Default: 0.1
            num_pre_loss: The number of the recorded losses to be considered to calculate the standard deviation. Default: 10

        Returns:
            None.
        """
        self.args = args

        self.cid = cid
        self.loss = loss

        self.batch_size = batch_size
        self.rand_percent = rand_percent
        self.layer_idx = layer_idx
        self.eta = eta
        self.threshold = threshold
        self.num_pre_loss = num_pre_loss
        self.device = device

        self.seg_loss = DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')

        self.weight = 0.4
        self.norm_transform = tio.ZNormalization(masking_method=lambda x: x > 0)

    def batch_forward(self, sam_model, image_embedding, gt3D, low_res_masks, points=None):
        
        sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
            points=points,
            boxes=None,
            masks=low_res_masks,
        )
        low_res_masks, iou_predictions = sam_model.mask_decoder(
            image_embeddings=image_embedding.to(self.device), # (B, 256, 64, 64)
            image_pe=sam_model.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
            multimask_output=False,
        )
        prev_masks = F.interpolate(low_res_masks, size=gt3D.shape[-3:], mode='trilinear', align_corners=False)
        return low_res_masks, prev_masks

    def interaction(self, args, sam_model, image_embedding, gt3D, num_clicks):
        return_loss = 0
        prev_masks = torch.zeros_like(gt3D).to(gt3D.device)
        low_res_masks = F.interpolate(prev_masks.float(), size=(args.img_size//4,args.img_size//4,args.img_size//4))
        random_insert = np.random.randint(2, 9)
        for num_click in range(num_clicks):
            points_input, labels_input = self.get_points(prev_masks, gt3D)
            # print(num_click, 'points_input, labels_input', points_input, labels_input)

            if num_click == random_insert or num_click == num_clicks - 1:
                low_res_masks, prev_masks = self.batch_forward(sam_model, image_embedding, gt3D, low_res_masks, points=None)
            else:
                low_res_masks, prev_masks = self.batch_forward(sam_model, image_embedding, gt3D, low_res_masks, points=[points_input, labels_input])
            loss = self.seg_loss(prev_masks, gt3D)
            return_loss += loss
        return prev_masks, return_loss
    
    def get_points(self, prev_masks, gt3D):
        batch_points, batch_labels = click_methods[self.args.click_type](prev_masks, gt3D)

        points_co = torch.cat(batch_points, dim=0).to(self.device)
        points_la = torch.cat(batch_labels, dim=0).to(self.device)

        self.click_points.append(points_co)
        self.click_labels.append(points_la)

        points_multi = torch.cat(self.click_points, dim=1).to(self.device)
        labels_multi = torch.cat(self.click_labels, dim=1).to(self.device)

        if self.args.multi_click:
            points_input = points_multi
            labels_input = labels_multi
        else:
            points_input = points_co
            labels_input = points_la
        return points_input, labels_input
    
    def set_optimizer(self):
        if self.args.multi_gpu:
            sam_model = self.model.module
        else:
            sam_model = self.model


    def set_lr_scheduler(self):
        if self.args.lr_scheduler == "multisteplr":
            self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                                self.args.step_size,
                                                                self.args.gamma)
        elif self.args.lr_scheduler == "steplr":
            self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                                self.args.step_size[0],
                                                                self.args.gamma)
        elif self.args.lr_scheduler == 'coswarm':
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer)
        else:
            self.lr_scheduler = torch.optim.lr_scheduler.LinearLR(self.optimizer, 0.1)

    def adaptive_local_aggregation(self, args,
                            global_model: nn.Module,
                            local_model: nn.Module) -> None:
        """
        Generates the Dataloader for the randomly sampled local training data and 
        preserves the lower layers of the update. 

        Args:
            global_model: The received global/aggregated model. 
            local_model: The trained local model. 

        Returns:
            None.
        """
      
        # obtain the references of the parameters
        params_g = list(global_model.parameters())
        params = list(local_model.parameters())

        non_frozen_params = [param for param in params if param.requires_grad]
        non_frozen_params_g = [param for param in params_g if param.requires_grad]
        
        # deactivate ALA at the 1st communication iteration
        if torch.sum(non_frozen_params[0] - non_frozen_params_g[0]) == 0:
            return
        
        # preserve all the updates in the lower layers
        for non_frozen_param, non_frozen_param_g in zip(non_frozen_params[:-self.layer_idx], non_frozen_params_g[:-self.layer_idx]):
            non_frozen_param.data = non_frozen_param_g.data.clone()

        # temp local model only for weight learning
        model_t = copy.deepcopy(local_model)
        params_t = list(model_t.parameters())
        non_frozen_params_t = [param for param in params_t if param.requires_grad]

        # only consider higher layers
        non_frozen_params_p = non_frozen_params[-self.layer_idx:]
        non_frozen_params_gp = non_frozen_params_g[-self.layer_idx:]
        non_frozen_params_tp = non_frozen_params_t[-self.layer_idx:]

        # frozen the lower layers to reduce computational cost in Pytorch
        for param in non_frozen_params_t[:-self.layer_idx]:
            param.requires_grad = False
        
        # initialize the higher layers in the temp local model
        for param_t, param, param_g in zip(non_frozen_params_tp, non_frozen_params_p, 
                                                   non_frozen_params_gp):
            param_t.data = param + self.weight *(param_g - param)
        

        # obtain initialized local model
        for param, param_t in zip(non_frozen_params_p, non_frozen_params_tp):
            param.data = param_t.data.clone()

        
def create_subset_dataloader(original_dataloader, subset_size):
    dataset_size = len(original_dataloader.dataset)
    subset_indices = torch.randperm(dataset_size)[:subset_size]
    subset_sampler = SubsetRandomSampler(subset_indices)
    subset_dataloader = DataLoader(original_dataloader.dataset, 
                                    batch_size=original_dataloader.batch_size, 
                                    sampler=subset_sampler, 
                                    num_workers=original_dataloader.num_workers, 
                                    drop_last=original_dataloader.drop_last)

    return subset_dataloader

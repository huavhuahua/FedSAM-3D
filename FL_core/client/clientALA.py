import os
import copy
import torch
import torch.nn as nn
from monai.losses import DiceCELoss
import matplotlib.pyplot as plt
import numpy as np
import logging
from tqdm import tqdm
from typing import Optional
import torch.nn.functional as F
import torchio as tio
from torch.utils.data import DataLoader
from torch.cuda import amp
import torch.distributed as dist
from sklearn.preprocessing import label_binarize
from sklearn import metrics
from utils.data_paths import img_datas, dataset_weight
from FL_core.ALA import ALA
from utils.data_loader import get_dataloaders
from utils.click_method import get_next_click3D_torch_2
join = os.path.join

click_methods = {
    'random': get_next_click3D_torch_2,
}


class clientALA(object):
    def __init__(self, args, cid):
        self.args = args
        self.model = copy.deepcopy(args.model)     
        self.dataset = img_datas                   
        self.device = args.device                  
        self.cid = cid
        self.loss = nn.CrossEntropyLoss()
        self.batch_size = args.batch_size
        self.rand_percent = args.rand_percent
        self.layer_idx = args.layer_idx
        self.eta = args.eta
        self.device = args.device

        self.set_loss_fn()
        self.set_optimizer()
        self.set_lr_scheduler()
        self.step_best_dice = 0.0
        self.step_best_loss = np.inf
        self.best_loss = np.inf
        self.best_dice = 0.0
        self.losses = []
        self.dices = []
        self.ious = []

        self.c_weight = dataset_weight[cid]

        self.norm_transform = tio.ZNormalization(masking_method=lambda x: x > 0)

        self.ALA = ALA(args, self.cid, self.loss, self.batch_size, 
                    self.rand_percent, self.layer_idx, self.eta, self.device)
        
    def train(self, round):
        epoch_loss = 0
        epoch_dice = 0
        self.scaler = amp.GradScaler()
        dataloader = get_dataloaders(self.args, self.cid)
        self.model.train()
        tbar = tqdm(dataloader)

        self.optimizer.zero_grad()
        step_loss = 0

        total_dice = 0
        total_batches = 0

        for step, (image3D, gt3D) in enumerate(tbar):
            image3D = self.norm_transform(image3D.squeeze(dim=1)) # (N, C, W, H, D)
            image3D = image3D.unsqueeze(dim=1)

            image3D = image3D.to(self.device)
            gt3D = gt3D.to(self.device).type(torch.long)

            with amp.autocast():
                image_embedding = self.model.image_encoder(image3D)

                self.click_points = []
                self.click_labels = []

                pred_list = []
                prev_masks, loss = self.interaction(self.model, image_embedding, gt3D, num_clicks=11)   
                prev_masks = torch.sigmoid(prev_masks)
            
            epoch_loss += loss.item()
            cur_loss = loss.item()
            loss /= self.args.accumulation_steps
            self.scaler.scale(loss).backward()

            if step % self.args.accumulation_steps == 0 and step != 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

                print_loss = step_loss / self.args.accumulation_steps
                step_loss = 0
                # print(prev_masks.max(), gt3D.max(), prev_masks.min(), gt3D.min())
                print_dice = self.get_dice_score(prev_masks, gt3D)
                total_dice += print_dice
                total_batches += 1
            else:
                step_loss += cur_loss

            if not self.args.multi_gpu or (self.args.multi_gpu and self.args.rank == 0):
                if step % self.args.accumulation_steps == 0 and step != 0:
                    print(f'Client:{self.cid}, Epoch: {round}, Step: {step}, Loss: {print_loss}, Dice: {print_dice}')
                    if print_dice > self.step_best_dice:
                        self.step_best_dice = print_dice
                        if print_dice > 0.9:
                            self.save_checkpoint(
                                round,
                                self.model.state_dict(),
                                describe=f'client{self.cid}_{round}_step_dice:{print_dice}_best'
                            )
                    if print_loss < self.step_best_loss:
                        self.step_best_loss = print_loss
        epoch_dice = total_dice / total_batches if total_batches != 0 else 0 
        epoch_loss /= step

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        if self.args.multi_gpu:
            dist.barrier()
        
        if not self.args.multi_gpu or (self.args.multi_gpu and self.args.rank == 0):
            self.losses.append(epoch_loss)
            self.dices.append(epoch_dice)
            print(f'EPOCH: {round}, Loss: {epoch_loss}')
            print(f'EPOCH: {round}, Dice: {epoch_dice}')

            if self.args.multi_gpu:
                state_dict = self.model.module.state_dict()
            else:
                state_dict = self.model.state_dict()
                
            # save latest checkpoint
            self.save_checkpoint(
                round, 
                state_dict,                     
                describe=f'client{self.cid}_latest'
                )

            # save train loss best checkpoint
            if epoch_loss < self.best_loss: 
                self.best_loss = epoch_loss
                self.save_checkpoint(                        
                    round,
                    state_dict,
                    describe=f'client{self.cid}_loss_best'
                )
                
            # save train dice best checkpoint
            if epoch_dice > self.best_dice: 
                self.best_dice = epoch_dice
                self.save_checkpoint(
                    round,
                    state_dict,
                    describe=f'client{self.cid}_dice_best'
                )

            self.plot_result(self.losses, 'Dice + Cross Entropy Loss', f'client{self.cid}_Loss')
            self.plot_result(self.dices, 'Dice', f'client{self.cid}_Dice')
         

    def plot_result(self, plot_data, description, save_name):
        plt.plot(plot_data)
        plt.title(description)
        plt.xlabel('Epoch')
        plt.ylabel(f'{save_name}')
        plt.savefig(join(self.args.work_dir, self.args.task_name, f'{save_name}.png'))
        plt.close()      


    def set_loss_fn(self):
        self.seg_loss = DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')

    def set_optimizer(self):
        if self.args.multi_gpu:
            sam_model = self.model.module
        else:
            sam_model = self.model

        self.optimizer = torch.optim.AdamW([
            {'params': sam_model.image_encoder.parameters()}, # , 'lr': self.args.lr * 0.1},
            {'params': sam_model.prompt_encoder.parameters() , 'lr': self.args.lr * 0.1},
            {'params': sam_model.mask_decoder.parameters(), 'lr': self.args.lr * 0.1},
        ], lr=self.args.lr, betas=(0.9,0.999), weight_decay=self.args.weight_decay)

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

    def get_points(self, prev_masks, gt3D):
        batch_points, batch_labels = click_methods[self.args.click_type](prev_masks, gt3D)

        points_co = torch.cat(batch_points, dim=0).to(self.args.device)
        points_la = torch.cat(batch_labels, dim=0).to(self.args.device)

        self.click_points.append(points_co)
        self.click_labels.append(points_la)

        points_multi = torch.cat(self.click_points, dim=1).to(self.args.device)
        labels_multi = torch.cat(self.click_labels, dim=1).to(self.args.device)

        if self.args.multi_click:
            points_input = points_multi
            labels_input = labels_multi
        else:
            points_input = points_co
            labels_input = points_la
        return points_input, labels_input
    
    def save_checkpoint(self, epoch, state_dict, describe="last"):
        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": state_dict,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "lr_scheduler_state_dict": self.lr_scheduler.state_dict(),
            "losses": self.losses,
            "dices": self.dices,
            "best_loss": self.best_loss,
            "best_dice": self.best_dice,
            "args": self.args,
            "used_datas": img_datas,
        }, join(self.args.work_dir, self.args.task_name, f"sam_model_{describe}.pth"))

    def batch_forward(self, sam_model, image_embedding, gt3D, low_res_masks, points=None):
        
        sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
            points=points,
            boxes=None,
            masks=low_res_masks,
        )
        low_res_masks, iou_predictions = sam_model.mask_decoder(
            image_embeddings=image_embedding.to(self.device), 
            image_pe=sam_model.prompt_encoder.get_dense_pe(), 
            sparse_prompt_embeddings=sparse_embeddings, 
            dense_prompt_embeddings=dense_embeddings, 
            multimask_output=False,
        )
        prev_masks = F.interpolate(low_res_masks, size=gt3D.shape[-3:], mode='trilinear', align_corners=False)
        return low_res_masks, prev_masks


    def interaction(self, sam_model, image_embedding, gt3D, num_clicks):
        return_loss = 0
        prev_masks = torch.zeros_like(gt3D).to(gt3D.device)
        low_res_masks = F.interpolate(prev_masks.float(), size=(self.args.img_size//4,self.args.img_size//4,self.args.img_size//4))
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
    
    def get_dice_score(self, prev_masks, gt3D):
        def compute_dice(mask_pred, mask_gt):
            mask_threshold = 0.5

            mask_pred = (mask_pred > mask_threshold)
            mask_gt = (mask_gt > 0)
            
            volume_sum = mask_gt.sum() + mask_pred.sum()
            if volume_sum == 0:
                return np.NaN
            volume_intersect = (mask_gt & mask_pred).sum()
            return 2*volume_intersect / volume_sum
    
        pred_masks = (prev_masks > 0.5)
        true_masks = (gt3D > 0)
        dice_list = []
        for i in range(true_masks.shape[0]):
            dice_list.append(compute_dice(pred_masks[i], true_masks[i]))
        return (sum(dice_list)/len(dice_list)).item() 


    def local_initialization(self, args, received_global_model):     
        self.ALA.adaptive_local_aggregation(args, received_global_model, self.model)


import argparse
import datetime
from FL_core.server.serverALA import FedALA
import logging
import numpy as np
import os
join = os.path.join
import random
from segment_anything.build_sam3D import sam_model_registry3D
from segment_anything.set_network import freeze_control
import time
import torch
from torch.backends import cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

# region add_argument
# endregion

def main(args):
    time_list = []
    mp.set_sharing_strategy('file_system') 
    device_config(args)
    start = time.time()

    if args.multi_gpu:
        mp.spawn(
            main_worker,   
            nprocs=args.world_size,  
            args=(args, )   
        )
    else:
        random.seed(2023)
        np.random.seed(2023)
        torch.manual_seed(2023)

        # region prepare model
        args.model = build_model_adapter(args)

        # endregion

        server = FedALA(args)
        server.train(args)

        time_list.append(time.time()-start)
    print(f"\nAverage time cost: {round(np.average(time_list), 2)}s.")

    print("All done!")


# region build_model
def build_model_adapter(args):
    sam_model = sam_model_registry3D[args.model_type](checkpoint=None).to(args.device)
    sam_model = freeze_control(args, sam_model)
    if args.multi_gpu:
        sam_model = DDP(sam_model, device_ids=[args.rank], output_device=args.rank)
    return sam_model
# endregion


# region main_worker()
def main_worker(rank, args):
    setup(rank, args.world_size)

    torch.cuda.set_device(rank)
    args.num_workers = int(args.num_workers / args.ngpus_per_node)
    args.device = torch.device(f"cuda:{rank}")
    args.rank = rank

    init_seeds(2023 + rank)

    cur_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.INFO if rank in [-1, 0] else logging.WARN,
        filemode='w',
        filename=os.path.join(LOG_OUT_DIR, f'output_{cur_time}.log'))
    

    # Training
    # dataloaders = get_dataloaders(args)
    # model = build_model_adapter(args)
    # trainer = BaseTrainer(model, dataloaders, args)
    # trainer.train()

    cleanup()

def setup(rank, world_size):
    # initialize the process group
    dist.init_process_group(
        backend='nccl',
        init_method=f'tcp://127.0.0.1:{args.port}',
        world_size=world_size,
        rank=rank
    )

def init_seeds(seed=0, cuda_deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    if cuda_deterministic:  # slower, more reproducible
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:  # faster, less reproducible
        cudnn.deterministic = False
        cudnn.benchmark = True

def cleanup():
    dist.destroy_process_group()
# endregion
    
        
def device_config(args):
    try:
        if not args.multi_gpu:
            # Single GPU
            if args.device == 'mps':
                args.device = torch.device('mps')
            else:
                args.device = torch.device(f"cuda:{args.gpu_ids[0]}")
        else:
            # Multi GPU
            args.nodes = 1
            args.ngpus_per_node = len(args.gpu_ids)
            args.world_size = args.nodes * args.ngpus_per_node

    except RuntimeError as e:
        print(e)


if __name__ == "__main__":
    total_start = time.time()

    parser = argparse.ArgumentParser()
    # region add_argument
    parser.add_argument('--task_name', type=str, default='task_name')
    parser.add_argument('--work_dir', type=str, default='./work_dir/fl_train')
    parser.add_argument('--num_clients', type=int, default=3, help="Total number of clients")
    parser.add_argument('--join_ratio', type=float, default=1.0, help="Ratio of clients per round")
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=8e-4)
    parser.add_argument('--weight_decay', type=float, default=0.1)
    parser.add_argument('--lr_scheduler', type=str, default='multisteplr')
    parser.add_argument('--step_size', type=list, default=[120, 180])
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--rand_percent', type=int, default=100)
    parser.add_argument('--layer_idx', type=int, default=2)
    parser.add_argument('--eta', type=float, default=1.0)
    parser.add_argument('--img_size', type=int, default=128)

    parser.add_argument('--num_workers', type=int, default=24)
    parser.add_argument('--accumulation_steps', type=int, default=10)

    parser.add_argument('--model_type', type=str, default='vit_b_ori')
    parser.add_argument('--sam_ckpt', type=str, default='./sam_ckpt/sam_med3d_turbo.pth')
    parser.add_argument('--global_rounds', type=int, default=200, help='global epoches')
    parser.add_argument('--random_join_ratio', type=bool, default=False, help="Random ratio of clients per round")
    parser.add_argument('--device', type=str, default="cuda")

    parser.add_argument('--gpu_ids', type=int, nargs='+', default=[3])

    parser.add_argument('--multi_gpu', action='store_true', default=False)
    parser.add_argument('-prev', type=int, default=0, help="Previous Running times")
    parser.add_argument('--click_type', type=str, default='random')
    parser.add_argument('--multi_click', action='store_true', default=False)
    # endregion
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("\ncuda is not avaiable.\n")
        args.device = "cpu"
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(i) for i in args.gpu_ids])
    logger = logging.getLogger(__name__)
    LOG_OUT_DIR = join(args.work_dir, args.task_name)
    MODEL_SAVE_PATH = join(args.work_dir, args.task_name)
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

    main(args)

    print('Time: ', time.time()-total_start, 's')


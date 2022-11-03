
import numpy as np

import torch
from torch.utils.data import DataLoader
from easydict import EasyDict
import os
from torch.utils.tensorboard import SummaryWriter
import coloredlogs
from tqdm import tqdm
import sys
from fire import Fire
import logging
import visualDet3D.data.kitti.dataset
from visualDet3D.utils.timer import Timer
from visualDet3D.networks.utils.registry import DETECTOR_DICT, DATASET_DICT, PIPELINE_DICT
from visualDet3D.utils.utils import LossLogger, jfc_from_file
from visualDet3D.networks.optimizers import optimizers, schedulers
from visualDet3D.networks.utils.utils import BackProjection, BBox3dProjector, get_num_parameters
from visualDet3D.evaluator.kitti.evaluate import evaluate
from _path_init import *
import pprint


def main(config="config/config.py", experiment_name="default", world_size=1, local_rank=-1):
   

   
    jfc = jfc_from_file(config)

    
    jfc.dist = EasyDict()
    jfc.dist.world_size = world_size
    jfc.dist.local_rank = local_rank
    dist = local_rank >= 0 
    logng     = local_rank <= 0
    evaluate  = local_rank <= 0

   
    recorder_dir = os.path.join(jfc.path.log_path, experiment_name + f"config={config}")
    
    if logng: 
        if os.path.isdir(recorder_dir):
            os.system("rm -r {}".format(recorder_dir))
        writer = SummaryWriter(recorder_dir)

        
       
        formatted_jfc = pprint.pformat(jfc)
        writer.add_text("config.py", formatted_jfc.replace(' ', '&nbsp;').replace('\n', '  \n')) # add space for markdown style in tensorboard text
    else:
        writer = None

    
    if dist:
        jfc.trainer.gpu = local_rank 
    gpu = min(jfc.trainer.gpu, torch.cuda.device_count() - 1)
    torch.backends.cudnn.benchmark = getattr(jfc.trainer, 'cudnn', False)
    torch.cuda.set_device(gpu)
    if dist:
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
 
    
    training_dataset = DATASET_DICT[jfc.data.training_dataset](jfc)
    validation_dataset = DATASET_DICT[jfc.data.val_dataset](jfc, "validation")

    train_dataloader = DataLoader(training_dataset, num_workers=jfc.data.num_workers,
                                  batch_size=jfc.data.batch_size, collate_fn=training_dataset.collate_fn, shuffle=local_rank<0, drop_last=True,
                                  sampler=torch.utils.data.DistributedSampler(training_dataset, num_replicas=world_size, rank=local_rank, shuffle=True) if local_rank >= 0 else None)
    val_dataloader = DataLoader(validation_dataset, num_workers=jfc.data.num_workers,
                                batch_size=jfc.data.batch_size, collate_fn=validation_dataset.collate_fn, shuffle=False, drop_last=True)

    
    detector = DETECTOR_DICT[jfc.detector.name](jfc.detector)

   
    old_checkpoint = getattr(jfc.path, 'pretrained_checkpoint', None)
    if old_checkpoint is not None:
        state_dict = torch.load(old_checkpoint, map_location='cpu')
        detector.load_state_dict(state_dict)

   
    if dist:
        detector = torch.nn.SyncBatchNorm.convert_sync_batchnorm(detector)
        detector = torch.nn.parallel.DistributedDataParallel(detector.cuda(), device_ids=[gpu], output_device=gpu)
    else:
        detector = detector.cuda()
    detector.train()

    
    if logng:
        string1 = detector.__str__().replace(' ', '&nbsp;').replace('\n', '  \n')
        writer.add_text("Model architecture", string1)
        num_parameters = get_num_parameters(detector)
        print(f'No. of parameters used in trainig: {num_parameters}')
    
    
    optim = optimizers.build_optimizer(jfc.optimizer, detector)

   
    scheduler_cfg = getattr(jfc, 'scheduler', None)
    schdlr = schedulers.build_scheduler(scheduler_cfg, optim)
    iter_based = getattr(scheduler_cfg, "iter_based", False)

    
    training_loss_logger =  LossLogger(writer, 'train') if logng else None

    
    if 'train_function' in jfc.trainer:
        training_dection = PIPELINE_DICT[jfc.trainer.train_function]
   
    if 'eval_function' in jfc.trainer:
        evaluate_detection = PIPELINE_DICT[jfc.trainer.eval_function]
    else:
        evaluate_detection = None
  
    timer = Timer()

    print('Num training images: {}'.format(len(training_dataset)))

    step_global = 0

    for num_epochs in range(jfc.trainer.max_epochs):
       
        detector.train()
        if training_loss_logger:
            training_loss_logger.reset()
        for iter_num, data in enumerate(train_dataloader):
            training_dection(data, detector, optim, writer, training_loss_logger, step_global, num_epochs, jfc)

            step_global += 1

            if iter_based:
                schdlr.step()

            if logng and step_global % jfc.trainer.dp_iteration == 0:
                
                if 'total_loss' not in training_loss_logger.loss_stats:
                    print(f"\nIn epoch {num_epochs}, iteration:{iter_num}, step_global:{step_global}, total_loss not found in logger.")
                else:
                    log_str = 'Epoch: {} --- Iteration: {}  --- Running loss: {:1.5f} --- eta:{}'.format(
                        num_epochs, iter_num, training_loss_logger.loss_stats['total_loss'].avg,
                        timer.compute_eta(step_global, len(train_dataloader) * jfc.trainer.max_epochs))
                    print(log_str, end='\r')
                    writer.add_text("training_log/train", log_str, step_global)
                    training_loss_logger.log(step_global)

        if not iter_based:
            schdlr.step()

       
        if logng:
            torch.save(detector.module.state_dict() if dist else detector.state_dict(), os.path.join(
                jfc.path.checkpoint_path, '{}_latest.pth'.format(
                    jfc.detector.name)
                )
            )
        if logng and (num_epochs + 1) % jfc.trainer.sv_iteration == 0:
            torch.save(detector.module.state_dict() if dist else detector.state_dict(), os.path.join(
                jfc.path.checkpoint_path, '{}_{}.pth'.format(
                    jfc.detector.name,num_epochs)
                )
            )

        
        if evaluate and evaluate_detection is not None and jfc.trainer.tt_iteration > 0 and (num_epochs + 1) % jfc.trainer.tt_iteration == 0:
            print("\n/Testing at epoch {} ".format(num_epochs))
            evaluate_detection(jfc, detector.module if dist else detector, validation_dataset, writer, num_epochs)


        if dist:
            torch.distributed.barrier() 

        if logng:
            writer.flush()

if __name__ == '__main__':
    Fire(main)

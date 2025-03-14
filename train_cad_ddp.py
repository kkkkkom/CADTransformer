import os
import argparse
import torch
from tqdm import tqdm
from dataset import CADDataLoader, DataLoaderX
from utils.utils_model import create_logger
from config import config, update_config
from models.model import CADTransformer
from utils.utils_model import OffsetLoss
from eval import do_eval, get_eval_criteria
from torch.nn.utils.rnn import pad_sequence

torch.backends.cudnn.benchmark = True
torch.autograd.set_detect_anomaly(True)

num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
distributed = num_gpus > 1

from pathlib import Path
import os
import shutil
import json
import subprocess


def get_kaggle_username():
    with open("/kaggle/input/api-token/kaggle.json") as f:
        kaggle_json = json.load(f)
    username = kaggle_json["username"]
    return username


def create_kaggle_metadata(title):
    # if Path("/kaggle").exists():
    os.makedirs("/root/.kaggle/", exist_ok=True)
    shutil.copy("/kaggle/input/api-token/kaggle.json", "/root/.kaggle/kaggle.json")
    os.chmod("/root/.kaggle/kaggle.json", 600)
    username = get_kaggle_username()
    dataset_metadata = {
        "title": title,
        "id": f"{username}/{title}",
        "licenses": [{"name": "unknown"}],
    }
    with open("/kaggle/working/dataset-metadata.json", "w") as f:
        json.dump(dataset_metadata, f)
    return dataset_metadata


def dataset_exists(dataset_id):
    username = get_kaggle_username()
    result = subprocess.run(
        ["kaggle", "datasets", "list", "--user", username, "--search", dataset_id],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    output = result.stdout.decode("utf-8")
    return dataset_id in output


def save_kaggle_dataset(dataset_metadata, source_dir):
    zip_path = f"{Path(source_dir).name}.zip"
    subprocess.run(f"rm {zip_path}", shell=True)
    res = subprocess.run(f"zip -r {zip_path} {source_dir}", shell=True, stderr=subprocess.STDOUT)
    print(f"[DEBUG] zip res: {res.returncode}")

    res = subprocess.run(f"mkdir -p tmp_zip_dir", shell=True, stderr=subprocess.STDOUT)
    print(f"[DEBUG] mkdir res: {res.returncode}")

    subprocess.run(f"rm tmp_zip_dir/*", shell=True)
    res = subprocess.run(f"mv {zip_path} tmp_zip_dir/", shell=True, stderr=subprocess.STDOUT)
    print(f"[DEBUG] mv res: {res.returncode}")

    shutil.move("/kaggle/working/dataset-metadata.json", "tmp_zip_dir/dataset-metadata.json")
    # if Path("/kaggle").exists():
    if not dataset_exists(dataset_metadata["id"]):
        print(f"[DEBUG] Creating new dataset ..")
        kaggle_cmd = f"kaggle datasets create -p tmp_zip_dir --dir-mode zip"
    else:
        print(f"[DEBUG] Updating dataset ..")
        kaggle_cmd = f'kaggle datasets version -p tmp_zip_dir --dir-mode zip -m "Updated model with new training" --delete-old-versions'
    result = subprocess.run(kaggle_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output = result.stdout.decode()
    error = result.stderr.decode()
    print(output)
    print(error)
    print(f"[DEBUG] Dataset saved.")

print("Import Done.")


def custom_collate(batch):
    # Unpack all 9 elements separately
    tensors_0, tensors_1, tensors_2, tensors_3, tensors_4, tensors_5, tensors_6, tensors_7, str_8 = zip(*batch)

    # Example: If some tensors need padding, apply pad_sequence
    tensors_0 = pad_sequence(tensors_0, batch_first=True, padding_value=0) if isinstance(tensors_0[0],
                                                                                         torch.Tensor) else torch.stack(
        tensors_0)
    tensors_1 = pad_sequence(tensors_1, batch_first=True, padding_value=0) if isinstance(tensors_1[0],
                                                                                         torch.Tensor) else torch.stack(
        tensors_1)
    tensors_2 = pad_sequence(tensors_2, batch_first=True, padding_value=0) if isinstance(tensors_2[0],
                                                                                         torch.Tensor) else torch.stack(
        tensors_2)
    tensors_3 = pad_sequence(tensors_3, batch_first=True, padding_value=0) if isinstance(tensors_3[0],
                                                                                         torch.Tensor) else torch.stack(
        tensors_3)
    tensors_4 = pad_sequence(tensors_4, batch_first=True, padding_value=0) if isinstance(tensors_4[0],
                                                                                         torch.Tensor) else torch.stack(
        tensors_4)
    tensors_5 = pad_sequence(tensors_5, batch_first=True, padding_value=0) if isinstance(tensors_5[0],
                                                                                         torch.Tensor) else torch.stack(
        tensors_5)
    tensors_6 = pad_sequence(tensors_6, batch_first=True, padding_value=0) if isinstance(tensors_6[0],
                                                                                         torch.Tensor) else torch.stack(
        tensors_6)
    tensors_7 = pad_sequence(tensors_7, batch_first=True, padding_value=0) if isinstance(tensors_7[0],
                                                                                         torch.Tensor) else torch.stack(
        tensors_7)

    return tensors_0, tensors_1, tensors_2, tensors_3, tensors_4, tensors_5, tensors_6, tensors_7, str_8


# def custom_collate(batch):
#     tensors, labels = zip(*batch)  # Assuming your dataset returns (tensor, label)
#     tensors_padded = pad_sequence(tensors, batch_first=True, padding_value=0)
#     return tensors_padded, torch.tensor(labels)

def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation network')
    parser.add_argument('--cfg',
                        type=str,
                        default="config/hrnet48.yaml",
                        help='experiment configure file name'
                        )
    parser.add_argument('--val_only',
                        action="store_true",
                        help='flag to do evaluation on val set')
    parser.add_argument('--test_only',
                        action="store_true",
                        help='flag to do evaluation on test set')
    parser.add_argument('--data_root', type=str,
                        default="/ssd1/zhiwen/projects/CADTransformer/data/floorplan_v1")
    parser.add_argument('--embed_backbone', type=str,
                        default="hrnet48")
    parser.add_argument('--pretrained_model', type=str,
                        default="./pretrained_models/HRNet_W48_C_ssld_pretrained.pth")
    parser.add_argument("--local-rank", type=int, default=0)
    parser.add_argument("--log_step", type=int,
                        default=100,
                        help='steps for logging')
    parser.add_argument("--img_size", type=int,
                        default=700,
                        help='image size of rasterized image')
    parser.add_argument("--max_prim", type=int,
                        default=12000,
                        help='maximum primitive number for each batch')
    parser.add_argument("--load_ckpt", type=str,
                        default='',
                        help='load checkpoint')
    parser.add_argument("--resume_ckpt", type=str,
                        default='',
                        help='continue train while loading checkpoint')
    parser.add_argument("--log_dir", type=str,
                        default='',
                        help='logging directory')
    parser.add_argument('--seed', type=int, default=304)
    parser.add_argument('--debug', action="store_true")
    parser.add_argument('--visualize', action="store_true")
    parser.add_argument('--epoch', type=int, default=1)
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)
    args = parser.parse_args()
    # args.local_rank = os.environ['LOCAL_RANK']
    return args


def main():
    args = parse_args()
    print(f"[DEBUG] args={args}")
    cfg = update_config(config, args)
    # print(f"[DEBUG] cfg={cfg}")

    os.makedirs(cfg.log_dir, exist_ok=True)
    if cfg.eval_only:
        logger = create_logger(cfg.log_dir, 'val')
    elif cfg.test_only:
        logger = create_logger(cfg.log_dir, 'test')
    else:
        logger = create_logger(cfg.log_dir, 'train')

    # Distributed Train Config
    if args.device=="cuda":
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://",
        )
        device = torch.device('cuda:{}'.format(args.local_rank))
    else:
        torch.distributed.init_process_group(
            backend="gloo", init_method="env://"
        )
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


    # Create Model
    model = CADTransformer(cfg)
    CE_loss = torch.nn.CrossEntropyLoss().cuda()

    # Create Optimizer
    if cfg.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=cfg.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=cfg.weight_decay
        )
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg.learning_rate, momentum=0.9)

    if args.device=="cuda":
        model = torch.nn.parallel.DistributedDataParallel(
            module=model.to(device), broadcast_buffers=False,
            device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
    else:
        model = torch.nn.parallel.DistributedDataParallel(
            module=model.to(device), broadcast_buffers=False,
            find_unused_parameters=True)
    model.train()

    # Load/Resume ckpt
    # start_epoch = 0
    start_epoch = -1
    if cfg.load_ckpt != '':
        if os.path.exists(cfg.load_ckpt):
            checkpoint = torch.load(cfg.load_ckpt, map_location=torch.device("cpu"))
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info("=> loaded checkpoint '{}' (epoch {})".format(
                cfg.load_ckpt, checkpoint['epoch']))
        else:
            logger.info("=>Failed: no checkpoint found at '{}'".format(cfg.load_ckpt))
            exit(0)

    if cfg.resume_ckpt != '':
        if os.path.exists(cfg.resume_ckpt):
            checkpoint = torch.load(cfg.resume_ckpt, map_location=torch.device("cpu"))
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['model_state_dict'])
            epoch = checkpoint['epoch']
            logger.info(f'=> resume checkpoint: {cfg.resume_ckpt} (epoch: {epoch})')
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(device)
        else:
            logger.info("=>Failed: no checkpoint found at '{}'".format(cfg.resume_ckpt))
            exit(0)
    # Set up Dataloader
    torch.multiprocessing.set_start_method('spawn', force=True)
    val_dataset = CADDataLoader(split='val', do_norm=cfg.do_norm, cfg=cfg)
    val_dataloader = DataLoaderX(args.local_rank, dataset=val_dataset,
                                 batch_size=cfg.test_batch_size, shuffle=False,
                                 num_workers=cfg.WORKERS, drop_last=False)
    # Eval Only
    if args.local_rank == 0:
        if cfg.eval_only:
            eval_F1 = do_eval(model, val_dataloader, logger, cfg)
            exit(0)

    test_dataset = CADDataLoader(split='test', do_norm=cfg.do_norm, cfg=cfg)
    test_dataloader = DataLoaderX(args.local_rank, dataset=test_dataset,
                                  batch_size=cfg.test_batch_size, shuffle=False,
                                  num_workers=cfg.WORKERS, drop_last=False)
    # Test Only
    if args.local_rank == 0:
        if cfg.test_only:
            eval_F1 = do_eval(model, test_dataloader, logger, cfg)
            exit(0)

    train_dataset = CADDataLoader(split='train', do_norm=cfg.do_norm, cfg=cfg)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
    train_dataloader = DataLoaderX(args.local_rank, dataset=train_dataset,
                                   sampler=train_sampler, batch_size=cfg.batch_size,
                                   num_workers=cfg.WORKERS, drop_last=True)

    def bn_momentum_adjust(m, momentum):
        if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
            m.momentum = momentum

    best_F1, eval_F1 = 0, 0
    best_epoch = 0
    global_epoch = 0

    print("> start epoch", start_epoch)
    logger.info(f"[DEBUG] cfg.epoch={cfg.epoch}")
    for epoch in range(start_epoch+1, cfg.epoch):
        logger.info(f"=> {cfg.log_dir}")

        logger.info("\n\n")
        logger.info(f'Epoch 1-based: {global_epoch + 1} ({epoch + 1}/{cfg.epoch})')
        lr = max(cfg.learning_rate * (cfg.lr_decay ** (epoch // cfg.step_size)),
                 cfg.LEARNING_RATE_CLIP)
        if epoch <= cfg.epoch_warmup:
            lr = cfg.learning_rate_warmup

        logger.info(f'Learning rate: {lr}')
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        momentum = cfg.MOMENTUM_ORIGINAL * (cfg.MOMENTUM_DECCAY ** (epoch // cfg.step_size))
        if momentum < 0.01:
            momentum = 0.01
        logger.info(f'BN momentum updated to: {momentum}')
        model = model.apply(lambda x: bn_momentum_adjust(x, momentum))
        model = model.train()

        # training loops
        with tqdm(train_dataloader, total=len(train_dataloader), smoothing=0.9) as _tqdm:
            for i, (image, xy, target, rgb_info, nns, offset_gt, inst_gt, index, basename) in enumerate(_tqdm):
                optimizer.zero_grad()

                seg_pred = model(image, xy, rgb_info, nns)
                seg_pred = seg_pred.contiguous().view(-1, cfg.num_class + 1)
                target = target.view(-1, 1)[:, 0]

                loss_seg = CE_loss(seg_pred, target)
                loss = loss_seg
                loss.backward()
                optimizer.step()
                _tqdm.set_postfix(loss=loss.item(), l_seg=loss_seg.item())

                if i % args.log_step == 0 and args.local_rank == 0:
                    logger.info(f'Train loss: {round(loss.item(), 5)}, loss seg: {round(loss_seg.item(), 5)})')

        # Save last
        if args.local_rank == 0:
            logger.info('Save last model...')
            savepath = os.path.join(cfg.log_dir, 'last_model.pth')
            state = {
                'epoch': epoch,
                'best_F1': best_F1,
                'best_epoch': best_epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(state, savepath)

            tmp_checkpoint = torch.load(savepath, map_location=torch.device("cpu"))
            logger.info(f"tmp checkpoint: {tmp_checkpoint['epoch']}")
            del tmp_checkpoint
            debug_suffix = ""
            if args.debug:
                debug_suffix = "-debug"
            metadata = create_kaggle_metadata(f"{get_kaggle_username()}-cad-model-no-nns-rename{debug_suffix}")
            save_kaggle_dataset(metadata, "/kaggle/working/CADTransformer/logs")


        # assert validation?
        eval = get_eval_criteria(epoch)

        if args.local_rank == 0:
            if eval:
                logger.info('> do validation')
                eval_F1 = do_eval(model, val_dataloader, logger, cfg)
        # Save ckpt
        if args.local_rank == 0:
            if eval_F1 > best_F1:
                best_F1 = eval_F1
                best_epoch = epoch
                logger.info(f'Save model... Best F1:{best_F1}, Best Epoch:{best_epoch}')
                savepath = os.path.join(cfg.log_dir, 'best_model.pth')
                state = {
                    'epoch': epoch,
                    'best_F1': best_F1,
                    'best_epoch': best_epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)

        global_epoch += 1


if __name__ == '__main__':
    main()

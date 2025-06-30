# -*- coding:utf-8 -*-
"""
@file name  : train_script.py
@author     : TingsongYu https://github.com/TingsongYu
@date       : 2023-02-04
@brief      : 肺炎Xray图像分类训练脚本
功能：
1. 加载和预处理肺炎X光图像数据集
2. 训练和验证深度学习分类模型
3. 支持多种模型架构和数据增强策略
4. 记录训练过程并保存最佳模型
"""
import os
import time
import datetime
import torchvision
import torch
import torch.nn as nn
import albumentations as A
import matplotlib
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from albumentations.pytorch import ToTensorV2
matplotlib.use('Agg')

# import utils.my_utils as utils
from datasets.pneumonia_dataset import PneumoniaDataset


def get_args_parser(add_help=True):
    """定义训练脚本的所有可配置参数
    
    Args:
        add_help (bool): 是否添加帮助信息
        
    Returns:
        argparse.ArgumentParser: 参数解析器对象
    """
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch Classification Training", add_help=add_help)

    parser.add_argument("--data-path", default=r"E:\ChestXRay2017\chest_xray", type=str, help="dataset path")
    parser.add_argument("--model", default="resnet50", type=str, help="model name; resnet50 or convnext or convnext-tiny")
    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument(
        "-b", "--batch-size", default=8, type=int, help="images per gpu, the total batch size is $NGPU x batch_size"
    )
    parser.add_argument("--epochs", default=50, type=int, metavar="N", help="number of total epochs to run")
    parser.add_argument(
        "-j", "--workers", default=4, type=int, metavar="N", help="number of data loading workers (default: 4)"
    )
    parser.add_argument("--opt", default="sgd", type=str, help="optimizer")
    parser.add_argument("--random-seed", default=42, type=int, help="random seed")
    parser.add_argument("--lr", default=0.01, type=float, help="initial learning rate")
    parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
    parser.add_argument(
        "--wd",
        "--weight-decay",
        default=1e-4,
        type=float,
        metavar="W",
        help="weight decay (default: 1e-4)",
        dest="weight_decay",
    )
    parser.add_argument("--lr-step-size", default=20, type=int, help="decrease lr every step-size epochs")
    parser.add_argument("--lr-gamma", default=0.1, type=float, help="decrease lr by a factor of lr-gamma")
    parser.add_argument("--print-freq", default=20, type=int, help="print frequency")
    parser.add_argument("--output-dir", default="./Result", type=str, help="path to save outputs")
    parser.add_argument("--resume", default="", type=str, help="path of checkpoint")
    parser.add_argument("--start-epoch", default=0, type=int, metavar="N", help="start epoch")
    parser.add_argument('--autoaug', action='store_true', default=False, help='use torchvision autoaugment')
    parser.add_argument('--useplateau', action='store_true', default=False, help='use torchvision autoaugment')

    return parser


def main(args):
    device = args.device
    data_dir = args.data_path
    # result_dir = args.output_dir
    # ------------------------------------  log ------------------------------------
    # logger, log_dir = utils.make_logger(result_dir)  # 创建日志记录器
    # writer = SummaryWriter(log_dir=log_dir)  # 创建TensorBoard写入器
    # ------------------------------------ step1: dataset ------------------------------------
    """数据加载和预处理阶段:
    1. 定义数据增强策略
    2. 创建数据集对象
    3. 构建数据加载器
    """

    normMean = [0.5]
    normStd = [0.5]
    input_size = (224, 224)

    if args.autoaug:
        auto_aug_list = torchvision.transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET)
        train_transform = transforms.Compose([
            auto_aug_list,
            transforms.Resize(256),
            transforms.RandomCrop(input_size, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(normMean, normStd),
        ])

        valid_transform = transforms.Compose([
            transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.Normalize(normMean, normStd),
        ])
    else:
        train_transform = A.Compose([
            A.RandomScale(scale_limit=0.2, p=0.5),
            A.Resize(256, 256),
            A.RandomCrop(224, 224),  # Randomly shift
            A.Rotate(limit=30, p=0.5),
            A.HorizontalFlip(p=0.5),
            A.Normalize(normMean, normStd, max_pixel_value=255.),  # mean, std， 基于0-1，像素值要求0-255，并通过max_pixel_value=255，来实现整体数据变换为0-1
            ToTensorV2(),  # 仅数据转换，不会除以255
        ])

        valid_transform = A.Compose([
            A.Resize(224, 224),
            A.Normalize(normMean, normStd, max_pixel_value=255.),  # mean, std， 基于0-1，像素值要求0-255，并通过max_pixel_value=255，来实现整体数据变换为0-1
            ToTensorV2(),  # 仅数据转换，不会除以255
        ])

    # chest_xray.zip 解压，获得 chest_xray/train, chest_xray/test
    # 数据可从 https://data.mendeley.com/datasets/rscbjbr9sj/2 下载
    train_dir = os.path.join(data_dir, 'train')
    valid_dir = os.path.join(data_dir, 'test')
    train_set = PneumoniaDataset(train_dir, transform=train_transform)
    valid_set = PneumoniaDataset(valid_dir, transform=valid_transform)

    # 构建DataLoder
    train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    valid_loader = DataLoader(dataset=valid_set, batch_size=8, num_workers=args.workers)

    # ------------------------------------ step2: model ------------------------------------
    """模型准备阶段:
    1. 加载预训练模型
    2. 修改模型结构以适应单通道输入
    3. 调整分类层输出维度
    """
    if args.model == 'resnet50':
        model = torchvision.models.resnet50(pretrained=True)  # 加载预训练的ResNet50
    elif args.model == 'convnext':
        model = torchvision.models.convnext_base(pretrained=True)
    elif args.model == 'convnext-tiny':
        model = torchvision.models.convnext_tiny(pretrained=True)
    else:
        logger.error(f'unexpect model --> :{args.model}')
    model_name = model._get_name()

    if 'ResNet' in model_name:
        # 替换第一层： 因为预训练模型输入是3通道，而本案例是灰度图，输入是1通道
        model.conv1 = nn.Conv2d(1, 64, (7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        num_ftrs = model.fc.in_features  # 替换最后一层
        model.fc = nn.Linear(num_ftrs, 2)
    elif 'ConvNeXt' in model_name:
        # 替换第一层： 因为预训练模型输入是3通道，而本案例是灰度图，输入是1通道
        num_kernel = 128 if args.model == 'convnext' else 96
        model.features[0][0] = nn.Conv2d(1, num_kernel, (4, 4), stride=(4, 4))  # convnext base/ tiny
        # 替换最后一层
        num_ftrs = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(num_ftrs, 2)

    model.to(device)

    # ------------------------------------ step3: optimizer, lr scheduler ------------------------------------
    """优化器和学习率调度器配置:
    1. 定义损失函数
    2. 设置优化器
    3. 配置学习率调度策略
    """
    criterion = nn.CrossEntropyLoss()  # 使用交叉熵损失函数
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)  # 选择优化器
    if args.useplateau:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                               factor=0.2, patience=10, cooldown=5, mode='max')
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size,
                                            gamma=args.lr_gamma)  # 设置学习率下降策略

    # ------------------------------------ step4: iteration ------------------------------------
    """训练和验证循环:
    1. 执行训练epoch
    2. 执行验证epoch 
    3. 记录训练指标
    4. 保存最佳模型
    """
    best_acc, best_epoch = 0, 0  # 初始化最佳准确率和对应epoch
    logger.info(args)
    # logger.info(train_loader, valid_loader)
    logger.info("Start training")
    start_time = time.time()
    epoch_time_m = utils.AverageMeter()
    end = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        # 训练
        loss_m_train, acc_m_train, mat_train = \
            utils.ModelTrainer.train_one_epoch(train_loader, model, criterion, optimizer, scheduler,
                                               epoch, device, args, logger, classes)
        # 验证
        loss_m_valid, acc_m_valid, mat_valid = \
            utils.ModelTrainer.evaluate(valid_loader, model, criterion, device, classes)

        epoch_time_m.update(time.time() - end)
        end = time.time()

        lr_current = scheduler.optimizer.param_groups[0]['lr'] if args.useplateau else scheduler.get_last_lr()[0]
        logger.info(
            'Epoch: [{:0>3}/{:0>3}]  '
            'Time: {epoch_time.val:.3f} ({epoch_time.avg:.3f})  '
            'Train Loss avg: {loss_train.avg:>6.4f}  '
            'Valid Loss avg: {loss_valid.avg:>6.4f}  '
            'Train Acc@1 avg:  {top1_train.avg:>7.4f}   '
            'Valid Acc@1 avg: {top1_valid.avg:>7.4f}    '
            'LR: {lr}'.format(
                epoch, args.epochs, epoch_time=epoch_time_m, loss_train=loss_m_train, loss_valid=loss_m_valid,
                top1_train=acc_m_train, top1_valid=acc_m_valid, lr=lr_current))

        # 学习率更新
        if args.useplateau:
            scheduler.step(acc_m_valid.avg)
        else:
            scheduler.step()
        # 记录
        writer.add_scalars('Loss_group', {'train_loss': loss_m_train.avg,
                                          'valid_loss': loss_m_valid.avg}, epoch)
        writer.add_scalars('Accuracy_group', {'train_acc': acc_m_train.avg,
                                              'valid_acc': acc_m_valid.avg}, epoch)
        conf_mat_figure_train = utils.show_conf_mat(mat_train, classes, "train", log_dir, epoch=epoch,
                                        verbose=epoch == args.epochs - 1, save=True)
        conf_mat_figure_valid = utils.show_conf_mat(mat_valid, classes, "valid", log_dir, epoch=epoch,
                                        verbose=epoch == args.epochs - 1, save=True)
        writer.add_figure('confusion_matrix_train', conf_mat_figure_train, global_step=epoch)
        writer.add_figure('confusion_matrix_valid', conf_mat_figure_valid, global_step=epoch)
        writer.add_scalar('learning rate', lr_current, epoch)

        # ------------------------------------ 模型保存 ------------------------------------
        if best_acc < acc_m_valid.avg or epoch == args.epochs - 1:
            best_epoch = epoch if best_acc < acc_m_valid.avg else best_epoch
            best_acc = acc_m_valid.avg if best_acc < acc_m_valid.avg else best_acc
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "lr_scheduler_state_dict": scheduler.state_dict(),
                "epoch": epoch,
                "args": args,
                "best_acc": best_acc}
            pkl_name = "checkpoint_{}.pth".format(epoch) if epoch == args.epochs - 1 else "checkpoint_best.pth"
            path_checkpoint = os.path.join(log_dir, pkl_name)
            torch.save(checkpoint, path_checkpoint)
            logger.info(f'save ckpt done! best acc:{best_acc}, epoch:{epoch}')

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info("Training time {}".format(total_time_str))


classes = ["NORMAL", "PNEUMONIA"]


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    utils.setup_seed(args.random_seed)
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main(args)

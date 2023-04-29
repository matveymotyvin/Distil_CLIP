# Import necessary libraries
from argparse import ArgumentParser
import pytorch_lightning as pl
import pytorch_lightning.callbacks as plc
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

# Import user-defined modules
from data import DInterface
from model import MInterface
from utils import load_model_path_by_args


def load_callbacks():
    """
    This function creates a list of callbacks for Trainer in the main function.
    """
    # Create a list of callbacks
    callbacks = [
        plc.ModelCheckpoint(
            monitor='val_acc',
            filename='best-{epoch:02d}-{val_acc:.3f}',
            save_top_k=2,
            auto_insert_metric_name=True,
            mode='max',
            save_last=True
        ),
    ]

    # If lr_scheduler is True, add LearningRateMonitor to the callbacks list
    if args.lr_scheduler:
        print("Load the LearningRateMonitor")
        callbacks.append(plc.LearningRateMonitor(logging_interval='epoch'))

    # Return the list of callbacks
    return callbacks


def main(args):
    """
    This function trains the model.
    """
    # Set the random seed
    pl.seed_everything(args.seed)

    # Load the model path
    load_path = load_model_path_by_args(args)

    # Create the data module
    data_module = DInterface(**vars(args))

    # Create the TensorBoardLogger
    logger = TensorBoardLogger(save_dir=args.save_dir, name=args.name, version=args.version)
    args.logger = logger

    # If load_path is None, create a new model. Otherwise, resume the training from the checkpoint.
    if load_path is None:
        model = MInterface(**vars(args))
    else:
        model = MInterface(**vars(args))
        args.resume_from_checkpoint = load_path

    # Load the callbacks
    args.callbacks = load_callbacks()

    # Create the Trainer
    trainer = Trainer.from_argparse_args(args,
                                         enable_progress_bar=True,
                                         accelerator='gpu',
                                         devices=1)

    # Train the model
    trainer.fit(model, data_module)


if __name__ == '__main__':
    # Create the ArgumentParser
    parser = ArgumentParser()

    # Experiment setting for Tensorboard
    parser.add_argument('--save_dir', default='./checkpoint', type=str)
    parser.add_argument('--name', default=None, type=str)
    parser.add_argument('--version', default=None, type=int)

    # Trainer Control

    # Basic Training Control
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--seed', default=2023, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--weight_decay', default=1e-5, type=float)


    # LR Scheduler
    parser.add_argument('--lr_scheduler', choices=['step', 'cosine'], type=str, default='step')
    parser.add_argument('--lr_decay_steps', default=10, type=int)
    parser.add_argument('--lr_decay_rate', default=0.5, type=float)
    parser.add_argument('--lr_decay_min_lr', default=1e-5, type=float)

    # Restart Control
    parser.add_argument('--load_best', action='store_true')
    parser.add_argument('--load_dir', default=None, type=str)
    parser.add_argument('--load_ver', default=None, type=str)
    parser.add_argument('--load_v_num', default=None, type=int)

    # Training Info
    parser.add_argument('--dataset', default='image_dataset', type=str)
    parser.add_argument('--data_dir', default='/path/to/data', type=str)
    parser.add_argument('--model_name', default='model_image_distilled', type=str)
    parser.add_argument('--teacher_name', default='ViT-B/32', type=str)
    parser.add_argument('--no_augment', action='store_true')
    parser.add_argument('--log_dir', default='runs', type=str)
    parser.add_argument('--cache_dir', default='cache', type=str)

    # Image Model Hyperparameters
    parser.add_argument('--input_resolution', default=224, type=int)
    parser.add_argument('--patch_size', default=32, type=int)
    parser.add_argument('--width', default=576, type=int)
    parser.add_argument('--layers', default=6, type=int)
    parser.add_argument('--heads', default=24, type=int)

    # Language Model Hyperparameters
    """
     context_length, vocab_size, transformer_width, transformer_layers, transformer_heads, output_dim
    """
    parser.add_argument('--context_length', default=77, type=int)
    parser.add_argument('--vocab_size', default=49408, type=int)
    parser.add_argument('--transformer_width', default=128, type=int)
    parser.add_argument('--transformer_layers', default=6, type=int)
    parser.add_argument('--transformer_heads', default=8, type=int)

    # share Hyperparameters
    parser.add_argument('--output_dim', default=512, type=int)

    # Other
    parser.add_argument('--aug_prob', default=0.5, type=float)

    # Loss Settings
    parser.add_argument('--t', default=4, type=int)
    parser.add_argument('--weight', default=[0.6, 0.35, 0.05], nargs='+', type=list)
    parser.add_argument('--loss_scale', default=[1, 1, 1], nargs='+', type=list)
    parser.add_argument('--loss', default=['kl', 'l1', 'ce'], nargs='+', type=list)

    parser.add_argument('--device', default='cuda', type=str)
    

    parser = Trainer.add_argparse_args(parser)

    args = parser.parse_args()

    # List Arguments
    args.mean_sen = [0.485, 0.456, 0.406]
    args.std_sen = [0.229, 0.224, 0.225]

    for key, value in args.__dict__.items():
        print(key, ' : ', value)

    main(args)

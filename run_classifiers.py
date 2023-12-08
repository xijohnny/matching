from classifiers.datamodules import ImageDataModule
from classifiers.models import EncoderClassifier, ImageEncoder
from pytorch_lightning import loggers, Trainer
import argparse
from pytorch_lightning.callbacks import ModelCheckpoint

def parse_arguments():
    parser = argparse.ArgumentParser(description = "Guide Classifier")
    parser.add_argument("--max_epochs", metavar = "MAX_EPOCHS", type = int, default = 50)
    parser.add_argument("--batch_size", metavar = "BATCH_SIZE", type = int, default = 256)
    parser.add_argument("--lr", metavar = "LEARNING_RATE", type = float, default = 0.0001)
    #parser.add_argument("--devices", metavar = "NUMBER OF GPUS", type = int, default = 1)
    return parser.parse_args()

if __name__ == "__main__":

    args = parse_arguments()

    data_path = "/mnt/ps/home/CORP/johnny.xi/sandbox/proteomics/53m_proteomics.parquet"  ## replace with your parquet file

    checkpoint_callback = ModelCheckpoint(
        dirpath = "results/checkpoints/",
        filename = "ImageClassifier-{epoch:02d}-{val_loss:.2f}",
        monitor = "val_loss"
    )

    data = ImageDataModule(data_path = data_path, batch_size = args.batch_size)
    model = EncoderClassifier(encoder = ImageEncoder, lr = args.lr)

    logdir = "checkpoints/" 
    wandb_logger = loggers.WandbLogger(save_dir = logdir, project = "Guide-Classifier")

    trainer = Trainer(accelerator = "gpu", 
                    max_epochs = args.max_epochs,
                    default_root_dir = logdir,
                    devices = -1,
                    log_every_n_steps = 1,
                    num_sanity_val_steps=2,
                    callbacks = [checkpoint_callback], 
                    logger = wandb_logger,
                    precision = "16"
                    #use_distributed_sampler=False                    
                    )
    
    if trainer.global_rank == 0:
        wandb_logger.experiment.config.update(vars(args))

    trainer.fit(model = model, datamodule = data)
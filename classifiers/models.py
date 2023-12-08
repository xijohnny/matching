import torch
from torch import nn, optim
from torchvision.models.densenet import densenet161
from pytorch_lightning import LightningModule
from typing import Type
import sys
sys.path.insert(1, "/mnt/ps/home/CORP/johnny.xi/sandbox/photosynthetic")

from .datamodules import Normalizer
from .utils import compute_class_weights, replace_submodules

def densenet(
    constructor: Type[densenet161] = densenet161,
    use_pretrained_imagenet_weights: bool = True,
    memory_efficient: bool = True,
) -> torch.nn.Module:
    """Construct a Densenet backbone, default to DN161."""
    backbone = constructor(
        weights="DEFAULT" if use_pretrained_imagenet_weights else None,
        memory_efficient=memory_efficient,
    ).features
    return replace_submodules(backbone, **{"conv0": {"_class_": torch.nn.LazyConv2d}})


class BaseClassifier(LightningModule):
    def __init__(self, 
                 lr: float = 0.0005, 
                 wd: float = 0.0001):
        super().__init__()

        self.lr = lr
        self.wd = wd
        self.loss = torch.nn.CrossEntropyLoss()
        ## subclasses should define self.clf1, self.clf2 classifiers for each modality in their init

    def forward(self, x):
        logits = self.clf(x)
        return logits

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)
        self.log("val_loss", loss)

    def on_validation_epoch_end(self): 
        pass

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)
        self.log("test_loss", loss)

    def on_test_epoch_end(self):
        pass

    def setup(self, stage:str):
        self.n_classes, weights = compute_class_weights(self.trainer.datamodule.labels)
        self.loss.weight = torch.from_numpy(weights).float()

    def on_train_epoch_end(self):
        pass

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr = self.lr, weight_decay = self.wd)
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr = self.lr, total_steps = self.trainer.max_epochs, pct_start = 0.1)
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]
    

class ImageEncoder(torch.nn.Module):
    def __init__(self, n_hidden: int = 1024):
        super(ImageEncoder, self).__init__()        

        self.n_hidden = n_hidden
        
        self.preprocess = nn.Sequential(Normalizer(),
        nn.LazyInstanceNorm2d(affine = False, track_running_stats = False))
        self.backbone = densenet(use_pretrained_imagenet_weights = True, memory_efficient = True)
        self.postprocess = nn.Sequential(nn.ReLU(),
            nn.AdaptiveAvgPool2d(output_size = 1),
            nn.Flatten(start_dim = 1),
            nn.LazyBatchNorm1d(),
            nn.LazyLinear(out_features = n_hidden),
            nn.ReLU(),
            nn.LazyBatchNorm1d(),
            nn.LazyLinear(out_features = n_hidden))

    def forward(self, x):
        x = self.preprocess(x)
        x = self.backbone(x)
        x = self.postprocess(x)
        return x
    
class EncoderClassifier(BaseClassifier):
    def __init__(self, 
                encoder,
                n_classes: int = 117,
                encoder_kwargs = {}, 
                **kwargs):
        super().__init__(**kwargs)

        self.encoder = encoder(**encoder_kwargs)

    def setup(self, stage: str):
        super().setup(stage)
        self.clf = nn.Sequential(self.encoder,
        nn.Linear(self.encoder.n_hidden, self.n_classes))
        
        if isinstance(self.encoder, ImageEncoder):
            dummy = torch.randn((2,6,512,512))
            self.forward(dummy) ## initialize lazy modules

        
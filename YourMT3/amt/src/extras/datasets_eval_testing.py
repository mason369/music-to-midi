from utils.datasets_eval import AudioFileDataset
from torch.utils.data import DataLoader
import pytorch_lightning as pl


def test():

    ds = AudioFileDataset()
    dl = DataLoader(
        ds, batch_size=None, collate_fn=lambda k: k
    )  # empty collate_fn is required to use mixed types.

    for x, y in dl:
        break

    class MyModel(pl.LightningModule):

        def __init__(self, **kwargs):
            super().__init__()

        def forward(self, x):
            return x

        def training_step(self, batch, batch_idx):
            return 0

        def validation_step(self, batch, batch_idx):
            print(batch)
            return 0

        def train_dataloader(self):
            return dl

        def val_dataloader(self):
            return dl

        def configure_optimizers(self):
            return None

    model = MyModel()
    trainer = pl.Trainer()
    trainer.validate(model)
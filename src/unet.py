"""3D U-Net (MONAI): model, training loop and sliding-window prediction."""
import numpy as np
import torch
from monai.data import DataLoader, Dataset
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.networks.nets import UNet
from monai.transforms import (Compose, RandAffined, RandCropByPosNegLabeld, RandFlipd, RandScaleIntensityd,
                              RandShiftIntensityd)
from monai.utils import set_determinism

from utils import keep_largest_component

PATCH = (96, 96, 96)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

keys = ["image", "label"]
# data augmentation: each time a volume is used, it is randomly modified so the network sees more variety
train_transforms = Compose([
    # 2 patches of 96x96x96 voxels per volume, half of them centred on the LA
    RandCropByPosNegLabeld(keys=keys, label_key="label", spatial_size=PATCH, pos=1, neg=1, num_samples=2),
    # small rotations (up to 15 degrees) and zoom (+-10%)
    RandAffined(keys=keys, prob=0.3, rotate_range=(0.26, 0.26, 0.26), scale_range=(0.1, 0.1, 0.1),
                mode=("bilinear", "nearest")),
    # mirror image
    RandFlipd(keys=keys, prob=0.5, spatial_axis=0),
    # small changes of contrast and brightness
    RandScaleIntensityd(keys="image", factors=0.1, prob=0.5),
    RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),
])


def make_model():
    # 5 levels (16 to 256 filters), each level halves the image size; 2 output channels: background and LA
    return UNet(spatial_dims=3, in_channels=1, out_channels=2,
                channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2), num_res_units=2)


def train(volumes, n_iters, seed, callback=None, every=250):
    """Trains a U-Net on a list of (normalised image, mask) pairs.
    callback(model, iteration) is called every `every` iterations (used for validation)."""
    set_determinism(seed=seed)
    ds = Dataset([{"image": x[None], "label": y[None].astype(np.float32)} for x, y in volumes], train_transforms)
    loader = DataLoader(ds, batch_size=2, shuffle=True, num_workers=0)  # 2 volumes x 2 patches = 4 patches

    model = make_model().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    # the learning rate decreases from 1e-3 to 0 during training (as in nnU-Net), so the end of training is stable
    scheduler = torch.optim.lr_scheduler.PolynomialLR(opt, total_iters=n_iters, power=0.9)
    # mixed precision (float16) to use less GPU memory and train faster
    scaler = torch.amp.GradScaler(device.type)
    loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)

    it = 0
    model.train()
    while it < n_iters:
        for batch in loader:
            x, y = batch["image"].to(device), batch["label"].to(device)
            # one training step: prediction, loss, gradients, weight update
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.float16):
                loss = loss_fn(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            scheduler.step()
            it += 1
            if callback is not None and it % every == 0:
                callback(model, it)
                model.train()
            if it == n_iters:
                break
    return model


def predict(model, x):
    """LA probability map of a normalised volume (sliding window over the whole volume)."""
    model.eval()
    with torch.no_grad(), torch.autocast(device.type, dtype=torch.float16):
        inp = torch.from_numpy(x)[None, None].to(device)
        # the volume is cut into overlapping 96^3 patches, predicted one by one, then put back together
        logits = sliding_window_inference(inp, PATCH, sw_batch_size=4, predictor=model, overlap=0.25)
    # softmax turns the network outputs into probabilities; channel 1 = LA
    return torch.softmax(logits.float(), dim=1)[0, 1].cpu().numpy()


def segment(model, x, threshold=0.5):
    """Probability >= 0.5 then largest component, as for the random forest. Also returns the probability map."""
    prob = predict(model, x)
    return keep_largest_component(prob >= threshold), prob

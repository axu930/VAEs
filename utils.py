import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.transforms import v2 
import numpy as np
import matplotlib.pyplot as plt


#TODO shuffling

class DataLoaderLite:

    def __init__(self, data):
        self.data = data
        self.current_position = 0
        self.len = len(data)

    def get_Batch(self, batch_size):
        self.current_position %= self.len
        if self.current_position + batch_size >= len(self.data):
            overflow = self.current_position + batch_size - self.len
            images = (self.data[self.current_position:]['image'] + self.data[:overflow]['image'])
            labels = (self.data[self.current_position:]['label'] + self.data[:overflow]['label'])
            print(type(images),type(labels))
            self.current_position = overflow
        else:
            images = self.data[self.current_position:self.current_position + batch_size]['image']
            labels = self.data[self.current_position:self.current_position + batch_size]['label']
            self.current_position = self.current_position + batch_size 
        images = enc(images)
        images = torch.stack(images)
        labels = F.one_hot(torch.tensor(labels), num_classes=10)

        return images, labels

# image pre/post processing
enc = v2.Compose([
    v2.ToImage(), 
    v2.ToDtype(torch.float32, scale=True),
    ])

dec = v2.Compose([
    v2.Lambda(lambd=(lambda x: 255 * x)),
    v2.ToDtype(torch.uint8),
    ])


device = 'mps:0'

def sample(model, samples):
    # in/out: normalized image tensors
    before = samples
    samples = samples.to(device)
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        out, _ = model(samples)
    after = out.detach().cpu()
    after = torch.clamp(after,0,1)
    return before, after


def show(imgs):
    to_pil = v2.ToPILImage()
    if not isinstance(imgs, list):
        imgs = [imgs]
    _, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = to_pil(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

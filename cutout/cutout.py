import torch
import numpy as np




def sinle_style_transfer(x1,x2):

    img1 = x1
    img2 = x2

    for channel in range(img1.shape[0]):
        img1_channel_mean = torch.mean(img1[channel])
        img1_channel_std = torch.std(img1[channel])

        img2_channel_mean = torch.mean(img2[channel])
        img2_channel_std = torch.std(img2[channel])

        img1[channel] = (img1[channel] - img1_channel_mean) / img1_channel_std
        img1[channel] = img1[channel] * img2_channel_std + img2_channel_mean


    return img1


class SSCutout(object):
    """Randomly mask out one or more patches from an image.

    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """

    def __init__(self, n_holes, length, alpha):
        self.n_holes = n_holes
        self.length = length
        self.alpha = alpha

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)
        cut_region = 0

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.
            cut_region = img[:, y1:y2, x1:x2]

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)

        img_style = sinle_style_transfer(img.clone(),cut_region.clone())
        lam = np.random.beta(self.alpha, self.alpha)

        mixImage = lam*img + (1-lam)*img_style
        img = mixImage * mask

        return img


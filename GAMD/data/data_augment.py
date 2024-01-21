import random
import torchvision.transforms as transforms
import torchvision.transforms.functional as F


class PairRandomCrop(transforms.RandomCrop):

    def __call__(self, image, cp, ep, label):

        if self.padding is not None:
            image = F.pad(image, self.padding, self.fill, self.padding_mode)
            label = F.pad(label, self.padding, self.fill, self.padding_mode)
            cp = F.pad(cp, self.padding, self.fill, self.padding_mode)
            ep = F.pad(ep, self.padding, self.fill, self.padding_mode)

        # pad the width if needed
        if self.pad_if_needed and image.size[0] < self.size[1]:
            image = F.pad(image, (self.size[1] - image.size[0], 0), self.fill, self.padding_mode)
            label = F.pad(label, (self.size[1] - label.size[0], 0), self.fill, self.padding_mode)
            cp = F.pad(cp, (self.size[1] - label.size[0], 0), self.fill, self.padding_mode)
            ep = F.pad(ep, (self.size[1] - label.size[0], 0), self.fill, self.padding_mode)
        # pad the height if needed
        if self.pad_if_needed and image.size[1] < self.size[0]:
            image = F.pad(image, (0, self.size[0] - image.size[1]), self.fill, self.padding_mode)
            label = F.pad(label, (0, self.size[0] - image.size[1]), self.fill, self.padding_mode)
            cp = F.pad(cp, (0, self.size[0] - image.size[1]), self.fill, self.padding_mode)
            ep = F.pad(ep, (0, self.size[0] - image.size[1]), self.fill, self.padding_mode)

        i, j, h, w = self.get_params(image, self.size)

        return F.crop(image, i, j, h, w), F.crop(cp, i, j, h, w), F.crop(ep, i, j, h, w), F.crop(label, i, j, h, w),


class PairCompose(transforms.Compose):
    def __call__(self, image, cp, ep, label):
        for t in self.transforms:
            image, cp, ep, label = t(image, cp, ep, label)
        return image, cp, ep, label


class PairRandomHorizontalFilp(transforms.RandomHorizontalFlip):
    def __call__(self, img, cp, ep, label):
        """
        Args:
            img (PIL Image): Image to be flipped.

        Returns:
            PIL Image: Randomly flipped image.
        """
        if random.random() < self.p:
            return F.hflip(img),F.hflip(cp),F.hflip(ep),F.hflip(label)
        return img,cp,ep,label


class PairToTensor(transforms.ToTensor):
    def __call__(self, pic, cp, ep, label):
        """
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """
        return F.to_tensor(pic),F.to_tensor(cp),F.to_tensor(ep),F.to_tensor(label)

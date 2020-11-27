"""
该脚本用于pytorch没有的图像增强
"""
from PIL.ImageFilter import GaussianBlur
import random
import torchvision.transforms.functional as functional


class RandomGaussianBlur(object):
    """
    随机高斯模糊

    Args:
    radius_min (int): 模糊最小半径, 默认是0
    radius_max (int): 模糊最大半径, 默认是2
    p (float): 发生概率, 默认是0.5
    """
    def __init__(self, radius_min=0, radius_max=2, p=0.5):
        self.radius_min = radius_min
        self.radius_max = radius_max
        self.p = p

    def blur(self, img):
        if random.random() < self.p:
            radius = random.randint(self.radius_min, self.radius_max)
            img = img.filter(GaussianBlur(radius=radius))
        return img

    def __call__(self, img):
        return self.blur(img)


class RandomHue(object):
    """
    随机h通道变换

    Args:
    hue_factor_min (float): 变换最小值, 默认是-0.1, 不超过-0.5
    hue_factor_max (float): 变换最小值, 默认是0.1, 不超过0.5
    p (float): 发生概率, 默认是0.5
    """
    def __init__(self, hue_factor_min=-0.1, hue_factor_max=0.1, p=0.5):
        self.hue_factor_min = hue_factor_min
        self.hue_factor_max = hue_factor_max
        self.p = p

    def hue(self, img):
        if random.random() < self.p:
            hue_factor = (self.hue_factor_max-self.hue_factor_min)*random.random() + self.hue_factor_min
            img = functional.adjust_hue(img, hue_factor)
        return img

    def __call__(self, img):
        return self.hue(img)
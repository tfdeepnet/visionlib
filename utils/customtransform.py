import albumentations as A
import albumentations.pytorch as AP
import numpy as np

class AlbumentationTransforms:
    def __init__(self, transforms_list=[]):
        transforms_list.append(AP.ToTensor())
        self.transforms_new = A.Compose(transforms_list)

    def __call__(self,img):
        img = np.array(img)
        img = self.transforms_new(image = img)['image']
        return img
import albumentations as A

def train_transforms(frame_size, crop_size, mean=0.0, std=1.0, norm=True):
    transforms = []
    transforms.append(A.SmallestMaxSize(max_size=frame_size, always_apply=True))
    transforms.append(A.CenterCrop(frame_size, frame_size, always_apply=True))
    transforms.append(A.Resize(crop_size, crop_size, always_apply=True))
    if norm:
        transforms.append(A.Normalize(mean=mean, std=std, always_apply=True))
    return A.Compose(transforms)


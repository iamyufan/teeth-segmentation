import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from PIL import Image


DATA_ROOT = "./Tufts_Dental_Database"
IMAGE_DIR = os.path.join(DATA_ROOT, "Radiographs")
MASKS_DIR = os.path.join(DATA_ROOT, "Transformed_masks")
ANNOTATION_PATH = os.path.join(DATA_ROOT, "Segmentation", "teeth_bbox.json")


class TuftsDentalDataset(Dataset):
    def __init__(self, annotations, image_dir, masks_dir, transforms=None):
        self.annotations = annotations
        self.image_dir = image_dir
        self.masks_dir = masks_dir
        self.transforms = transforms

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        """
        Returns
        -------
        image : torch.Tensor
            The image tensor of shape (C, H, W).
        target : dict
            The target dictionary containing the following keys:
            - boxes (torch.Tensor):
                The bounding box coordinates of shape (N, 4)
            - labels (torch.Tensor): The label of each bounding box of shape (N,).
            - masks (torch.Tensor): The segmentation masks of shape (N, H, W).
        """
        # Get the annotation and image & mask paths
        annotation = self.annotations[idx]
        img_idx = annotation["External ID"].split(".")[0]
        image_path = os.path.join(self.image_dir, img_idx + ".JPG")
        mask_path = os.path.join(self.masks_dir, img_idx + ".npy")

        # Load the image and mask
        image = np.array(Image.open(image_path).convert("RGB"))
        image = image.astype(np.float32) / 255.0
        stored_mask = np.load(mask_path)

        # Get the targets (boxes, labels, masks) for each instance
        masks = []
        boxes = []
        labels = []

        for i, obj in enumerate(annotation["Label"]["objects"]):
            # Exclude objects whose title is not integer
            if not obj["title"].isdigit():
                continue

            # Convert the mask into multiple binary masks
            # The stored mask is a 2D array, where each pixel value is
            # one unique instance. The pixel value is incremented by one
            # for each new instance from 1
            # i.e., the pixel with value i + 1 is the mask for the i-th instance
            mask = stored_mask == (i + 1)
            masks.append(mask.astype(np.uint8))

            # Get the bounding box coordinates
            y_min, x_min, y_max, x_max = obj["bounding box"]
            boxes.append([x_min, y_min, x_max, y_max])

            # Use the title as the label
            labels.append(int(obj["title"]))

        # Convert everything into a torch.Tensor
        image = torch.as_tensor(image, dtype=torch.float32).permute(2, 0, 1)
        masks = torch.as_tensor(np.stack(masks, axis=0), dtype=torch.uint8)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        # Create the target dictionary
        target = {"boxes": boxes, "labels": labels, "masks": masks}

        # Apply the transforms if any
        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def visualize_one_sample(self, idx, target_idx=0):
        sample = self.__getitem__(idx)
        sample_image = sample[0].numpy().transpose(1, 2, 0)
        sample_target = sample[1]

        print(sample_target["boxes"].shape)
        print(sample_target["labels"].shape)
        print(sample_target["masks"].shape)

        # Visualize the first bounding box and the first mask with pyplot
        box = sample_target["boxes"][target_idx].numpy()
        mask = sample_target["masks"][target_idx].numpy().squeeze()

        # Visualize the bounding box with pyplot
        x_min, y_min, x_max, y_max = box

        plt.figure(figsize=(10, 10))
        plt.imshow(sample_image)
        plt.axis("off")
        plt.gca().add_patch(
            plt.Rectangle(
                (x_min, y_min),
                x_max - x_min,
                y_max - y_min,
                fill=False,
                edgecolor="red",
                lw=2,
            )
        )

        # Visualize the mask with pyplot
        plt.figure(figsize=(10, 10))
        plt.imshow(mask, cmap="gray")
        plt.axis("off")


def collate_fn(batch):
    """
    Custom collate function to handle variable number of objects per image.
    Args:
        batch (list): List of tuples (image, target).
    Returns:
        tuple: A tuple containing the images and targets.
    """
    return tuple(zip(*batch))


class TuftsDentalDataModule:
    def __init__(
        self,
        annotations_path=ANNOTATION_PATH,
        image_dir=IMAGE_DIR,
        masks_dir=MASKS_DIR,
        batch_size=4,
    ):
        self.annotations_path = annotations_path
        self.image_dir = image_dir
        self.masks_dir = masks_dir
        self.batch_size = batch_size
        self.setup()

    def setup(self):
        # Load annotations
        with open(self.annotations_path, "r") as f:
            annotations = json.load(f)

        # Filter out annotations without objects
        self.annotations = [ann for ann in annotations if ann["Label"]["objects"]]

        # Randomly sample 80% of the data for training and 20% for validation
        num_samples = len(self.annotations)
        num_train = int(0.8 * num_samples)
        indices = np.random.permutation(num_samples)
        train_indices = indices[:num_train]
        val_indices = indices[num_train:]

        # Split the data into training and validation sets
        self.train_annotations = [self.annotations[i] for i in train_indices]
        self.val_annotations = [self.annotations[i] for i in val_indices]

        # Initialize the datasets
        self.train_dataset = TuftsDentalDataset(
            self.train_annotations, self.image_dir, self.masks_dir
        )
        self.val_dataset = TuftsDentalDataset(
            self.val_annotations, self.image_dir, self.masks_dir
        )
        print("Training samples:", len(self.train_dataset))
        print("Validation samples:", len(self.val_dataset))

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )

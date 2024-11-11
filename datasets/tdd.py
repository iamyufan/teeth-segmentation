import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageDraw
import numpy as np


class TuftsDentalDataset(Dataset):
    def __init__(self, annotations, image_dir, transforms=None):
        self.annotations = annotations
        self.image_dir = image_dir
        self.transforms = transforms

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        # Get annotation and image path
        annotation = self.annotations[idx]
        img_path = os.path.join(self.image_dir, annotation["External ID"])
        image = np.array(Image.open(img_path).convert("RGB"))
        image = image.astype(np.float32) / 255.0

        # Parse polygons for masks
        objects = annotation["Label"]["objects"]
        masks = []
        boxes = []
        labels = []

        for obj in objects:
            # Exclude objects whose title is not integer
            if not obj["title"].isdigit():
                continue

            # Get the polygons and convert to a mask
            polygons = obj["polygons"]
            mask = self.polygons_to_mask(polygons, (image.shape[1], image.shape[0]))
            masks.append(mask)

            # Get the bounding box
            y_min, x_min, y_max, x_max = obj["bounding box"]
            boxes.append([x_min, y_min, x_max, y_max])

            # Use the 'title' as the class label
            labels.append(int(obj["title"]))

        # Convert everything to PyTorch tensors
        image = torch.as_tensor(image, dtype=torch.float32).permute(2, 0, 1)
        masks = torch.as_tensor(np.stack(masks), dtype=torch.uint8)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        # Create the target dictionary
        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
        }

        if self.transforms:
            image, target = self.transforms(image, target)

        return image, target

    @staticmethod
    def polygons_to_mask(polygons, image_size):
        """
        Converts polygons to a binary mask.
        Args:
            polygons (list): List of polygons, where each polygon is a list of [x, y] points.
            image_size (tuple): Size of the image (width, height).
        Returns:
            np.ndarray: Binary mask of the object.
        """
        mask = Image.new("L", image_size, 0)
        draw = ImageDraw.Draw(mask)

        for polygon in polygons:
            # Flatten the polygon coordinates
            if len(polygon) > 2:  # Ensure it is a valid polygon
                flat_polygon = [tuple(point) for point in polygon]
                draw.polygon(flat_polygon, outline=1, fill=1)

        return np.array(mask)

    @staticmethod
    def get_bounding_box(mask):
        """
        Get the bounding box of a binary mask.
        Args:
            mask (np.ndarray): Binary mask of the object.
        Returns:
            list: Bounding box in [xmin, ymin, xmax, ymax] format.
        """
        pos = np.where(mask)
        xmin = np.min(pos[1])
        ymin = np.min(pos[0])
        xmax = np.max(pos[1])
        ymax = np.max(pos[0])
        return [xmin, ymin, xmax, ymax]


def collate_fn(batch):
    """
    Custom collate function to handle variable number of objects per image.
    Args:
        batch (list): List of tuples (image, target).
    Returns:
        images (list): List of images.
        targets (list): List of target dictionaries.
    """
    images, targets = zip(*batch)
    return list(images), list(targets)


class TuftsDentalDataModule:
    def __init__(self, annotations_path, image_dir, batch_size=4):
        self.annotations_path = annotations_path
        self.image_dir = image_dir
        self.batch_size = batch_size

    def setup(self):
        # Load annotations
        with open(self.annotations_path, "r") as f:
            self.annotations = json.load(f)
            
        # Only use the first 100 annotations for demonstration purposes
        self.annotations = self.annotations[:100]

        # Filter out annotations without objects
        self.annotations = [ann for ann in self.annotations if ann["Label"]["objects"]]

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
        self.train_dataset = TuftsDentalDataset(self.train_annotations, self.image_dir)
        self.val_dataset = TuftsDentalDataset(self.val_annotations, self.image_dir)

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

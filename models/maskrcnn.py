import os
import wandb
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.utils import draw_segmentation_masks, draw_bounding_boxes


NUM_CLASSES = 33

OUTPUT_DIR = "./outputs"
CHECKPOINT_DIR = "./checkpoints"


class MaskRCNNModel:
    def __init__(self, num_classes=NUM_CLASSES):
        # Initialize Mask R-CNN model
        self.model = maskrcnn_resnet50_fpn(pretrained=True)
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = (
            torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
                in_features, num_classes
            )
        )
        self.model.roi_heads.mask_predictor = (
            torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(
                self.model.roi_heads.mask_predictor.conv5_mask.in_channels,
                256,
                num_classes,
            )
        )

    def configure_optimizers(self):
        params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.SGD(
            params, lr=self.learning_rate, momentum=0.9, weight_decay=0.0005
        )
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=self.step_size, gamma=self.gamma
        )

    def training_step(self, batch, batch_idx):
        images, targets = batch
        images = list(image.to(self.device) for image in images)
        targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

        # Forward pass
        loss_dict = self.model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        return losses

    def _train_one_epoch(self, train_loader, epoch_idx):
        self.model.train()

        epoch_loss = 0
        for batch_idx, batch in enumerate(train_loader):
            # Zero the gradients
            self.optimizer.zero_grad()
            # Forward pass
            batch_losses = self.training_step(batch, batch_idx)
            # Backward pass
            batch_losses.backward()
            self.optimizer.step()
            # Accumulate the total loss
            epoch_loss += batch_losses.item()

        # Compute the average loss for the epoch
        avg_epoch_loss = epoch_loss / len(train_loader)
        # Update the learning rate
        self.lr_scheduler.step()
        return avg_epoch_loss

    def train(
        self,
        train_loader,
        num_epochs=10,
        learning_rate=0.005,
        step_size=3,
        gamma=0.1,
        checkpoint_dir=CHECKPOINT_DIR,
        output_dir=OUTPUT_DIR,
    ):
        # Set the device
        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        print("Training on:", device)
        self.device = device
        self.model.to(device)

        # Set up the optimizer and learning rate scheduler
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.step_size = step_size
        self.gamma = gamma
        self.configure_optimizers()
        self.checkpoint_dir = checkpoint_dir
        self.output_dir = output_dir

        # Lists to store the training loss
        self.train_loss_list = []

        # Training loop
        for epoch in range(self.num_epochs):
            # Train for one epoch
            avg_epoch_loss = self._train_one_epoch(train_loader, epoch)
            self.train_loss_list.append(avg_epoch_loss)
            # Print loss for the epoch
            print(f"Epoch {epoch+1} \t Loss: {avg_epoch_loss}")

            # Save the model checkpoint
            checkpoint_path = f"{self.checkpoint_dir}/model_epoch_{epoch+1}.pth"
            torch.save(self.model.state_dict(), checkpoint_path)

        # Visualize the training loss
        self.visualize_training_loss(
            self.train_loss_list, self.output_dir, self.num_epochs
        )

    def visualize_one_prediction(self, image, target):
        # Visualize some predictions
        self.model.eval()

        with torch.no_grad():
            image = image.to(self.model.device)
            prediction = self.model([image])[0]

            # Move to CPU for visualization
            img_cpu = image.cpu()
            img_vis = img_cpu * 255
            img_vis = img_vis.byte()

            # Get predictions
            boxes = prediction["boxes"].cpu()
            labels = prediction["labels"].cpu()
            scores = prediction["scores"].cpu()
            masks = prediction["masks"].cpu()
            # Filter out low scoring results
            keep = scores >= 0.5
            boxes = boxes[keep]
            labels = labels[keep]
            masks = masks[keep]
            # Convert masks to binary masks
            binary_masks = masks > 0.5  # shape: (N, 1, H, W)
            binary_masks = binary_masks.squeeze(1).bool()  # Now shape: (N, H, W)
            # Get labels as strings
            labels_str = [str(label.item()) for label in labels]
            # Draw masks and boxes on image
            img_pred = self.draw_boxes_and_masks(
                img_vis, binary_masks, boxes, labels_str
            )

            # Now prepare the ground truth visualization
            gt_boxes = target["boxes"]
            gt_labels = target["labels"]
            gt_masks = target["masks"].bool()
            gt_labels_str = [str(label.item()) for label in gt_labels]
            img_gt = self.draw_boxes_and_masks(
                img_vis, gt_masks, gt_boxes, gt_labels_str
            )

        return img_pred, img_gt

    @staticmethod
    def visualize_training_loss(train_loss_list, output_dir, num_epochs):
        plt.figure()
        plt.plot(range(1, num_epochs + 1), train_loss_list)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss")
        plt.savefig(f"{output_dir}/training_loss_epoch_{num_epochs}.png")
        plt.close()

    @staticmethod
    def draw_boxes_and_masks(img_vis, binary_masks, boxes, labels_str):
        # Draw masks and boxes on image
        img_with_masks = draw_segmentation_masks(
            image=img_vis, masks=binary_masks, alpha=0.5
        )
        img_with_boxes = draw_bounding_boxes(
            image=img_with_masks, boxes=boxes, labels=labels_str
        )
        # Convert to numpy array for plotting
        img_np = img_with_boxes.permute(1, 2, 0).numpy()
        img_np = np.clip(img_np, 0, 255).astype(np.uint8)
        return img_np

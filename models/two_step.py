import os
import wandb
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision.utils import draw_segmentation_masks, draw_bounding_boxes
from segment_anything import (
    sam_model_registry,
    SamPredictor,
)  # Ensure segment_anything is installed

# Placeholder import for AnchorDETR
# You need to replace this with the actual import based on your AnchorDETR implementation
from anchor_detr import AnchorDETR  # Replace with actual AnchorDETR import

NUM_CLASSES = 33

OUTPUT_DIR = "./outputs"
CHECKPOINT_DIR = "./checkpoints"
SAM_MODEL_TYPE = (
    "vit_h"  # Choose the appropriate SAM model type: 'vit_b', 'vit_l', 'vit_h'
)


class TwoStepModel:
    def __init__(
        self,
        num_classes=NUM_CLASSES,
        sam_model_type=SAM_MODEL_TYPE,
        sam_checkpoint_path="path_to_sam_checkpoint.pth",
    ):
        # Initialize AnchorDETR model
        self.object_detector = AnchorDETR(num_classes=num_classes, pretrained=True)

        # Initialize SAM model
        self.sam = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint_path)
        self.sam_predictor = SamPredictor(self.sam)

        # Initialize SAM's neck (assuming it's a part of SAM that can be fine-tuned)
        # Modify this part based on the actual SAM implementation
        self.sam_neck = self.sam.mask_decoder  # Example: using mask_decoder as the neck

    def configure_optimizers(
        self, learning_rate_det=1e-4, learning_rate_sam=1e-4, step_size=7, gamma=0.1
    ):
        # Optimizer for object detector
        det_params = [p for p in self.object_detector.parameters() if p.requires_grad]
        self.optimizer_det = torch.optim.AdamW(
            det_params, lr=learning_rate_det, weight_decay=1e-4
        )
        self.lr_scheduler_det = torch.optim.lr_scheduler.StepLR(
            self.optimizer_det, step_size=step_size, gamma=gamma
        )

        # Optimizer for SAM neck
        sam_neck_params = [p for p in self.sam_neck.parameters() if p.requires_grad]
        self.optimizer_sam = torch.optim.AdamW(
            sam_neck_params, lr=learning_rate_sam, weight_decay=1e-4
        )
        self.lr_scheduler_sam = torch.optim.lr_scheduler.StepLR(
            self.optimizer_sam, step_size=step_size, gamma=gamma
        )

    def training_step_det(self, batch, batch_idx):
        images, targets = batch
        images = list(image.to(self.device) for image in images)
        targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

        # Forward pass
        loss_dict = self.object_detector(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        return losses

    def training_step_sam(self, batch, batch_idx):
        images, targets = batch
        images = list(image.to(self.device) for image in images)
        targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

        # Assuming that SAM's neck is trained using bounding boxes and masks
        # Extract bounding boxes from targets
        for img, target in zip(images, targets):
            boxes = target["boxes"]
            masks = target["masks"]
            # Convert boxes to prompts for SAM
            # This is a simplified example; adjust based on SAM's requirements
            self.sam_predictor.set_image(img.cpu().numpy())
            masks_pred = []
            for box, mask in zip(boxes, masks):
                box = box.cpu().numpy()
                mask = mask.cpu().numpy()
                # Create prompts (e.g., bounding boxes)
                prompt = {"boxes": box[np.newaxis, :]}  # SAM expects a batch of boxes
                # Generate masks using SAM
                masks_pred.append(
                    self.sam_predictor.predict_boxes(boxes=prompt["boxes"])[0]
                )
            # Compute loss between masks_pred and ground truth masks
            # This part depends on how SAM's neck is being trained
            # Placeholder: assume a simple binary cross-entropy loss
            # You need to replace this with the actual loss computation
            loss = self.compute_sam_loss(masks_pred, masks)

        return loss

    def compute_sam_loss(self, masks_pred, masks_gt):
        # Placeholder loss function
        # Replace with actual loss computation based on SAM's training requirements
        loss_fn = torch.nn.BCELoss()
        loss = loss_fn(masks_pred, masks_gt)
        return loss

    def _train_one_epoch_det(self, train_loader, epoch_idx):
        self.object_detector.train()
        epoch_loss = 0
        for batch_idx, batch in enumerate(train_loader):
            self.optimizer_det.zero_grad()
            batch_losses = self.training_step_det(batch, batch_idx)
            batch_losses.backward()
            self.optimizer_det.step()
            epoch_loss += batch_losses.item()
        avg_epoch_loss = epoch_loss / len(train_loader)
        self.lr_scheduler_det.step()
        return avg_epoch_loss

    def _train_one_epoch_sam(self, train_loader, epoch_idx):
        self.sam_neck.train()
        epoch_loss = 0
        for batch_idx, batch in enumerate(train_loader):
            self.optimizer_sam.zero_grad()
            batch_loss = self.training_step_sam(batch, batch_idx)
            batch_loss.backward()
            self.optimizer_sam.step()
            epoch_loss += batch_loss.item()
        avg_epoch_loss = epoch_loss / len(train_loader)
        self.lr_scheduler_sam.step()
        return avg_epoch_loss

    def train(
        self,
        train_loader,
        num_epochs_det=200,
        num_epochs_sam=50,
        learning_rate_det=1e-4,
        learning_rate_sam=1e-4,
        step_size=70,
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
        self.object_detector.to(device)
        self.sam.to(device)

        # Configure optimizers
        self.configure_optimizers(
            learning_rate_det, learning_rate_sam, step_size, gamma
        )

        # Create directories if they don't exist
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        # Lists to store training loss
        self.train_loss_det = []
        self.train_loss_sam = []

        # Stage 1: Train Object Detector
        print("Starting Stage 1: Training Object Detector (AnchorDETR)")
        for epoch in range(num_epochs_det):
            avg_epoch_loss = self._train_one_epoch_det(train_loader, epoch)
            self.train_loss_det.append(avg_epoch_loss)
            print(
                f"Epoch [{epoch+1}/{num_epochs_det}] - Detector Loss: {avg_epoch_loss:.4f}"
            )

            # Save checkpoint
            checkpoint_path = os.path.join(
                checkpoint_dir, f"detector_epoch_{epoch+1}.pth"
            )
            torch.save(self.object_detector.state_dict(), checkpoint_path)

        # Visualize detector training loss
        self.visualize_training_loss(
            self.train_loss_det, output_dir, num_epochs_det, "Detector Training Loss"
        )

        # Stage 2: Train SAM Neck
        print("Starting Stage 2: Training SAM Neck")
        # Freeze object detector parameters
        for param in self.object_detector.parameters():
            param.requires_grad = False
        # Optionally, freeze parts of SAM if needed
        for param in self.sam.parameters():
            param.requires_grad = False
        for param in self.sam_neck.parameters():
            param.requires_grad = True

        for epoch in range(num_epochs_sam):
            avg_epoch_loss = self._train_one_epoch_sam(train_loader, epoch)
            self.train_loss_sam.append(avg_epoch_loss)
            print(
                f"Epoch [{epoch+1}/{num_epochs_sam}] - SAM Neck Loss: {avg_epoch_loss:.4f}"
            )

            # Save checkpoint
            checkpoint_path = os.path.join(
                checkpoint_dir, f"sam_neck_epoch_{epoch+1}.pth"
            )
            torch.save(self.sam_neck.state_dict(), checkpoint_path)

        # Visualize SAM neck training loss
        self.visualize_training_loss(
            self.train_loss_sam, output_dir, num_epochs_sam, "SAM Neck Training Loss"
        )

    def visualize_one_prediction(self, image, target):
        self.object_detector.eval()
        self.sam.eval()

        with torch.no_grad():
            image = image.to(self.device)
            target = {k: v.to(self.device) for k, v in target.items()}

            # Object Detection
            detections = self.object_detector([image])[0]
            boxes = detections["boxes"].cpu()
            labels = detections["labels"].cpu()
            scores = detections["scores"].cpu()
            masks = detections["masks"].cpu()

            # Filter detections with score >= 0.5
            keep = scores >= 0.5
            boxes = boxes[keep]
            labels = labels[keep]
            masks = masks[keep]

            # Prepare SAM prompts (bounding boxes)
            sam_prompts = boxes.numpy()

            # Set image for SAM
            self.sam_predictor.set_image(image.cpu().numpy())
            sam_masks = self.sam_predictor.predict_boxes(sam_prompts)
            sam_masks = sam_masks[
                "masks"
            ]  # Assuming predict_boxes returns a dict with 'masks'

            # Convert masks to binary masks
            binary_masks = sam_masks > 0.5  # Shape: (N, H, W)

            # Get labels as strings
            labels_str = [str(label.item()) for label in labels]

            # Convert image to CPU for visualization
            img_cpu = image.cpu()
            img_vis = img_cpu * 255
            img_vis = img_vis.byte()

            # Draw SAM masks and boxes on image
            img_pred = self.draw_boxes_and_masks(
                img_vis, binary_masks, boxes, labels_str
            )

            # Ground truth visualization
            gt_boxes = target["boxes"].cpu()
            gt_labels = target["labels"].cpu()
            gt_masks = target["masks"].cpu().bool()
            gt_labels_str = [str(label.item()) for label in gt_labels]
            img_gt = self.draw_boxes_and_masks(
                img_vis, gt_masks, gt_boxes, gt_labels_str
            )

        return img_pred, img_gt

    @staticmethod
    def visualize_training_loss(train_loss_list, output_dir, num_epochs, title):
        plt.figure()
        plt.plot(range(1, num_epochs + 1), train_loss_list)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(title)
        plt.savefig(
            os.path.join(
                output_dir, f"{title.replace(' ', '_').lower()}_{num_epochs}.png"
            )
        )
        plt.close()

    @staticmethod
    def draw_boxes_and_masks(img_vis, binary_masks, boxes, labels_str):
        # Draw masks on image
        img_with_masks = draw_segmentation_masks(
            image=img_vis, masks=binary_masks, alpha=0.5, colors="random"
        )
        # Draw bounding boxes on image
        img_with_boxes = draw_bounding_boxes(
            image=img_with_masks, boxes=boxes, labels=labels_str, colors="yellow"
        )
        # Convert to numpy array for plotting
        img_np = img_with_boxes.permute(1, 2, 0).numpy()
        img_np = np.clip(img_np, 0, 255).astype(np.uint8)
        return img_np

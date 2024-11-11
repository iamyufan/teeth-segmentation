import os
import wandb
import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.utils import draw_segmentation_masks


class MaskRCNNModel:
    def __init__(self, num_classes, learning_rate=0.001, step_size=5, gamma=0.1):
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
        self.learning_rate = learning_rate
        self.step_size = step_size
        self.gamma = gamma
        self.optimizer = None
        self.scheduler = None

    def configure_optimizers(self):
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.learning_rate
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=self.step_size, gamma=self.gamma
        )

    def training_step(self, batch, batch_idx):
        images, targets = batch
        loss_dict = self.model(images, targets)
        total_loss = sum(loss for loss in loss_dict.values())
        wandb.log({"training_loss": total_loss.item()})
        return total_loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx, log_images=False):
        images, targets = batch
        # loss_dict = self.model(images, targets)
        # total_loss = sum(loss for loss in loss_dict.values())
        # wandb.log({"validation_loss": total_loss.item()})

        # Log example images, ground truth masks, and predicted masks
        if log_images and batch_idx == 0:
            self.log_images(images, targets)

        # return total_loss

    @torch.no_grad()
    def log_images(self, images, targets):
        # Get predictions
        predictions = self.model(images)

        # Log images with ground truth and predicted masks
        for i, (img, target, prediction) in enumerate(
            zip(images, targets, predictions)
        ):
            img = (img * 255).byte()  # Convert back to byte for visualization
            img = img.permute(1, 2, 0).cpu().numpy()

            # Draw ground truth masks
            gt_masks = target["masks"].cpu().numpy()
            gt_overlay = draw_segmentation_masks(
                torch.tensor(img).permute(2, 0, 1),
                masks=torch.tensor(gt_masks, dtype=torch.bool),
                alpha=0.5,
            )

            # Draw predicted masks
            pred_masks = (prediction["masks"] > 0.5).squeeze(1).cpu()
            pred_overlay = draw_segmentation_masks(
                torch.tensor(img).permute(2, 0, 1),
                masks=pred_masks,
                alpha=0.5,
            )

            wandb.log(
                {
                    f"example_image_{i}": [
                        wandb.Image(gt_overlay, caption="Ground Truth"),
                        wandb.Image(pred_overlay, caption="Predictions"),
                    ]
                }
            )

    def train_model(self, train_loader, val_loader, device, num_epochs=10):
        self.model.to(device)
        self.configure_optimizers()

        # Create a checkpoint directory if it doesn't exist
        os.makedirs("checkpoints", exist_ok=True)

        for epoch in range(num_epochs):
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print("-" * 20)

            # Training Phase
            self.model.train()
            train_loss = 0.0

            for batch_idx, batch in enumerate(train_loader):
                images, targets = batch
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                total_loss = self.training_step((images, targets), batch_idx)
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()
                train_loss += total_loss.item()

                if batch_idx % 10 == 0:
                    print(f"Batch {batch_idx}: Loss = {total_loss.item()}")

            self.scheduler.step()
            print(f"Epoch {epoch + 1} Training Loss: {train_loss / len(train_loader)}")

            # Validation Phase
            self.model.eval()
            # val_loss = 0.0
            for batch_idx, batch in enumerate(val_loader):
                images, targets = batch
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                # val_loss += self.validation_step(
                #     (images, targets), batch_idx, log_images=True
                # )

            # print(f"Epoch {epoch + 1} Validation Loss: {val_loss / len(val_loader)}")
            
            # Save the model checkpoint after each epoch
            torch.save(self.model.state_dict(), f"checkpoints/mask_rcnn_epoch_{epoch + 1}.pth")

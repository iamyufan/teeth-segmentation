import os
import wandb
import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.utils import draw_segmentation_masks


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

        # Lists to store the training loss
        self.train_loss_list = []

        # Training loop
        for epoch in range(self.num_epochs):
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
            self.train_loss_list.append(avg_epoch_loss)

            # Update the learning rate
            self.lr_scheduler.step()
            # Print loss for the epoch
            print(f"Epoch {epoch+1} \t Loss: {avg_epoch_loss}")

            # Save the model checkpoint
            checkpoint_path = f"{checkpoint_dir}/model_epoch_{epoch+1}.pth"
            torch.save(self.model.state_dict(), checkpoint_path)

            # Plot training loss
            plt.figure()
            plt.plot(range(1, epoch + 2), self.train_loss_list)
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title("Training Loss")
            plt.savefig(f"{output_dir}/training_loss_epoch_{epoch+1}.png")
            plt.close()

    def training_step(self, batch, batch_idx):
        images, targets = batch
        images = list(image.to(self.device) for image in images)
        targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

        # Forward pass
        loss_dict = self.model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        return losses
    
    def visualize_one_prediction(self, image, target):
        # Visualize some predictions
        self.model.eval()
        with torch.no_grad():
            for i in range(3):  # Visualize 3 images
                img, target = val_dataset[i]
                img = img.to(device)
                prediction = model([img])[0]

                # Move to CPU for visualization
                img_cpu = img.cpu()
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
                img_with_masks = draw_segmentation_masks(img_vis, binary_masks, alpha=0.5)
                img_with_boxes = draw_bounding_boxes(
                    img_with_masks, boxes, labels=labels_str
                )

                # Convert to numpy array for plotting
                img_np = img_with_boxes.permute(1, 2, 0).numpy()
                img_np = np.clip(img_np, 0, 255).astype(np.uint8)

                # Save the predicted image
                plt.figure(figsize=(12, 8))
                plt.imshow(img_np)
                plt.axis('off')
                plt.title(f'Epoch {epoch+1}, Image {i+1} - Predictions')
                plt.savefig(f'{OUTPUT_DIR}/epoch_{epoch+1}_image_{i+1}_pred.png')
                plt.close()


                # Now prepare the ground truth visualization
                gt_boxes = target['boxes']
                gt_labels = target['labels']
                gt_masks = target['masks']

                # Ensure masks are boolean tensors for visualization
                gt_masks_bool = gt_masks.bool()

                # Get labels as strings
                gt_labels_str = [str(label.item()) for label in gt_labels]

                # Draw ground truth masks and boxes on image
                img_gt_with_masks = draw_segmentation_masks(img_vis.clone(), gt_masks_bool, alpha=0.5)
                img_gt_with_boxes = draw_bounding_boxes(img_gt_with_masks, gt_boxes, labels=gt_labels_str)

                # Convert to numpy array for plotting
                img_gt_np = img_gt_with_boxes.permute(1, 2, 0).numpy()
                img_gt_np = np.clip(img_gt_np, 0, 255).astype(np.uint8)

                # Save the ground truth image
                plt.figure(figsize=(12, 8))
                plt.imshow(img_gt_np)
                plt.axis('off')
                plt.title(f'Epoch {epoch+1}, Image {i+1} - Ground Truth')
                plt.savefig(f'{OUTPUT_DIR}/epoch_{epoch+1}_image_{i+1}_gt.png')
                plt.close()

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
            torch.save(
                self.model.state_dict(), f"checkpoints/mask_rcnn_epoch_{epoch + 1}.pth"
            )

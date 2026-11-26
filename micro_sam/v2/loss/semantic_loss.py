import torch
import torch.nn as nn

from micro_sam.v1.training.semantic_sam_trainer import CustomDiceLoss


class CustomCombinedLoss(nn.Module):
    """Weighted dice and cross entropy loss for semantic segmentation.

    The loss is 'dice_weight * dice + (1 - dice_weight) * cross_entropy'. The dice term is the multi-class
    dice of `micro_sam.v1.training.semantic_sam_trainer.CustomDiceLoss`.

    The predictions must be the raw class logits. The dice term applies a softmax and the cross entropy term
    applies a log softmax, so a model with a final activation would activate its predictions twice.

    Args:
        num_classes: The number of semantic classes, the background class included.
        dice_weight: The weight of the dice loss. One selects dice only. Zero selects cross entropy only.
    """
    def __init__(self, num_classes: int, dice_weight: float = 0.5):
        super().__init__()
        if not 0.0 <= dice_weight <= 1.0:
            raise ValueError(f"The dice weight is '{dice_weight}'. It must lie between zero and one.")

        self.dice_weight = dice_weight
        self.dice_loss = CustomDiceLoss(num_classes=num_classes)
        self.ce_loss = nn.CrossEntropyLoss()

        self.init_kwargs = {"num_classes": num_classes, "dice_weight": dice_weight}

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute the combined loss between the class logits and the semantic map.

        Args:
            pred: The class logits of shape (B, num_classes, Z, Y, X).
            target: The semantic map of shape (B, 1, Z, Y, X), holding the class ids.

        Returns:
            The combined loss.
        """
        target = target.to(pred.device, non_blocking=True)
        dice_loss = self.dice_loss(pred, target)
        ce_loss = self.ce_loss(pred, target.squeeze(1).long())
        return self.dice_weight * dice_loss + (1 - self.dice_weight) * ce_loss

"""
Logistic (softmax) regression model for the AMDNet23 classification project.

Used as a skorch module in the cross-validation main file, in the same way
as the CNN from CNN.py.

Based on the model in the MachineLearningMastery tutorial "Building a Logistic
Regression Classifier in PyTorch"
(https://machinelearningmastery.com/building-a-logistic-regression-classifier-in-pytorch/),
with the corrections from its reader comments:
    - forward() returns raw logits, without sigmoid. nn.CrossEntropyLoss
      applies log-softmax internally, so a sigmoid before it computed
      softmax(sigmoid(x)) and gave worse results.
    - nn.CrossEntropyLoss is kept (not BCEWithLogitsLoss), because the task
      is single-label multi-class. With more than two classes this is
      multinomial (softmax) logistic regression.
"""

import torch
import torch.nn as nn


class LogisticRegression(nn.Module):
    """
    Single linear layer mapping a flattened image to one logit per class.

    Date: 2026-11-25.
    Written by: Claude Opus 5.5 (Anthropic), based on the MachineLearningMastery
    tutorial model with the sigmoid removed as pointed out in its comments.
    Level: TODO - set after testing and validating it yourself.

    Args:
        n_inputs: Number of input features, i.e. channels * height * width of
            one preprocessed image (3 * 256 * 256 = 196608 for 256x256 RGB).
        n_outputs: Number of classes.
    """

    def __init__(self, n_inputs: int = 3 * 256 * 256, n_outputs: int = 4) -> None:
        # Date: 2026-11-25. Written by: Claude Opus 5.5 (Anthropic).
        # Level: TODO - set after testing and validating it yourself.
        super().__init__()
        self.linear = nn.Linear(n_inputs, n_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Date: 2026-11-25. Written by: Claude Opus 5.5 (Anthropic).
        # Level: TODO - set after testing and validating it yourself.
        # Flatten (N, C, H, W) -> (N, C*H*W) and return RAW logits. No sigmoid
        # or softmax here: nn.CrossEntropyLoss applies log-softmax itself, and
        # skorch applies softmax in predict_proba when it sees that loss.
        x = torch.flatten(x, start_dim=1)
        return self.linear(x)
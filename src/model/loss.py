# Standard library imports
from typing import Annotated

# Third-party imports
import torch
import torch.nn as nn
import torch.nn.functional as F


class AMSoftmaxLoss(nn.Module):
    """
    Additive Margin Softmax Loss module.

    AMSoftmax introduces a margin term in the softmax space to enforce
    a larger separation between classes, often applied in tasks like
    face recognition or speaker recognition. It modifies the logits
    by subtracting a margin from the logit of the true class, and then
    scales them by a factor.

    Parameters
    ----------
    embed_dim : int
        Dimension of the input embeddings.
    n_classes : int
        Number of classes or targets.
    margin : float, optional
        Margin to subtract from the logit of the true class. Defaults
        to 0.3.
    scale : float, optional
        Scaling factor applied to the logits after margin subtraction.
        Defaults to 30.0.

    Attributes
    ----------
    embed_dim : int
        Dimension of the input embeddings.
    n_classes : int
        Number of classes or targets.
    margin : float
        Margin subtracted from the logits of the target class.
    scale : float
        Factor used to scale the logits after margin subtraction.
    weight : torch.nn.Parameter
        Learnable weight matrix of shape (n_classes, embed_dim).

    Examples
    --------
    >>> import torch
    >>> from src.model.loss import AMSoftmaxLoss  # doctest: +SKIP
    >>> am_softmax = AMSoftmaxLoss(embed_dim=64, n_classes=5, margin=0.2, scale=32.0)
    >>> embeddings = torch.randn(4, 64)
    >>> targets = torch.randint(low=0, high=5, size=(4,))
    >>> loss, logits = am_softmax(embeddings, targets)
    >>> loss.item()  # doctest: +SKIP
    1.609...
    >>> logits.shape
    torch.Size([4, 5])
    """

    def __init__(
            self,
            embed_dim: Annotated[int, "Dimension of the input embeddings"],
            n_classes: Annotated[int, "Number of output classes"],
            margin: Annotated[float, "Margin for AMSoftmax"] = 0.3,
            scale: Annotated[float, "Scale for AMSoftmax"] = 30.0
    ) -> None:
        """
        Initialize the AMSoftmaxLoss module.

        Raises
        ------
        TypeError
            If any provided parameter has an incorrect type.
        ValueError
            If embed_dim or n_classes is non-positive.
        """
        super().__init__()

        if not isinstance(embed_dim, int):
            raise TypeError("Expected 'embed_dim' to be an integer.")
        if not isinstance(n_classes, int):
            raise TypeError("Expected 'n_classes' to be an integer.")
        if not isinstance(margin, (int, float)):
            raise TypeError("Expected 'margin' to be a float or int.")
        if not isinstance(scale, (int, float)):
            raise TypeError("Expected 'scale' to be a float or int.")

        if embed_dim <= 0:
            raise ValueError("'embed_dim' must be > 0.")
        if n_classes <= 0:
            raise ValueError("'n_classes' must be > 0.")
        if margin < 0:
            raise ValueError("'margin' must be >= 0.")
        if scale <= 0:
            raise ValueError("'scale' must be > 0.")

        self.embed_dim = embed_dim
        self.n_classes = n_classes
        self.margin = margin
        self.scale = scale

        self.weight = nn.Parameter(torch.Tensor(n_classes, embed_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(
            self,
            embeddings: Annotated[torch.Tensor, "(B, embed_dim) Input embeddings"],
            targets: Annotated[torch.Tensor, "(B,) Target class labels"]
    ) -> Annotated[
        tuple, "A tuple (loss, logits) where loss is a scalar and logits is (B, n_classes)"
    ]:
        """
        Compute the AMSoftmax loss and scaled logits.

        Parameters
        ----------
        embeddings : torch.Tensor
            Input embedding tensor of shape (batch_size, embed_dim).
        targets : torch.Tensor
            Target labels of shape (batch_size,).

        Returns
        -------
        tuple
            (loss, logits_scaled):

            - loss is a scalar representing the cross-entropy loss
              after margin and scale operations.
            - logits_scaled is the tensor of shape (batch_size, n_classes)
              used in the cross-entropy calculation.

        Examples
        --------
        >>> am_softmax = AMSoftmaxLoss(embed_dim=64, n_classes=5, margin=0.2, scale=32.0)
        >>> embeddings_test = torch.randn(4, 64)
        >>> targets_test = torch.randint(low=0, high=5, size=(4,))
        >>> loss_test, logits_scaled_test = am_softmax(embeddings_test, targets_test)
        >>> loss_test.item()  # doctest: +SKIP
        1.609...
        >>> logits_scaled_test.shape
        torch.Size([4, 5])
        """
        if not isinstance(embeddings, torch.Tensor):
            raise TypeError("Expected 'embeddings' to be a torch.Tensor.")
        if not isinstance(targets, torch.Tensor):
            raise TypeError("Expected 'targets' to be a torch.Tensor.")
        if embeddings.dim() != 2:
            raise ValueError("Expected 'embeddings' to have 2 dimensions (B, embed_dim).")
        if targets.dim() != 1:
            raise ValueError("Expected 'targets' to have 1 dimension (B,).")

        w_norm = F.normalize(self.weight)
        x_norm = F.normalize(embeddings)

        logits = torch.matmul(x_norm, w_norm.transpose(0, 1))

        one_hot = torch.zeros_like(logits)
        one_hot.scatter_(1, targets.view(-1, 1), 1.0)

        logits_with_margin = logits - one_hot * self.margin

        logits_scaled = logits_with_margin * self.scale

        loss = F.cross_entropy(logits_scaled, targets)
        return loss, logits_scaled


if __name__ == "__main__":
    am_softmax_loss_test_model = AMSoftmaxLoss(
        embed_dim=64,
        n_classes=5,
        margin=0.2,
        scale=32.0
    )

    embeddings_test_input = torch.randn(4, 64)
    targets_test_input = torch.randint(low=0, high=5, size=(4,))

    loss_test_output, logits_test_output = am_softmax_loss_test_model(
        embeddings_test_input,
        targets_test_input
    )

    print("AMSoftmaxLoss Loss:", loss_test_output.item())
    print("AMSoftmaxLoss Logits Shape:", logits_test_output.shape)

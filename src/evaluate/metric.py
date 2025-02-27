# Standard library imports
from typing import Annotated

# Third-party imports
import numpy as np


class Metrics:
    """
    Metrics class for evaluating binary classification performance in
    speaker verification tasks.

    This class provides static methods to compute Equal Error Rate (EER)
    and minimum Detection Cost Function (minDCF). The typical use cases
    involve threshold-based decisions, where `scores` represents similarity
    or likelihood, and `labels` indicates ground truth (1 for same speaker,
    0 for different speakers).

    Methods
    -------
    eer(scores, labels)
        Computes the Equal Error Rate (EER), where false acceptance rate (FPR)
        equals false rejection rate (FNR).
    mindcf(scores, labels, p_target=0.01, c_miss=1, c_fa=1)
        Computes the minimum Detection Cost Function (minDCF) over all
        possible score thresholds.

    Examples
    --------
    >>> scores = np.array([0.9, 0.8, 0.75, 0.2, 0.1])
    >>> labels = np.array([1, 1, 1, 0, 0])
    >>> eer_value = Metrics.eer(scores, labels)
    >>> min_dcf_value = Metrics.mindcf(scores, labels)
    >>> print(eer_value, min_dcf_value)
    0.0 0.0
    """

    @staticmethod
    def eer(
            scores: Annotated[np.ndarray, "Array of scores"],
            labels: Annotated[np.ndarray, "Array of binary labels"]
    ) -> Annotated[float, "Equal Error Rate"]:
        """
        Compute the Equal Error Rate (EER).

        The EER is the operating point on the ROC curve where the
        false positive rate (FPR) equals the false negative rate (FNR).
        Here, we treat label=1 as "positive" (same speaker) and label=0
        as "negative" (different speaker).

        Parameters
        ----------
        scores : np.ndarray
            A 1D array of predicted scores or similarity values.
        labels : np.ndarray
            A 1D array of binary labels (1 for same, 0 for different).

        Returns
        -------
        float
            The computed EER value, ranging between 0 and 1.

        Raises
        ------
        TypeError
            If scores or labels is not a numpy array.
        ValueError
            If scores and labels have different lengths.

        Examples
        --------
        >>> test_scores = np.array([0.9, 0.1, 0.5])
        >>> test_labels = np.array([1, 0, 1])
        >>> Metrics.eer(test_scores, test_labels)
        0.3333333333333333
        """
        if not isinstance(scores, np.ndarray):
            raise TypeError("Expected 'scores' to be a numpy ndarray.")
        if not isinstance(labels, np.ndarray):
            raise TypeError("Expected 'labels' to be a numpy ndarray.")
        if scores.shape[0] != labels.shape[0]:
            raise ValueError("'scores' and 'labels' must have the same length.")

        sort_idx = np.argsort(scores)[::-1]
        sorted_labels = labels[sort_idx]

        p = np.sum(sorted_labels == 1)
        n = np.sum(sorted_labels == 0)

        tpr = np.cumsum(sorted_labels == 1) / (p + 1e-8)
        fpr = np.cumsum(sorted_labels == 0) / (n + 1e-8)

        fnr = 1.0 - tpr

        diff = np.abs(fnr - fpr)
        min_index = np.argmin(diff)

        eer_value = (fpr[min_index] + fnr[min_index]) / 2
        return eer_value

    @staticmethod
    def mindcf(
            scores: Annotated[np.ndarray, "Array of scores"],
            labels: Annotated[np.ndarray, "Array of binary labels"],
            p_target: Annotated[float, "Prior probability of target"] = 0.01,
            c_miss: Annotated[float, "Cost of a miss"] = 1,
            c_fa: Annotated[float, "Cost of a false alarm"] = 1
    ) -> Annotated[float, "Minimum detection cost function"]:
        """
        Compute the minimum Detection Cost Function (minDCF).

        The detection cost function is calculated as:

        .. math::
            DCF = C_{miss} \\cdot FN\_rate \\cdot p\_target +
                  C_{fa} \\cdot FP\_rate \\cdot (1 - p\_target)

        This function searches for the threshold that minimizes the DCF
        across all possible score thresholds.

        Parameters
        ----------
        scores : np.ndarray
            A 1D array of predicted scores or similarity values.
        labels : np.ndarray
            A 1D array of binary labels (1 for same, 0 for different).
        p_target : float, optional
            Prior probability of the target (label=1). Defaults to 0.01.
        c_miss : float, optional
            Cost of a miss (false negative). Defaults to 1.
        c_fa : float, optional
            Cost of a false alarm (false positive). Defaults to 1.

        Returns
        -------
        float
            The minimum DCF value found, ranging between 0 and 1 or more,
            depending on the cost configuration.

        Raises
        ------
        TypeError
            If scores or labels is not a numpy array or any cost parameter
            is not float or int.
        ValueError
            If scores and labels have different lengths or cost parameters
            are negative.

        Examples
        --------
        >>> scores_test = np.array([0.2, 0.8, 0.5, 0.7])
        >>> labels_test = np.array([0, 1, 0, 1])
        >>> min_dcf_val = Metrics.mindcf(scores_test, labels_test)
        >>> min_dcf_val
        0.006...
        """
        if not isinstance(scores, np.ndarray):
            raise TypeError("Expected 'scores' to be a numpy ndarray.")
        if not isinstance(labels, np.ndarray):
            raise TypeError("Expected 'labels' to be a numpy ndarray.")
        if scores.shape[0] != labels.shape[0]:
            raise ValueError("'scores' and 'labels' must have the same length.")
        if not isinstance(p_target, (int, float)):
            raise TypeError("Expected 'p_target' to be a float or int.")
        if not isinstance(c_miss, (int, float)):
            raise TypeError("Expected 'c_miss' to be a float or int.")
        if not isinstance(c_fa, (int, float)):
            raise TypeError("Expected 'c_fa' to be a float or int.")
        if p_target < 0:
            raise ValueError("'p_target' must be >= 0.")
        if c_miss < 0:
            raise ValueError("'c_miss' must be >= 0.")
        if c_fa < 0:
            raise ValueError("'c_fa' must be >= 0.")

        sort_idx = np.argsort(scores)[::-1]
        sorted_labels = labels[sort_idx]

        p = np.sum(sorted_labels == 1)
        n = np.sum(sorted_labels == 0)

        min_cost = float('inf')
        tp = 0
        fp = 0

        for i in range(len(scores)):
            if sorted_labels[i] == 1:
                tp += 1
            else:
                fp += 1

            fn = p - tp
            _ = n - fp

            fnr = fn / (p + 1e-8)
            fpr = fp / (n + 1e-8)

            cost = c_miss * fnr * p_target + c_fa * fpr * (1 - p_target)
            if cost < min_cost:
                min_cost = cost

        return min_cost


if __name__ == "__main__":
    scores_perfect = np.array([0.9, 0.8, 0.75, 0.2, 0.1])
    labels_perfect = np.array([1, 1, 1, 0, 0])

    eer_perfect = Metrics.eer(scores_perfect, labels_perfect)
    dcf_perfect = Metrics.mindcf(scores_perfect, labels_perfect)

    print("CASE 1: Perfect separation")
    print(f"EER     : {eer_perfect:.4f}")
    print(f"minDCF  : {dcf_perfect:.4f}\n")

    scores_mixed = np.array([0.1, 0.9, 0.4, 0.7, 0.2])
    labels_mixed = np.array([0, 1, 0, 1, 0])

    eer_mixed = Metrics.eer(scores_mixed, labels_mixed)
    dcf_mixed = Metrics.mindcf(scores_mixed, labels_mixed)

    print("CASE 2: Mixed separation (but still perfect when sorted)")
    print(f"EER     : {eer_mixed:.4f}")
    print(f"minDCF  : {dcf_mixed:.4f}\n")

    scores_overlap = np.array([0.2, 0.8, 0.5, 0.6, 0.3, 0.55])
    labels_overlap = np.array([0, 1, 0, 1, 1, 0])

    eer_overlap = Metrics.eer(scores_overlap, labels_overlap)
    dcf_overlap = Metrics.mindcf(scores_overlap, labels_overlap)

    print("CASE 3: Overlapping scenario")
    print(f"EER     : {eer_overlap:.4f}")
    print(f"minDCF  : {dcf_overlap:.4f}\n")

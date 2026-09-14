"""DeepSeaNet: underwater object detection on the Brackish dataset.

Reference implementation of the components described in

    Sanyam Jain, "DeepSeaNet: Improving Underwater Object Detection using
    EfficientDet", ICAPAI 2024, IEEE. doi:10.1109/ICAPAI61893.2024.10541265

The package is deliberately small and dependency-light. It provides:

``deepseanet.models``
    Swish activation, MBConv backbone, the BiFPN and proposed BiSkFPN necks,
    the classification/regression head and the assembled detector.
``deepseanet.losses``
    Multi-class focal loss, smooth-L1 box loss, L2 weight penalty and the
    combined objective ``L = L_cls + a*L_box + b*L_reg``.
``deepseanet.attacks``
    Universal adversarial perturbations and the curriculum used for
    adversarial training.
``deepseanet.explain``
    GradCAM++ class activation maps.
``deepseanet.data``
    Brackish preprocessing, annotation format conversion and dataset splits.

See ``docs/REPRODUCIBILITY.md`` for how this code relates to the numbers in
the paper and to the notebooks committed under ``notebooks/``.
"""

from __future__ import annotations

__version__ = "2.0.0"
__author__ = "Sanyam Jain"

BRACKISH_CLASSES = (
    "fish",
    "small_fish",
    "crab",
    "shrimp",
    "jellyfish",
    "starfish",
)
"""Object classes carried by the Brackish bounding boxes, in label-id order."""

__all__ = ["BRACKISH_CLASSES", "__version__", "__author__"]

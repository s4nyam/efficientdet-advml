"""Box maths: conversions, IoU, encode/decode round-trip and NMS."""

from __future__ import annotations

import torch

from deepseanet.utils.boxes import (
    batched_nms,
    box_iou,
    cxcywh_to_xyxy,
    decode_boxes,
    encode_boxes,
    nms,
    xyxy_to_cxcywh,
)


def test_format_conversion_round_trips():
    boxes = torch.tensor([[10.0, 20.0, 50.0, 80.0], [0.0, 0.0, 4.0, 4.0]])
    assert torch.allclose(cxcywh_to_xyxy(xyxy_to_cxcywh(boxes)), boxes, atol=1e-6)


def test_iou_of_identical_boxes_is_one():
    boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    assert torch.allclose(box_iou(boxes, boxes), torch.ones(1, 1), atol=1e-5)


def test_iou_of_disjoint_boxes_is_zero():
    a = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
    b = torch.tensor([[5.0, 5.0, 6.0, 6.0]])
    assert float(box_iou(a, b)) == 0.0


def test_iou_half_overlap():
    a = torch.tensor([[0.0, 0.0, 2.0, 1.0]])
    b = torch.tensor([[1.0, 0.0, 3.0, 1.0]])
    # intersection 1, union 3
    assert abs(float(box_iou(a, b)) - 1 / 3) < 1e-5


def test_encode_decode_round_trips():
    anchors = torch.tensor([[0.0, 0.0, 32.0, 32.0], [10.0, 10.0, 26.0, 42.0]])
    boxes = torch.tensor([[2.0, 3.0, 30.0, 29.0], [12.0, 14.0, 24.0, 38.0]])
    assert torch.allclose(decode_boxes(encode_boxes(boxes, anchors), anchors), boxes, atol=1e-4)


def test_nms_keeps_highest_score_and_drops_overlaps():
    boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0], [1.0, 1.0, 11.0, 11.0], [50.0, 50.0, 60.0, 60.0]])
    scores = torch.tensor([0.9, 0.8, 0.7])
    keep = nms(boxes, scores, iou_threshold=0.5)
    assert keep.tolist() == [0, 2]


def test_nms_on_empty_input():
    keep = nms(torch.zeros(0, 4), torch.zeros(0))
    assert keep.numel() == 0


def test_batched_nms_does_not_suppress_across_classes():
    boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0], [0.0, 0.0, 10.0, 10.0]])
    scores = torch.tensor([0.9, 0.8])
    same = batched_nms(boxes, scores, torch.tensor([0, 0]))
    diff = batched_nms(boxes, scores, torch.tensor([0, 1]))
    assert same.numel() == 1
    assert diff.numel() == 2

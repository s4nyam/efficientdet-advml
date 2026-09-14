"""Command-line interface: ``deepseanet <command>``.

Run ``deepseanet --help`` for the full list. The commands exist so the common
operations - preparing the dataset, converting annotations, inspecting a model,
summarising the committed runs - can be done without opening a notebook.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import BRACKISH_CLASSES, __version__

__all__ = ["main", "build_parser"]


def _cmd_prepare(args: argparse.Namespace) -> int:
    from .data import (
        class_distribution,
        drop_empty_labels,
        normalize_labels,
        pair_images_and_labels,
    )

    pairs, unmatched = pair_images_and_labels(args.images, args.labels)
    print(f"paired frames        : {len(pairs)}")
    print(f"images with no label : {len(unmatched)}")

    kept = drop_empty_labels(pairs, delete=args.delete)
    print(f"frames with boxes    : {len(kept)}")
    print(
        f"frames with no boxes : {len(pairs) - len(kept)}" + ("  (deleted)" if args.delete else "")
    )

    if args.normalize:
        n = normalize_labels(
            args.labels, width=args.width, height=args.height, class_offset=args.class_offset
        )
        print(f"labels normalized    : {n}")

    dist = class_distribution(args.labels, BRACKISH_CLASSES)
    print("\nboxes per class:")
    for name, count in dist.items():
        print(f"  {name:<12} {count:>7}")
    print(f"  {'total':<12} {sum(dist.values()):>7}")
    return 0


def _cmd_convert(args: argparse.Namespace) -> int:
    from .data import coco_to_yolo, yolo_to_coco, yolo_to_voc

    names = args.classes.split(",") if args.classes else list(BRACKISH_CLASSES)
    if args.to == "coco":
        coco = yolo_to_coco(args.images, args.labels, args.output, names)
        print(
            f"wrote {args.output}: {len(coco['images'])} images, {len(coco['annotations'])} boxes"
        )
    elif args.to == "voc":
        n = yolo_to_voc(args.images, args.labels, args.output, names)
        print(f"wrote {n} XML files to {args.output}")
    elif args.to == "yolo":
        n = coco_to_yolo(args.input, args.output)
        print(f"wrote {n} label files to {args.output}")
    return 0


def _cmd_split(args: argparse.Namespace) -> int:
    from .data import (
        grouped_split,
        materialize_split,
        pair_images_and_labels,
        random_split,
        video_id_of,
        write_data_yaml,
    )

    pairs, _ = pair_images_and_labels(args.images, args.labels)
    ratios = (args.train, args.val, args.test)
    if args.strategy == "grouped":
        split = grouped_split(pairs, lambda p: video_id_of(p.image), ratios, seed=args.seed)
    else:
        split = random_split(pairs, ratios, seed=args.seed)

    print(f"strategy: {args.strategy}   seed: {args.seed}")
    for name, count in split.sizes.items():
        print(f"  {name:<6} {count:>7}")

    if args.output:
        written = materialize_split(split, args.output, copy=not args.link)
        yaml_path = write_data_yaml(Path(args.output) / "data.yaml", BRACKISH_CLASSES, root=".")
        print(f"\nwrote {sum(written.values())} frames to {args.output}")
        print(f"wrote {yaml_path}")
    return 0


def _cmd_model(args: argparse.Namespace) -> int:
    try:
        import torch
    except ImportError:
        print("this command needs PyTorch: pip install 'deepseanet[torch]'", file=sys.stderr)
        return 1
    from .models import DeepSeaNet, DeepSeaNetConfig

    cfg = DeepSeaNetConfig(
        num_classes=args.num_classes, phi=args.phi, neck=args.neck, image_size=args.image_size
    )
    model = DeepSeaNet(cfg).eval()
    breakdown = model.parameter_breakdown()

    print(
        f"DeepSeaNet  neck={args.neck}  phi={args.phi}  input={args.image_size}x{args.image_size}"
    )
    print(f"{'component':<12} {'params':>14}")
    for name, count in breakdown.items():
        print(f"{name:<12} {count:>14,}")
    print(f"{'total':<12} {model.num_parameters():>14,}")

    if args.forward:
        with torch.no_grad():
            cls_logits, box_deltas, anchors = model(
                torch.randn(1, 3, args.image_size, args.image_size)
            )
        print(
            f"\nforward: cls {tuple(cls_logits.shape)}  box {tuple(box_deltas.shape)}  "
            f"anchors {tuple(anchors.shape)}"
        )
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    from .evaluate import collect_runs, format_table

    summaries = collect_runs(args.results)
    if args.json:
        print(json.dumps([s.as_row() for s in summaries], indent=2))
    else:
        print(format_table(summaries))
    return 0


def _cmd_verify(args: argparse.Namespace) -> int:
    import hashlib

    manifest = json.loads(Path(args.manifest).read_text())
    root = Path(args.root)
    failures = 0
    for entry in manifest["files"]:
        path = root / entry["path"]
        if not path.exists():
            print(f"MISSING  {entry['path']}")
            failures += 1
            continue
        digest = hashlib.md5(path.read_bytes()).hexdigest()
        if digest == entry["md5"]:
            print(f"OK       {entry['path']}  ({entry.get('note', '')})".rstrip())
        else:
            print(f"MISMATCH {entry['path']}  expected {entry['md5']}, got {digest}")
            failures += 1
    return 1 if failures else 0


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser."""
    parser = argparse.ArgumentParser(
        prog="deepseanet",
        description="DeepSeaNet: underwater object detection on the Brackish dataset.",
    )
    parser.add_argument("--version", action="version", version=f"deepseanet {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("prepare", help="pair, clean and normalise a frames/labels directory pair")
    p.add_argument("--images", required=True, help="directory of extracted frames")
    p.add_argument("--labels", required=True, help="directory of YOLO .txt label files")
    p.add_argument("--normalize", action="store_true", help="rescale pixel coords to [0, 1]")
    p.add_argument("--width", type=int, default=960)
    p.add_argument("--height", type=int, default=540)
    p.add_argument("--class-offset", type=int, default=0, help="use -1 for one-based class ids")
    p.add_argument("--delete", action="store_true", help="delete frames that have no boxes")
    p.set_defaults(func=_cmd_prepare)

    p = sub.add_parser("convert", help="convert between YOLO, COCO and Pascal VOC")
    p.add_argument("--to", choices=["coco", "voc", "yolo"], required=True)
    p.add_argument("--images", help="image directory (YOLO source)")
    p.add_argument("--labels", help="label directory (YOLO source)")
    p.add_argument("--input", help="COCO json (when --to yolo)")
    p.add_argument("--output", required=True)
    p.add_argument("--classes", help="comma-separated class names")
    p.set_defaults(func=_cmd_convert)

    p = sub.add_parser("split", help="build train/val/test splits")
    p.add_argument("--images", required=True)
    p.add_argument("--labels", required=True)
    p.add_argument(
        "--strategy",
        choices=["grouped", "random"],
        default="grouped",
        help="grouped keeps whole source videos on one side (recommended)",
    )
    p.add_argument("--train", type=float, default=0.7)
    p.add_argument("--val", type=float, default=0.2)
    p.add_argument("--test", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", help="write the split to disk here")
    p.add_argument("--link", action="store_true", help="hard-link instead of copying")
    p.set_defaults(func=_cmd_split)

    p = sub.add_parser("model", help="inspect the DeepSeaNet architecture")
    p.add_argument("--neck", choices=["biskfpn", "bifpn"], default="biskfpn")
    p.add_argument("--phi", type=int, default=0)
    p.add_argument("--num-classes", type=int, default=6)
    p.add_argument("--image-size", type=int, default=512)
    p.add_argument("--forward", action="store_true", help="also run one forward pass")
    p.set_defaults(func=_cmd_model)

    p = sub.add_parser("report", help="summarise the committed training logs")
    p.add_argument("--results", default="results", help="results directory")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=_cmd_report)

    p = sub.add_parser("verify", help="check downloaded checkpoints against the manifest")
    p.add_argument("--manifest", default="results/checkpoints.json")
    p.add_argument("--root", default=".")
    p.set_defaults(func=_cmd_verify)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

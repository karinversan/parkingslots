from __future__ import annotations

import argparse
import csv
from pathlib import Path

from parking_vision.data.layouts import Layout, Slot, save_layout


def main():
    parser = argparse.ArgumentParser(description="Convert a simple CSV slot file into project JSON layout.")
    parser.add_argument("--input-csv", required=True, help="CSV columns: slot_id,x1,y1,x2,y2,x3,y3,x4,y4")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--camera-id", default="camera-01")
    parser.add_argument("--image-width", type=int, required=True)
    parser.add_argument("--image-height", type=int, required=True)
    args = parser.parse_args()

    slots = []
    with Path(args.input_csv).open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            polygon = [
                [int(row["x1"]), int(row["y1"])],
                [int(row["x2"]), int(row["y2"])],
                [int(row["x3"]), int(row["y3"])],
                [int(row["x4"]), int(row["y4"])],
            ]
            slots.append(Slot(slot_id=row["slot_id"], polygon=polygon))

    layout = Layout(
        camera_id=args.camera_id,
        image_width=args.image_width,
        image_height=args.image_height,
        slots=slots,
    )
    save_layout(layout, args.output_json)
    print({"output_json": args.output_json, "num_slots": len(slots)})


if __name__ == "__main__":
    main()

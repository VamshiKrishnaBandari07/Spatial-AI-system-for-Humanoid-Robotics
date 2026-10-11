"""Human-review crops from computed observations; never change annotations."""

import html
import json
from pathlib import Path
import cv2
import numpy as np
from .association import semantic_label


def create_object_review(output):
    output = Path(output)
    destination = output / "visualization" / "object_review"
    destination.mkdir(parents=True, exist_ok=True)
    cards, records = [], []
    for state in ("before", "after"):
        source = output / f"{state}_model_outputs.json"
        if not source.is_file():
            continue
        for row in json.loads(source.read_text()):
            frame = str(row["frame_id"]).split(":")[-1]
            if not frame.isdigit():
                raise ValueError("Invalid observation frame identifier")
            detections = row["detections"]
            for index, (box, label, score) in enumerate(
                zip(detections["boxes"], detections["labels"], detections["raw_scores"])
            ):
                observation = f"{state}_{frame}_{index}"
                evidence = (
                    output / "evidence" / "supporting_frames" / f"{observation}.png"
                )
                if not evidence.is_file():
                    continue
                image = cv2.imread(str(evidence))
                if image is None:
                    raise ValueError(f"Unreadable evidence: {observation}")
                h, w = image.shape[:2]
                x0, y0, x1, y1 = np.rint(
                    np.clip(box, [0, 0, 0, 0], [w, h, w, h])
                ).astype(int)
                if x1 <= x0 or y1 <= y0:
                    continue
                name = semantic_label(label)
                clear = name in {
                    "chair",
                    "table",
                    "cabinet",
                    "box",
                    "bag",
                    "backpack",
                    "bottle",
                    "cup",
                    "bowl",
                    "plate",
                    "pan",
                    "kettle",
                    "plant",
                    "laptop",
                    "monitor",
                    "phone",
                    "book",
                    "tissue box",
                    "toiletry",
                    "toiletry bottle",
                }
                display = name.title() if clear else "Unidentified object"
                filename = f"{observation}.jpg"
                if not cv2.imwrite(str(destination / filename), image[y0:y1, x0:x1]):
                    raise ValueError("Could not save object crop")
                record = {
                    "observation_id": observation,
                    "state": state,
                    "frame_id": row["frame_id"],
                    "display_name": display,
                    "raw_label": label,
                    "raw_detector_score": score,
                    "identity_verified": False,
                    "crop": filename,
                }
                records.append(record)
                cards.append(
                    f'<article><h3>{html.escape(display)}</h3><img src="{filename}" alt="Object crop"><p>{state} · {html.escape(observation)}</p><p>Model proposal: {html.escape(str(label))}</p></article>'
                )
    (destination / "observations.json").write_text(json.dumps(records, indent=2))
    page = (
        '<!doctype html><meta charset="utf-8"><title>Object review</title><style>body{font-family:system-ui;background:#101820;color:#eee;padding:24px}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:16px}article{background:#203040;padding:16px;border-radius:12px}img{width:100%;height:200px;object-fit:contain}p{overflow-wrap:anywhere}</style><h1>Object observations</h1><p>Names are model proposals. Crops are observations, not unique verified identities or confirmed changes. Ambiguous categories remain unidentified. Undetected objects are not represented.</p><main>'
        + "".join(cards)
        + "</main>"
    )
    (destination / "index.html").write_text(page)
    return destination / "index.html"

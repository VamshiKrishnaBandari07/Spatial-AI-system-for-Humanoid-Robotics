"""Source-audited partial evaluation and private visual artifacts for the approved pair."""

import argparse
from collections import Counter
import json
from pathlib import Path
import numpy as np

from spatialmemory.benchmarking import sha256
from spatialmemory.evaluation import classification_metrics
from spatialmemory.native_review import review_native, read_ascii_instances
from spatialmemory.rgbd_validation import transform_points


def evaluate(review, report):
    if report["status"] != "complete" or not report["registration"]["valid"]:
        raise ValueError("Successful gated inference required before evaluation")
    if report["scene_id"] != review["scene_id"]:
        raise ValueError("Evaluation scene mismatch")
    events = [e for e in report["events"] if e.get("object_id")]
    rows = []
    for obj in review["objects"]:
        if obj["change_type"] == "UNKNOWN":
            continue
        aliases = {"trash can": "trash", "trash": "trash"}
        label = aliases.get(obj["label"], obj["label"])
        candidates = []
        for event in events:
            if aliases.get(event["semantic_label"], event["semantic_label"]) != label:
                continue
            position = event.get("before_position") or event.get("after_position")
            reference = obj.get("before_position") or obj.get("after_position")
            if (
                position is not None
                and reference is not None
                and np.linalg.norm(np.asarray(position) - reference) <= 0.25
            ):
                candidates.append(event)
        # Ambiguous identities or abstentions cannot become verified positive detections.
        prediction = candidates[0]["change"] if len(candidates) == 1 else "UNKNOWN"
        rows.append(
            {
                "native_object_id": obj["object_id"],
                "truth": obj["change_type"],
                "prediction": prediction,
                "candidate_events": [e["id"] for e in candidates],
                "mapping_status": "unique gated semantic/location match"
                if len(candidates) == 1
                else "unmatched or ambiguous; abstained",
            }
        )
    if not rows:
        raise ValueError("No explicitly supported canonical evaluation labels")
    result = classification_metrics(
        [r["truth"] for r in rows], [r["prediction"] for r in rows]
    )
    result.update(
        {
            "scope": "Positive-object classification on the explicitly supported native MOVED subset only; not full inventory detection precision or a general benchmark",
            "native_split": review["native_split"],
            "source_transform_assisted": True,
            "objects": rows,
            "matching_rule": "Fixed trash/trash-can alias and 0.25m position gate; ambiguous identities remain UNKNOWN",
            "full_detection_precision": None,
            "full_detection_recall": None,
            "full_detection_F1": None,
            "object_matching_accuracy": None,
            "object_matching_reason": "No verified prediction-to-native identity mapping for all instances; no fabricated matching accuracy",
            "unsupported_categories": review["unsupported_categories"],
            "unscored_native_objects": sum(
                o["change_type"] == "UNKNOWN" for o in review["objects"]
            ),
            "semantic_event_counts": dict(Counter(e["change"] for e in events)),
            "all_event_counts": dict(Counter(e["change"] for e in report["events"])),
            "probability_calibration": "Unavailable: no independent labeled calibration split; raw scores and geometric RMSE are not calibrated probabilities",
        }
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    review = review_native(plan)
    report_path = args.run / "report.json"
    report = json.loads(report_path.read_text())
    result = evaluate(review, report)
    result["inference_report_sha256"] = sha256(report_path)
    result["native_source_hashes"] = review["source_hashes"]
    result["registration"] = report["registration"]
    result["runtime_seconds"] = report["runtime_seconds"]
    result["artifacts"] = str(args.run / "visualization")
    args.report_output.write_text(json.dumps(result, indent=2) + "\n")
    private = args.run / "visualization"
    private.mkdir(exist_ok=True)
    (private / "evaluation_manifest.json").write_text(
        json.dumps(review, indent=2) + "\n"
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    before, _ = read_ascii_instances(
        Path(plan["raw_destination"])
        / plan["reference"]
        / "labels.instances.annotated.v2.ply"
    )
    after, _ = read_ascii_instances(
        Path(plan["raw_destination"])
        / plan["rescan"]
        / "labels.instances.annotated.v2.ply"
    )
    T = np.asarray(review["T_reference_rescan"])
    fig = plt.figure(figsize=(14, 6))
    for i, (title, cloud) in enumerate(
        [
            ("Native scans before shared transform", after),
            (
                "Verified native rescan → reference transform",
                transform_points(after, T),
            ),
        ]
    ):
        ax = fig.add_subplot(1, 2, i + 1, projection="3d")
        ax.scatter(*before[::5].T, s=0.4, alpha=0.3, label="Reference")
        ax.scatter(*cloud[::4].T, s=0.4, alpha=0.3, label="Rescan")
        ax.set_title(title)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.legend()
    fig.tight_layout()
    fig.savefig(private / "native_alignment_overlay.png", dpi=180)
    plt.close(fig)
    with np.load(args.run / "world.npz", allow_pickle=False) as world:
        b = world["before"]
        a = world["after"]
        b = b[:: max(1, len(b) // 5000)]
        a = a[:: max(1, len(a) // 5000)]
    payload = {
        "before": b.tolist(),
        "after": a.tolist(),
        "events": [
            {
                "position": e.get("before_position") or e.get("after_position"),
                "text": e["semantic_label"] + ": " + e["change"],
            }
            for e in report["events"]
            if e.get("object_id")
        ],
        "truth": [
            {
                "position": o["after_position"],
                "text": "Native GT " + o["label"] + ": " + o["change_type"],
            }
            for o in review["objects"]
            if o["change_type"] != "UNKNOWN"
        ],
    }
    html = """<!doctype html><meta charset="utf-8"><title>3RScan evidence review</title><style>body{margin:0;background:#101923;color:#eef;font:16px system-ui}header{padding:16px}canvas{width:100%;height:75vh}label{margin-right:20px}</style><header><h1>Aligned 3RScan evidence</h1><p>Blue: reference · orange: rescan. Drag to rotate; scroll to zoom. Model labels and native ground truth are separate.</p><label><input id="labels" type="checkbox" checked>Model decisions</label><label><input id="truth" type="checkbox">Native verified MOVED subset</label></header><canvas id="view"></canvas><script>const data=PAYLOAD;const c=document.getElementById('view'),ctx=c.getContext('2d');let rx=-0.4,ry=0.2,scale=1,last=null,fitted=false;const center=data.before.reduce((a,p)=>a.map((v,i)=>v+p[i]/data.before.length),[0,0,0]);function project(p){let [x,y,z]=p.map((v,i)=>v-center[i]);let X=x*Math.cos(ry)+z*Math.sin(ry),Z=-x*Math.sin(ry)+z*Math.cos(ry);let Y=y*Math.cos(rx)-Z*Math.sin(rx);return[c.width/2+X*scale,c.height/2-Y*scale]}function draw(){c.width=c.clientWidth;c.height=c.clientHeight;if(!fitted){const radius=Math.max(...data.before.concat(data.after).map(p=>Math.hypot(...p.map((v,i)=>v-center[i]))),0.1);scale=Math.min(c.width,c.height)*0.38/radius;fitted=true}ctx.clearRect(0,0,c.width,c.height);for(let [cloud,col]of[[data.before,'#45aaff'],[data.after,'#ffad50']]){ctx.fillStyle=col;for(let p of cloud){let [x,y]=project(p);ctx.fillRect(x,y,1.5,1.5)}}for(let [items,on,col]of[[data.events,document.getElementById('labels').checked,'#ddd'],[data.truth,document.getElementById('truth').checked,'#ff5577']]){if(on){ctx.fillStyle=col;for(let p of items){if(p.position){let [x,y]=project(p.position);ctx.fillText(p.text,x,y)}}}}}c.onpointerdown=e=>{last=[e.clientX,e.clientY];c.setPointerCapture(e.pointerId)};c.onpointermove=e=>{if(last){ry+=(e.clientX-last[0])*0.01;rx+=(e.clientY-last[1])*0.01;last=[e.clientX,e.clientY];draw()}};c.onpointerup=()=>last=null;c.addEventListener('wheel',e=>{e.preventDefault();scale*=Math.exp(-e.deltaY*0.001);draw()},{passive:false});for(let id of['labels','truth'])document.getElementById(id).onchange=draw;onresize=draw;draw();</script>"""
    (private / "evidence_viewer.html").write_text(
        html.replace("PAYLOAD", json.dumps(payload).replace("<", "\\u003c"))
    )
    print(
        json.dumps(
            {"subset_metrics": result["per_class"], "visualization": str(private)},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

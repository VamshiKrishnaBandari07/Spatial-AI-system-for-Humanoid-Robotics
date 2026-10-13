"""Quarter-turn neural detection ensemble with native-coordinate outputs.

Scores remain raw detector scores; rotated predictions are correlated, not
independent evidence. This stage makes no physical change or identity decision.
"""
from __future__ import annotations
from typing import Any, Protocol
import numpy as np
from PIL import Image


class Detector(Protocol):
    def infer(self, image: Image.Image, prompt: str) -> dict[str, Any]: ...


def native_box(box: list[float], width: int, height: int, turns: int) -> list[float]:
    """Invert np.rot90 for continuous pixel-edge xyxy boxes (not pixel centers)."""
    b=np.asarray(box,float)
    if b.shape!=(4,) or not np.isfinite(b).all() or np.any(b[2:]<=b[:2]):
        raise ValueError('Invalid detection box')
    if width<=0 or height<=0 or turns not in (0,1,2,3):raise ValueError('Invalid image geometry')
    x=np.array([b[0],b[2],b[2],b[0]]);y=np.array([b[1],b[1],b[3],b[3]])
    if turns==1:x,y=width-y,x
    elif turns==2:x,y=width-x,height-y
    elif turns==3:x,y=y,height-x
    return np.r_[np.maximum([x.min(),y.min()],0),np.minimum([x.max(),y.max()],[width,height])].tolist()


def detect_orientations(detector: Detector, image: Image.Image, prompt: str,
                        threshold: float=.35) -> dict[str, Any]:
    """Invoke the same neural detector at 0/90/180/270 degrees; fuse by box IoU.

    NMS is label-independent because multiple textual labels may describe the
    same object. All raw predictions are retained for audit before suppression.
    """
    if not 0<=threshold<=1:raise ValueError('Invalid threshold')
    pixels=np.asarray(image.convert('RGB'));raw=[]
    for k in range(4):
        prediction=detector.infer(Image.fromarray(np.rot90(pixels,k).copy()),prompt)
        for box,score,label in zip(prediction['boxes'],prediction['raw_scores'],prediction['labels']):
            if not np.isfinite(score) or score<threshold:continue
            b=native_box(box,image.width,image.height,k)
            if b[2]<=b[0] or b[3]<=b[1]:continue
            raw.append(dict(box=b,raw_score=float(score),label=label,rotation_ccw_degrees=90*k))
    retained=[]
    for candidate in sorted(raw,key=lambda r:-r['raw_score']):
        a=np.asarray(candidate['box']);duplicate=False
        for previous in retained:
            b=np.asarray(previous['box']);wh=np.maximum(0,np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2]))
            intersection=float(np.prod(wh));union=float(np.prod(a[2:]-a[:2])+np.prod(b[2:]-b[:2])-intersection)
            if intersection/union>.5:duplicate=True;break
        if not duplicate:retained.append(candidate)
    return dict(detections=retained,raw_predictions=raw,calibrated=False,
                independent_views=1,neural_calls=4,physical_change_verified=False)


def main() -> None:
    """Save neural proposals and annotated image for an explicit input crop."""
    import argparse
    import json
    from pathlib import Path
    from PIL import ImageDraw
    from .models import Expert, ModelSpec
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--prompt',required=True)
    parser.add_argument('--cache',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    expert=Expert(ModelSpec('IDEA-Research/grounding-dino-tiny',
        'a2bb814dd30d776dcf7e30523b00659f4f141c71','detection'),args.cache).load()
    with Image.open(args.image) as source:
        image=source.convert('RGB')
    result=detect_orientations(expert,image,args.prompt)
    result['model']={'identifier':expert.spec.identifier,'revision':expert.spec.revision}
    (args.output/'detections.json').write_text(json.dumps(result,indent=2))
    draw=ImageDraw.Draw(image)
    for row in result['detections']:
        draw.rectangle(row['box'],outline='lime',width=2)
    image.save(args.output/'detections.jpg')


if __name__=='__main__':
    main()

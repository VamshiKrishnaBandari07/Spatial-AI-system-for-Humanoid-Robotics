"""Model-free planar visual-change candidates. No semantic or 3D change claims.

Consumes timestamped frame manifests from existing ingestion. Each before frame
anchors a viewpoint group; after frames join only after held-out homography checks.
"""
from __future__ import annotations
import argparse
import html
import json
from pathlib import Path
import cv2
import numpy as np


def _escape(value: object) -> str:
    """Escape text interpolated into classical index.html."""
    return html.escape(str(value), quote=True)


def _border_fraction(bbox, shape, margin: int = 8) -> float:
    """Fraction of bbox area that lies within ``margin`` px of the image border."""
    h, w = shape[:2]
    x0, y0, x1, y1 = bbox
    area = max(1, (x1 - x0) * (y1 - y0))
    inner_x0, inner_y0 = margin, margin
    inner_x1, inner_y1 = max(margin, w - margin), max(margin, h - margin)
    ix0, iy0 = max(x0, inner_x0), max(y0, inner_y0)
    ix1, iy1 = min(x1, inner_x1), min(y1, inner_y1)
    inner = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    return 1.0 - inner / area


def features(image):
    return cv2.SIFT_create(nfeatures=2500).detectAndCompute(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), None)


def align(first, second, shape):
    """Return AFTER->BEFORE homography with held-out native-pixel diagnostics."""
    ka, da = first; kb, db = second
    if da is None or db is None or min(len(da), len(db)) < 2:
        return None
    matcher = cv2.BFMatcher()
    def match(a,b):
        return {m.queryIdx:m.trainIdx for pair in matcher.knnMatch(a,b,k=2)
                if len(pair)==2 for m,n in [pair] if m.distance < .7*n.distance}
    forward, reverse = match(da,db),match(db,da)
    pairs = [(a,b) for a,b in forward.items() if reverse.get(b)==a]
    if len(pairs)<20:return None
    pa=np.float32([ka[a].pt for a,b in pairs]);pb=np.float32([kb[b].pt for a,b in pairs])
    hold=np.arange(len(pairs))%4==0
    H,inliers=cv2.findHomography(pb[~hold],pa[~hold],cv2.RANSAC,2.,maxIters=3000,confidence=.995)
    if H is None or not np.isfinite(H).all() or abs(np.linalg.det(H))<1e-12:return None
    err=np.linalg.norm(cv2.perspectiveTransform(pb[hold,None],H)[:,0]-pa[hold],axis=1)
    support=int(inliers.sum());fraction=float(np.mean(err<2))
    h,w=shape[:2]
    selected=pa[~hold][inliers.ravel().astype(bool)]
    coverage=cv2.contourArea(cv2.convexHull(selected))/(h*w)
    if support<15 or fraction<.6 or np.median(err)>=2 or coverage<.08:return None
    return H,{'matches':len(pairs),'fit_inliers':support,'heldout_count':len(err),'heldout_fraction_under_2px':fraction,'heldout_median_px':float(np.median(err)),'inlier_hull_fraction':coverage}


def difference(before, after, H):
    """Photometric normalization + blurred Lab difference; valid overlap only."""
    h,w=before.shape[:2]
    warped=cv2.warpPerspective(after,H,(w,h))
    valid=cv2.warpPerspective(np.ones(after.shape[:2],np.uint8),H,(w,h),flags=cv2.INTER_NEAREST)>0
    valid=cv2.erode(valid.astype('uint8'),np.ones((15,15),np.uint8)).astype(bool)
    if valid.sum()<100:
        return warped,np.zeros((h,w),np.uint8),[],{'rejected_border_regions':0}
    a=cv2.cvtColor(before,cv2.COLOR_BGR2LAB).astype(float)
    b=cv2.cvtColor(warped,cv2.COLOR_BGR2LAB).astype(float)
    # Robust channel offset mitigates global illumination shifts, not local shadows.
    offset=np.median((a-b)[valid],axis=0)
    b=np.clip(b+offset,0,255)
    delta=np.linalg.norm(cv2.GaussianBlur(a,(7,7),0)-cv2.GaussianBlur(b,(7,7),0),axis=2)
    values=delta[valid];median=np.median(values);mad=np.median(np.abs(values-median))
    threshold=max(25.,float(median+3*1.4826*mad))
    mask=((delta>threshold)&valid).astype('uint8')*255
    mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
    mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((9,9),np.uint8))
    mask[~valid]=0
    n,labels,stats,_=cv2.connectedComponentsWithStats(mask)
    regions=[]
    rejected_border=0
    for i in range(1,n):
        x,y,bw,bh,area=stats[i]
        if area<max(100,.001*h*w) or area>=.3*h*w:
            continue
        bbox=[int(x),int(y),int(x+bw),int(y+bh)]
        # Warped-border / parallax strips hug the invalid margin after erosion.
        if _border_fraction(bbox, before.shape) >= 0.45:
            rejected_border += 1
            continue
        regions.append({
            'bbox_xyxy':bbox,
            'area_pixels':int(area),
            'difference_threshold':threshold,
            'border_fraction':round(_border_fraction(bbox, before.shape), 4),
            'physical_change_class':None,
            'classification':'VISUAL_CHANGE_CANDIDATE',
        })
    return warped,mask,regions,{'rejected_border_regions':rejected_border}


def run(root:Path,output:Path,max_frames:int=32):
    if not 2<=max_frames<=100:raise ValueError('max_frames must be 2..100')
    output.mkdir(parents=True,exist_ok=False)
    cv2.setRNGSeed(0)
    rows={};images={};desc={}
    for role in ('before','after'):
        entries=json.loads((root/role/'manifest.json').read_text())['frames']
        entries=sorted(entries,key=lambda r:r['timestamp'])
        indices=np.unique(np.linspace(0,len(entries)-1,min(max_frames,len(entries))).round().astype(int))
        rows[role]=[entries[i] for i in indices]; images[role]=[]
        for row in rows[role]:
            path=Path(row['path']).resolve()
            if not path.is_relative_to((root/role).resolve()):raise ValueError('Frame escapes run directory')
            im=cv2.imread(str(path))
            if im is None:raise ValueError(f'Missing frame {path}')
            images[role].append(im)
        desc[role]=[features(im) for im in images[role]]
    groups=[]
    for i,before in enumerate(images['before']):
        candidates=[]
        for j,after in enumerate(images['after']):
            result=align(desc['before'][i],desc['after'][j],before.shape)
            if result is not None:candidates.append((j,*result))
        candidates.sort(key=lambda r:r[2]['fit_inliers'],reverse=True)
        if not candidates:continue
        group={'before_id':rows['before'][i]['id'],'before_timestamp':rows['before'][i]['timestamp'],'pairs':[]}
        border_rejects=0
        for j,H,metrics in candidates[:3]:
            after=images['after'][j]; warped,mask,regions,diff_stats=difference(before,after,H)
            border_rejects+=diff_stats['rejected_border_regions']
            name=f'b{i:03}_a{j:03}';folder=output/name;folder.mkdir()
            cv2.imwrite(str(folder/'before.jpg'),before);cv2.imwrite(str(folder/'after_native.jpg'),after)
            cv2.imwrite(str(folder/'after_aligned.jpg'),warped);cv2.imwrite(str(folder/'mask.png'),mask)
            overlay=before.copy()
            for k,region in enumerate(regions):
                x,y,x2,y2=region['bbox_xyxy'];cv2.rectangle(overlay,(x,y),(x2,y2),(0,0,255),2)
                cv2.imwrite(str(folder/f'candidate_{k:03}.jpg'),np.concatenate([before[y:y2,x:x2],warped[y:y2,x:x2]],axis=1))
            cv2.imwrite(str(folder/'comparison.jpg'),np.concatenate([overlay,warped],axis=1))
            group['pairs'].append({'after_id':rows['after'][j]['id'],'after_timestamp':rows['after'][j]['timestamp'],'folder':name,'alignment':metrics,'homography_after_to_before':H.tolist(),'regions':regions,'rejected_border_regions':diff_stats['rejected_border_regions']})
        # Agreement in one anchor grid is supporting evidence, not independent 3D views.
        for pair in group['pairs']:
            for region in pair['regions']:
                x,y,x2,y2=region['bbox_xyxy'];support=0
                for other in group['pairs']:
                    if any(min(x2,r['bbox_xyxy'][2])>max(x,r['bbox_xyxy'][0]) and min(y2,r['bbox_xyxy'][3])>max(y,r['bbox_xyxy'][1]) for r in other['regions']):support+=1
                region['supporting_pair_count']=support
        group['rejected_border_regions']=border_rejects
        groups.append(group)
    region_instances=sum(len(p['regions']) for g in groups for p in g['pairs'])
    report={
        'method':'SIFT + held-out RANSAC homography + robust Lab difference',
        'classification':'VISUAL_CHANGE_CANDIDATE',
        'physical_change_verified':False,
        'region_count_is_not_object_count':True,
        'limitations':'Planar alignment only; parallax, shadows, reflections and occlusion can cause false positives. No semantic labels, calibrated confidence, physical change class or 3D verification. Groups are overlapping anchor views, not unique objects. Region totals are instance boxes, not distinct objects.',
        'frames_per_role':{k:len(v) for k,v in rows.items()},
        'summary':{
            'viewpoint_groups':len(groups),
            'aligned_pairs':sum(len(g['pairs']) for g in groups),
            'candidate_region_instances':region_instances,
            'note':'candidate_region_instances must not be reported as object or removal counts',
        },
        'groups':groups,
    }
    (output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    cards=[]
    for g in groups:
        for p in g['pairs']:
            cards.append(
                f'<h3>{_escape(g["before_id"])} → {_escape(p["after_id"])}: '
                f'{len(p["regions"])} candidate regions (not object count)</h3>'
                f'<img width="900" src="{_escape(p["folder"])}/comparison.jpg">'
            )
            cards.append(
                f'<p>Held-out alignment median: {p["alignment"]["heldout_median_px"]:.2f} pixels. '
                f'<a href="{_escape(p["folder"])}/mask.png">Difference mask</a> · '
                f'<a href="{_escape(p["folder"])}/after_native.jpg">Original after frame</a></p>'
            )
            for k,region in enumerate(p['regions']):
                cards.append(
                    f'<a href="{_escape(p["folder"])}/candidate_{k:03}.jpg">'
                    f'<img alt="Candidate {k}, before left and aligned after right" width="240" '
                    f'src="{_escape(p["folder"])}/candidate_{k:03}.jpg"></a>'
                )
    disclaimer=_escape(report['limitations'])
    (output/'index.html').write_text(
        '<!doctype html><meta charset="utf-8"><title>Visual change candidates</title>'
        '<h1>Model-free visual change candidates</h1><p>'+disclaimer+'</p>'+''.join(cards)
    )
    print(json.dumps({
        'groups':len(groups),
        'pairs':sum(len(g['pairs']) for g in groups),
        'regions':region_instances,
        'region_count_is_not_object_count':True,
        'output':str(output),
    }))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--max-frames',type=int,default=32)
    args=p.parse_args();run(args.run.resolve(),args.output.resolve(),args.max_frames)

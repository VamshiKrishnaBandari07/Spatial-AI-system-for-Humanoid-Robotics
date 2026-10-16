"""Execute object-view rankings as native crop detections and appearance proposals.

This recovery pass does not interpret detector absence as physical removal.
"""
from __future__ import annotations
import argparse
import json
import sqlite3
from pathlib import Path
import numpy as np
from PIL import Image
from .pixel_mapping import native_to_geometry_matrix
from .models import Expert, ModelSpec
from .paths import models_dir


def native_region(region, native_shape, geometry_shape, mode, context=1.5):
    """Map reconstruction-grid xyxy bounds to a clipped native context crop."""
    box = np.asarray(region,float)
    if box.shape != (4,) or not np.isfinite(box).all() or np.any(box[2:] <= box[:2]):
        raise ValueError('Invalid projected region')
    inverse = np.linalg.inv(native_to_geometry_matrix(native_shape,geometry_shape,mode))
    corners = np.array([[box[0],box[1],1],[box[2],box[3],1]]) @ inverse.T
    lo,hi = corners[:,:2]; center=(lo+hi)/2; half=(hi-lo)*context/2
    h,w=native_shape
    out=np.r_[np.maximum(0,np.floor(center-half)),np.minimum([w,h],np.ceil(center+half))].astype(int)
    return out.tolist() if np.all(out[2:]>out[:2]) else None


def match_candidate(scores, minimum=.85, margin=.05):
    """Appearance proposal only: scores are not calibrated identity probabilities."""
    scores=np.asarray(scores,float)
    if scores.ndim != 1 or not np.isfinite(scores).all():raise ValueError('Invalid similarities')
    if not len(scores):return None
    order=np.argsort(-scores)
    if scores[order[0]] < minimum:return None
    if len(scores)>1 and scores[order[0]]-scores[order[1]] < margin:return None
    return int(order[0])


def run_recheck(run: Path, output: Path, max_objects: int | None = None, max_views: int = 2, rotation_aware: bool = False):
    if (max_objects is not None and not 1<=max_objects<=128) or not 1<=max_views<=4:raise ValueError('Invalid recovery budget')
    run=run.resolve();output.mkdir(parents=True,exist_ok=False)
    report=json.loads((run/'report.json').read_text())
    if report['status']!='complete':raise ValueError('Completed source run required')
    frames={}
    for role in ['before','after']:
        frames.update({r['id']:r for r in json.loads((run/role/'manifest.json').read_text())['frames']})
    with sqlite3.connect((run/'spatial_memory.sqlite').as_uri()+'?mode=ro',uri=True) as db:
        observations={r['id']:r for (text,) in db.execute('SELECT data FROM observations') for r in [json.loads(text)]}
    events=json.loads((run/'changes/changes.json').read_text())['events']
    objects={o['id']:o for o in report['spatial_memory']['before']['objects']}
    eligible=[]
    for event in events:
        obj=objects.get(event['object_id']);vis=event['evidence'].get('visibility',{})
        if obj and vis.get('ranked_object_view_ids'):
            # Use the largest available native observation, not a tiny first fragment.
            observation=max((observations[oid] for oid in obj['observation_ids']),
                            key=lambda o: (o['provenance']['box'][2]-o['provenance']['box'][0]) *
                                          (o['provenance']['box'][3]-o['provenance']['box'][1]))
            box=observation['provenance']['box']
            eligible.append(((box[2]-box[0])*(box[3]-box[1]),obj,observation,vis))
    # Deduplicate repeated event records only. Similar-looking physical tracks
    # must remain separate until joint identity matching can justify a merge.
    eligible=list({row[1]['id']:row for row in eligible}.values())
    eligible.sort(key=lambda r:r[1]['id'])
    selected=eligible if max_objects is None else eligible[:max_objects]
    coverage={'eligible_ids':[r[1]['id'] for r in eligible],
              'selected_ids':[r[1]['id'] for r in selected],
              'excluded_no_ranked_view':[oid for oid in objects if oid not in {r[1]['id'] for r in eligible}],
              'deferred_by_budget':[r[1]['id'] for r in eligible if r not in selected],
              'max_views':max_views,'physical_tracks_merged':False}
    (output/'coverage.json').write_text(json.dumps(coverage,indent=2))
    detector=Expert(ModelSpec('IDEA-Research/grounding-dino-tiny','a2bb814dd30d776dcf7e30523b00659f4f141c71','detection'),str(models_dir())).load()
    appearance=Expert(ModelSpec('facebook/dinov2-small','ed25f3a31f01632728cabb09d1542f84ab7b0056','appearance'),str(models_dir())).load()
    results=[]
    def image_for(fid):
        path=Path(frames[fid]['path']).resolve()
        if not path.is_relative_to(run):raise ValueError('Evidence path escapes source run')
        return Image.open(path).convert('RGB')
    for _,obj,observation,visibility in selected:
        src=image_for(observation['frame_id']);box=observation['provenance']['box']
        source=src.crop(box);source.save(output/f'{obj["id"]}_before.jpg')
        descriptor=np.asarray(appearance.infer(source)['descriptor'])
        views=[]
        byid={v['frame_id']:v for v in visibility['views']}
        for fid in visibility['ranked_object_view_ids'][:max_views]:
            view=byid[fid];image=image_for(fid)
            shape=view['raybox_diagnostics']['depth_shape']
            region=native_region(view['object_projection']['region_xyxy'],(image.height,image.width),shape,report['config']['neural_preprocessing'])
            if region is None:continue
            patch=image.crop(region);name=f'{obj["id"]}_{fid.replace(":","_")}.jpg';patch.save(output/name)
            orientation_audit = None
            if rotation_aware:
                from .oriented_detection import detect_orientations
                orientation_audit = detect_orientations(
                    detector, patch, report['config']['object_prompt'],
                    report['config']['detection_threshold'])
                rows = orientation_audit['detections']
                detected = {'boxes': [r['box'] for r in rows],
                            'raw_scores': [r['raw_score'] for r in rows],
                            'labels': [r['label'] for r in rows]}
            else:
                detected=detector.infer(patch,report['config']['object_prompt'])
            candidates=[]
            for local,score,label in zip(detected['boxes'],detected['raw_scores'],detected['labels']):
                if score<report['config']['detection_threshold']:continue
                b=np.asarray(local,float);b=np.clip(b,[0,0,0,0],[patch.width,patch.height,patch.width,patch.height])
                if np.any(b[2:]<=b[:2]):continue
                emb=np.asarray(appearance.infer(patch.crop(b.tolist()))['descriptor'])
                candidates.append({'label_proposal':label,'native_box':(b+[region[0],region[1],region[0],region[1]]).tolist(),'raw_detector_score':score,'appearance_cosine':float(descriptor@emb)})
            best=match_candidate([c['appearance_cosine'] for c in candidates])
            views.append({'frame_id':fid,'crop':name,'native_region':region,'candidates':candidates,
                          'appearance_proposal_index':best,'identity_verified':False,
                          'occupancy':view['raybox_diagnostics'],'absence_is_removal':False, 'orientation_audit':orientation_audit})
        results.append({'object_id':obj['id'],'source_frame':observation['frame_id'],'views':views})
        (output/'recheck.json').write_text(json.dumps({'status':'running','fresh_detection_and_embeddings':True,'objects':results,'source_geometry_reused':True,'confirmed_changes':[]},indent=2))
    (output/'recheck.json').write_text(json.dumps({'status':'complete_recovery_proposals',
        'fresh_detection_and_embeddings':True,'objects':results,'coverage':coverage,
        'source_geometry_reused':True,'confirmed_changes':[]},indent=2))
    return results


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--max-objects',type=int);p.add_argument('--max-views',type=int,default=2)
    p.add_argument('--rotation-aware',action='store_true')
    a=p.parse_args();run_recheck(a.run,a.output,a.max_objects,a.max_views,a.rotation_aware)

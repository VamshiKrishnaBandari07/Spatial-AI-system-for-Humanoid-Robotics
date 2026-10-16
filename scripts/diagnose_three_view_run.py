"""Read saved VGGT predictions; evaluate cycle-consistent SIFT tracks without camera fitting.

Descriptor tracks are candidates, not verified static landmarks. Outputs are diagnostics,
not change predictions. C observations are never used in triangulation or camera fitting.
"""
import argparse
import itertools
import json
from pathlib import Path
import cv2
import numpy as np
from spatialmemory.contracts import Frame
from spatialmemory.ingestion import neural_keyframes
from spatialmemory.pixel_mapping import native_to_geometry_matrix
from spatialmemory.three_view_diagnostic import triangulate_and_validate_three_view


def mutual_matches(first, second, ratio=.7):
    if first is None or second is None or min(len(first), len(second)) < 2:
        return {}
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    def directed(a, b):
        return {m.queryIdx: m.trainIdx for pair in matcher.knnMatch(a, b, k=2)
                if len(pair) == 2 for m, n in [pair] if m.distance < ratio * n.distance}
    forward, reverse = directed(first, second), directed(second, first)
    return {a: b for a, b in forward.items() if reverse.get(b) == a}


def diagnose(run, output):
    output.mkdir(parents=True, exist_ok=False)
    cv2.setRNGSeed(0)
    config = json.loads((run / 'report.json').read_text())['config']
    results = []
    pair_counts = []
    sift = cv2.SIFT_create(nfeatures=4000)
    for role in ('before', 'after'):
        manifest = json.loads((run / role / 'manifest.json').read_text())
        frames = neural_keyframes([Frame(**f) for f in manifest['frames']], config['neural_frames_per_video'])
        with np.load(run / 'neural' / role / 'predictions.npz', allow_pickle=False) as data:
            images = [cv2.imread(f.path) for f in frames]
            if any(im is None for im in images):
                raise ValueError('Missing source frame')
            features = [sift.detectAndCompute(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), None) for im in images]
            shape = data['world_points'].shape[2:4]
            mappings = [native_to_geometry_matrix(im.shape[:2], shape, config['neural_preprocessing']) for im in images]
            intrinsics = [np.linalg.inv(a) @ k for a, k in zip(mappings, data['intrinsics'][0])]
            poses = []
            for extrinsic in data['extrinsics'][0]:
                pose = np.eye(4); pose[:3] = extrinsic; poses.append(pose)
            matches = {(i,j): mutual_matches(features[i][1], features[j][1]) for i,j in itertools.permutations(range(len(frames)),2)}
            pair_counts.extend({"role":role,"frames":[frames[i].id,frames[j].id],"mutual_matches":len(m)} for (i,j),m in matches.items() if i<j)
            for a,b,c in itertools.permutations(range(len(frames)),3):
                if a > b:
                    continue
                tracks = [(i,j,matches[a,c][i]) for i,j in matches[a,b].items()
                          if i in matches[a,c] and matches[b,c].get(j) == matches[a,c][i]]
                row = {'role':role,'frames':[frames[i].id for i in (a,b,c)],'candidate_tracks':len(tracks)}
                if not tracks:
                    results.append(row);continue
                pixels = [np.array([features[v][0][t[column]].pt for t in tracks]) for column,v in enumerate((a,b,c))]
                mapped = np.column_stack([pixels[2],np.ones(len(tracks))]) @ mappings[c].T
                # Bilinear sampling at the same C-feature locations; no geometry-based track filtering.
                direct = cv2.remap(data['world_points'][0,c],mapped[:,0].astype('float32')[:,None],mapped[:,1].astype('float32')[:,None],cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=float('nan'))[:,0,:]
                finite = np.isfinite(direct).all(1)
                row['finite_direct_tracks'] = int(finite.sum())
                if not finite.any():
                    results.append(row);continue
                pixels = [p[finite] for p in pixels]; direct = direct[finite]
                args = []
                for v,p in zip((a,b,c),pixels):args.extend([intrinsics[v],poses[v],p])
                r = triangulate_and_validate_three_view(*args,direct)
                geometric = r['triangulation_valid_mask']
                ab = geometric & (r['reprojection_errors_a'] < 2) & (r['reprojection_errors_b'] < 2)
                def median(values):
                    good = values[np.isfinite(values)]
                    return float(np.median(good)) if len(good) else None
                row.update(geometric_tracks=int(geometric.sum()),ab_supported_tracks=int(ab.sum()),all_views_supported=int(r['final_valid_mask'].sum()),
                           heldout_median_px=median(r['reprojection_errors_c'][ab]),direct_c_median_same_tracks_px=median(r['direct_head_reprojection_errors']['view_c'][ab]))
                name=f'{role}_{a}_{b}_{c}'
                np.savez_compressed(output/f'{name}.npz',pixels_a=pixels[0],pixels_b=pixels[1],pixels_c=pixels[2],direct=direct,points=r['triangulated_points'],valid=r['final_valid_mask'],error_a=r['reprojection_errors_a'],error_b=r['reprojection_errors_b'],error_c=r['reprojection_errors_c'])
                # Raw candidate evidence for manual verification; includes rejected matches.
                canvas=np.concatenate([images[a],images[b],images[c]],axis=1)
                offsets=np.cumsum([0,images[a].shape[1],images[b].shape[1]])
                for n in range(min(len(direct),80)):
                    color=(0,200,0) if r['final_valid_mask'][n] else (0,0,255)
                    coords=[tuple(np.round(p[n]+[offsets[j],0]).astype(int)) for j,p in enumerate(pixels)]
                    for q in coords:cv2.circle(canvas,q,3,color,1)
                    cv2.line(canvas,coords[0],coords[1],color,1);cv2.line(canvas,coords[1],coords[2],color,1)
                cv2.imwrite(str(output/f'{name}.jpg'),canvas)
                results.append(row)
    report={'scope':'saved prediction diagnostic; automatic feature candidates, static identity unverified; no camera optimization','seed':0,'ratio':.7,'minimum_parallax_deg':2,'strict_limit_native_px':2,'triplets':results,'pairwise_matches':pair_counts}
    (output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    print(json.dumps(report,indent=2))

if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();diagnose(args.run.resolve(),args.output.resolve())

"""Independent local planar context alignment for crop recovery, not 3D evidence.

Object pixels are excluded from matching so disappearance need not destroy the
correspondences. A held-out subset validates the mapping; parallax can still
invalidate its use on objects away from the support plane. No removal is inferred.
"""
from __future__ import annotations
from dataclasses import dataclass
import cv2
import numpy as np
from numpy.typing import NDArray


@dataclass
class ContextMatch:
    homography: NDArray[np.float64]
    polygon: NDArray[np.float64]
    inliers: int
    heldout_error_px: float
    coverage: float


def fit_context(source: NDArray, target: NDArray, box: NDArray,
                max_error_px: float = 2.0,
                diagnostics: dict | None = None) -> ContextMatch | None:
    """Map native source xyxy box into target using nearby static context.

    Images are uint8 BGR (H,W,3); H maps source homogeneous pixels to target.
    Deterministic held-out correspondences are never used for fitting. A mapping
    requires surrounding support and >=4 held-out matches within the pixel gate.
    """
    def reject(reason: str) -> None:
        if diagnostics is not None:
            diagnostics['rejection'] = reason
        return None

    box = np.asarray(box, dtype=float)
    if box.shape != (4,) or not np.isfinite(box).all() or np.any(box[2:] <= box[:2]):
        raise ValueError('Invalid xyxy box')
    for image in (source, target):
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise ValueError('Expected uint8 BGR images')
    if not np.isfinite(max_error_px) or max_error_px <= 0:
        raise ValueError('Invalid error limit')
    h,w = source.shape[:2]
    if np.any(box[:2]<0) or np.any(box[2:]>[w,h]):
        raise ValueError('Box outside source')
    center=(box[:2]+box[2:])/2
    radius=np.maximum(box[2:]-box[:2],64)*2
    lo=np.maximum(0,np.floor(center-radius)).astype(int)
    hi=np.minimum([w,h],np.ceil(center+radius)).astype(int)
    mask=np.zeros((h,w),np.uint8);mask[lo[1]:hi[1],lo[0]:hi[0]]=255
    x1,y1=np.floor(box[:2]).astype(int);x2,y2=np.ceil(box[2:]).astype(int)
    mask[y1:y2,x1:x2]=0
    sift=cv2.SIFT_create(nfeatures=3000)
    ka,da=sift.detectAndCompute(source,mask); kb,db=sift.detectAndCompute(target,None)
    if da is None or db is None or len(db)<2:return reject('missing_descriptors')
    matches=[a for pair in cv2.BFMatcher().knnMatch(da,db,k=2)
             if len(pair)==2 for a,b in [pair] if a.distance<.7*b.distance]
    # A target feature cannot support multiple source correspondences.
    matches=list({m.trainIdx:m for m in sorted(matches,key=lambda m:-m.distance)}.values())
    matches.sort(key=lambda m:(m.queryIdx,m.trainIdx))
    if len(matches)<20:return reject('too_few_matches')
    a=np.float64([ka[m.queryIdx].pt for m in matches]);b=np.float64([kb[m.trainIdx].pt for m in matches])
    validation=np.arange(len(a))%4==0
    cv2.setRNGSeed(0)
    H,inliers=cv2.findHomography(a[~validation],b[~validation],cv2.RANSAC,max_error_px)
    if H is None or not np.isfinite(H).all() or abs(np.linalg.det(H))<1e-10:return reject('invalid_homography')
    good=inliers.ravel().astype(bool)
    if good.sum()<12:return reject('too_few_inliers')
    support=a[~validation][good]
    # Reject extrapolation beyond the spatial support of matched background.
    hull=cv2.convexHull(support.astype(np.float32))
    if cv2.pointPolygonTest(hull,tuple(center),False)<0:return reject('extrapolated_box')
    pred=cv2.perspectiveTransform(a[validation,None,:],H)[:,0]
    errors=np.linalg.norm(pred-b[validation],axis=1)
    if diagnostics is not None:
        diagnostics.update(matches=len(matches), inliers=int(good.sum()),
                           heldout_errors_px=errors.tolist())
    accepted=errors<=max_error_px
    if accepted.sum()<4 or accepted.mean()<.75:return reject('heldout_reprojection_failed')
    corners=np.array([[x1,y1],[x2,y1],[x2,y2],[x1,y2]],dtype=float)
    homogeneous=np.c_[corners,np.ones(4)]@H.T
    z=homogeneous[:,2]
    if np.any(abs(z)<1e-8) or not (np.all(z>0) or np.all(z<0)):return reject('projection_pole')
    polygon=homogeneous[:,:2]/z[:,None]
    area=abs(cv2.contourArea(polygon.astype(np.float32)))
    if area<16:return reject('tiny_projection')
    th,tw=target.shape[:2]
    screen=np.float32([[0,0],[tw,0],[tw,th],[0,th]])
    visible,_=cv2.intersectConvexConvex(polygon.astype(np.float32),screen)
    coverage=float(visible/area)
    if coverage<.9:return reject('partial_view')
    return ContextMatch(H,polygon,int(good.sum()),float(np.max(errors[accepted])),coverage)

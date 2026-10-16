"""Hash-verified, local LoFTR crop matching. Model scores are not calibrated."""

from copy import deepcopy
from pathlib import Path
import hashlib
import cv2
import numpy as np
from numpy.typing import NDArray

CHECKPOINT_SHA256 = "be9ff88b323ec27889114719f668ae41aff7034b56a4c4acbd46b8b180b87ed3"
MODEL_REVISION = "0fb3b456cf176746709c38fa9cf80695a1c82dac"


def prepare_crop(gray: NDArray[np.uint8], mask: NDArray[np.uint8], size: int = 256) -> tuple[NDArray[np.float32], NDArray[np.float64]]:
    """Return 1x1xHxW float image and 3x3 pixel-centre crop-to-native affine."""
    if gray.ndim != 2 or gray.shape != mask.shape or gray.dtype != np.uint8:
        raise ValueError("Matching needs same-sized native grayscale image and mask")
    y,x = np.nonzero(mask)
    if not len(x):
        raise ValueError("Cannot match an empty object mask")
    cx,cy = (x.min()+x.max())/2, (y.min()+y.max())/2
    hx,hy = max(32,int(x.max()-x.min()+1)),max(32,int(y.max()-y.min()+1))
    x0,y0 = max(0,int(cx-hx)),max(0,int(cy-hy))
    x1,y1 = min(gray.shape[1],int(cx+hx+1)),min(gray.shape[0],int(cy+hy+1))
    crop = gray[y0:y1,x0:x1]
    scale = size/max(crop.shape)
    width,height = max(8,int(crop.shape[1]*scale)//8*8),max(8,int(crop.shape[0]*scale)//8*8)
    image = cv2.resize(crop,(width,height)).astype(np.float32)/255
    sx,sy = crop.shape[1]/width,crop.shape[0]/height
    affine = np.array([[sx,0,x0+(sx-1)/2],[0,sy,y0+(sy-1)/2],[0,0,1.]])
    return image[None,None],affine


def _sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    """Stream the file so large checkpoints are not read into memory at once."""
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(chunk), b''):
            digest.update(block)
    return digest.hexdigest()


class LearnedCropMatcher:
    def __init__(self, checkpoint: Path) -> None:
        checkpoint = Path(checkpoint)
        # Verify integrity before importing or deserializing anything, so an
        # unverified file is rejected without optional neural dependencies.
        if _sha256_file(checkpoint) != CHECKPOINT_SHA256:
            raise ValueError("LoFTR checkpoint checksum mismatch")
        import torch
        from kornia.feature import LoFTR
        from kornia.feature.loftr.loftr import default_cfg
        config = deepcopy(default_cfg)
        config['coarse']['temp_bug_fix'] = True
        self.model = LoFTR(pretrained=None,config=config).eval()
        weights = torch.load(checkpoint,map_location='cpu',weights_only=True)
        self.model.load_state_dict(weights['state_dict'],strict=True)

    def match(self, gray_a: NDArray[np.uint8], mask_a: NDArray[np.uint8],
              gray_b: NDArray[np.uint8], mask_b: NDArray[np.uint8]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        import torch
        a,aa = prepare_crop(gray_a,mask_a)
        b,ab = prepare_crop(gray_b,mask_b)
        with torch.inference_mode():
            result = self.model({'image0':torch.from_numpy(a),'image1':torch.from_numpy(b)})
        pa = result['keypoints0'].cpu().numpy()
        pb = result['keypoints1'].cpu().numpy()
        scores = result['confidence'].cpu().numpy()
        pa = (np.c_[pa,np.ones(len(pa))] @ aa.T)[:,:2]
        pb = (np.c_[pb,np.ones(len(pb))] @ ab.T)[:,:2]
        accepted = np.isfinite(pa).all(axis=1) & np.isfinite(pb).all(axis=1) & np.isfinite(scores) & (scores >= .8)
        for pixels,mask in [(pa,mask_a),(pb,mask_b)]:
            xy = np.rint(np.nan_to_num(pixels)).astype(int)
            valid = (xy[:,0]>=0)&(xy[:,1]>=0)&(xy[:,0]<mask.shape[1])&(xy[:,1]<mask.shape[0])
            inside = np.zeros(len(xy),bool)
            inside[valid] = mask[xy[valid,1],xy[valid,0]] > 0
            accepted &= inside
        ids = np.where(accepted)[0]
        ids = ids[np.argsort(scores[ids])[::-1]][:100]
        return pa[ids],pb[ids]

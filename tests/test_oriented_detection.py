import numpy as np
from PIL import Image
from spatialmemory.oriented_detection import native_box,detect_orientations


def test_all_rotations_return_native_edges():
    original=[10,20,40,60]
    rotated=[original,[20,60,60,90],[60,140,90,180],[140,10,180,40]]
    for k,box in enumerate(rotated):assert np.allclose(native_box(box,100,200,k),original)


def test_rotations_are_not_independent_observations():
    class Fake:
        calls=0
        def infer(self,image,prompt):
            boxes=[[10,20,40,60],[20,60,60,90],[60,140,90,180],[140,10,180,40]]
            b=boxes[self.calls];self.calls+=1
            return dict(boxes=[b],raw_scores=[.8],labels=['object'])
    detector=Fake();result=detect_orientations(detector,Image.new('RGB',(100,200)),'object.')
    assert detector.calls==4
    assert len(result['detections'])==1
    assert len(result['raw_predictions'])==4
    assert result['independent_views']==1
    assert not result['physical_change_verified']

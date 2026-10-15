import json
from spatialmemory.local_visual import validate_local_output


def evidence():
    return [{'id':'before:1','role':'before'},{'id':'after:2','role':'after'}]


def test_prose_and_truncation_never_become_change_proposals():
    for text,truncated in [('The cup disappeared.',False),('{"changes":[],"limitations":"none"}',True)]:
        result=validate_local_output(text,evidence(),truncated)
        assert result['status']=='invalid_structured_output'
        assert result['changes']==[]


def test_structured_proposals_require_supplied_frames_and_stay_unverified():
    change=dict(label='container',change='REMOVED',before_frames=['before:1'],
                after_frames=['after:2'],observation='Absent from observed position',
                visibility_evidence='Surface visible',uncertainty='Other locations not observed')
    text=json.dumps(dict(changes=[change],limitations='Visual only'))
    result=validate_local_output(text,evidence(),False)
    assert result['status']=='visual_proposals'
    assert len(result['changes'])==1
    assert not result['physical_3d_verified']
    change['after_frames']=['after:invented']
    assert validate_local_output(json.dumps(dict(changes=[change],limitations='Visual only')),evidence(),False)['changes']==[]


def test_unpinned_model_rejected_before_output_creation(tmp_path):
    import pytest
    from spatialmemory.local_visual import compare_local
    with pytest.raises(ValueError,match='profile'):
        compare_local(tmp_path,tmp_path/'out',model_profile='untrusted/latest')
    assert not (tmp_path/'out').exists()

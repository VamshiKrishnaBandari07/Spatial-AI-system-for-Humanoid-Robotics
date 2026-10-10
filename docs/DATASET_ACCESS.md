# Real-data access and benchmark inputs

The initial workspace search, before the approved acquisition, is recorded in [workspace_data_search.json](../reports/workspace_data_search.json), including exact paths. It found 44 videos and 15 annotation candidates. Available videos are prepared research sequences, explicit synthetic augmentations and fixtures, fixture uploads, or dependency demonstration clips. Seven annotation files describe synthetic region transformations; eight are inference outputs rather than annotation-schema ground truth. None supplied independently reviewed real change annotations. No 3RScan scan assets were found at that stage; the approved acquisition below supersedes that absence finding.

## 3RScan

Use the [official project](https://waldjohannau.github.io/RIO/) and its [access form](https://forms.gle/NvL5dvB4tSFrHfQH6). Read the Terms of Use, provide your actual applicant details, and obtain approval and the official downloader. No request or agreement was submitted on your behalf; you subsequently confirmed your own official approval and agreement. Restrict future acquisitions to one documented reference/rescan relationship and necessary file types; retain the native relationship metadata and its checksum.

The local directory `data/datasets/3rscan/raw/` now contains the approved pair. Layout:

```text
data/datasets/3rscan/
  downloader/3RScan.json
  raw/
    <approved-reference-id>/sequence.zip
    <approved-reference-id>/extracted/
    <approved-rescan-id>/sequence.zip
    <approved-rescan-id>/extracted/
  processed/
```

The scan-ID placeholders refer to the actual IDs listed below; the relationship JSON is retained in the downloader directory. Preserve native RGB/depth frames, calibration, timestamps, camera poses, instance annotations and documented object relationships. The existing `spatialmemory.datasets.convert_3rscan(root, pairs_json, reference, rescan, output, split)` adapter rejects undocumented pair relationships. Native change metadata must still be converted and reviewed; ingestion alone does not produce evaluation-ready labels. Follow [annotation_protocol.md](annotation_protocol.md) for provenance and pair gates. Keep all raw files outside Git.

## Smartphone recordings

Empty directories are prepared at `data/raw/scene_001/before/`, `data/raw/scene_001/after/`, `data/annotations/scene_001/` and `data/manifests/`. Copy [scene_manifest.json](../examples/scene_manifest.json) into the manifest directory and supply actual same-room recordings at its expected video paths. Relative paths resolve against the repository root. Optional camera/depth/pose/IMU fields can remain null.

Start from [smartphone_annotations.template.json](../examples/smartphone_annotations.template.json). It deliberately contains no objects, invented coordinates or declared real provenance. Use the application's **Annotation workspace** to inspect both original recordings, assign persistent object identities, capture frame observations and export annotations. Document the same-location relationship and actual physical changes. Record annotation provenance and obtain an independent review before evaluation. Change `data_origin` to `REAL` only for verified real recordings. Leave metric scale unverified unless measured. Record the complete object inventory before reporting detection precision and recall.

Inference requires only the two videos. A genuine supervised benchmark additionally requires validated same-room relationship evidence, physical-change documentation, reviewed labels, successful shared registration and explicit object correspondence. Model predictions are not ground truth.

## TUM RGB-D

The [official download page](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download) and file-format documentation are now accessible. The official overview states CC BY 4.0 for data unless otherwise stated, and BSD-2-Clause for accompanying source code. The proposed small `freiburg1_xyz` archive redirects to `webshare.cvg.cit.tum.de`, whose request is still blocked by environment network policy with HTTP tunnel 403; see [tum_access_check.json](../reports/tum_access_check.json). No TUM files or sequence size have been verified locally. The additional `webshare.cvg.cit.tum.de` domain is saved in the environment draft; review, save and publish the environment settings to activate it, then retry the official archive. Do not substitute an unofficial mirror or infer source calibration.

When accessible, verify the source license and archive size before acquisition, preserve timestamp lists and pose conventions, associate frames within a documented tolerance, and use the officially specified depth scale and calibration. TUM localization/reconstruction sequences must not be labeled as ground-truth changed scenes. Existing ARKitScenes and 7-Scenes data remain available for independently documented reconstruction controls.

## Uploaded downloader inspection

The supplied Gist archive was inspected and its downloader retained locally under ignored `data/datasets/3rscan/downloader/download_3rscan.py`. The official HTTPS metadata and Terms of Use are accessible; the complete relationship JSON contains 478 entries. The Terms of Use explicitly require approval and restrict use to non-commercial research and education; commercial-employer applicants must also have authority to bind their employer. At inspection, no agreement had been accepted on your behalf and no scan assets had been downloaded. The reference/rescan relationship is recorded in `reports/3rscan_candidate_download_plan.json`, with its subsequent download/validation status. Relationship metadata alone is not proof that evaluation-ready labels are available.

## Approved subset acquired

The user subsequently confirmed official approval and agreement to Terms of Use. One native training reference/rescan pair was acquired from the official HTTPS host: reference `02b33dfb-be2b-2d54-92d2-cd012b2b3c40`, rescan `fcf66d9e-622d-291c-84c2-bb23dfe31327`. Raw sequences, semantic annotations and labeled geometry are under `data/datasets/3rscan/raw/<scan-id>/`. The integrated manifest is `data/datasets/3rscan/processed/02b33dfb-be2b-2d54-92d2-cd012b2b3c40__fcf66d9e-622d-291c-84c2-bb23dfe31327/manifest.json`. All 257 native frame sets passed RGB/depth/pose validation; 3rscan_acquisition.json and 3rscan_native_validation.json report computed checks. Previous absence/access statements above are historical stages, not the current acquisition status.

The canonical `annotations.draft.json` in that processed directory is explicitly UNREVIEWED. Native metadata describes rigid changes for a toilet, scale and trash can; movement-versus-rotation mapping, full inventory coverage, reviewer provenance and metric shared-transform units require validation. Do not treat missing objects in incomplete coverage as removed, or all unlisted objects as unchanged. No benchmark metrics are claimed from this draft. The current neural profile uses RGB and retains optional sensor metadata without fusing it.

Native input validation can be repeated without downloads or model changes:

```bash
python scripts/check_3rscan_inputs.py \
  --plan reports/3rscan_candidate_download_plan.json \
  --manifest data/datasets/3rscan/processed/02b33dfb-be2b-2d54-92d2-cd012b2b3c40__fcf66d9e-622d-291c-84c2-bb23dfe31327/manifest.json
```

The unchanged neural CPU profile was attempted on the genuine pair. Both visits produced finite neural outputs, but shared registration failed with insufficient nondegenerate consensus. The failed run and partial outputs are retained at `/workspace/spatialmemory-3rscan-pair-20261010`; the report is `reports/3rscan_inference_attempt.json`. No completed change predictions or supervised accuracy metrics were produced. Fixing or configuring valid cross-session alignment is separate from dataset acquisition; no architecture or training changes were made during this acquisition.

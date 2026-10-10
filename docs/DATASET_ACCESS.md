# Real-data access and benchmark inputs

The workspace search is recorded in [workspace_data_search.json](../reports/workspace_data_search.json), including exact paths. It found 44 videos and 15 annotation candidates. Available videos are prepared research sequences, explicit synthetic augmentations and fixtures, fixture uploads, or dependency demonstration clips. Seven annotation files describe synthetic region transformations; eight are inference outputs rather than annotation-schema ground truth. None supplies independently reviewed real change annotations. No 3RScan scan assets were found.

## 3RScan

Use the [official project](https://waldjohannau.github.io/RIO/) and its [access form](https://forms.gle/NvL5dvB4tSFrHfQH6). Read the Terms of Use, provide your actual applicant details, and obtain approval and the official downloader. No request or agreement has been submitted on your behalf. Restrict the first acquisition to one documented reference/rescan relationship and necessary file types; retain the native relationship metadata and its checksum.

The empty local directory `data/datasets/3rscan/raw/` is prepared for approved files. Expected layout after acquisition:

```text
data/datasets/3rscan/
  raw/
    3RScan.json
    <approved-reference-id>/sequence/
    <approved-rescan-id>/sequence/
  processed/
```

These are expected paths, not downloaded assets. Preserve native RGB/depth frames, calibration, timestamps, camera poses, instance annotations and documented object relationships. The existing `spatialmemory.datasets.convert_3rscan(root, pairs_json, reference, rescan, output, split)` adapter rejects undocumented pair relationships. Native change metadata must still be converted and reviewed; ingestion alone does not produce evaluation-ready labels. Follow [annotation_protocol.md](annotation_protocol.md) for provenance and pair gates. Keep all raw files outside Git.

## Smartphone recordings

Empty directories are prepared at `data/raw/scene_001/before/`, `data/raw/scene_001/after/`, `data/annotations/scene_001/` and `data/manifests/`. Copy [scene_manifest.json](../examples/scene_manifest.json) into the manifest directory and supply actual same-room recordings at its expected video paths. Relative paths resolve against the repository root. Optional camera/depth/pose/IMU fields can remain null.

Start from [smartphone_annotations.template.json](../examples/smartphone_annotations.template.json). It deliberately contains no objects, invented coordinates or declared real provenance. Use the application's **Annotation workspace** to inspect both original recordings, assign persistent object identities, capture frame observations and export annotations. Document the same-location relationship and actual physical changes. Record annotation provenance and obtain an independent review before evaluation. Change `data_origin` to `REAL` only for verified real recordings. Leave metric scale unverified unless measured. Record the complete object inventory before reporting detection precision and recall.

Inference requires only the two videos. A genuine supervised benchmark additionally requires validated same-room relationship evidence, physical-change documentation, reviewed labels, successful shared registration and explicit object correspondence. Model predictions are not ground truth.

## TUM RGB-D

The [official download page](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download) and proposed small `freiburg1_xyz` archive were checked. Both requests were blocked by the environment network policy with HTTP tunnel 403; see [tum_access_check.json](../reports/tum_access_check.json). No TUM files, calibration, license or sequence size have been verified locally. The `cvg.cit.tum.de` domain addition is saved in the environment draft; review, save and publish the environment settings to activate it, then retry the official source. Do not substitute an unofficial mirror or infer source calibration.

When accessible, verify the source license and archive size before acquisition, preserve timestamp lists and pose conventions, associate frames within a documented tolerance, and use the officially specified depth scale and calibration. TUM localization/reconstruction sequences must not be labeled as ground-truth changed scenes. Existing ARKitScenes and 7-Scenes data remain available for independently documented reconstruction controls.

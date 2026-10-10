# Known limitations and diagnosed failures

- Missing real videos/annotations: real-world acceptance and performance remain unrun.
- Hugging Face access returned proxy HTTP 403 on 2026-10-09. Required domains saved in environment draft; saving does not activate them. Access subsequently succeeded after the user updated settings. Four pretrained model inference smoke checks passed; historical HTTP 403 is resolved.
- No exposed CUDA GPU. VGGT two-frame CPU memory/runtime are measured in docs/vggt_smoke.json; larger workloads and SAM 2 video tracking remain unprofiled. CPU geometry works on synthetic textured scenes only.
- Pure rotation, low texture, repeated appearance, blur and poor overlap can prevent sparse SfM; the pipeline reports failure and does not invent a world model.
- Foreground-dominated registration, repeated textures and heuristic intrinsics can produce misleading correspondence fits. Detected foreground exclusion and held-out correspondence gates are implemented; undetected foreground, metric scale, independent scene validation and calibrated covariance remain unresolved.
- Sparse point absence cannot prove removal/addition. Candidate regions remain UNKNOWN without positive free-space observations.
- Identical objects preserve ambiguous associations. Semantic reasoning is integrated, but real-video MOVED accuracy remains unmeasured.
- Decoder orientation was disabled by default in the installed OpenCV build; explicit enabling was implemented and checked with MOV rotation metadata. Calibration rotation must still be supplied by the caller.
- Small-image LK pyramid levels caused track losses; coarsest-level size now adapts to the support window.
- Clean CPU installation initially imported optional torch before the unloaded-model guard; the guard was moved before the import. Neural extras are no longer required for that diagnostic.
- Starlette's current TestClient warns about httpx deprecation. HTTP app requests work; this warning is not suppressed.
- Optional sensor data is validated and retained; depth reprojection, IMU fusion and externally supplied pose reconstruction are not yet connected.
- Uploaded recordings are local files requiring user-managed retention. The app is a local single-worker developer tool without public deployment authentication.
- Automatic pipeline resume, full bundle adjustment, benchmark ablations and actual model comparison remain unimplemented.

Hugging Face Xet initially attempted a read-only home cache despite model cache_dir; writable HF_HOME/HF_XET_CACHE fixed downloads. Grounding DINO produced an unverified table label on synthetic panels; do not promote it to a verified object identity.

- A semantic stage initially failed when NumPy integer bounding boxes reached JSON serialization. Native integer conversion fixed it; successful fresh neural/browser runs verified the correction. The original partial failure is not counted as passing.
- Long recordings previously exhausted the frame budget near the beginning. Budgeted extraction now samples the full presentation-timestamp range, including the final frame; source timestamps remain intact.
- Predicted neural free space is not a measured empty-room observation. Addition/removal rules demand distinct registered views and produce probable rather than verified labels.
- SAM 2 is used as an image predictor. Dense flow is a diagnostic propagation fallback, not a tested implementation of SAM 2 video-memory tracking.

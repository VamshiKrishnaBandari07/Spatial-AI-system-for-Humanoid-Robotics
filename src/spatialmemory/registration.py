import numpy as np
from .contracts import Registration
from .geometry import match


def similarity(source, target):
    if len(source) < 3:
        raise ValueError("At least three correspondences required")
    a = source - source.mean(0)
    b = target - target.mean(0)
    if np.linalg.matrix_rank(a) < 2 or np.linalg.matrix_rank(b) < 2:
        raise ValueError("Degenerate alignment")
    U, s, V = np.linalg.svd(b.T @ a / len(a))
    D = np.eye(3)
    D[-1, -1] = np.sign(np.linalg.det(U @ V))
    R = U @ D @ V
    scale = float(np.sum(s * np.diag(D)) / np.mean(np.sum(a * a, axis=1)))
    t = target.mean(0) - scale * R @ source.mean(0)
    return scale, R, t


def transform(points, scale, R, t):
    return scale * (points @ R.T) + t


def ransac(source, target, threshold=0.08, seed=7, iterations=500):
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    rng = np.random.default_rng(seed)
    best = np.zeros(len(source), bool)
    if len(source) >= 6:
        for _ in range(iterations):
            ids = rng.choice(len(source), 3, replace=False)
            try:
                s, R, t = similarity(source[ids], target[ids])
            except (ValueError, np.linalg.LinAlgError):
                continue
            mask = (
                np.linalg.norm(transform(source, s, R, t) - target, axis=1) < threshold
            )
            if mask.sum() > best.sum():
                best = mask
    if best.sum() < 6:
        return Registration(
            valid=False,
            scale=1,
            rotation=np.eye(3).tolist(),
            translation=[0, 0, 0],
            inliers=int(best.sum()),
            correspondences=len(source),
            rmse=None,
            threshold=threshold,
            reason="Insufficient nondegenerate consensus",
        ), best
    s, R, t = similarity(source[best], target[best])
    errors = np.linalg.norm(transform(source, s, R, t) - target, axis=1)
    best = errors < threshold
    valid = best.sum() >= 6 and best.mean() >= 0.35
    return Registration(
        valid=bool(valid),
        scale=s,
        rotation=R.tolist(),
        translation=t.tolist(),
        inliers=int(best.sum()),
        correspondences=len(source),
        rmse=float(np.sqrt(np.mean(errors[best] ** 2))) if best.any() else None,
        threshold=threshold,
        reason="Validated sparse correspondence consensus; static background assumed"
        if valid
        else "Low inlier ratio",
    ), best


def register(before, after, config, before_masks=None, after_masks=None):
    pairs = set()
    excluded = 0
    before_masks = before_masks or {}
    after_masks = after_masks or {}

    def foreground(reconstruction, frame, key, masks):
        mask = masks.get(frame)
        if mask is None:
            return False
        x, y = np.rint(reconstruction.features[frame][0][key]).astype(int)
        return 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1] and bool(mask[y, x])

    for fa, (_, da, _) in before.features.items():
        for fb, (_, db, _) in after.features.items():
            for i, j in match(da, db):
                if foreground(before, fa, i, before_masks) or foreground(
                    after, fb, j, after_masks
                ):
                    excluded += 1
                    continue
                if (fa, i) in before.tracks and (fb, j) in after.tracks:
                    pairs.add((before.tracks[(fa, i)], after.tracks[(fb, j)]))
    # One-to-one identity constraints reduce repetitive-texture ambiguity.
    from collections import Counter

    ca = Counter(a for a, b in pairs)
    cb = Counter(b for a, b in pairs)
    pairs = [(a, b) for a, b in sorted(pairs) if ca[a] == 1 and cb[b] == 1]
    target = np.array([before.points[a] for a, b in pairs]).reshape(-1, 3)
    source = np.array([after.points[b] for a, b in pairs]).reshape(-1, 3)
    # Random held-out correspondences check generalization of the fit.
    # This is not an independent scene-level accuracy or covariance estimate.
    if len(source) >= 20:
        rng = np.random.default_rng(config.seed)
        order = rng.permutation(len(source))
        n = max(4, len(source) // 5)
        test = order[:n]
        train = order[n:]
        candidate, _ = ransac(
            source[train], target[train], config.registration_threshold, config.seed
        )
        errors = np.linalg.norm(
            transform(
                source[test],
                candidate.scale,
                np.array(candidate.rotation),
                np.array(candidate.translation),
            )
            - target[test],
            axis=1,
        )
        good = errors < config.registration_threshold
        reg, _ = ransac(source, target, config.registration_threshold, config.seed)
        reg.heldout_count = n
        reg.heldout_inliers = int(good.sum())
        reg.heldout_rmse = (
            float(np.sqrt(np.mean(errors[good] ** 2))) if good.any() else None
        )
        if not candidate.valid or good.mean() < 0.35:
            reg.valid = False
            reg.reason = "Held-out registration validation failed"
    else:
        reg = ransac(source, target, config.registration_threshold, config.seed)[0]
    reg.foreground_matches_excluded = excluded
    return reg

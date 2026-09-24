"""传播副本来源定位（局部特征匹配 + RANSAC，只读）。

设计概述
========
* 以某水印任务的含水印原图 ``watermarked.png`` 为**参照图**，用户上传一张从该图
  经裁切、缩放或 JPEG 转存得到的**传播副本**。
* 两侧用 SIFT 提取局部特征，BFMatcher 做 KNN 匹配并施加 Lowe 最近/次近距离比
  筛选，再以 RANSAC 估计「上传图 → 参照图」的单应矩阵（透视变换）。
* 把上传图四角经单应矩阵投影到参照图坐标系，得到来源四边形。
* **仅当内点数量、内点比例、投影四边形的凸性/边长比/面积/边界以及内点对四边形
  的区域支撑全部通过门限时**才返回四角坐标、覆盖率与多边形标注预览；任一环节
  不可信都返回 ``located=False`` 与明确的失败原因，**绝不给出伪位置**。
* 全程只读：上传图仅驻留内存，参照图只读打开，不写任务目录、不产生检测记录、
  不追加报告归档。
"""

from __future__ import annotations

import base64

import cv2
import numpy as np

# ---------------------------------------------------------------- 门限参数

MIN_UPLOAD_SIDE = 64              # 上传图最短边下限（px）
MAX_FEATURES = 4000               # 每侧最多参与匹配的 SIFT 特征数
MATCH_RATIO = 0.75                # Lowe 最近/次近距离比
RANSAC_REPROJ_PX = 4.0            # RANSAC 重投影门限（参照图像素）
RANSAC_CONFIDENCE = 0.995

MIN_REFERENCE_KEYPOINTS = 16      # 参照图特征点过少则无法定位
MIN_QUERY_KEYPOINTS = 20          # 上传图特征点过少（如纯色/过度压缩）
MIN_INLIERS = 12                  # RANSAC 内点绝对数量下限
MIN_INLIER_RATIO = 0.55           # 内点占通过比值筛选匹配的比例下限
MIN_REGION_SUPPORT = 0.75         # 落在投影四边形内（含边界）的内点比例下限

MIN_AREA_FRACTION = 0.005         # 投影四边形面积 / 参照图面积：0.5%
MAX_AREA_FRACTION = 1.02          # 允许整图 JPEG 转存（≈100%）及小幅数值外扩
BOUND_MARGIN = 0.02               # 四角允许越出参照图边界的相对余量
MAX_EDGE_RATIO = 6.0              # 四边形最长边 / 最短边上限（透视退化）
MIN_EDGE_PX = 8.0                 # 四边形最短边绝对下限（参照图像素）

PREVIEW_MAX_WIDTH = 1024

CORNER_LABELS = ("top_left", "top_right", "bottom_right", "bottom_left")
CORNER_TAGS = ("TL", "TR", "BR", "BL")


class LocateError(ValueError):
    """请求级参数错误（上传图无法解码 / 尺寸不合法），映射为 HTTP 400。"""


def _fail(reason: str, detail: str, stats: dict) -> dict:
    return {
        "located": False,
        "reason": reason,
        "reason_detail": detail,
        "stats": stats,
        "corners": None,
        "coverage": None,
        "preview": None,
    }


def _stats(reference_shape, query_shape) -> dict:
    return {
        "reference_width": int(reference_shape[1]),
        "reference_height": int(reference_shape[0]),
        "query_width": int(query_shape[1]),
        "query_height": int(query_shape[0]),
        "keypoints_reference": None,
        "keypoints_query": None,
        "good_matches": None,
        "inliers": None,
        "inlier_ratio": None,
        "rmse_px": None,
        "polygon_area_fraction": None,
        "inlier_region_support": None,
    }


def _polygon_is_valid(poly: np.ndarray) -> tuple[bool, float, float]:
    """凸性与边长合理性。返回 (是否有效, 最短边长, 最长/最短比)。"""
    if not np.all(np.isfinite(poly)):
        return False, 0.0, float("inf")
    if not bool(cv2.isContourConvex(poly.astype(np.float32))):
        return False, 0.0, float("inf")
    edges = np.roll(poly, -1, axis=0) - poly
    lengths = np.hypot(edges[:, 0], edges[:, 1])
    shortest = float(lengths.min())
    ratio = float(lengths.max() / shortest) if shortest > 1e-9 else float("inf")
    if shortest < MIN_EDGE_PX or ratio > MAX_EDGE_RATIO:
        return False, shortest, ratio
    return True, shortest, ratio


def _inlier_rmse(H: np.ndarray, src: np.ndarray, dst: np.ndarray,
                 inlier_mask: np.ndarray) -> float:
    if not int(inlier_mask.sum()):
        return 0.0
    mapped = cv2.perspectiveTransform(
        src[inlier_mask].reshape(-1, 1, 2), H).reshape(-1, 2)
    err = mapped - dst[inlier_mask]
    return float(np.sqrt(np.mean(err[:, 0] ** 2 + err[:, 1] ** 2)))


def _region_support(poly: np.ndarray, dst_inliers: np.ndarray) -> float:
    """落在投影四边形内（含小幅边界容差）的内点占比。"""
    if len(dst_inliers) == 0:
        return 0.0
    # 角点重投影误差可能让边缘内点落到四边形外侧几个像素：把多边形相对
    # 质心外扩 2% 再判定。
    centroid = poly.mean(axis=0)
    grown = (centroid + (poly - centroid) * 1.02).astype(np.float32)
    inside = sum(
        cv2.pointPolygonTest(grown, (float(x), float(y)),
                             measureDist=False) >= 0
        for x, y in dst_inliers
    )
    return inside / len(dst_inliers)


def _make_preview(reference: np.ndarray, poly: np.ndarray,
                  dst_inliers: np.ndarray, area_fraction: float,
                  inliers: int, good: int, rmse: float) -> str:
    """在参照图上绘制来源四边形 / 四角 / 内点，返回 JPEG data URL。"""
    ref_h, ref_w = reference.shape[:2]
    scale = min(1.0, PREVIEW_MAX_WIDTH / ref_w)
    preview_w, preview_h = (
        max(1, int(round(ref_w * scale))),
        max(1, int(round(ref_h * scale))),
    )
    if scale < 1.0:
        base = cv2.resize(reference, (preview_w, preview_h),
                          interpolation=cv2.INTER_AREA)
    else:
        base = reference.copy()
    base = cv2.cvtColor(base, cv2.COLOR_RGB2BGR)

    poly_s = (poly * scale).astype(np.int32)

    # 半透明橙色填充
    overlay = base.copy()
    cv2.fillPoly(overlay, [poly_s], color=(60, 165, 255))
    cv2.addWeighted(overlay, 0.22, base, 0.78, 0, dst=base)
    # 红色四边形边框 + 黄色四角
    cv2.polylines(base, [poly_s], isClosed=True, color=(70, 70, 255),
                  thickness=2, lineType=cv2.LINE_AA)
    for (x, y), tag in zip(poly_s, CORNER_TAGS):
        cv2.circle(base, (int(x), int(y)), 5, (0, 230, 255), 2,
                   lineType=cv2.LINE_AA)
        cv2.putText(base, tag, (int(x) + 7, int(y) - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 230, 255), 2,
                    cv2.LINE_AA)
    # 青色：RANSAC 内点在参照图上的位置
    for x, y in dst_inliers:
        cv2.circle(base, (int(round(x * scale)), int(round(y * scale))),
                   2, (200, 230, 60), -1, lineType=cv2.LINE_AA)

    lines = [
        f"coverage {area_fraction * 100:.1f}%   inliers {inliers}/{good}"
        f"   rmse {rmse:.2f}px",
        "TL/TR/BR/BL = upload corners in reference image",
    ]
    font = cv2.FONT_HERSHEY_SIMPLEX
    y0 = 8
    for i, text in enumerate(lines):
        (tw, th), _ = cv2.getTextSize(text, font, 0.5, 1)
        y = y0 + (i + 1) * (th + 8)
        cv2.rectangle(base, (6, y - th - 5), (12 + tw, y + 4),
                      (16, 20, 30), -1)
        cv2.putText(base, text, (9, y), font, 0.5, (235, 240, 250), 1,
                    cv2.LINE_AA)

    ok, buf = cv2.imencode(".jpg", base,
                           [cv2.IMWRITE_JPEG_QUALITY, 88])
    if not ok:
        raise LocateError("标注预览编码失败。")
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def locate_source(reference: np.ndarray, query: np.ndarray) -> dict:
    """以 ``reference``(RGB) 为参照定位 ``query``(RGB) 的来源四边形。

    成功（``located=True``）返回四角坐标 / 覆盖率 / 标注预览；
    匹配不足或几何退化时返回 ``located=False`` 与原因码 + 中文说明。
    """
    if reference.ndim != 3 or query.ndim != 3:
        raise LocateError("图片必须为三通道 RGB 图。")
    ref_h, ref_w = reference.shape[:2]
    q_h, q_w = query.shape[:2]
    if min(q_h, q_w) < MIN_UPLOAD_SIDE:
        raise LocateError(
            f"上传图片最短边至少需要 {MIN_UPLOAD_SIDE}px（当前 {min(q_h, q_w)}px）。")

    stats = _stats(reference.shape, query.shape)

    g_ref = cv2.cvtColor(reference, cv2.COLOR_RGB2GRAY)
    g_qry = cv2.cvtColor(query, cv2.COLOR_RGB2GRAY)
    sift = cv2.SIFT_create(nfeatures=MAX_FEATURES)
    kp_ref, des_ref = sift.detectAndCompute(g_ref, None)
    kp_qry, des_qry = sift.detectAndCompute(g_qry, None)
    n_ref = 0 if des_ref is None else len(kp_ref)
    n_qry = 0 if des_qry is None else len(kp_qry)
    stats["keypoints_reference"] = n_ref
    stats["keypoints_query"] = n_qry

    if n_ref < MIN_REFERENCE_KEYPOINTS:
        return _fail(
            "too_few_keypoints",
            f"参照图仅提取到 {n_ref} 个局部特征（需 ≥ {MIN_REFERENCE_KEYPOINTS}），"
            "无法作为定位参照。", stats)
    if n_qry < MIN_QUERY_KEYPOINTS:
        return _fail(
            "too_few_keypoints",
            f"上传图仅提取到 {n_qry} 个局部特征（需 ≥ {MIN_QUERY_KEYPOINTS}），"
            "可能过度模糊、纯色区域过大或压缩过重。", stats)

    # KNN（上传图 → 参照图）+ Lowe 距离比筛选
    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(des_qry, des_ref, k=2)
    good = [m for pair in pairs if len(pair) == 2
            for m, n in [pair] if m.distance < MATCH_RATIO * n.distance]
    stats["good_matches"] = len(good)

    if len(good) < MIN_INLIERS:
        return _fail(
            "insufficient_matches",
            f"通过距离比筛选的可信匹配仅 {len(good)} 对"
            f"（需 ≥ {MIN_INLIERS}），上传图与参照图可能不是同一来源，"
            "或画面被裁切得过多、压缩过重。", stats)

    src = np.float64([kp_qry[m.queryIdx].pt for m in good])
    dst = np.float64([kp_ref[m.trainIdx].pt for m in good])
    H, mask = cv2.findHomography(
        src, dst, cv2.RANSAC, RANSAC_REPROJ_PX, maxIters=4000,
        confidence=RANSAC_CONFIDENCE)
    if H is None or not np.all(np.isfinite(H)):
        return _fail("homography_failed",
                     "RANSAC 无法从匹配点估计稳定的透视变换（几何关系不一致）。",
                     stats)

    inlier_mask = mask.ravel().astype(bool)
    n_in = int(inlier_mask.sum())
    in_ratio = n_in / len(good)
    rmse = _inlier_rmse(H, src, dst, inlier_mask)
    stats["inliers"] = n_in
    stats["inlier_ratio"] = round(in_ratio, 4)
    stats["rmse_px"] = round(rmse, 3)

    if n_in < MIN_INLIERS:
        return _fail(
            "too_few_inliers",
            f"RANSAC 内点仅 {n_in} 个（需 ≥ {MIN_INLIERS}），"
            "匹配缺乏几何一致性，可能混入了重复纹理的错误匹配。", stats)
    if in_ratio < MIN_INLIER_RATIO:
        return _fail(
            "inlier_ratio_low",
            f"内点占可信匹配的比例仅 {in_ratio * 100:.0f}%"
            f"（需 ≥ {MIN_INLIER_RATIO * 100:.0f}%），整体匹配几何不一致，"
            "不予定位。", stats)

    # 上传图四角 → 参照图坐标（顺序 TL, TR, BR, BL）
    q_corners = np.float64(
        [[0, 0], [q_w - 1, 0], [q_w - 1, q_h - 1], [0, q_h - 1]]
    ).reshape(-1, 1, 2)
    try:
        poly = cv2.perspectiveTransform(q_corners, H).reshape(-1, 2)
    except cv2.error:
        return _fail("geometry_degenerate",
                     "投影四边形计算失败（单应矩阵退化）。", stats)

    ok, shortest, edge_ratio = _polygon_is_valid(poly)
    if not ok:
        return _fail(
            "geometry_degenerate",
            "投影四边形几何退化（非凸、自交或边长比过大"
            f" {edge_ratio:.1f}:1，上限 {MAX_EDGE_RATIO:.0f}:1），结果不可信。",
            stats)

    area = float(cv2.contourArea(poly.astype(np.float32)))
    area_fraction = area / float(ref_w * ref_h)
    support = _region_support(poly, dst[inlier_mask])
    stats["polygon_area_fraction"] = round(area_fraction, 6)
    stats["inlier_region_support"] = round(support, 4)

    if not (MIN_AREA_FRACTION <= area_fraction <= MAX_AREA_FRACTION):
        return _fail(
            "area_untrusted",
            f"投影四边形面积占参照图 {area_fraction * 100:.2f}%，超出可信区间 "
            "0.5%~102%（100% 对应整图转存），"
            "疑似错误匹配导致的缩放失真。", stats)

    margin_x, margin_y = BOUND_MARGIN * ref_w, BOUND_MARGIN * ref_h
    if (poly[:, 0].min() < -margin_x or poly[:, 0].max() > ref_w + margin_x
            or poly[:, 1].min() < -margin_y
            or poly[:, 1].max() > ref_h + margin_y):
        return _fail(
            "out_of_bounds",
            "投影四边形明显越出参照图边界（超出 2% 容差），"
            "变换与「上传图是参照图的一块区域」这一前提矛盾。", stats)

    if support < MIN_REGION_SUPPORT:
        return _fail(
            "region_support_low",
            f"仅 {support * 100:.0f}% 的几何内点落在投影四边形内"
            f"（需 ≥ {MIN_REGION_SUPPORT * 100:.0f}%），四边形缺乏匹配点支撑。",
            stats)

    corners = [
        {"label": label, "x": round(float(x), 2), "y": round(float(y), 2)}
        for label, (x, y) in zip(CORNER_LABELS, poly)
    ]
    min_x, min_y = poly.min(axis=0)
    max_x, max_y = poly.max(axis=0)
    coverage = {
        "area_fraction": round(area_fraction, 6),
        "bbox": {
            "min_x": round(float(min_x), 2), "min_y": round(float(min_y), 2),
            "max_x": round(float(max_x), 2), "max_y": round(float(max_y), 2),
        },
        "bbox_width_fraction": round(float((max_x - min_x) / ref_w), 6),
        "bbox_height_fraction": round(float((max_y - min_y) / ref_h), 6),
    }

    preview_url = _make_preview(
        reference, poly, dst[inlier_mask], area_fraction,
        n_in, len(good), rmse)

    return {
        "located": True,
        "reason": None,
        "reason_detail": None,
        "stats": stats,
        "corners": corners,
        "coverage": coverage,
        "preview": {
            "media_type": "image/jpeg",
            "encoding": "base64-data-url",
            "width": int(round(ref_w * min(1.0, PREVIEW_MAX_WIDTH / ref_w))),
            "image_data_url": preview_url,
        },
    }

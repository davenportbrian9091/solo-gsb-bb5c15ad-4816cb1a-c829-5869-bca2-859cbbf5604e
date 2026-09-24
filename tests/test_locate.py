"""传播副本来源定位：几何门限纯逻辑 + 端到端 API（只读不改写）。

运行（需项目依赖）：
    WM_DATA_DIR 由本测试自动指向临时目录；
    python -m pytest tests/test_locate.py -q
或直接：
    python tests/test_locate.py
"""
import io
import os
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

IMG_SIZE = 512


# ---------------------------------------------------------------- 测试图

def _make_textured(size: int = IMG_SIZE, seed: int = 7) -> np.ndarray:
    """生成富含局部特征的合成图（矩形/圆/线段 + 渐变 + 噪声）。"""
    rng = np.random.RandomState(seed)
    yy, xx = np.mgrid[0:size, 0:size]
    rgb = np.empty((size, size, 3), np.uint8)
    rgb[..., 0] = (180 + 40 * np.sin(xx / 90)).clip(0, 255).astype(np.uint8)
    rgb[..., 1] = (170 + 40 * np.cos(yy / 110)).clip(0, 255).astype(np.uint8)
    rgb[..., 2] = (160 + 30 * np.sin((xx + yy) / 150)).clip(0, 255).astype(np.uint8)
    for _ in range(140):
        x0, y0 = rng.randint(0, size, 2)
        w0, h0 = rng.randint(12, 90), rng.randint(12, 90)
        col = tuple(int(v) for v in rng.randint(0, 256, 3))
        if rng.rand() < 0.5:
            cv2.rectangle(rgb, (x0, y0),
                          (min(size - 1, x0 + w0), min(size - 1, y0 + h0)), col, -1)
        else:
            cv2.circle(rgb, (x0, y0), rng.randint(8, 50), col, -1)
    for _ in range(60):
        p1 = tuple(int(v) for v in rng.randint(0, size, 2))
        p2 = tuple(int(v) for v in rng.randint(0, size, 2))
        cv2.line(rgb, p1, p2, tuple(int(v) for v in rng.randint(0, 256, 3)),
                 int(rng.randint(1, 4)))
    rgb = (rgb.astype(np.int16)
           + rng.normal(0, 6, rgb.shape)).clip(0, 255).astype(np.uint8)
    return rgb


def _jpeg(arr: np.ndarray, quality: int = 50) -> np.ndarray:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG", quality=quality, subsampling=2)
    return np.array(Image.open(buf).convert("RGB"))


def _resize(arr: np.ndarray, wh: tuple[int, int]) -> np.ndarray:
    return np.array(Image.fromarray(arr).resize(wh, Image.BILINEAR))


def _png_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------- 几何门限

def test_polygon_is_valid_gates():
    from app.locate import _polygon_is_valid

    # 正常矩形
    ok, shortest, ratio = _polygon_is_valid(
        np.float64([[0, 0], [100, 0], [100, 80], [0, 80]]))
    assert ok and shortest == 80 and ratio == 1.25

    # 自交（蝴蝶形，非凸）
    ok, _, _ = _polygon_is_valid(
        np.float64([[0, 0], [100, 100], [100, 0], [0, 100]]))
    assert not ok

    # 针状退化：边长比超 6:1
    ok, _, ratio = _polygon_is_valid(
        np.float64([[0, 0], [400, 0], [400, 1], [0, 1]]))
    assert not ok and ratio > 6.0

    # 最短边 < 8px：前两个顶点仅相距 3px 的退化四边形
    ok, shortest, _ = _polygon_is_valid(
        np.float64([[0, 0], [3, 0], [100, 100], [0, 100]]))
    assert not ok and shortest < 8

    # 非有限值
    ok, _, _ = _polygon_is_valid(
        np.float64([[0, 0], [np.inf, 0], [100, 100], [0, 100]]))
    assert not ok


def test_region_support():
    from app.locate import _region_support

    poly = np.float64([[10, 10], [110, 10], [110, 110], [10, 110]])
    inside = np.float64([[20, 20], [100, 100], [60, 60]])
    outside = np.float64([[500, 500], [0, 0]])
    pts = np.vstack([inside, outside])
    # 3 内 / 2 外（顶点略在外扩容差外）
    assert _region_support(poly, pts) == 0.6
    assert _region_support(poly, inside) == 1.0
    assert _region_support(poly, np.zeros((0, 2))) == 0.0


def test_fail_payload_has_no_coordinates():
    """失败结果绝不携带坐标 / 覆盖率 / 预览。"""
    from app.locate import _fail

    out = _fail("insufficient_matches", "原因", {"good_matches": 3})
    assert out["located"] is False
    assert out["corners"] is None and out["coverage"] is None
    assert out["preview"] is None and out["reason"] == "insufficient_matches"


# ---------------------------------------------------------------- 算法端到端

def _watermarked(seed: int = 7):
    from app import dctwm as wm
    plain = _make_textured(seed=seed)
    return wm.embed_watermark(plain, "定位测试-2026", strength="medium").image


def test_locate_crop_scale_jpeg():
    from app.locate import locate_source

    ref = _watermarked()
    x0, y0, x1, y1 = 130, 90, 400, 360
    query = _jpeg(_resize(ref[y0:y1, x0:x1], (150, 150)), quality=40)
    r = locate_source(ref, query)
    assert r["located"] is True, r["reason_detail"]
    by = {c["label"]: (c["x"], c["y"]) for c in r["corners"]}
    truth = {"top_left": (x0, y0), "top_right": (x1, y0),
             "bottom_right": (x1, y1), "bottom_left": (x0, y1)}
    for label, (tx, ty) in truth.items():
        px, py = by[label]
        assert abs(px - tx) <= 4 and abs(py - ty) <= 4, (label, px, py, tx, ty)

    cov = r["coverage"]
    assert 0.20 < cov["area_fraction"] < 0.35
    assert cov["bbox"]["min_x"] <= x0 + 4 and cov["bbox"]["max_x"] >= x1 - 4
    s = r["stats"]
    assert s["inliers"] >= 12 and s["inlier_ratio"] >= 0.55
    assert s["inlier_region_support"] >= 0.75
    assert r["preview"]["media_type"] == "image/jpeg"
    assert r["preview"]["image_data_url"].startswith("data:image/jpeg;base64,")


def test_locate_full_image_jpeg_resave():
    """整图 JPEG 转存（无裁切、无缩放）也必须能定位且面积 ≈ 100%。"""
    from app.locate import locate_source

    ref = _watermarked()
    r = locate_source(ref, _jpeg(ref, quality=35))
    assert r["located"] is True, r["reason_detail"]
    assert 0.95 <= r["coverage"]["area_fraction"] <= 1.02
    s = r["stats"]
    assert s["inliers"] >= 50 and s["inlier_ratio"] >= 0.8


def test_locate_unrelated_image_fails_without_coordinates():
    from app.locate import locate_source

    ref = _watermarked(seed=7)
    other = _make_textured(seed=999)
    r = locate_source(ref, other)
    assert r["located"] is False
    assert r["reason"] in (
        "too_few_inliers", "inlier_ratio_low", "insufficient_matches")
    assert r["corners"] is None and r["coverage"] is None
    assert r["preview"] is None and r["reason_detail"]


def test_locate_min_side_validation():
    from app.locate import locate_source, LocateError, MIN_UPLOAD_SIDE

    ref = _watermarked()
    tiny = _resize(ref[50:150, 50:150], (40, 40))
    try:
        locate_source(ref, tiny)
    except LocateError as e:
        assert str(MIN_UPLOAD_SIDE) in str(e)
    else:
        raise AssertionError("过短边图片应被拒绝")


# ---------------------------------------------------------------- API 端到端

def _fresh_client():
    tmp = tempfile.mkdtemp(prefix="wm-locate-")
    os.environ["WM_DATA_DIR"] = tmp
    os.environ["WM_MAX_JOBS"] = "200"
    for mod in ("app.main", "app.archive"):
        sys.modules.pop(mod, None)
    from fastapi.testclient import TestClient
    from app.main import app, DATA_DIR  # noqa: E402

    return TestClient(app), Path(DATA_DIR)


def _embed_job(client, seed=7):
    r = client.post(
        "/api/embed",
        files={"image": ("img.png", _png_bytes(_make_textured(seed=seed)),
                         "image/png")},
        data={"text": "定位测试-2026", "strength": "medium"},
    )
    assert r.status_code == 200, r.text
    return r.json()["job_id"]


def test_api_locate_success():
    client, data_dir = _fresh_client()
    job = _embed_job(client)

    ref = np.array(Image.open(data_dir / job / "watermarked.png").convert("RGB"))
    x0, y0, x1, y1 = 130, 90, 400, 360
    query = _jpeg(_resize(ref[y0:y1, x0:x1], (160, 150)), quality=45)
    r = client.post(
        f"/api/jobs/{job}/locate-source",
        files={"image": ("copy.jpg", _jpeg_to_bytes(query), "image/jpeg")},
    )
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["located"] is True
    assert data["job_id"] == job and data["stored"] is False
    assert [c["label"] for c in data["corners"]] == [
        "top_left", "top_right", "bottom_right", "bottom_left"]
    for c in data["corners"]:
        assert set(c) == {"label", "x", "y"}
        assert -10 < c["x"] < IMG_SIZE + 10 and -10 < c["y"] < IMG_SIZE + 10
    assert set(data["stats"]) >= {
        "keypoints_reference", "keypoints_query", "good_matches",
        "inliers", "inlier_ratio", "rmse_px", "polygon_area_fraction",
        "inlier_region_support",
        "reference_width", "reference_height", "query_width", "query_height"}
    assert data["coverage"]["area_fraction"] > 0
    assert data["preview"]["image_data_url"].startswith("data:image/")


def _jpeg_to_bytes(arr):
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def test_api_locate_unrelated_is_200_with_reason():
    client, _ = _fresh_client()
    job = _embed_job(client, seed=7)
    r = client.post(
        f"/api/jobs/{job}/locate-source",
        files={"image": ("other.png", _png_bytes(_make_textured(seed=999)),
                         "image/png")},
    )
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["located"] is False
    assert data["reason"] and data["reason_detail"]
    assert data["corners"] is None and data["coverage"] is None
    assert data["preview"] is None


def test_api_locate_read_only():
    """定位前后任务目录与归档字节级不变。"""
    client, data_dir = _fresh_client()
    job = _embed_job(client)

    # 先产生 target / 检测记录 / 归档，证明定位不动这些东西
    assert client.post("/api/attack",
                       json={"job_id": job, "jpeg_quality": 70,
                             "crop_fraction": 0.75}).status_code == 200
    det = client.post("/api/detect", data={"job_id": job})
    assert det.status_code == 200
    det_id = det.json()["detection_id"]

    job_dir = data_dir / job
    before = {p.relative_to(job_dir).as_posix(): (p.stat().st_mtime_ns,
                                                  p.read_bytes())
              for p in job_dir.rglob("*") if p.is_file()}
    arch = data_dir / "archive.log.jsonl"
    arch_before = arch.read_bytes() if arch.exists() else None

    ref = np.array(Image.open(job_dir / "watermarked.png").convert("RGB"))
    query = _jpeg(_resize(ref[100:380, 120:420], (200, 200)), quality=55)
    r = client.post(
        f"/api/jobs/{job}/locate-source",
        files={"image": ("c.jpg", _jpeg_to_bytes(query), "image/jpeg")},
    )
    assert r.status_code == 200 and r.json()["located"] is True

    after = {p.relative_to(job_dir).as_posix(): (p.stat().st_mtime_ns,
                                                 p.read_bytes())
             for p in job_dir.rglob("*") if p.is_file()}
    assert after == before
    assert (arch.read_bytes() if arch.exists() else None) == arch_before
    # 没有新增检测记录
    det_files = sorted(p.name for p in (job_dir / "detections").iterdir())
    assert det_files == [f"{det_id}.json", f"{det_id}.png",
                         f"{det_id}.source.png"]


def test_api_locate_errors():
    client, _ = _fresh_client()
    job = _embed_job(client)

    # 空上传
    r = client.post(f"/api/jobs/{job}/locate-source",
                    files={"image": ("x.jpg", b"", "image/jpeg")})
    assert r.status_code == 400
    # 非图片
    r = client.post(f"/api/jobs/{job}/locate-source",
                    files={"image": ("x.jpg", b"not-an-image", "image/jpeg")})
    assert r.status_code == 400
    # 最短边 < 64
    tiny = _png_bytes(np.full((50, 200, 3), 128, np.uint8))
    r = client.post(f"/api/jobs/{job}/locate-source",
                    files={"image": ("tiny.png", tiny, "image/png")})
    assert r.status_code == 400
    # 任务不存在
    r = client.post(
        "/api/jobs/000000000000/locate-source",
        files={"image": ("c.jpg", _jpeg_to_bytes(_jpeg(_watermarked())),
                         "image/jpeg")},
    )
    assert r.status_code == 404


def test_api_locate_rejected_for_upload_only_job():
    client, data_dir = _fresh_client()
    emb = client.post(
        "/api/embed",
        files={"image": ("img.png", _png_bytes(_make_textured()), "image/png")},
        data={"text": "盲检测", "strength": "medium"})
    wm_bytes = (data_dir / emb.json()["job_id"] / "watermarked.png").read_bytes()
    det = client.post("/api/detect",
                      files={"image": ("up.png", wm_bytes, "image/png")})
    up_job = det.json()["job_id"]
    assert det.json()["upload_only"] is True
    r = client.post(
        f"/api/jobs/{up_job}/locate-source",
        files={"image": ("c.jpg", wm_bytes, "image/png")})
    assert r.status_code == 404


if __name__ == "__main__":
    import inspect
    sys.modules.pop("app.main", None)
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and inspect.isfunction(v)]
    for fn in fns:
        fn()
        print(f"[PASS] {fn.__name__}")
    print(f"\n{len(fns)} 个来源定位测试全部通过 ✔")

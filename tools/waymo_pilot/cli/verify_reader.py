#!/usr/bin/env python
"""Exercise the Waymo reader on a SYNTHETIC tfrecord. Runs in the waymo env, no real data.

The reader's camera path -- calibration parsing, the nine-element intrinsic vector, the
camera-to-vehicle extrinsic, the per-image pose and timestamps, and the axis permutation --
can be checked without the dataset by writing a Frame proto we construct ourselves and
reading it back through the same ``read_segment`` the pilot will use. The LiDAR path needs
real compressed range images and is NOT covered here; that is stated rather than implied.

    /media/SSD1/MINH_DATASETS/envs/waymo/bin/python tools/waymo_pilot/cli/verify_reader.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from tools.waymo_pilot import config as C, waymo as W                         # noqa: E402

#: a deliberately asymmetric calibration, so a transposed or swapped field is visible
FX, FY, CX, CY = 2055.5, 2060.25, 959.5, 641.25
WIDTH, HEIGHT = 1920, 1280
#: camera 1.5 m forward, 0.25 m left, 2.1 m up of the vehicle origin, Waymo axes
CAM_T = np.array([1.5, 0.25, 2.1])


def _frame_proto(dataset_pb2, tf):
    f = dataset_pb2.Frame()
    f.context.name = "synthetic_segment"
    f.timestamp_micros = 1_550_000_000_000_000
    f.pose.transform.extend(np.eye(4).ravel().tolist())
    cal = f.context.camera_calibrations.add()
    cal.name = W.CAMERA_ID["FRONT"]
    cal.intrinsic.extend([FX, FY, CX, CY, 0.1, -0.2, 0.001, 0.002, 0.03])
    ext = np.eye(4)
    ext[:3, 3] = CAM_T
    cal.extrinsic.transform.extend(ext.ravel().tolist())
    cal.width, cal.height = WIDTH, HEIGHT
    cal.rolling_shutter_direction = 1
    img = f.images.add()
    img.name = W.CAMERA_ID["FRONT"]
    # a solid colour: JPEG is lossy, so a single bright pixel in an 8x8 block does not
    # survive the DCT and the check would be testing the codec, not the reader
    rgb = np.zeros((16, 16, 3), np.uint8)
    rgb[:, :] = (220, 30, 40)
    img.image = tf.io.encode_jpeg(tf.convert_to_tensor(rgb)).numpy()
    vp = np.eye(4)
    vp[:3, 3] = [10.0, 20.0, 0.5]
    img.pose.transform.extend(vp.ravel().tolist())
    img.pose_timestamp = 1_550_000_000.021        # 21 ms after the frame timestamp
    img.camera_trigger_time = 1_550_000_000.018
    img.camera_readout_done_time = 1_550_000_000.044
    img.shutter = 0.0005
    return f


def main() -> int:
    try:
        import tensorflow.compat.v1 as tf
        from waymo_open_dataset import dataset_pb2
    except Exception as exc:                                        # noqa: BLE001
        print(f"BLOCKED: this script needs the waymo env ({type(exc).__name__}: {exc})",
              file=sys.stderr)
        return 2
    tf.enable_eager_execution()

    tmp = tempfile.mkdtemp(prefix="waymo_reader_")
    path = os.path.join(tmp, "synthetic.tfrecord")
    with tf.io.TFRecordWriter(path) as wtr:
        wtr.write(_frame_proto(dataset_pb2, tf).SerializeToString())

    got = list(W.read_segment(path))
    ok, fails = True, []

    def expect(name, cond, detail=""):
        nonlocal ok
        if not cond:
            ok = False
            fails.append(f"{name}: {detail}")
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))

    expect("one frame parsed", len(got) == 1, f"got {len(got)}")
    if not got:
        return 1
    idx, fr = got[0]
    cam = fr["cameras"]["FRONT"]

    expect("segment name", fr["segment"] == "synthetic_segment", fr["segment"])
    expect("intrinsics unpacked in Waymo order (fu, fv, cu, cv)",
           np.allclose([cam.K[0, 0], cam.K[1, 1], cam.K[0, 2], cam.K[1, 2]],
                       [FX, FY, CX, CY]), f"K diag/offsets {cam.K[0,0]},{cam.K[1,1]},"
                                          f"{cam.K[0,2]},{cam.K[1,2]}")
    expect("distortion carried, not dropped",
           np.allclose(cam.distortion, [0.1, -0.2, 0.001, 0.002, 0.03]), str(cam.distortion))
    expect("image size", (cam.width, cam.height) == (WIDTH, HEIGHT),
           f"{cam.width}x{cam.height}")
    expect("extrinsic is camera->vehicle", np.allclose(cam.cam_to_vehicle[:3, 3], CAM_T),
           str(cam.cam_to_vehicle[:3, 3]))
    expect("per-image pose is vehicle->world at image time",
           np.allclose(fr["image_pose"]["FRONT"][:3, 3], [10.0, 20.0, 0.5]),
           str(fr["image_pose"]["FRONT"][:3, 3]))
    expect("jpeg decodes to the written pixels",
           _jpeg_ok(tf, fr["images"]["FRONT"]), "")

    # the axis permutation, applied through the real calibration object
    p_fwd = CAM_T + np.array([10.0, 0.0, 0.0])          # 10 m ahead OF THE CAMERA
    u, v, z = W.project(p_fwd[None], cam)
    expect("forward point hits the principal point",
           abs(u[0] - CX) < 1e-6 and abs(v[0] - CY) < 1e-6 and abs(z[0] - 10.0) < 1e-9,
           f"u={u[0]:.3f} v={v[0]:.3f} z={z[0]:.3f} (want {CX}, {CY}, 10)")
    u_l, _v, _z = W.project((CAM_T + [10.0, 1.0, 0.0])[None], cam)
    _u, v_u, _z2 = W.project((CAM_T + [10.0, 0.0, 1.0])[None], cam)
    expect("left of the vehicle is left in the image", u_l[0] < CX, f"u={u_l[0]:.1f}")
    expect("up in the vehicle is up in the image", v_u[0] < CY, f"v={v_u[0]:.1f}")

    sync = W.sync_error_s(fr["image_timing"], fr["frame_timestamp_s"])
    expect("sync error measured, not assumed", abs(sync["FRONT"] - 0.021) < 1e-6,
           f"{sync['FRONT'] * 1000:.1f} ms")
    expect("sync error within the pilot's bound", sync["FRONT"] <= C.MAX_SYNC_ERROR_S,
           f"{sync['FRONT']:.3f}s <= {C.MAX_SYNC_ERROR_S}s")

    print("\nNOT covered here: the LiDAR range-image path, which needs real compressed "
          "range images. It is exercised the first time a real segment is extracted.")
    print(f"\n{'READER VERIFIED' if ok else 'READER FAILURE: ' + '; '.join(fails)}")
    return 0 if ok else 1


def _jpeg_ok(tf, raw: bytes) -> bool:
    """``read_segment`` returns the JPEG as raw bytes; the extractor stores them verbatim."""
    img = tf.io.decode_jpeg(tf.constant(bytes(raw))).numpy()
    return img.shape == (16, 16, 3) and abs(float(img[..., 0].mean()) - 220.0) < 6.0


if __name__ == "__main__":
    raise SystemExit(main())

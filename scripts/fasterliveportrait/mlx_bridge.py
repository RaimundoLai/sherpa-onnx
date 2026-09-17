#!/usr/bin/env python3
"""Run the optional Apple-Silicon MLX talking-head backend.

This adapter intentionally lives outside the C++ addon.  The addon remains the
stable ONNX API; Node sends a resized RGB source frame and AuraFace's five
landmarks here, while this process runs the MIT-licensed FasterLivePortrait-MLX
reference implementation and MLX on the Apple GPU.

The reference checkout is supplied with --reference-root instead of being
vendored into sherpa-onnx.  This keeps the existing ONNX build independent of
Python/MLX and makes the backend opt-in.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import soundfile as sf


class AuraFaceSeedAnalysis:
    """Face-analysis boundary backed by AuraFace/SCRFD in the Node addon.

    The five-point seed is deliberately synthesized from AuraFace's bbox. The
    ONNX bridge crops from that bbox, while the MLX reference crop helper
    estimates its crop from landmark extents. Passing the raw five points here
    would therefore produce a much tighter source crop and a different output
    even with identical FLP weights.
    """

    def __init__(self, bbox: np.ndarray):
        if bbox.shape != (4,) or not np.isfinite(bbox).all():
            raise ValueError("AuraFace seed bbox must have shape [4]")
        left, top, right, bottom = bbox.astype(np.float32, copy=False)
        if right <= left or bottom <= top:
            raise ValueError("AuraFace seed bbox must have positive size")
        center_x = (left + right) * 0.5
        center_y = (top + bottom) * 0.5
        self.landmarks = np.asarray(
            [
                [left, top],
                [right, top],
                [center_x, center_y],
                [left, bottom],
                [right, bottom],
            ],
            dtype=np.float32,
        )

    def predict(self, _img_bgr):
        return [self.landmarks.copy()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-root", required=True)
    parser.add_argument("--weights-root", required=True)
    parser.add_argument("--source-rgb", required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--face-landmarks", required=True)
    parser.add_argument("--face-bbox", required=True)
    parser.add_argument("--audio", required=True)
    parser.add_argument(
        "--motion-f32",
        help="Optional ONNX-generated raw motion coefficients (float32 N x motion-dim)",
    )
    parser.add_argument("--motion-frames", type=int, default=0)
    parser.add_argument("--motion-dim", type=int, default=0)
    parser.add_argument("--motion-fps", type=float, default=25.0)
    parser.add_argument("--motion-diffusion-steps", type=int, default=0)
    parser.add_argument("--output-raw", required=True)
    parser.add_argument("--max-seconds", type=float, default=0.0)
    parser.add_argument("--output-fps", type=float, default=25.0)
    parser.add_argument("--diffusion-steps", type=int, default=0)
    parser.add_argument(
        "--profile",
        choices=("fast", "quality"),
        default=os.environ.get("SHERPA_ONNX_MLX_PROFILE", "quality"),
        help="MLX precision profile: fast uses bf16; quality keeps FLP core models in fp32",
    )
    parser.add_argument(
        "--cfg",
        action="store_true",
        help="Enable classifier-free guidance, matching the ONNX cfg=true option",
    )
    parser.add_argument("--cfg-scale", type=float, default=2.8)
    return parser.parse_args()


def set_model_paths(cfg, weights_root: Path, reference_root: Path) -> None:
    """Point the human MLX config at the explicit local weight snapshot."""
    liveportrait = weights_root / "liveportrait_mlx"
    joyvasa = weights_root / "JoyVASA"

    cfg.models.warping_spade.model_path = [
        str(liveportrait / "warping_module.npz"),
        str(liveportrait / "spade_generator.npz"),
    ]
    cfg.models.motion_extractor.model_path = str(liveportrait / "motion_extractor.npz")
    cfg.models.landmark.model_path = str(liveportrait / "landmark.npz")
    cfg.models.face_analysis.model_path = str(liveportrait / "landmark.npz")
    cfg.models.app_feat_extractor.model_path = str(
        liveportrait / "appearance_feature_extractor.npz"
    )
    cfg.models.stitching.model_path = str(liveportrait / "stitching.npz")
    cfg.models.stitching_eye_retarget.model_path = str(liveportrait / "stitching_eye.npz")
    cfg.models.stitching_lip_retarget.model_path = str(liveportrait / "stitching_lip.npz")

    cfg.joyvasa_models.motion_mlx_model_path = str(
        joyvasa / "motion_generator" / "motion_generator_hubert_chinese_mlx.npz"
    )
    cfg.joyvasa_models.audio_mlx_model_path = str(
        joyvasa / "audio_encoder" / "hubert_chinese_mlx.npz"
    )
    cfg.joyvasa_models.motion_template_path = str(
        joyvasa / "motion_template" / "motion_template.pkl"
    )
    cfg.infer_params.mask_crop_path = str(reference_root / "assets" / "mask_template.png")


def validate_files(args: argparse.Namespace, reference_root: Path, weights_root: Path) -> None:
    required = [
        reference_root / "configs" / "mlx_infer.yaml",
        weights_root / "liveportrait_mlx" / "appearance_feature_extractor.npz",
        weights_root / "liveportrait_mlx" / "landmark.npz",
        weights_root / "liveportrait_mlx" / "motion_extractor.npz",
        weights_root / "liveportrait_mlx" / "spade_generator.npz",
        weights_root / "liveportrait_mlx" / "warping_module.npz",
        weights_root / "liveportrait_mlx" / "stitching.npz",
        weights_root / "liveportrait_mlx" / "stitching_eye.npz",
        weights_root / "liveportrait_mlx" / "stitching_lip.npz",
        weights_root / "JoyVASA" / "motion_template" / "motion_template.pkl",
    ]
    if not args.motion_f32:
        required.extend(
            [
                weights_root / "JoyVASA" / "audio_encoder" / "hubert_chinese_mlx.npz",
                weights_root / "JoyVASA" / "motion_generator" / "motion_generator_hubert_chinese_mlx.npz",
            ]
        )
    missing = [str(path) for path in required if not path.exists()]
    missing.extend(str(path) for path in (Path(args.source_rgb), Path(args.audio)) if not path.exists())
    if missing:
        raise FileNotFoundError("Missing MLX runtime files: " + ", ".join(missing))


def load_audio_for_generation(audio_path: Path, max_seconds: float, work_dir: Path) -> Path:
    if max_seconds <= 0:
        return audio_path
    if not np.isfinite(max_seconds):
        raise ValueError("max_seconds must be finite")
    audio, sample_rate = sf.read(str(audio_path), dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    limit = max(1, int(np.floor(max_seconds * sample_rate)))
    audio = audio[:limit]
    trimmed = work_dir / "audio-trimmed.wav"
    sf.write(str(trimmed), audio, sample_rate, subtype="PCM_16")
    return trimmed


def load_external_motion(
    motion_path: Path,
    frame_count: int,
    motion_dim: int,
    motion_fps: float,
    template_path: Path,
) -> dict:
    """Decode motion generated by the Node/ONNX JoyVASA bridge.

    The JSON and pickle templates are byte-for-byte equivalent in the bundled
    assets. Decoding here keeps the MLX renderer independent from the ONNX
    rendering models while preserving the ONNX motion trajectory exactly.
    """
    if frame_count <= 0 or motion_dim <= 0:
        raise ValueError("external motion requires positive frame count and dimension")
    if not np.isfinite(motion_fps) or motion_fps <= 0:
        raise ValueError("external motion fps must be positive")
    raw = np.fromfile(str(motion_path), dtype=np.float32)
    expected = frame_count * motion_dim
    if raw.size != expected:
        raise ValueError(
            f"external motion contains {raw.size} float32 values; expected {expected}"
        )
    with template_path.open("rb") as handle:
        template = pickle.load(handle)
    from src.utils import utils

    def vector(name):
        return np.asarray(template[name], dtype=np.float32).reshape(-1)

    mean_exp = vector("mean_exp")
    std_exp = vector("std_exp")
    min_scale = float(vector("min_scale")[0])
    max_scale = float(vector("max_scale")[0])
    min_t = vector("min_t")
    max_t = vector("max_t")
    min_pitch = float(vector("min_pitch")[0])
    max_pitch = float(vector("max_pitch")[0])
    min_yaw = float(vector("min_yaw")[0])
    max_yaw = float(vector("max_yaw")[0])
    min_roll = float(vector("min_roll")[0])
    max_roll = float(vector("max_roll")[0])
    if motion_dim < 70 or mean_exp.size != 63 or std_exp.size != 63:
        raise ValueError("external motion/template dimensions are incompatible")

    motions = []
    for row in raw.reshape(frame_count, motion_dim):
        exp = (row[:63] * std_exp + mean_exp).reshape(1, 21, 3).astype(np.float32)
        scale = np.asarray(
            [[row[63] * (max_scale - min_scale) + min_scale]], dtype=np.float32
        )
        t = (row[64:67] * (max_t - min_t) + min_t).reshape(1, 3).astype(np.float32)
        pitch = np.asarray(
            [[row[67] * (max_pitch - min_pitch) + min_pitch]], dtype=np.float32
        )
        yaw = np.asarray(
            [[row[68] * (max_yaw - min_yaw) + min_yaw]], dtype=np.float32
        )
        roll = np.asarray(
            [[row[69] * (max_roll - min_roll) + min_roll]], dtype=np.float32
        )
        rotation = utils.get_rotation_matrix(
            pitch.reshape(-1), yaw.reshape(-1), roll.reshape(-1)
        ).reshape(1, 3, 3).astype(np.float32)
        motions.append(
            {
                "exp": exp,
                "scale": scale,
                "t": t,
                "R": rotation,
                "pitch": pitch,
                "yaw": yaw,
                "roll": roll,
            }
        )
    return {"n_frames": frame_count, "output_fps": motion_fps, "motion": motions}


def paste_back_bbox(source: np.ndarray, generated: np.ndarray, bbox: np.ndarray) -> np.ndarray:
    """Paste a generated crop using the same soft mask as the ONNX bridge."""
    height, width = source.shape[:2]
    generated_height, generated_width = generated.shape[:2]
    left, top, right, bottom = bbox.astype(np.float32, copy=False)
    face_width = right - left
    face_height = bottom - top
    side = max(float(face_width), float(face_height)) * 2.3
    center_x = (left + right) * 0.5
    center_y = (top + bottom) * 0.5 - side * 0.125

    yy, xx = np.indices((height, width), dtype=np.float32)
    x_from_center = xx - center_x
    y_from_center = yy - center_y
    current_x = x_from_center * generated_width / side + generated_width * 0.5
    current_y = y_from_center * generated_height / side + generated_height * 0.5
    valid = (
        (current_x >= 0)
        & (current_y >= 0)
        & (current_x < generated_width)
        & (current_y < generated_height)
    )
    nx = x_from_center / (side * 0.5)
    ny = y_from_center / (side * 0.5)
    radius = np.sqrt(nx * nx + ny * ny)
    alpha = np.where(
        radius <= 0.55,
        1.0,
        np.where(radius >= 0.92, 0.0, (0.92 - radius) / 0.37),
    ).astype(np.float32)
    valid &= alpha > 0

    output = source.copy()
    if not np.any(valid):
        return output

    valid_x = current_x[valid]
    valid_y = current_y[valid]
    x0 = np.floor(valid_x).astype(np.int32)
    y0 = np.floor(valid_y).astype(np.int32)
    x1 = np.minimum(generated_width - 1, x0 + 1)
    y1 = np.minimum(generated_height - 1, y0 + 1)
    dx = (valid_x - x0).astype(np.float32)
    dy = (valid_y - y0).astype(np.float32)
    top_left = generated[y0, x0].astype(np.float32)
    top_right = generated[y0, x1].astype(np.float32)
    bottom_left = generated[y1, x0].astype(np.float32)
    bottom_right = generated[y1, x1].astype(np.float32)
    top_value = top_left * (1.0 - dx[:, None]) + top_right * dx[:, None]
    bottom_value = bottom_left * (1.0 - dx[:, None]) + bottom_right * dx[:, None]
    animated = top_value * (1.0 - dy[:, None]) + bottom_value * dy[:, None]
    source_pixels = output[valid].astype(np.float32)
    output[valid] = np.clip(
        alpha[valid, None] * animated + (1.0 - alpha[valid, None]) * source_pixels,
        0,
        255,
    ).astype(np.uint8)
    return output


def main() -> None:
    args = parse_args()
    if args.width <= 0 or args.height <= 0:
        raise ValueError("source dimensions must be positive")
    if args.output_fps <= 0 or args.output_fps > 25:
        raise ValueError("output_fps must be in (0, 25]")
    if args.diffusion_steps not in (0, 50):
        raise ValueError(
            "MLX JoyVASA weights currently use the exported 50-step schedule; "
            "use diffusionSteps=50 for the MLX backend"
        )

    reference_root = Path(args.reference_root).expanduser().resolve()
    weights_root = Path(args.weights_root).expanduser().resolve()
    validate_files(args, reference_root, weights_root)

    # Make the reference checkout importable without installing this project.
    sys.path.insert(0, str(reference_root))
    from omegaconf import OmegaConf
    from src import models as model_registry
    from src.pipelines.faster_live_portrait_pipeline import FasterLivePortraitPipeline
    from src.pipelines.joyvasa_audio_to_motion_pipeline import JoyVASAAudio2MotionPipeline

    source_rgb = np.fromfile(args.source_rgb, dtype=np.uint8)
    expected = args.width * args.height * 3
    if source_rgb.size != expected:
        raise ValueError(f"source RGB size {source_rgb.size} does not match {expected}")
    source_rgb = source_rgb.reshape(args.height, args.width, 3)
    face_landmarks = np.asarray(json.loads(args.face_landmarks), dtype=np.float32)
    if face_landmarks.shape != (5, 2):
        raise ValueError(f"face landmarks must have shape [5, 2], got {face_landmarks.shape}")
    face_bbox = np.asarray(json.loads(args.face_bbox), dtype=np.float32)
    if face_bbox.shape != (4,):
        raise ValueError(f"face bbox must have shape [4], got {face_bbox.shape}")

    with tempfile.TemporaryDirectory(prefix="sherpa-onnx-mlx-") as temp_name:
        work_dir = Path(temp_name)
        source_path = work_dir / "source.png"
        cv2.imwrite(str(source_path), cv2.cvtColor(source_rgb, cv2.COLOR_RGB2BGR))
        audio_path = load_audio_for_generation(Path(args.audio), args.max_seconds, work_dir)

        cfg = OmegaConf.load(str(reference_root / "configs" / "mlx_infer.yaml"))
        set_model_paths(cfg, weights_root, reference_root)
        # The reference MLX config uses bf16 for the expensive FLP models. It
        # is fast, but it is not the closest match to the float32 ONNX graph.
        # Keep that path available and make a deliberate quality profile for
        # comparisons and applications where identity/detail matter more than
        # throughput.
        if args.profile == "quality":
            for model_name in ("warping_spade", "motion_extractor", "app_feat_extractor"):
                cfg.models[model_name].dtype = "fp32"
        # Node has already resized the source and has already used AuraFace to
        # find it.  Keep the Python side at the exact supplied dimensions.
        cfg.infer_params.source_max_dim = max(args.width, args.height)
        # Node already chose the final dimensions. Do not trim an odd-sized
        # frame here, otherwise the AuraFace bbox and the source image would
        # no longer share the same coordinate system.
        cfg.infer_params.source_division = 1
        # The Node ONNX path owns paste-back so both backends use the same
        # interpolation and soft alpha mask. The reference MLX mask is kept
        # out of this comparison because it can leave a visible crop boundary.
        cfg.infer_params.flag_pasteback = False
        cfg.infer_params.flag_do_crop = True
        cfg.infer_params.flag_stitching = True
        cfg.infer_params.flag_relative_motion = True
        cfg.infer_params.flag_crop_driving_video = True
        # Match the existing Node/ONNX transform exactly. The MLX reference
        # defaults to expression-friendly motion, locked scale, and lip
        # normalization; those are useful reference-pipeline choices but they
        # change the mouth trajectory relative to our ONNX bridge.
        cfg.infer_params.flag_normalize_lip = False
        cfg.infer_params.driving_option = "pose-friendly"
        cfg.infer_params.flag_lock_driving_motion_scale = False
        cfg.infer_params.cfg_scale = 1.2

        class AuraFaceSeedModel(AuraFaceSeedAnalysis):
            def __init__(self, **_kwargs):
                super().__init__(face_bbox)

        # Replace the reference Haar-based face-analysis constructor before
        # model loading.  The actual detector is AuraFace/SCRFD in Node; this
        # keeps the MLX process from loading or invoking a second detector.
        model_registry.AuraFaceSeedModel = AuraFaceSeedModel
        cfg.models.face_analysis.name = "AuraFaceSeedModel"
        cfg.models.face_analysis.model_path = ""
        pipe = FasterLivePortraitPipeline(cfg=cfg, is_animal=False)
        if not pipe.prepare_source(str(source_path), realtime=False):
            raise RuntimeError(pipe.prepare_source_error or "MLX source preparation failed")

        if args.motion_f32:
            driving = load_external_motion(
                Path(args.motion_f32),
                args.motion_frames,
                args.motion_dim,
                args.motion_fps,
                Path(cfg.joyvasa_models.motion_template_path),
            )
            if args.max_seconds > 0:
                external_frames = max(1, int(np.ceil(args.max_seconds * driving["output_fps"])))
                driving["motion"] = driving["motion"][:external_frames]
                driving["n_frames"] = min(driving["n_frames"], external_frames)
            diffusion_steps = 0
        else:
            joy = JoyVASAAudio2MotionPipeline(
                motion_mlx_model_path=cfg.joyvasa_models.motion_mlx_model_path,
                audio_mlx_model_path=cfg.joyvasa_models.audio_mlx_model_path,
                motion_template_path=cfg.joyvasa_models.motion_template_path,
                cfg_mode=cfg.infer_params.cfg_mode,
                cfg_scale=cfg.infer_params.cfg_scale,
                cfg_cond=["audio"] if args.cfg else [],
            )
            joy.cfg_scale = args.cfg_scale
            driving = joy.gen_motion_sequence(str(audio_path))
            diffusion_steps = int(joy.motion_generator.diffusion_sched.num_steps)
        motion_fps = float(driving["output_fps"])
        output_fps = min(float(args.output_fps), motion_fps)
        frame_count = max(1, int(np.ceil(driving["n_frames"] / motion_fps * output_fps)))

        source = pipe.src_imgs[0]
        source_info = pipe.src_infos[0]
        output_path = Path(args.output_raw)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("wb") as output:
            for frame_index in range(frame_count):
                motion_index = min(
                    driving["n_frames"] - 1,
                    int(np.floor(frame_index * motion_fps / output_fps)),
                )
                motion_info = [driving["motion"][motion_index], None, None]
                generated_crop, frame = pipe.run_with_pkl(
                    motion_info,
                    source,
                    source_info,
                    first_frame=frame_index == 0,
                )
                if frame is None:
                    raise RuntimeError(f"MLX failed to render frame {frame_index}")
                if generated_crop is None:
                    raise RuntimeError(f"MLX returned no generated crop for frame {frame_index}")
                frame = np.asarray(frame, dtype=np.uint8)
                generated_crop = np.asarray(generated_crop, dtype=np.uint8)
                if generated_crop.ndim != 3 or generated_crop.shape[2] != 3:
                    raise RuntimeError(
                        f"Unexpected MLX generated crop shape: {generated_crop.shape}"
                    )
                output.write(
                    paste_back_bbox(source, generated_crop, face_bbox).tobytes()
                )
                if frame_index == 0 or frame_index + 1 == frame_count or (frame_index + 1) % 25 == 0:
                    print(
                        json.dumps(
                            {"event": "progress", "frame": frame_index + 1, "frames": frame_count},
                            ensure_ascii=False,
                        ),
                        file=sys.stderr,
                        flush=True,
                    )

        print(
            json.dumps(
                {
                    "width": int(source.shape[1]),
                    "height": int(source.shape[0]),
                    "fps": output_fps,
                    "frames": frame_count,
                    "duration": frame_count / output_fps,
                    "provider": "mlx",
                    "profile": f"mlx-{args.profile}",
                    "precision": "fp32" if args.profile == "quality" else "bf16",
                    "cfg": bool(args.cfg),
                    "diffusionSteps": int(
                        args.motion_diffusion_steps or diffusion_steps
                    ),
                    "motionBackend": "onnx" if args.motion_f32 else "mlx",
                    "face": {"landmarks": face_landmarks.reshape(-1).tolist()},
                    "audioSamples": int(round(driving["n_frames"] / motion_fps * 16000)),
                }
            )
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"MLX bridge error: {exc}", file=sys.stderr)
        raise

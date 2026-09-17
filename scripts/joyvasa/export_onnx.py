#!/usr/bin/env python3
"""Export the JoyVASA audio and diffusion networks to ONNX.

JoyVASA's original sampler is deliberately not exported as one graph.  It
contains a checkpoint-configured stochastic diffusion loop and
classifier-free guidance.
This exporter produces two deterministic graphs instead:

* ``audio_encoder.onnx``: 16 kHz audio window -> ``[B, n_motions, feature_dim]``
* ``motion_generator.onnx``: one denoising step ->
  ``[B, n_prev_motions + n_motions, motion_feat_dim]``

The Node addon runs the checkpoint-configured diffusion schedule and random sampling in host code.
The script uses the FasterLivePortrait JoyVASA fork because it accepts an
explicit local Hugging Face audio-model directory.  It also works with an
equivalent checkout whose source tree contains ``src/models/JoyVASA``.

Example:

  uv run --python .venv-joyvasa/bin/python scripts/joyvasa/export_onnx.py \
    --source-root /path/to/FasterLivePortrait \
    --motion-checkpoint /path/to/motion_generator_hubert_chinese.pt \
    --audio-model-path /path/to/hubert-base-ls960 \
    --output-dir ./joyvasa-onnx

The checkpoint and audio model are intentionally command-line inputs.  They
are large model assets and are not downloaded by this repository script.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import tempfile
import types
from pathlib import Path
from typing import Any

import torch.nn as nn


def _import_joyvasa(source_root: Path):
    """Import the JoyVASA fork without making it a repository dependency."""

    source_root = source_root.resolve()
    if not (source_root / "src").is_dir():
        raise FileNotFoundError(f"--source-root does not contain src: {source_root}")
    sys.path.insert(0, str(source_root))

    if (source_root / "src/models/JoyVASA/dit_talking_head.py").is_file():
        prefix = "src.models.JoyVASA"
    elif (source_root / "src/modules/dit_talking_head.py").is_file():
        raise RuntimeError(
            "The original JoyVASA checkout hard-codes its audio model path. "
            "Use the FasterLivePortrait JoyVASA fork (src/models/JoyVASA), "
            "or provide a fork that accepts audio_encoder_path."
        )
    else:
        raise FileNotFoundError(
            "Could not find src/models/JoyVASA/dit_talking_head.py in "
            f"{source_root}"
        )

    # FasterLivePortrait's broad ``src.models.__init__`` imports cv2 and the
    # complete portrait stack. The exporter only needs JoyVASA, so install
    # lightweight package objects and let Python import the four local
    # JoyVASA modules without pulling those unrelated runtime dependencies.
    src_package = types.ModuleType("src")
    src_package.__path__ = [str(source_root / "src")]
    models_package = types.ModuleType("src.models")
    models_package.__path__ = [str(source_root / "src/models")]
    sys.modules["src"] = src_package
    sys.modules["src.models"] = models_package

    dit = importlib.import_module(f"{prefix}.dit_talking_head")
    helper = importlib.import_module(f"{prefix}.helper")
    common = importlib.import_module(f"{prefix}.common")
    # JoyVASA's custom HuBERT/wav2vec2 forward methods request attention
    # tensors. Recent Transformers defaults to SDPA, which rejects that
    # request; eager attention is exportable and keeps the upstream behavior.
    for module_name, class_name in (
        (f"{prefix}.hubert", "HubertModel"),
        (f"{prefix}.wav2vec2", "Wav2Vec2Model"),
    ):
        audio_module = importlib.import_module(module_name)
        audio_class = getattr(audio_module, class_name)
        original_loader = audio_class.from_pretrained

        def load_with_eager_attention(cls, *loader_args, _loader=original_loader, **loader_kwargs):
            loader_kwargs["attn_implementation"] = "eager"
            return _loader(*loader_args, **loader_kwargs)

        audio_class.from_pretrained = classmethod(load_with_eager_attention)
    # The upstream helper defaults its attention mask to CUDA even when the
    # surrounding model is explicitly constructed on CPU. Exporting on a
    # CPU-only machine is supported by forcing that non-parameter buffer to
    # the selected export device.
    def cpu_enc_dec_mask(T, S, frame_width=2, expansion=0, device="cpu"):
        return common.enc_dec_mask(T, S, frame_width, expansion, device="cpu")

    dit.enc_dec_mask = cpu_enc_dec_mask
    return dit, helper, common


def _get(value: Any, name: str, default: Any) -> Any:
    result = getattr(value, name, None)
    return default if result is None else result


def _load_model(args: argparse.Namespace):
    import torch

    dit, helper, common = _import_joyvasa(Path(args.source_root))
    checkpoint_path = Path(args.motion_checkpoint).expanduser().resolve()
    audio_model_path = Path(args.audio_model_path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Motion checkpoint does not exist: {checkpoint_path}")
    if not audio_model_path.is_dir():
        raise FileNotFoundError(f"Audio model directory does not exist: {audio_model_path}")

    # Explicit weights_only=False is required for the Namespace stored by the
    # official checkpoint and is safe here because this is a local checkpoint
    # supplied by the caller.
    model_data = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    if "args" not in model_data or "model" not in model_data:
        raise ValueError("JoyVASA checkpoint must contain 'args' and 'model'")
    model_args = helper.NullableArgs(model_data["args"])

    # These are the arguments used by FasterLivePortrait's own pipeline, with
    # defensive defaults for older JoyVASA checkpoints.
    constructor_args = dict(
        device="cpu",
        target=_get(model_args, "target", "sample"),
        architecture=_get(model_args, "architecture", "decoder"),
        motion_feat_dim=int(_get(model_args, "motion_feat_dim", 76)),
        fps=int(_get(model_args, "fps", 25)),
        n_motions=int(_get(model_args, "n_motions", 100)),
        n_prev_motions=int(_get(model_args, "n_prev_motions", 10)),
        audio_model=_get(model_args, "audio_model", "hubert"),
        feature_dim=int(_get(model_args, "feature_dim", 512)),
        n_diff_steps=int(_get(model_args, "n_diff_steps", 500)),
        diff_schedule=_get(model_args, "diff_schedule", "cosine"),
        cfg_mode=_get(model_args, "cfg_mode", "incremental"),
        guiding_conditions=_get(model_args, "guiding_conditions", "audio,"),
        audio_encoder_path=str(audio_model_path),
    )
    model = dit.DitTalkingHead(**constructor_args)

    # Some official checkpoints omit this deterministic buffer.  Keeping the
    # model-created positional encoding is equivalent and avoids a strict-load
    # mismatch across JoyVASA revisions.
    state = {
        key: value
        for key, value in model_data["model"].items()
        if key != "denoising_net.TE.pe"
    }
    load_result = model.load_state_dict(state, strict=False)
    model.eval()
    model.to("cpu")
    if load_result.unexpected_keys:
        print("warning: unexpected checkpoint keys:", load_result.unexpected_keys)
    if load_result.missing_keys:
        print("warning: missing checkpoint keys:", load_result.missing_keys)
    return model, model_args, common, constructor_args


class _AudioFeatureEncoder(nn.Module):
    """Torch module wrapper with tensor-only output for ONNX export."""

    def __init__(self, model, common, frame_num: int):
        super().__init__()
        self.audio_encoder = model.audio_encoder
        self.audio_feature_map = model.audio_feature_map
        self._pad_audio = common.pad_audio
        self._fps = int(model.fps)
        self._frame_num = int(frame_num)

    def forward(self, audio):
        import torch.nn.functional as F

        audio = self._pad_audio(audio)
        hidden_states = self.audio_encoder(
            audio, self._fps, frame_num=self._frame_num * 2
        ).last_hidden_state
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = F.interpolate(
            hidden_states,
            size=self._frame_num,
            align_corners=False,
            mode="linear",
        )
        hidden_states = hidden_states.transpose(1, 2)
        return self.audio_feature_map(hidden_states)


class _Denoiser(nn.Module):
    """Torch module wrapper for one deterministic diffusion denoise step."""

    def __init__(self, denoising_net, use_indicator: bool):
        super().__init__()
        self.denoising_net = denoising_net
        self._use_indicator = bool(use_indicator)

    def forward(
        self,
        motion_feat,
        audio_feat,
        prev_motion_feat,
        prev_audio_feat,
        step,
        indicator=None,
    ):
        if self._use_indicator:
            return self.denoising_net(
                motion_feat,
                audio_feat,
                prev_motion_feat,
                prev_audio_feat,
                step,
                indicator,
            )
        return self.denoising_net(
            motion_feat,
            audio_feat,
            prev_motion_feat,
            prev_audio_feat,
            step,
            None,
        )


def _export_graph(module, inputs: tuple, output_path: Path, input_names: list[str],
                  output_names: list[str], dynamic_axes: dict[str, dict[int, str]]):
    import torch

    output_path.parent.mkdir(parents=True, exist_ok=True)
    # The legacy exporter is still the most predictable route for the
    # TransformerDecoder used by JoyVASA.  It is available in torch 2.14 and
    # keeps the resulting graph consumable by ONNX Runtime 1.18+.
    torch.onnx.export(
        module,
        inputs,
        str(output_path),
        opset_version=18,
        dynamo=False,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
        training=torch.onnx.TrainingMode.EVAL,
    )


def _verify_graph(path: Path, feeds: dict[str, Any]):
    import numpy as np
    import onnx
    import onnxruntime as ort

    onnx.checker.check_model(onnx.load(str(path)))
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    result = session.run(None, feeds)
    if not result or not all(np.isfinite(item).all() for item in result if item.dtype.kind == "f"):
        raise RuntimeError(f"ONNX Runtime verification failed for {path}")
    print(f"verified {path} ({len(result)} output tensor(s))")


def _tensor_to_list(tensor) -> list[float]:
    return tensor.detach().cpu().float().reshape(-1).tolist()


def _metadata(model, model_args, constructor_args, output_dir: Path, audio_path: Path,
              motion_path: Path, use_indicator: bool) -> dict[str, Any]:
    return {
        "format": "sherpa-onnx-joyvasa-1",
        "source": "FasterLivePortrait/src/models/JoyVASA",
        "huggingfaceRepo": "jdh-algo/JoyVASA",
        "audioModelRepo": "TencentGameMate/chinese-hubert-base",
        "audio_model_path": str(audio_path),
        "motion_checkpoint": str(motion_path),
        "models": {
            "audioEncoder": "audio_encoder.onnx",
            "motionGenerator": "motion_generator.onnx",
        },
        "audioInput": {
            "sampleRate": 16000,
            "samplesPerWindow": round(16000 * constructor_args["n_motions"] / constructor_args["fps"]),
            "fps": constructor_args["fps"],
        },
        "motion": {
            "nMotions": constructor_args["n_motions"],
            "nPrevMotions": constructor_args["n_prev_motions"],
            "motionFeatDim": constructor_args["motion_feat_dim"],
            "featureDim": constructor_args["feature_dim"],
            "fps": constructor_args["fps"],
            "target": constructor_args["target"],
            "nDiffSteps": constructor_args["n_diff_steps"],
            "diffSchedule": constructor_args["diff_schedule"],
            "cfgMode": constructor_args["cfg_mode"],
            "guidingConditions": constructor_args["guiding_conditions"],
            "useIndicator": use_indicator,
        },
        # These learned initial states are needed for the first audio window.
        # They are small compared with the model and make the Node sampler
        # numerically match the Python pipeline instead of silently using zeros.
        "initialState": {
            "startMotionFeat": _tensor_to_list(model.start_motion_feat),
            "startAudioFeat": _tensor_to_list(model.start_audio_feat),
            "nullAudioFeat": _tensor_to_list(model.null_audio_feat)
            if hasattr(model, "null_audio_feat") else None,
        },
    }


def export_real(args: argparse.Namespace) -> None:
    import numpy as np
    import torch

    model, model_args, common, constructor_args = _load_model(args)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    n_motions = constructor_args["n_motions"]
    n_prev_motions = constructor_args["n_prev_motions"]
    motion_dim = constructor_args["motion_feat_dim"]
    feature_dim = constructor_args["feature_dim"]
    samples_per_window = round(16000 * n_motions / constructor_args["fps"])
    # The flag used by the Python pipeline can be present even when the
    # checkpoint's DenoisingNetwork was built without an indicator channel.
    # Inspect the actual network so metadata and ONNX inputs cannot disagree.
    use_indicator = bool(getattr(model.denoising_net, "use_indicator", False))

    audio_module = _AudioFeatureEncoder(model, common, n_motions)
    audio_module.eval()
    audio_input = torch.zeros((1, samples_per_window), dtype=torch.float32)
    _export_graph(
        audio_module,
        (audio_input,),
        output_dir / "audio_encoder.onnx",
        ["audio"],
        ["audio_features"],
        {"audio": {0: "batch"}, "audio_features": {0: "batch"}},
    )

    denoiser = _Denoiser(model.denoising_net, use_indicator)
    denoiser.eval()
    motion_input = torch.zeros((1, n_motions, motion_dim), dtype=torch.float32)
    audio_features = torch.zeros((1, n_motions, feature_dim), dtype=torch.float32)
    prev_motion = torch.zeros((1, n_prev_motions, motion_dim), dtype=torch.float32)
    prev_audio = torch.zeros((1, n_prev_motions, feature_dim), dtype=torch.float32)
    step = torch.ones((1,), dtype=torch.int64)
    inputs: tuple = (motion_input, audio_features, prev_motion, prev_audio, step)
    input_names = ["motion_feat", "audio_feat", "prev_motion_feat", "prev_audio_feat", "step"]
    dynamic_axes = {
        "motion_feat": {0: "batch"},
        "audio_feat": {0: "batch"},
        "prev_motion_feat": {0: "batch"},
        "prev_audio_feat": {0: "batch"},
        "step": {0: "batch"},
        "motion_target": {0: "batch"},
    }
    if use_indicator:
        indicator = torch.ones((1, n_motions), dtype=torch.float32)
        inputs = inputs + (indicator,)
        input_names.append("indicator")
        dynamic_axes["indicator"] = {0: "batch"}
    _export_graph(
        denoiser,
        inputs,
        output_dir / "motion_generator.onnx",
        input_names,
        ["motion_target"],
        dynamic_axes,
    )

    if args.verify:
        _verify_graph(
            output_dir / "audio_encoder.onnx",
            {"audio": np.zeros((1, samples_per_window), dtype=np.float32)},
        )
        denoiser_feeds = {
            "motion_feat": np.zeros((1, n_motions, motion_dim), dtype=np.float32),
            "audio_feat": np.zeros((1, n_motions, feature_dim), dtype=np.float32),
            "prev_motion_feat": np.zeros((1, n_prev_motions, motion_dim), dtype=np.float32),
            "prev_audio_feat": np.zeros((1, n_prev_motions, feature_dim), dtype=np.float32),
            "step": np.ones((1,), dtype=np.int64),
        }
        if use_indicator:
            denoiser_feeds["indicator"] = np.ones((1, n_motions), dtype=np.float32)
        _verify_graph(output_dir / "motion_generator.onnx", denoiser_feeds)

    metadata = _metadata(
        model,
        model_args,
        constructor_args,
        output_dir,
        Path(args.audio_model_path).expanduser().resolve(),
        Path(args.motion_checkpoint).expanduser().resolve(),
        use_indicator,
    )
    (output_dir / "joyvasa.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {output_dir / 'joyvasa.json'}")


def _self_test(output_dir: str | None = None) -> None:
    """Exercise the two exported graph contracts without downloading weights."""

    import numpy as np
    import onnx
    import onnxruntime as ort
    import torch
    import torch.nn as nn

    class ToyAudio(nn.Module):
        def forward(self, audio):
            # [B, samples] -> [B, 3, 4], intentionally simple but shaped like
            # the real audio feature graph.
            return audio.reshape(audio.shape[0], 3, -1).mean(-1, keepdim=True).expand(-1, -1, 4)

    class ToyDenoiser(nn.Module):
        def forward(self, motion_feat, audio_feat, prev_motion_feat, prev_audio_feat, step):
            # Keep every input live in the exported graph, matching the real
            # denoiser's five-input contract.
            zero = audio_feat.mean() * 0 + prev_audio_feat.mean() * 0 + step.float().mean() * 0
            return torch.cat([prev_motion_feat + zero, motion_feat + zero], dim=1)

    if output_dir:
        root = Path(output_dir).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
    else:
        root = Path(tempfile.mkdtemp(prefix="joyvasa-export-test-"))
    audio_path = root / "audio_encoder.onnx"
    motion_path = root / "motion_generator.onnx"
    _export_graph(
        ToyAudio().eval(),
        (torch.zeros((1, 12), dtype=torch.float32),),
        audio_path,
        ["audio"],
        ["audio_features"],
        {"audio": {0: "batch", 1: "samples"}, "audio_features": {0: "batch"}},
    )
    _export_graph(
        ToyDenoiser().eval(),
        (
            torch.zeros((1, 3, 2)),
            torch.zeros((1, 3, 4)),
            torch.zeros((1, 2, 2)),
            torch.zeros((1, 2, 4)),
            torch.ones((1,), dtype=torch.int64),
        ),
        motion_path,
        ["motion_feat", "audio_feat", "prev_motion_feat", "prev_audio_feat", "step"],
        ["motion_target"],
        {"motion_feat": {0: "batch"}, "audio_feat": {0: "batch"},
         "prev_motion_feat": {0: "batch"}, "prev_audio_feat": {0: "batch"},
         "step": {0: "batch"}, "motion_target": {0: "batch"}},
    )
    for path in (audio_path, motion_path):
        onnx.checker.check_model(onnx.load(str(path)))
        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        feeds = {}
        for item in session.get_inputs():
            shape = tuple(dim if isinstance(dim, int) else 1 for dim in item.shape)
            if item.name == "audio": shape = (1, 12)
            feeds[item.name] = np.zeros(shape, dtype=np.float32)
        if "step" in feeds:
            feeds["step"] = np.ones((1,), dtype=np.int64)
        session.run(None, feeds)
    print("JoyVASA exporter self-test passed (no model weights downloaded).")
    if output_dir:
        print(f"wrote test graphs to {root}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", help="FasterLivePortrait checkout")
    parser.add_argument("--motion-checkpoint", help="JoyVASA motion_generator_*.pt")
    parser.add_argument("--audio-model-path", help="Local HuBERT/wav2vec2 model directory")
    parser.add_argument("--output-dir", default="joyvasa-onnx")
    parser.add_argument("--verify", action="store_true", help="Run ONNX checker and CPU ORT smoke tests")
    parser.add_argument("--self-test", action="store_true", help="Test exporter contracts without weights")
    parser.add_argument("--self-test-output-dir", help="Keep self-test graphs in this directory")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.self_test:
        _self_test(args.self_test_output_dir)
        return
    missing = [name for name in ("source_root", "motion_checkpoint", "audio_model_path")
               if not getattr(args, name)]
    if missing:
        parser.error("missing required arguments: " + ", ".join("--" + item.replace("_", "-") for item in missing))
    export_real(args)


if __name__ == "__main__":
    main()

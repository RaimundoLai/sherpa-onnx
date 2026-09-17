#!/usr/bin/env python3
"""Create a lower-resolution FasterLivePortrait warping graph.

The published warping graph has a fixed 64x64 feature grid and produces a
512x512 image.  Its weights can be reused at a 32x32 grid because all spatial
upsampling stages are scale-by-two operations.  This tool rewrites only the
static shape tensors and graph metadata; it does not quantize or change any
weights.

The generated graph consumes feature_3d [1, 32, 16, 32, 32] and produces
out [1, 3, 256, 256].  The caller must downsample the appearance feature from
64x64 to 32x32 before invoking it.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import onnx
import numpy as np
from onnx import checker, helper, numpy_helper, shape_inference


def _scale_shape_initializer(tensor: onnx.TensorProto, factor: float) -> bool:
    """Scale spatial entries in known static shape tensors."""
    if tensor.data_type != onnx.TensorProto.INT64:
        return False
    values = numpy_helper.to_array(tensor).copy()
    changed = False
    # These tensors are one-dimensional shape vectors.  Find the adjacent
    # equal H/W pair rather than scaling every occurrence: a value such as
    # 256 or 512 earlier in a vector can be a channel count.  One reshape has
    # a trailing -1, so its H/W pair is not the final two entries.
    if values.ndim != 1 or values.size < 4:
        return False
    # Prefer the last pair when a vector contains [N, C, 256, 256]; otherwise
    # the channel value would be mistaken for the first spatial dimension.
    for index in range(values.size - 2, -1, -1):
        first = int(values[index])
        second = int(values[index + 1])
        if first == second and first in (64, 128, 256, 512):
            values[index] = int(round(first * factor))
            values[index + 1] = int(round(second * factor))
            changed = True
            break
    if changed:
        replacement = numpy_helper.from_array(values.astype(np.int64), tensor.name)
        tensor.CopyFrom(replacement)
    return changed


def _rewrite_spatial_constants(model: onnx.ModelProto, size: int) -> list[str]:
    """Resize the exported dense-motion coordinate constants."""
    tensors = {tensor.name: tensor for tensor in model.graph.initializer}
    changed = []

    zero_name = "/dense_motion_network/Constant_58_output_0"
    zero = tensors.get(zero_name)
    if zero is not None:
        replacement = numpy_helper.from_array(
            np.zeros((1, 1, 16, size, size), dtype=np.float32), zero_name)
        zero.CopyFrom(replacement)
        changed.append(zero_name)

    # This is the fixed [-1, 1] sampling lattice used by the dense-motion
    # GridSample path.  Rebuild it at the new resolution instead of taking
    # every other value, so both endpoints remain exactly -1 and +1.
    axis = np.linspace(-1.0, 1.0, size, dtype=np.float32)
    depth = np.linspace(-1.0, 1.0, 16, dtype=np.float32)
    z, y, x = np.meshgrid(depth, axis, axis, indexing="ij")
    lattice = np.stack((x, y, z), axis=-1)[None, None, ...]
    for name, count in (
        ("/dense_motion_network/Reshape_output_0", 1),
        ("/dense_motion_network/Tile_8_output_0", 21),
    ):
        tensor = tensors.get(name)
        if tensor is None:
            continue
        value = np.tile(lattice, (1, count, 1, 1, 1, 1))
        tensor.CopyFrom(numpy_helper.from_array(value.astype(np.float32), name))
        changed.append(name)
    return changed


def optimize(input_path: Path, output_path: Path, factor: float) -> None:
    if factor != 0.5:
        raise ValueError("Only factor=0.5 is supported by this graph rewrite")

    model = onnx.load(str(input_path), load_external_data=True)
    feature = next(item for item in model.graph.input if item.name == "feature_3d")
    output = next(item for item in model.graph.output if item.name == "out")
    feature_shape = feature.type.tensor_type.shape.dim
    output_shape = output.type.tensor_type.shape.dim
    feature_shape[2].dim_value = 16
    feature_shape[3].dim_value = 32
    feature_shape[4].dim_value = 32
    output_shape[2].dim_value = 256
    output_shape[3].dim_value = 256

    changed = []
    for tensor in model.graph.initializer:
        if _scale_shape_initializer(tensor, factor):
            changed.append(tensor.name)
    changed.extend(_rewrite_spatial_constants(model, 32))

    # Existing value_info entries describe the old fixed shapes.  Remove them
    # and let ONNX shape inference rebuild the metadata from the rewritten
    # graph, including the GridSample path.
    model.graph.ClearField("value_info")
    model = shape_inference.infer_shapes(model)
    checker.check_model(model)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(output_path))
    print(f"wrote {output_path}")
    print(f"rewrote {len(changed)} shape initializers")
    print("inputs:", [(x.name, [d.dim_value for d in x.type.tensor_type.shape.dim])
                       for x in model.graph.input])
    print("outputs:", [(x.name, [d.dim_value for d in x.type.tensor_type.shape.dim])
                        for x in model.graph.output])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--factor", type=float, default=0.5)
    args = parser.parse_args()
    optimize(args.input, args.output, args.factor)


if __name__ == "__main__":
    main()

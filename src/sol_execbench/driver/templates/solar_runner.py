#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""SOL ExecBench SOLAR runner -- self-contained script for SOLAR analysis.

Reuses the same staging files (definition.json, workload.jsonl, config.json)
produced by ProblemPackager.  Generates a SOLAR-compatible model.py wrapping
the reference run() function and runs the SOLAR 4-stage pipeline.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

import torch

STAGING_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(STAGING_DIR))

from sol_execbench.core import Definition, Workload  # noqa: E402

# ── Bundled SOLAR architecture configs ───────────────────────────────────────
# SOLAR's pip-installed wheel may not ship configs/arch/ YAML files.
# We bundle them alongside this script in configs/arch/ and write to disk
# before calling predict().

_ARCH_CONFIGS_DIR = Path(__file__).parent / "configs" / "arch"


def _detect_solar_arch() -> str:
    """Auto-detect GPU architecture and map to nearest SOLAR arch config."""
    if not torch.cuda.is_available():
        return "B200"
    props = torch.cuda.get_device_properties(0)
    major = props.major
    if major >= 100:
        return "B200"
    if major == 90:
        return "H100_PCIe"
    # Fallback
    return "B200"


def _ensure_arch_config(arch_name: str, output_dir: Path) -> str:
    """Ensure the SOLAR arch config YAML file exists and return its path.

    SOLAR's pip wheel may not ship configs/arch/*.yaml.  If the config is not
    found by SOLAR's built-in lookup, we copy the bundled config to the
    output directory and return that path.
    """
    # Check if SOLAR can find it natively
    try:
        from solar.perf import EinsumGraphPerfModel  # noqa: E402

        model = EinsumGraphPerfModel()
        cfg = model._load_arch_config(arch_name)
        if cfg:
            return arch_name  # SOLAR found it by name
    except Exception:
        pass

    # Copy bundled config to output_dir
    bundled = _ARCH_CONFIGS_DIR / f"{arch_name}.yaml"
    if not bundled.exists():
        # Fallback to B200
        arch_name = "B200"
        bundled = _ARCH_CONFIGS_DIR / f"{arch_name}.yaml"
    config_path = output_dir / f"{arch_name}.yaml"
    config_path.write_text(bundled.read_text())
    return str(config_path)


def _generate_model_py(
    definition: Definition, workload: Workload, output_dir: Path
) -> Path:
    """Generate a SOLAR-compatible model.py wrapping the reference run()."""
    resolved_axes = definition.get_resolved_axes_values(workload.axes)

    # Build input tensor expressions for get_inputs()
    input_exprs: list[str] = []
    param_names: list[str] = []
    for name, spec in definition.inputs.items():
        input_spec = workload.inputs.get(name)
        param_names.append(name)

        if input_spec is not None and input_spec.type == "scalar":
            # Scalar inputs: literal Python value
            val = input_spec.value
            input_exprs.append(repr(val))
        else:
            # Tensor inputs: torch.randn with resolved shape
            shape_parts = []
            if spec.shape:
                for dim_name in spec.shape:
                    shape_parts.append(str(resolved_axes.get(dim_name, dim_name)))
            shape_str = ", ".join(shape_parts)
            dtype_str = f"torch.{spec.dtype.value}"
            input_exprs.append(f"torch.randn(({shape_str}), dtype={dtype_str})")

    model_py = (
        "import torch\n"
        "import torch.nn as nn\n"
        "\n"
        "# === Reference code ===\n" + definition.reference + "\n\n"
        "# === SOLAR wrapper ===\n"
        "class Model(nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "\n"
        "    def forward(self, " + ", ".join(param_names) + "):\n"
        "        return run(" + ", ".join(param_names) + ")\n"
        "\n\n"
        "def get_inputs():\n"
        "    return [" + ", ".join(input_exprs) + "]\n"
        "\n\n"
        "def get_init_inputs():\n"
        "    return []\n"
    )

    model_path = output_dir / "model.py"
    model_path.write_text(model_py)
    return model_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run SOLAR analysis on a reference kernel"
    )
    parser.add_argument(
        "--workload-uuid",
        required=True,
        help="UUID of the workload to analyze",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Base directory for SOLAR output files",
    )
    parser.add_argument(
        "--arch-config",
        default="",
        help="Path to SOLAR arch config YAML. Auto-detects GPU arch if omitted",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir) / str(args.workload_uuid) / "solar"
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # -- Load definition & workloads --
        definition = Definition(
            **json.loads((STAGING_DIR / "definition.json").read_text())
        )
        workloads_raw = []
        wkl_path = STAGING_DIR / "workload.jsonl"
        if wkl_path.exists():
            for line in wkl_path.read_text().splitlines():
                line = line.strip()
                if line:
                    workloads_raw.append(json.loads(line))

        workloads = [Workload(**w) for w in workloads_raw]

        # Find matching workload
        target_workload = None
        for wl in workloads:
            if str(wl.uuid) == args.workload_uuid:
                target_workload = wl
                break
        if target_workload is None:
            (output_dir / "error.txt").write_text(
                f"Workload UUID {args.workload_uuid} not found\n"
            )
            sys.exit(0)

        # -- Generate model.py --
        model_path = _generate_model_py(definition, target_workload, output_dir)

        # -- Detect architecture and ensure config is available --
        if args.arch_config:
            # User-supplied arch config path — use directly
            arch_config = args.arch_config
        else:
            arch = _detect_solar_arch()
            arch_config = _ensure_arch_config(arch, output_dir)

        # -- Run SOLAR 4-stage pipeline --
        from solar.graph import PyTorchProcessor  # noqa: E402
        from solar.einsum import PyTorchToEinsum  # noqa: E402
        from solar.analysis import EinsumGraphAnalyzer  # noqa: E402
        from solar.perf import EinsumGraphPerfModel  # noqa: E402

        # Determine precision from first input dtype
        first_input = next(iter(definition.inputs.values()))
        precision = (
            "fp16"
            if "float16" in first_input.dtype.value
            or "bfloat16" in first_input.dtype.value
            else "fp32"
        )

        # Stage 1: PyTorch graph extraction
        processor = PyTorchProcessor()
        processor.process_model_file(str(model_path), str(output_dir))
        pytorch_graph_path = output_dir / "pytorch_graph.yaml"
        if not pytorch_graph_path.exists():
            (output_dir / "error.txt").write_text(
                "SOLAR Stage 1 (PyTorchProcessor) produced no output. "
                "The reference may use Triton/custom CUDA ops that cannot be traced.\n"
            )
            sys.exit(0)

        # Stage 2: Einsum conversion (enable_rename to produce einsum_graph_renamed.yaml)
        converter = PyTorchToEinsum()
        converter.convert_graph(pytorch_graph_path, output_dir, enable_rename=True)
        einsum_renamed_path = output_dir / "einsum_graph_renamed.yaml"
        if not einsum_renamed_path.exists():
            (output_dir / "error.txt").write_text(
                "SOLAR Stage 2 (PyTorchToEinsum) produced no output.\n"
            )
            sys.exit(0)

        # Stage 3: Hardware-independent analysis
        analyzer = EinsumGraphAnalyzer()
        analyzer.analyze_graph(einsum_renamed_path, output_dir, precision=precision)
        analysis_path = output_dir / "analysis.yaml"
        if not analysis_path.exists():
            (output_dir / "error.txt").write_text(
                "SOLAR Stage 3 (EinsumGraphAnalyzer) produced no output.\n"
            )
            sys.exit(0)

        # Stage 4: Hardware performance prediction
        perf_model = EinsumGraphPerfModel()
        perf_model.predict(
            analysis_path, output_dir, arch_config=arch_config, precision=precision
        )

        print(f"[solar] analysis complete for {args.workload_uuid}", file=sys.stderr)

    except Exception as e:
        (output_dir / "error.txt").write_text(
            f"SOLAR analysis failed: {e}\n\n{traceback.format_exc()}\n"
        )
        # Non-fatal: exit 0 so the parent eval_driver continues
        sys.exit(0)


if __name__ == "__main__":
    main()

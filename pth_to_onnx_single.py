# -*- coding: utf-8 -*-
"""
pth_to_onnx_single.py

目的：
把现有 run10_2000.pth 导出成真正的单文件 gomoku_single.onnx。

关键点：
1. 复用你现有 pth_to_onnx.py 中已经验证正确的网络结构。
2. 强制 dynamo=False，使用传统 ONNX exporter。
3. external_data=False，禁止拆出 .data。
4. 导出后检查 ONNX 是否还包含 external_data。
5. 用 ONNX Runtime 与 PyTorch 输出做数值一致性验证。

使用：
    python pth_to_onnx_single.py

要求：
    pth_to_onnx.py
    run10_2000.pth
    pth_to_onnx_single.py
放在同一目录。
"""

import os
import glob
from pathlib import Path

import numpy as np
import torch
import onnx
import onnxruntime as ort
from onnx import TensorProto

from pth_to_onnx import (
    BOARD_SIZE,
    HIDDEN_CHANNELS,
    NUM_BLOCKS,
    VALUE_DIM,
    ValueCNN,
    WebGomokuModel,
    load_state_dict_safely,
    strip_module_prefix,
)


def find_model(folder: Path) -> Path:
    files = list(folder.glob("*.pth"))

    if not files:
        raise FileNotFoundError(
            "当前目录没有找到 .pth 模型文件。"
        )

    # 优先 run10_2000.pth
    preferred = folder / "run10_2000.pth"

    if preferred.exists():
        return preferred

    files.sort(
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )

    return files[0]


def main():
    base = Path(__file__).resolve().parent

    model_path = find_model(base)
    output_path = base / "gomoku_single.onnx"

    print("=" * 64)
    print("Gomoku PTH -> 真正单文件 ONNX")
    print("=" * 64)
    print("Python :", os.sys.executable)
    print("PTH    :", model_path)
    print("输出   :", output_path)

    # -----------------------------------------------------
    # 构建与加载模型
    # -----------------------------------------------------
    model = ValueCNN(
        in_channels=3,
        hidden_channels=HIDDEN_CHANNELS,
        num_blocks=NUM_BLOCKS,
        value_dim=VALUE_DIM
    )

    state_dict = load_state_dict_safely(
        str(model_path)
    )

    state_dict = strip_module_prefix(
        state_dict
    )

    missing, unexpected = model.load_state_dict(
        state_dict,
        strict=False
    )

    if missing or unexpected:
        print("missing   :", missing)
        print("unexpected:", unexpected)
        raise RuntimeError(
            "PTH 参数与当前网络结构不完全一致。"
        )

    model.eval()

    web_model = WebGomokuModel(model)
    web_model.eval()

    params = sum(
        p.numel()
        for p in model.parameters()
    )

    print(f"模型参数量：{params:,}")

    # -----------------------------------------------------
    # 测试输入
    # -----------------------------------------------------
    x = torch.zeros(
        1,
        3,
        BOARD_SIZE,
        BOARD_SIZE,
        dtype=torch.float32
    )

    # empty channel
    x[:, 2, :, :] = 1.0

    # current player
    x[0, 0, 7, 7] = 1.0
    x[0, 2, 7, 7] = 0.0

    x[0, 0, 7, 8] = 1.0
    x[0, 2, 7, 8] = 0.0

    # opponent
    x[0, 1, 8, 7] = 1.0
    x[0, 2, 8, 7] = 0.0

    with torch.no_grad():
        pt_value, pt_policy = web_model(x)

    print()
    print("PyTorch：")
    print(
        "  Value      =",
        f"{pt_value.item():.8f}"
    )
    print(
        "  Policy sum =",
        f"{pt_policy.sum().item():.8f}"
    )
    print(
        "  Policy max =",
        f"{pt_policy.max().item():.8f}"
    )

    # -----------------------------------------------------
    # 删除旧输出，避免误判
    # -----------------------------------------------------
    if output_path.exists():
        output_path.unlink()

    possible_data = Path(
        str(output_path) + ".data"
    )

    if possible_data.exists():
        possible_data.unlink()

    # -----------------------------------------------------
    # 关键：传统 exporter + 禁止 external data
    # -----------------------------------------------------
    print()
    print("开始导出真正单文件 ONNX ...")

    torch.onnx.export(
        web_model,
        x,
        str(output_path),

        export_params=True,
        opset_version=17,
        do_constant_folding=True,

        input_names=["board"],
        output_names=["value", "policy"],

        # 最关键的两个参数
        dynamo=False,
        external_data=False
    )

    print("导出完成。")

    # -----------------------------------------------------
    # 检查是否真的无 external data
    # -----------------------------------------------------
    onnx_model = onnx.load_model(
        str(output_path),
        load_external_data=False
    )

    external = []

    for tensor in onnx_model.graph.initializer:
        if (
            tensor.data_location == TensorProto.EXTERNAL
            or len(tensor.external_data) > 0
        ):
            external.append(tensor.name)

    if external:
        print()
        print("发现 external_data：")
        for name in external:
            print(" ", name)

        raise RuntimeError(
            "导出仍包含 external data，未达到单文件要求。"
        )

    onnx.checker.check_model(
        onnx_model
    )

    size_kb = (
        output_path.stat().st_size
        / 1024
    )

    print()
    print(
        f"单文件大小：{size_kb:.1f} KB"
    )
    print(
        "external_data 数量：0"
    )

    if possible_data.exists():
        raise RuntimeError(
            "意外生成了 .data 文件。"
        )

    # -----------------------------------------------------
    # ORT 验证
    # -----------------------------------------------------
    sess = ort.InferenceSession(
        str(output_path),
        providers=[
            "CPUExecutionProvider"
        ]
    )

    ort_value, ort_policy = sess.run(
        ["value", "policy"],
        {
            "board":
                x.detach()
                 .cpu()
                 .numpy()
        }
    )

    value_error = np.max(
        np.abs(
            pt_value.detach().cpu().numpy()
            - ort_value
        )
    )

    policy_error = np.max(
        np.abs(
            pt_policy.detach().cpu().numpy()
            - ort_policy
        )
    )

    print()
    print("PyTorch / ONNX 一致性：")
    print(
        "  Value 最大误差  =",
        f"{value_error:.10e}"
    )
    print(
        "  Policy 最大误差 =",
        f"{policy_error:.10e}"
    )

    if value_error > 1e-5 or policy_error > 1e-5:
        raise RuntimeError(
            "数值误差超过 1e-5。"
        )

    # -----------------------------------------------------
    # 再做一次“孤立文件”验证
    # -----------------------------------------------------
    # 直接加载这个单文件，旁边不需要任何 .data。
    print()
    print("✓ 真正单文件 ONNX 导出成功")
    print("✓ 不需要 gomoku.onnx.data")
    print("✓ ONNX Runtime 推理通过")
    print()
    print("请上传到 GitHub：")
    print(output_path)
    print("=" * 64)


if __name__ == "__main__":
    main()

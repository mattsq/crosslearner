import torch
import torch.nn as nn
from typing import Union, Tuple


def export_model(
    model: nn.Module,
    example_inputs: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
    path: str,
    *,
    onnx: bool = False,
    opset_version: int = 17,
) -> None:
    """Export ``model`` to TorchScript or ONNX format.

    Args:
        model: Neural network module to export.
        example_inputs: Sample inputs used for tracing / exporting.
        path: Destination file path.
        onnx: If ``True`` export in ONNX format, otherwise TorchScript.
        opset_version: ONNX opset to use when ``onnx`` is ``True``.
    """
    model.eval()
    if onnx:
        torch.onnx.export(
            model,
            example_inputs,
            path,
            input_names=["input"],
            output_names=["output"],
            opset_version=opset_version,
        )
    else:
        scripted = torch.jit.script(model)
        scripted.save(path)

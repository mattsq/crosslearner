import numpy as np
import torch

from crosslearner.models.acx import ACX
from crosslearner.export import export_model


def test_export_model_torchscript(tmp_path):
    model = ACX(p=3)
    x = torch.randn(4, 3)
    path = tmp_path / "model.pt"
    export_model(model, x, str(path))
    scripted = torch.jit.load(str(path))
    out1 = model(x)
    out2 = scripted(x)
    for a, b in zip(out1, out2):
        assert torch.allclose(a, b)


def test_export_model_onnx(tmp_path):
    model = ACX(p=3)
    x = torch.randn(2, 3)
    path = tmp_path / "model.onnx"
    export_model(model, x, str(path), onnx=True)
    import onnxruntime as ort

    sess = ort.InferenceSession(str(path))
    res = sess.run(None, {"input": x.numpy()})
    expected = [t.detach().numpy() for t in model(x)]
    for r, e in zip(res, expected):
        assert np.allclose(r, e, atol=1e-5)

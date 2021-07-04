import torch
import torchvision

# Traceする対象のメソッドを記述
def test_sample(tensor):
    return torch.sum(tensor)

# Traceする際に用いるサンプル入力を用意
test_sample_input_trace = torch.tensor([1, 2, 3])
# TraceしてTorchScript Modelとして取得
test_sample_trace = torch.jit.trace(test_sample, test_sample_input_trace)
# PyTorchの学習済みモデルもTraceできる
traced_from_pytorch_model = torch.jit.trace(torchvision.models.resnet18(), torch.rand(1, 3, 224, 224))

# TraceしたものをTorchScript ModelとしてSave
traced_from_pytorch_model.save("traced_from_pytorch_model.pt")
# PyTorch側でLoad / 実行できるのはもちろん・・・・
#load_traced_from_pytorch_model = torch.jit.load("traced_from_pytorch_model.pt")
#load_traced_from_pytorch_model(input_tensor)

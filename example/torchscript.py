import torch
import torchvision

# An instance of your model.
model = torchvision.models.resnet18(pretrained=True)

# An example input you would normally provide to your model's forward() method.
example = torch.rand(1, 3, 224, 224)

# Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
traced_script_module = torch.jit.trace(model, example)
traced_script_module.save("traced_from_pytorch_model.pt")
#compiled_model = torch.jit.script(model)
#torch.jit.save(compiled_model, "traced_from_pytorch_model.pt")

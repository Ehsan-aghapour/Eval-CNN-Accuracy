import torch
import torchvision.models as models
import torch.quantization

# Load a pretrained model
model = models.mobilenet_v2(pretrained=True).eval()

# Manually set the quantization configuration for layers you want to quantize
quantize_layers = [4, 6]
for i, layer in enumerate(model.features):
    if i in quantize_layers:
        # Set the layer to use per tensor affine quantization
        layer.qconfig = torch.quantization.get_default_qconfig('fbgemm')

# Apply the quantization configuration to the model
model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
torch.quantization.prepare(model, inplace=True)

# Calibrate the prepared model with representative dataset
# Here you should pass the calibration dataset through the model
# For the example, we use a dummy input
dummy_input = torch.randn(1, 3, 224, 224)
model(dummy_input)

# Convert only the specified layers to their quantized versions
torch.quantization.convert(model, inplace=True)

# Save or use the quantized model
# ...

# Print the model to verify the quantization
print(model)


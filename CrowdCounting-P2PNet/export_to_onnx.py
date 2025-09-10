import torch
from models import build_model

# This must be the path to the PyTorch model you've been using
WEIGHT_PATH = './weights/SHTechA.pth'

print("Loading the P2PNet PyTorch model...")
class MockArgs: backbone = 'vgg16_bn'; row = 2; line = 2
model = build_model(MockArgs())
checkpoint = torch.load(WEIGHT_PATH, map_location='cpu')
model.load_state_dict(checkpoint['model'])
model.eval()
print("Model loaded successfully.")

# Create a dummy input tensor with a typical size
dummy_input = torch.randn(1, 3, 384, 640) 

onnx_file_path = "p2pnet.onnx"

print(f"Exporting model to {onnx_file_path}...")
torch.onnx.export(model,
                  dummy_input,
                  onnx_file_path,
                  opset_version=11,
                  input_names=['input'],
                  output_names=['pred_logits', 'pred_points'])

print(f"Export to ONNX complete! You now have a '{onnx_file_path}' file.")
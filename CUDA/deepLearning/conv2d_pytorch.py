import time
import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv2DExample(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(Conv2DExample, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
    
    def forward(self, x):
        return self.conv(x)

def main():
    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Example parameters
    batch_size = 8
    in_channels = 64
    in_height = 512
    in_width = 512
    out_channels = 128
    kernel_size = 3
    padding = 1
    stride = 1
    warmup_iterations = 5
    timed_iterations = 50
    
    # Create deterministic input tensor that matches C++ examples
    total_input_elements = batch_size * in_channels * in_height * in_width
    input_values = torch.arange(total_input_elements, device=device, dtype=torch.float32)
    input_values = (input_values % 255) / 255.0
    x = input_values.view(batch_size, in_channels, in_height, in_width)
    print(f"Input shape: {x.shape}")
    
    # Create convolution layer and align weights/biases with C++ initialization
    conv_layer = Conv2DExample(in_channels, out_channels, kernel_size, stride, padding).to(device)
    with torch.no_grad():
        weight_elements = out_channels * in_channels * kernel_size * kernel_size
        weight_values = torch.arange(weight_elements, device=device, dtype=torch.float32)
        weight_values = (weight_values % 10) / 10.0
        conv_layer.conv.weight.copy_(weight_values.view_as(conv_layer.conv.weight))
        if conv_layer.conv.bias is not None:
            conv_layer.conv.bias.zero_()

        # Warm-up runs
        for _ in range(warmup_iterations):
            _ = conv_layer(x)

    if device.type == 'cuda':
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        with torch.no_grad():
            for _ in range(timed_iterations):
                output = conv_layer(x)
        end_event.record()
        torch.cuda.synchronize()
        total_kernel_ms = start_event.elapsed_time(end_event)
    else:
        start = time.perf_counter()
        with torch.no_grad():
            for _ in range(timed_iterations):
                output = conv_layer(x)
        total_kernel_ms = (time.perf_counter() - start) * 1000.0

    avg_kernel_ms = total_kernel_ms / timed_iterations
    checksum = output.double().sum().item()
    
    print(f"Output shape: {output.shape}")
    print("Conv2D with PyTorch completed successfully!")
    print(f"Total kernel time over {timed_iterations} iterations: {total_kernel_ms:.3f} ms")
    print(f"Average kernel time per iteration: {avg_kernel_ms:.3f} ms")
    print(f"Output checksum: {checksum:.6f}")

if __name__ == "__main__":
    main()

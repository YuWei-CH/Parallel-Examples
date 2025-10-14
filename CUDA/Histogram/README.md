Histogram CUDA demos

Files:
- histo_kernel.cu: Histogram kernels (MC, SP, SP_TC2)
- histo_host.cu: Host program to generate input or load a file, run kernels, time, and verify
- Makefile: quick build for the demo
- data/: optional place for input files

Build:
  make

Run with generated random input (default 64M bytes):
  ./histo_host

Run with a specific size (e.g., 16M bytes):
  ./histo_host 16777216

Run with a binary file of bytes (0-255):
  ./histo_host data/your_bytes.bin

Output:
- Prints runtime for each kernel (MC, SP, SP_TC2)
- Verifies GPU results match CPU reference

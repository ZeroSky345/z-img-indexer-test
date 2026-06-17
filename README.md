# Z-Image Reproduction

This repository captures a working reproduction of the `Z-Image` project path documented in `DiffSynth-Studio`.

## Source

- Upstream project: `modelscope/DiffSynth-Studio`
- Reference doc: `docs/en/Model_Details/Z-Image.md`
- Inference script used: `examples/z_image/model_inference_low_vram/Z-Image-Turbo.py`

## Reproduction Environment

- Host: lab server `xfusion4`
- GPU used: `NVIDIA A100 80GB PCIe`
- Python: `3.10.12`
- DiffSynth-Studio checkout path on server: `/tmp/DiffSynth-Studio`
- Virtual environment path on server: `/tmp/diffsynth-venv`

## Important Compatibility Fix

The default `pip install -e .` path pulled a CUDA 13 build of PyTorch that could not initialize CUDA on the lab server because the NVIDIA driver was too old for that runtime.

Working replacement:

```bash
pip install --force-reinstall torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
```

After switching to the CUDA 12.4 build:

- `torch.cuda.is_available()` returned `True`
- GPU detection succeeded on the A100

## Reproduction Result

The official low-VRAM script completed successfully and generated:

- `image.jpg`

The generated file was validated as:

- format: `JPEG`
- resolution: `1024x1024`

## Included Files

- `docs/en/Model_Details/Z-Image.md`
- `examples/z_image/model_inference_low_vram/Z-Image-Turbo.py`
- `image.jpg`

## Notes

This repository is a focused reproduction snapshot, not a full mirror of the upstream `DiffSynth-Studio` history.

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository demonstrates **edge AI deployment** on NVIDIA Jetson Orin (8GB RAM) across two domains:

1. **Computer Vision**: ResNet18 image classification with TensorRT optimization (7-30x speedup vs ONNX Runtime)
2. **Large Language Models**: Qwen2.5 models via two technical routes:
   - **TensorRT-Edge-LLM** (0.5B): Runtime optimization approach, OpenAI-compatible API
   - **MLC-LLM** (1.5B/3B): Compiler-based approach, cross-compile + Jetson deployment

### Architecture

```
tensorRT_ONNX/
├── src/                    # Modular source code
│   ├── inference/          # TensorRT inference classes (TensorRT 10.3.0 compatible)
│   ├── calibration/        # INT8 calibration scripts
│   ├── export/            # PyTorch → ONNX export
│   ├── test/              # Engine validation and testing
│   └── utils/             # Helper utilities
├── Edge_llm_deploy/       # TensorRT-Edge-LLM HTTP server (OpenAI-compatible)
├── MLC_llm_deploy/        # MLC-LLM deployment suite (cross-compile + JIT)
└── DOC/                   # Technical documentation
```

## Common Commands

### Computer Vision (ResNet18)

```bash
# 1. Export PyTorch model to ONNX
python src/export/export_resnet.py

# 2. Generate TensorRT engines (FP32/FP16/INT8)
trtexec --onnx=resnet18.onnx --saveEngine=resnet18_fp16.engine --fp16

# 3. Run inference
python -c "from src.inference.tensorrt_inference_fixed import TensorRTInference; TensorRTInference('resnet18_fp16.engine').infer(input_data)"

# 4. INT8 calibration (requires ImageNet dataset)
python src/calibration/calibrate_resnet_simple.py
```

### LLM Deployment - TensorRT-Edge-LLM (0.5B)

```bash
# 1. Start HTTP server (OpenAI-compatible)
cd Edge_llm_deploy
python3 llm_server.py \
    --engine-dir ../qwen25_0.5b_trt/engine_new \
    --llm-inference ../TensorRT-Edge-LLM/build/examples/llm/llm_inference \
    --plugin ../TensorRT-Edge-LLM/build/libNvInfer_edgellm_plugin.so \
    --host 0.0.0.0 --port 8000

# 2. Test with OpenAI SDK
python -c "from openai import OpenAI; client = OpenAI(base_url='http://localhost:8000/v1', api_key='not-needed'); print(client.chat.completions.create(model='qwen2.5-0.5b', messages=[{'role': 'user', 'content': '你好'}]).choices[0].message.content)"

# 3. Performance testing (requires pre-built engine)
cd qwen25_0.5b_trt
./run_perf_test.sh
```

### LLM Deployment - MLC-LLM (1.5B/3B)

```bash
# On Jetson device:
cd MLC_llm_deploy/scrips/JIT_deploy_scrips

# 1. Environment check
./jetson-run.sh check

# 2. Enable 8GB swap (prevents JIT OOM)
./jetson-run.sh swap-on

# 3. Download model weights (~840MB for 1.5B, ~1.9GB for 3B)
./jetson-run.sh download
# For 3B model: MODEL_SIZE=3B MAX_SEQ_LEN=2048 ./jetson-run.sh download

# 4. Start service (OpenAI-compatible on :8000)
./jetson-run.sh serve
# For 3B model: MODEL_SIZE=3B MAX_SEQ_LEN=2048 ./jetson-run.sh serve

# 5. Performance benchmarking
./benchmark.sh
# For 3B model: MODEL_SIZE=3B ./benchmark.sh

# Cross-compile setup (on PC with CUDA GPU):
cd MLC_llm_deploy/scrips/pc-crosscompile_scrips
./setup-pc.sh          # Set up cross-compile environment
./crosscompile.sh      # Build MLC-LLM for sm_87 (Orin architecture)
```

### Testing and Validation

```bash
# Test TensorRT engine compatibility
python src/test/test_engine.py

# Get engine tensor information
python src/test/get_tensor_names.py

# Verify ONNX model validity
python src/test/verify_onnx.py

# Monitor Jetson performance (memory/power/temperature)
tegrastats --interval 1000
```

## Key Technical Concepts

### Environment Versions (Anchored)

| Component | Version |
|-----------|---------|
| Device | Jetson Orin (ARM64, sm_87) |
| JetPack | 6.x (L4T R36.x) |
| CUDA | 12.6 |
| TensorRT | 10.3.0 |
| TensorRT-Edge-LLM | 0.6.0 |
| MLC-LLM | 0.20.0 |
| Container | `dustynv/jetson-containers` based |

### Performance Targets

**ResNet18** (vs ONNX Runtime baseline):
- FP32: 7.4x speedup
- FP16: 15x speedup (recommended for production)
- INT8: 30x speedup (extreme edge scenarios)

**MLC-LLM Qwen2.5** (Jetson 8GB):
- 1.5B: 60 tok/s, TTFT 0.077s, peak 4.44GB RAM
- 3B: 25-35 tok/s, TTFT ~0.15s, peak ~5.5GB RAM (requires `MAX_SEQ_LEN=2048`)

### Technical Route Selection

| Requirement | Recommended Route |
|------------|------------------|
| Real-time response (<100ms) | TensorRT-Edge-LLM 0.5B |
| High-quality dialogue | MLC-LLM 1.5B |
| Complex reasoning | MLC-LLM 3B (with `MAX_SEQ_LEN=2048`) |
| Image recognition | TensorRT ResNet18 FP16 |

### Code Architecture Patterns

1. **TensorRT 10.3.0 API Compatibility**: All inference code uses modern API (`num_io_tensors`, `get_tensor_name`, `execute_v2`)
2. **Modular Source Structure**: Code organized by function in `src/` (inference, calibration, export, test, utils)
3. **Container-Based Deployment**: Docker-based reproducible environments
4. **Cross-Compilation Strategy**: MLC-LLM builds on PC (CUDA GPU) → runs on Jetson to avoid 8GB build-time constraints

### Important Constraints

- **8GB RAM Limit**: MLC-LLM 3B requires `MAX_SEQ_LEN=2048` (down from default 4096)
- **No Streaming in TensorRT-Edge-LLM v0.6.0**: Backend lacks streaming, `stream=True` returns full response
- **Swap Required for MLC-LLM**: 8GB swap needed before first JIT compilation to prevent OOM
- **Exclusive Frameworks**: `TensorRT-Edge-LLM/` and large model files excluded by `.gitignore`

## Development Workflow

1. **Model Development**: Add features in appropriate `src/` subdirectories
2. **Testing**: Use scripts in `src/test/` for validation
3. **Deployment**: Use route-specific scripts (Edge-LLM or MLC-LLM)
4. **Benchmarking**: Route-specific benchmark scripts before/after optimizations
5. **Documentation**: Update relevant README.md files in subdirectories

## File Structure Notes

- `.gitignore` excludes: TensorRT engines, ONNX models, PyTorch weights, datasets, and the `TensorRT-Edge-LLM/` framework directory
- Main README.md is in Chinese (project context: Chinese edge AI deployment)
- Technical documentation in `DOC/` provides detailed implementation guides

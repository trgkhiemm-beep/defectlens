Hardware: CPU: 11th Gen Intel(R) Core(TM) i5-1135G7 @ 2.40GHz, GPU: Intel(R) Iris(R) Xe Graphics (iGPU)  
Batch 1, 256x256, `transistor` sample, end to end (preprocess + embed + score). One process per configuration, 25 s cool-down before each, warm-up, then up to 30 runs or 6 s; 3 round-robin repeats, p50 = median over repeats [min-max]. Burst latency: a 15 W laptop is slower under sustained load.

| Runtime | Device | Embedder | Bank | p50 (ms) [spread] | p95 (ms) | FPS | Embed / Score p50 (ms) | Speed-up vs PyTorch r0.1 | Mean img AUROC | Load/compile (s) | Disk (MB) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| PyTorch eager | CPU | FP32 | r0.1 (17510) | 1925.3 [1646-2407] | 2548.2 | 0.5 | 879.0 / 1038.4 | x1.0 | (= OV FP32) | 1.0 | 197.4 |
| PyTorch eager | CPU | FP32 | r0.01 (1751) | 193.5 [188-624] | 792.4 | 5.2 | 157.1 / 37.0 | x9.9 | (= OV FP32) | 0.9 | 105.1 |
| OpenVINO | CPU | FP32 | r0.1 (17510) | 230.9 [229-553] | 1546.4 | 4.3 | 68.4 / 162.0 | x8.3 | 0.9888 | 0.3 | 146.1 |
| OpenVINO | CPU | FP32 | r0.01 (1751) | 87.8 [88-89] | 90.0 | 11.4 | 66.4 / 20.1 | x21.9 | 0.9880 | 0.2 | 99.9 |
| OpenVINO | CPU | INT8 | r0.1 (17510) | 183.3 [183-192] | 1382.5 | 5.5 | 21.0 / 160.9 | x10.5 | 0.9881 | 0.6 | 75.2 |
| OpenVINO | CPU | INT8 | r0.01 (1751) | 41.0 [41-42] | 42.5 | 24.4 | 19.7 / 20.2 | x46.9 | 0.9878 | 0.3 | 28.9 |
| OpenVINO | GPU | FP32 | r0.1 (17510) | 266.6 [135-350] | 376.0 | 3.8 | 61.3 / 196.4 | x7.2 | 0.9888 | 4.9 | 146.1 |
| OpenVINO | GPU | FP32 | r0.01 (1751) | 33.4 [33-126] | 42.9 | 30.0 | 21.0 / 10.6 | x57.7 | 0.9880 | 4.4 | 99.9 |
| OpenVINO | GPU | INT8 | r0.1 (17510) | 428.8 [310-450] | 570.2 | 2.3 | 65.3 / 365.3 | x4.5 | 0.9881 | 13.3 | 75.2 |
| OpenVINO | GPU | INT8 | r0.01 (1751) | 54.1 [50-62] | 62.4 | 18.5 | 29.3 / 21.4 | x35.6 | 0.9878 | 7.2 | 28.9 |

_code `16e4e17a-dirty`, torch 2.14.0+cpu, Windows-11-10.0.26200-SP0_

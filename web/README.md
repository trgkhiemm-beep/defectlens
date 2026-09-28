---
title: DefectLens
emoji: 🔍
colorFrom: blue
colorTo: gray
sdk: static
app_file: index.html
pinned: false
license: cc-by-nc-sa-4.0
short_description: PatchCore defect detection running in your browser
custom_headers:
  cross-origin-embedder-policy: require-corp
  cross-origin-opener-policy: same-origin
  cross-origin-resource-policy: cross-origin
---

# DefectLens (in-browser demo)

Unsupervised industrial visual inspection: a PatchCore model trained only on defect-free parts
(MVTec AD: transistor, capsule, metal nut, carpet) runs **entirely in the visitor's browser**
with ONNX Runtime Web (multi-threaded WASM; WebGPU optional). No server, and images never
leave the device.

The COOP/COEP headers above make the page cross-origin isolated, which multi-threaded WASM
needs (SharedArrayBuffer): 1 thread took ~15 s per image on an i5-1135G7, 4 threads ~0.7 s.

Source, training pipeline, OpenVINO edge build and benchmarks: see the GitHub repository.
Model trained on MVTec AD (CC BY-NC-SA 4.0): non-commercial use only.

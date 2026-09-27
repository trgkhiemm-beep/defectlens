---
title: DefectLens
emoji: 🔍
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
license: cc-by-nc-sa-4.0
short_description: Unsupervised defect detection, INT8 OpenVINO on CPU
---

# DefectLens

Unsupervised industrial visual inspection: PatchCore trained only on defect-free parts
(MVTec AD: transistor, capsule, metal nut, carpet), exported to OpenVINO with an INT8
backbone and running on this Space's CPU.

- UI: this page
- REST API: `/docs` (OpenAPI), `POST /api/v1/inspect`
- Monitoring: `/api/v1/metrics` (defect rate, latency, score-drift monitor)

Source code, training pipeline and benchmarks: see the GitHub repository.
Model trained on MVTec AD (CC BY-NC-SA 4.0): non-commercial use only.

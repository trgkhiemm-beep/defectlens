# Hướng dẫn test DefectLens

Tài liệu này hướng dẫn kiểm tra mọi phần của dự án: test tự động, **cách lấy ảnh test**, test model trên cả
thư mục ảnh, test API, giao diện, bản web trên Hugging Face, Docker, benchmark và pipeline trên Kaggle.
Mọi con số "kỳ vọng" dưới đây đều **đo thật** trên model đã deploy (commit của README).

> Quy ước lệnh: chạy trong thư mục gốc của repo, trên Windows (PowerShell hoặc CMD).
> `.venv\Scripts\python` là Python của dự án. Trên Linux/macOS thay bằng `.venv/bin/python`.

---

## 0. Muốn làm gì thì xem mục nào

| Mục tiêu | Lệnh chính | Mục |
|---|---|---|
| Chạy toàn bộ test tự động | `.venv\Scripts\python -m pytest -q` | [2](#2-test-tự-động-pytest) |
| Lấy ảnh test | 8 ảnh mẫu có sẵn, gói ảnh từ Kaggle, MVTec gốc, ảnh biên, ảnh tự chụp | [3](#3-lấy-ảnh-test) |
| Chạy model trên cả thư mục ảnh và chấm điểm | `scripts\inspect_folder.py` | [4](#4-test-model-trên-một-thư-mục-ảnh) |
| Test REST API | `uvicorn` + `curl.exe` | [5](#5-test-rest-api-fastapi) |
| Test giao diện Gradio | trình duyệt | [6](#6-test-giao-diện-gradio) |
| Test bản web / Hugging Face Space | trình duyệt | [7](#7-test-bản-web-và-hugging-face-space) |
| Đo tốc độ | `scripts\benchmark.py` | [9](#9-benchmark-hiệu-năng) |
| Chạy lại toàn bộ pipeline | Kaggle | [10](#10-test-pipeline-end-to-end-trên-kaggle) |

---

## 1. Chuẩn bị môi trường

```bash
python -m venv .venv
.venv\Scripts\python -m pip install --prefer-binary -r requirements.txt
```

- `--prefer-binary` là bắt buộc trên Windows: thiếu nó, `albumentations` sẽ cố build `stringzilla` từ mã nguồn và lỗi.
- Các mục 4 đến 9 cần **gói model** trong `deploy/model/`. Nếu chưa có, tạo từ `artifacts/edge/` (tải từ Kaggle, xem mục 10):

```bash
.venv\Scripts\python scripts\export_bundle.py --edge artifacts/edge --precision int8 --bank r0.01 --out deploy/model
```

---

## 2. Test tự động (pytest)

Toàn bộ test chạy trên **ảnh giả sinh ra trong lúc test** (không cần MVTec, không tải gì từ mạng) và backbone
`resnet18` với trọng số ngẫu nhiên, nên đo **tính đúng của code**, không đo độ chính xác của model.

```bash
.venv\Scripts\python -m pytest -q                     # tất cả, khoảng 8-10 phút
.venv\Scripts\python -m pytest tests\test_app.py -q   # một file
.venv\Scripts\python -m pytest -q -k "drift"          # các test có chữ "drift" trong tên
.venv\Scripts\python -m pytest -q -x --tb=short       # dừng ở lỗi đầu tiên, traceback ngắn
```

| File | Kiểm tra gì | Thời gian |
|---|---|---|
| `test_data_pipeline.py` | sinh lỗi tổng hợp đúng vùng, cổng validation bắt được dữ liệu hỏng / cũ / rò rỉ, split, DataLoader, gói ảnh test | nhanh |
| `test_patchcore.py` | AUROC / AUPRO đúng định nghĩa, chọn threshold, drift, coreset, PatchCore định vị lỗi, script train end-to-end | trung bình |
| `test_edge.py` | OpenVINO và ONNX khớp PyTorch, INT8 giữ đặc trưng, bộ blur tách đôi đúng, build edge, benchmark, export web | chậm nhất |
| `test_app.py` | API (health, inspect, lỗi 400/404), UI, không import torch, import không nạp model, drift monitor, chạy thư mục ảnh, mọi ảnh biên | trung bình |

**Kết quả đạt:** dòng cuối có dạng `48 passed` (có thể thêm vài `warnings`, không sao).

**Muốn giống hệt CI** (môi trường sạch, không có `deploy/model`): tạm đổi tên `deploy\model` thành `deploy\model.bak`,
chạy pytest, rồi đổi lại. Nếu vẫn pass thì CI sẽ pass.

---

## 3. Lấy ảnh test

### Quy tắc quan trọng nhất: chỉ dùng ảnh model chưa từng thấy

| Nguồn ảnh MVTec | Dùng để đánh giá? | Lý do |
|---|---|---|
| `train/good` | **Không** | Là chính memory bank của model: điểm thấp giả tạo, kết quả đẹp nhưng vô nghĩa |
| `val` (20% train/good + lỗi tổng hợp) | **Không** | Đã dùng để chọn threshold |
| `test/...` (split chính thức) | **Có** | Model và threshold chưa từng thấy |

Mọi cách lấy ảnh dưới đây đều lấy từ **split test**.

### 3.1. Có sẵn trong repo: 8 ảnh mẫu

`deploy/model/samples/` (và `web/samples/`): mỗi category 1 ảnh tốt (`*_good.png`) + 1 ảnh lỗi (`*_defect.png`),
256x256, lấy từ split test. Dùng để kiểm tra nhanh mọi thành phần có cho **đúng những con số này** không:

| Ảnh | Bản API / Docker (INT8, `deploy/model`): điểm / threshold / tỷ lệ | Bản web (FP32, Space): điểm / threshold / tỷ lệ | Verdict |
|---|---|---|---|
| transistor_good | 1.9177 / 2.1732 / 0.88 | 1.8785 / 2.1084 / 0.89 | OK |
| transistor_defect | 2.5954 / 2.1732 / 1.19 | 2.6190 / 2.1084 / 1.24 | DEFECT |
| capsule_good | 1.3337 / 1.4534 / 0.92 | 1.3483 / 1.4605 / 0.92 | OK |
| capsule_defect | 1.7176 / 1.4534 / 1.18 | 1.7484 / 1.4605 / 1.20 | DEFECT |
| metal_nut_good | 2.0878 / 2.2632 / 0.92 | 2.0217 / 2.2628 / 0.89 | OK |
| metal_nut_defect | 3.0957 / 2.2632 / 1.37 | 3.0491 / 2.2628 / 1.35 | DEFECT |
| carpet_good | 1.4983 / 1.5710 / 0.95 | 1.4913 / 1.5659 / 0.95 | OK |
| carpet_defect | 3.1707 / 1.5710 / 2.02 | 3.1990 / 1.5659 / 2.04 | DEFECT |

Hai cột khác nhau một chút là **đúng thiết kế**: bản API dùng embedder INT8, bản web dùng FP32 (xem README).
Sai khác trong cùng một cột giữa hai lần chạy phải dưới 0.5% (CPU) hoặc 2% (iGPU, tính FP16).

> Lưu ý đã biết: với `capsule_defect`, verdict DEFECT đúng nhưng vùng vượt threshold nằm trên nền phía trên viên
> thuốc, không nằm trên viên thuốc (xem README, mục Known limitations).

### 3.2. Khuyến nghị: gói ảnh test thật từ Kaggle (khoảng 110 ảnh, 15 MB)

Dữ liệu đã xử lý trên Kaggle (`/kaggle/working/data/processed`) có sẵn split test ở đúng độ phân giải model
dùng. Script `make_test_pack.py` lấy N ảnh mỗi loại lỗi, xếp vào thư mục **đặt tên theo nhãn**, kèm mask.

Trên Kaggle (nếu phiên mới, chạy lại `prepare_data.py` trước, xem mục 10):

```python
%cd /kaggle/working/defectlens
!git pull -q
!python scripts/make_test_pack.py --data /kaggle/working/data/processed --out /kaggle/working/test_pack --per-type 5 --with-masks
!cd /kaggle/working && zip -rq test_pack.zip test_pack && ls -lh test_pack.zip
```

- `--per-type 5`: 5 ảnh mỗi (category, loại lỗi), kể cả `good`. `--per-type 0` lấy **toàn bộ** split test (khoảng 470 ảnh, 60 MB).
- `--categories transistor capsule`: chỉ lấy vài category.

Tải `test_pack.zip` từ mục **Output** của notebook, giải nén vào **thư mục gốc repo** để có `test_pack/` (đã gitignore):

```
test_pack/
  manifest.csv                     category, label, defect_type, file
  transistor/good/*.png            ảnh tốt  -> kỳ vọng OK
  transistor/bent_lead/*.png       ảnh lỗi  -> kỳ vọng DEFECT
  transistor/ground_truth/bent_lead/*.png   mask vùng lỗi (trắng = lỗi)
  capsule/... metal_nut/... carpet/...
```

Nếu zip tạo ra thư mục lồng `test_pack/test_pack/`, chuyển nội dung lên một cấp.

### 3.3. Tải MVTec AD gốc (ảnh 700-1024 px)

Hai cách, đều do **bạn tự thực hiện** (cần tài khoản / chấp nhận giấy phép):

- **Trang chính thức:** mvtec.com, mục *MVTec AD* (điền form, khoảng 4.9 GB, giấy phép CC BY-NC-SA 4.0: chỉ dùng phi thương mại).
- **Kaggle CLI:** tải file `kaggle.json` (Kaggle > Settings > API > Create New Token) vào `%USERPROFILE%\.kaggle\`, rồi:

```bash
.venv\Scripts\python -m pip install kaggle
.venv\Scripts\kaggle datasets download -d <owner>/<dataset-slug> -p data/raw --unzip
```

`<owner>/<dataset-slug>` là phần cuối URL của dataset MVTec AD bạn đã dùng trên Kaggle (trang dataset, thanh địa chỉ).

Cấu trúc gốc đã **đặt tên thư mục theo nhãn**, nên dùng thẳng được với mục 4: trỏ `--images` vào
`data/raw/.../transistor/test` (không phải `train`). Ảnh gốc lớn hơn 256 px: model tự resize giống lúc train.

### 3.4. Ảnh tình huống biên (tự sinh, không cần tải)

```bash
.venv\Scripts\python scripts\make_edge_cases.py --sample deploy/model/samples/transistor_good.png --out test_inputs/edge_cases
```

Sinh 18 file từ một ảnh **tốt**. Kết quả đo thật qua API (category `transistor`):

| File | HTTP | Kết quả | Ý nghĩa |
|---|---|---|---|
| `empty.png` | 400 | `empty upload` | kiểm tra validate đầu vào |
| `not_an_image.txt` | 400 | `not a decodable image` | |
| `truncated.png` | 400 | `not a decodable image` | file hỏng giữa chừng |
| `too_large_over_10mb.bmp` (100 MB) | 400 | `image larger than 10 MB` | giới hạn upload |
| `as_jpeg_q90.jpg` | 200 | OK 0.91x | nén JPEG không ảnh hưởng |
| `upscaled_1024.png` | 200 | OK 0.88x | ảnh lớn được resize đúng |
| `with_alpha.png` | 200 | OK 0.88x | kênh alpha bị bỏ đúng |
| `non_square_wide.png` | 200 | OK 0.88x | ảnh bị kéo về hình vuông |
| `dark_lighting.png` / `glare.png` | 200 | OK 0.94x / 0.97x | chịu được thay đổi ánh sáng nhẹ |
| `grayscale.png` | 200 | **DEFECT 1.13x** | giới hạn: model cần ảnh màu |
| `tiny_32px.png` | 200 | **DEFECT 1.86x** | giới hạn: mất chi tiết |
| `blurred.png` / `sensor_noise.png` | 200 | **DEFECT 1.54x / 1.42x** | camera mất nét hoặc nhiễu gây loại nhầm |
| `rotated_10deg.png` / `shifted_20px.png` | 200 | **DEFECT 1.20x / 1.08x** | **đúng**: với transistor, lệch vị trí chính là lỗi `misplaced` |
| `random_noise.png` / `flat_grey.png` | 200 | DEFECT 1.93x / 1.63x, **không cảnh báo** | giới hạn: cảnh báo "ảnh lạ" (3x threshold) quá yếu |

Không có trường hợp nào được phép trả **500**: nếu thấy 500, đó là bug.

### 3.5. Ảnh tự chụp

Có thể chụp vật thật (ví dụ viên con nhộng, đai ốc), nhưng cần hiểu model được train trên ảnh MVTec: nền, ánh
sáng, góc chụp, độ phóng đại cố định. Ảnh điện thoại gần như chắc chắn **khác phân phối**, nên thường ra DEFECT
dù vật không lỗi. Dùng ảnh tự chụp để **minh họa giới hạn** (domain shift), **không dùng để đo độ chính xác**.

Muốn kết quả hợp lý nhất:
1. Nền trơn giống MVTec (capsule: nền trắng; metal_nut: nền đen), vật nằm giữa và chiếm khoảng 60-70% khung.
2. Cắt ảnh thành **hình vuông** (Photos / Paint trên Windows) trước khi upload, vì model kéo ảnh về 256x256.
3. Ánh sáng đều, không bóng đổ, lấy nét rõ.
4. Chụp 10 ảnh vật tốt + vài ảnh vật có lỗi (vết xước, mẻ), xếp vào `my_photos/good/` và `my_photos/scratch/`, rồi chạy mục 4.

### 3.6. Tạo ảnh lỗi nhân tạo từ ảnh tốt

Dùng đúng bộ sinh lỗi của pipeline (vết xước, dán texture, cắt-dán), giới hạn trong vùng vật thể:

```python
# .venv\Scripts\python make_synth.py
import os, sys, cv2, numpy as np
sys.path.insert(0, ".")
from src.data.synthetic import inspection_region, synthesize_defect

os.makedirs("test_inputs/synthetic/scratch", exist_ok=True)

img = cv2.imread("deploy/model/samples/capsule_good.png")
rng = np.random.default_rng(0)
region = inspection_region(img, foreground="color")          # transistor: roi=[0.28, 0.15, 0.78, 0.95]
for i in range(5):
    bad, mask, kind = synthesize_defect(img, rng, region=region)
    cv2.imwrite(f"test_inputs/synthetic/scratch/capsule_{i}_{kind}.png", bad)
```

Lỗi tổng hợp **dễ hơn lỗi thật** (bài học threshold trong README). Đo thật: 5 lỗi tổng hợp trên capsule có điểm
**2.1-2.7x threshold**, trong khi lỗi thật `capsule_defect` chỉ **1.18x**. Recall 100% trên lỗi tổng hợp vì thế
không nói lên độ chính xác thật; dùng chúng để kiểm tra model **định vị** đúng chỗ (xem `--overlays`).

---

## 4. Test model trên một thư mục ảnh

```bash
.venv\Scripts\python scripts\inspect_folder.py --images test_pack/transistor --category transistor
.venv\Scripts\python scripts\inspect_folder.py --images test_pack/capsule --category capsule --overlays
```

- Nhãn lấy từ **tên thư mục**: ảnh trong `good/` kỳ vọng OK, ảnh trong thư mục khác (`scratch/`, `bent_lead/`...) kỳ vọng DEFECT; `ground_truth/` bị bỏ qua.
- Kết quả trong `reports/inspect/<category>/`: `results.csv` (từng ảnh), `summary.json` (accuracy, tỷ lệ loại nhầm hàng tốt, recall theo từng loại lỗi, danh sách ảnh sai), và `overlays/` (ảnh gốc + heatmap) nếu có `--overlays`.
- Mỗi dòng in ra: tên file, verdict, tỷ lệ điểm/threshold, và `<-- WRONG` nếu sai.

**Kỳ vọng** (toàn bộ split test, bản INT8 1%, từ `deploy/model/MODEL_CARD.md`):

| Category | F1 | Tỷ lệ loại nhầm hàng tốt (FPR) | Ghi chú |
|---|---|---|---|
| transistor | 1.000 | 0.000 | |
| capsule | 0.972 | 0.043 | |
| metal_nut | 0.989 | 0.091 | 2 ảnh tốt luôn bị báo lỗi (giới hạn đã biết) |
| carpet | 0.927 | **0.500** | **dự kiến**: lệch dữ liệu val/test, xem README |

Với gói 5 ảnh mỗi loại, mỗi ảnh chiếm 20% của nhóm đó: chênh lệch vài ảnh so với bảng trên là bình thường.
Muốn so sánh chặt, dùng `--per-type 0` ở mục 3.2.

**Phân tích lỗi:** mở `overlays/` của các ảnh `WRONG`, đặt cạnh mask trong `ground_truth/`: heatmap có nằm
đúng chỗ lỗi không? Đây là cách tìm các trường hợp "đúng verdict, sai chỗ".

---

## 5. Test REST API (FastAPI)

Khởi động (cửa sổ terminal 1):

```bash
.venv\Scripts\python -m uvicorn app.main:create_app --factory --host 127.0.0.1 --port 7860
```

Khi thấy `Uvicorn running on http://127.0.0.1:7860`, mở terminal 2:

```bash
curl.exe http://127.0.0.1:7860/api/v1/health
curl.exe http://127.0.0.1:7860/api/v1/categories
curl.exe -F "file=@deploy/model/samples/transistor_defect.png" -F "category=transistor" http://127.0.0.1:7860/api/v1/inspect
curl.exe http://127.0.0.1:7860/api/v1/metrics
```

Kỳ vọng cho lệnh `inspect`: `"verdict":"DEFECT"`, `"score"` khoảng 2.5954, `"score_ratio"` khoảng 1.19,
`"timing_ms"` có `total` vài chục ms.

**Lấy heatmap về thành ảnh:**

```bash
curl.exe -F "file=@deploy/model/samples/carpet_defect.png" -F "category=carpet" -F "heatmap=true" http://127.0.0.1:7860/api/v1/inspect -o resp.json
.venv\Scripts\python -c "import json,base64; open('heatmap.png','wb').write(base64.b64decode(json.load(open('resp.json'))['heatmap_png_base64']))"
```

**Mã lỗi phải đúng:**

| Gửi gì | Kỳ vọng |
|---|---|
| category không tồn tại (`-F category=pizza`) | 404, liệt kê category hợp lệ |
| file rỗng, file text, PNG hỏng, file trên 10 MB | 400 kèm lý do (mục 3.4) |
| thiếu trường `file` hoặc `category` | 422 (FastAPI tự validate) |

**Tài liệu API tương tác:** mở http://127.0.0.1:7860/docs, bấm `POST /api/v1/inspect` > *Try it out*.

**Test drift monitor** (đã chạy thử, kết quả thật ở cuối đoạn mã). Mỗi category cần 100 ảnh **đạt** đầu tiên để
lập mốc (`warming_up`), rồi tối thiểu 30 ảnh để đánh giá. Dùng transistor: carpet không phù hợp vì loại nhầm
khoảng 50% hàng tốt, mà chỉ ảnh đạt mới được tính.

```python
# .venv\Scripts\python drift_demo.py   (server ở trên đang chạy)
import cv2, httpx
URL = "http://127.0.0.1:7860/api/v1"
img = cv2.imread("deploy/model/samples/transistor_good.png")
def send(im):
    httpx.post(f"{URL}/inspect", files={"file": ("x.png", cv2.imencode(".png", im)[1].tobytes())},
               data={"category": "transistor"}, timeout=60)
for _ in range(140):
    send(img)
print(httpx.get(f"{URL}/metrics").json()["categories"]["transistor"]["drift"])   # state: ok, ratio 1.0
blur = cv2.GaussianBlur(img, (0, 0), 0.8)     # "camera hơi mất nét": từng ảnh vẫn OK (0.98x threshold)
for _ in range(100):
    send(blur)
print(httpx.get(f"{URL}/metrics").json()["categories"]["transistor"]["drift"])   # state: drift, ratio khoảng 1.107
```

Ý nghĩa: không ảnh nào bị loại, nhưng điểm của hàng tốt đã tăng 10%. Monitor báo `drift` kèm hành động
`collect new normal samples and recalibrate the threshold` trước khi dây chuyền bắt đầu loại nhầm hàng loạt.
Lưu ý: bộ đếm nằm trong bộ nhớ, khởi động lại server là về 0.

**Test tải đồng thời** (API phải không lỗi khi nhiều request cùng lúc):

```python
import httpx, concurrent.futures as cf
def one(_):
    with open("deploy/model/samples/metal_nut_defect.png", "rb") as f:
        return httpx.post("http://127.0.0.1:7860/api/v1/inspect", files={"file": f},
                          data={"category": "metal_nut"}, timeout=60).json()["verdict"]
with cf.ThreadPoolExecutor(8) as ex:
    print(set(ex.map(one, range(40))))   # kỳ vọng: {'DEFECT'}
```

Đo thật với 8 luồng: mọi kết quả đúng, nhưng `latency_ms.p50` trong `/metrics` tăng lên khoảng 400 ms, vì service
xử lý **tuần tự** (một lock quanh model OpenVINO, vốn không gọi đồng thời được). Muốn tăng thông lượng: chạy
nhiều worker/tiến trình, hoặc dùng `AsyncInferQueue` của OpenVINO (hướng cải tiến, chưa làm).

---

## 6. Test giao diện Gradio

Cùng server ở mục 5, mở http://127.0.0.1:7860.

- [ ] Bấm từng ảnh ở mục *Examples* (ảnh và category được điền sẵn), rồi bấm *Inspect*: kết quả khớp cột INT8 ở mục 3.1.
- [ ] Upload một ảnh trong `test_pack/`: heatmap hiện ra, viền trắng bao vùng vượt threshold.
- [ ] Ô *Verdict* ghi `DEFECT (score = 1.19 x threshold)`, không có phần trăm (điểm không phải xác suất).
- [ ] Bấm *Inspect* khi chưa có ảnh: hiện thông báo lỗi thân thiện, không crash.
- [ ] Mở *Production monitoring* > *Refresh*: số ảnh đã kiểm tra tăng đúng.

---

## 7. Test bản web và Hugging Face Space

Space: https://huggingface.co/spaces/khiem05/defectlens (bản chạy trong trình duyệt, FP32).
Chạy thử ở máy trước khi upload: `.venv\Scripts\python scripts\serve_web.py` rồi mở http://127.0.0.1:8765.

- [ ] Dòng trạng thái: `Ready · WASM (4 threads)`. Nếu `1 thread`: header COOP/COEP không có hiệu lực, mỗi ảnh mất khoảng 15 giây.
- [ ] Bấm 8 ảnh mẫu: điểm khớp cột **bản web** ở mục 3.1 đến 4 chữ số thập phân.
- [ ] Kéo thả một ảnh vào ô upload, và chọn file bằng nút: cả hai cách đều chạy.
- [ ] Đổi category khi đã có ảnh: tự chấm lại, lần đầu mỗi category tải thêm khoảng 5-7 MB.
- [ ] Lần tải thứ hai (F5): model lấy từ cache, trạng thái `loaded in` nhanh hơn hẳn.
- [ ] Backend **WebGPU** (thử nghiệm): nếu chạy, ghi lại số ms so với WASM; nếu lỗi, trang phải tự quay về WASM.
- [ ] Mở trên điện thoại hoặc thu hẹp cửa sổ dưới 820 px: bố cục chuyển thành một cột, không cuộn ngang.
- [ ] DevTools (F12) > Console: không có lỗi đỏ.

Kỳ vọng tốc độ (i5-1135G7, WASM 4 luồng, tab đang mở): khoảng 0.6-0.7 giây mỗi ảnh; lần đầu tiên chậm hơn.

---

## 8. Test Docker (nếu máy có Docker)

```bash
.venv\Scripts\python scripts\export_bundle.py
docker compose up --build
```

- [ ] Sau khoảng 1 phút, `docker ps` báo container `healthy` (healthcheck gọi `/api/v1/health`).
- [ ] Các lệnh `curl.exe` của mục 5 cho cùng kết quả.
- [ ] Image không chứa PyTorch: `docker run --rm defectlens:latest python -c "import sys, app.main; print('torch' in sys.modules)"` in ra `False`.

Không có Docker vẫn yên tâm được: job `docker` của CI build image và kiểm tra điều trên ở mỗi lần push.

---

## 9. Benchmark hiệu năng

Cắm sạc, chọn *Power mode: Best performance*, đóng ứng dụng nặng (laptop mỏng tự hạ xung khi nóng):

```bash
.venv\Scripts\python scripts\benchmark.py --edge artifacts/edge --devices CPU GPU --embedders fp32 int8 --banks r0.1 r0.01
```

Mất khoảng 20 phút (mỗi cấu hình chạy trong tiến trình riêng, nghỉ 25 giây, lặp 3 vòng). Kết quả:
`artifacts/edge/benchmark.md`. Kỳ vọng trên i5-1135G7 (bank 1%): OpenVINO INT8 CPU khoảng 41 ms,
OpenVINO FP32 CPU khoảng 88 ms, PyTorch FP32 khoảng 190 ms. Đọc cột `[min-max]`: khoảng rộng nghĩa là máy bị
hạ xung trong lúc đo, nên chạy lại. Các cấu hình bank 10% luôn nhiễu vì tự làm nóng máy.

Đo nhanh một cấu hình (vài giây):

```bash
.venv\Scripts\python scripts\inspect_folder.py --images deploy/model/samples --category transistor
```

(dòng `latency_p50_ms` trong kết quả; ảnh mẫu không nằm trong thư mục con nên sẽ không được chấm đúng/sai).

---

## 10. Test pipeline end-to-end trên Kaggle

Notebook: GPU T4, Internet ON, thêm dataset MVTec AD làm input. Chạy từng dòng; lệnh nào lỗi thì dừng ở đó.

```python
%cd /kaggle/working
!rm -rf defectlens && git clone https://github.com/trgkhiemm-beep/defectlens.git
%cd defectlens
!pip install -q -r requirements.txt
import subprocess
run = lambda *a: subprocess.run(["python", *a], check=True)
run("scripts/prepare_data.py", "--src", "/kaggle/input/datasets/ipythonx", "--dst", "/kaggle/working/data/processed", "--overwrite")
run("scripts/train_patchcore.py", "--data", "/kaggle/working/data/processed", "--out", "/kaggle/working/artifacts/patchcore")
run("scripts/analyze_thresholds.py", "--artifacts", "/kaggle/working/artifacts/patchcore")
run("scripts/build_edge.py", "--data", "/kaggle/working/data/processed", "--out", "/kaggle/working/edge")
```

| Bước | Dấu hiệu đạt |
|---|---|
| `prepare_data` | dòng đầu `synthetic_v3 val_ratio=0.2`, dòng cuối `VALIDATION PASSED`; transistor `train/good=171 val/good=42` |
| `train_patchcore` | image AUROC: transistor khoảng 0.999, capsule 0.97, metal_nut 0.997, carpet 0.986; **dưới 0.95 là có bug** |
| `analyze_thresholds` | có bảng Recalibration; carpet có `Normal-score shift` khoảng x1.09 |
| `build_edge` | 9 dòng `[fp32/..]`, `[int8/..]`, `[int8mix/..]`; không có lỗi `exported scorer deviates` |

Nếu `prepare_data` báo `PREFLIGHT FAILED` hoặc `VALIDATION FAILED`: repo có file cũ trộn file mới, hoặc dữ liệu
không đạt. Đọc dòng lỗi, **không** train tiếp.

---

## 11. CI trên GitHub

Mỗi lần `git push`, tab **Actions** chạy 2 job: `test` (toàn bộ pytest) và `docker` (build image). Khi đỏ:
bấm vào lần chạy, mục **Annotations** có một lỗi tên `pytest summary` chứa các dòng `FAILED` / `ERROR` và
đuôi log, không cần mở log đầy đủ.

---

## 12. Lỗi thường gặp

| Triệu chứng | Nguyên nhân | Cách xử lý |
|---|---|---|
| `Failed building wheel for stringzilla` | cài `albumentations` trên Windows | thêm `--prefer-binary` |
| `FileNotFoundError ... deploy/model/manifest.json` | chưa có gói model | chạy `export_bundle.py` (mục 1) |
| `No module named 'onnx'` | cài thiếu | `pip install -r requirements.txt` lại |
| Web báo `WASM (1 thread)` | trang không cross-origin isolated | dùng `serve_web.py`, không dùng `python -m http.server` |
| Benchmark lúc nhanh lúc chậm | laptop hạ xung nhiệt | cắm sạc, Best performance, chạy lại |
| Điểm lệch vài % so với bảng 3.1 | chạy trên iGPU (FP16) hoặc so nhầm cột INT8 / FP32 | so đúng cột; CPU phải lệch dưới 0.5% |
| Tool báo không tìm thấy thư mục `Tài liệu` | tên thư mục OneDrive lưu dạng Unicode NFD | dùng đường dẫn ngắn `C:\Users\ADMIN\OneDrive\TAILIU~1\...` |
| `cannot do a partial commit during a merge` | có merge dở dang sau `git pull` | `git status`, hoàn tất merge trước |

---

## 13. Checklist trước khi demo / phỏng vấn

- [ ] CI xanh (badge trên README).
- [ ] Space mở được, trạng thái `WASM (4 threads)`, 8 ảnh mẫu đúng.
- [ ] Chạy mục 4 trên `test_pack/` và thuộc 2-3 ảnh sai tiêu biểu (kèm lý do từ heatmap).
- [ ] Chạy mục 3.4 và giải thích được vì sao ảnh xoay / mờ / xám ra DEFECT.
- [ ] Nhớ các con số: AUROC 0.989, F1 0.964, 41 ms / 24 FPS, 47.5 MB, 4.7x.

# imgdedup

CLI chạy hoàn toàn trên máy local để:

- tìm ảnh trùng bằng SHA-256, pHash và embedding hình ảnh;
- xem báo cáo trước khi xóa;
- chuyển ảnh trùng vào thùng rác có manifest khôi phục;
- chọn một tập ảnh đại diện có kích thước chính xác;
- xuất YOLO dataset đã loại ảnh trùng bằng SHA-256, pHash và embedding mà không sửa dữ liệu nguồn.

> **An toàn mặc định:** nếu không truyền `--inplace`, công cụ chỉ quét và tạo báo cáo, không thay đổi ảnh nguồn.

## Cài đặt

Yêu cầu Python `>=3.10,<3.11` và [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run python main.py --help
```

Lần chạy đầu có thể mất nhiều thời gian do phải tải model và tạo embedding. Những lần sau sẽ dùng cache cho các file không thay đổi.

## Cấu hình YAML (tùy chọn)

Có thể chạy hoàn toàn bằng file `config.yaml` mẫu ở thư mục gốc:

```bash
uv run python main.py --config config.yaml
```

File mẫu giải thích chi tiết mọi key và có cấu hình tham khảo cho cả ba workflow `dedup`, `remove-like` và `select`. Tên key dùng dạng `snake_case`, ví dụ `--batch-size` tương ứng `batch_size`. Các đường dẫn tương đối được tính từ thư mục hiện tại khi chạy lệnh.

Thứ tự ưu tiên là:

```text
mặc định chương trình < tùy chọn CLI < giá trị trong config.yaml
```

Vì vậy config luôn ghi đè CLI. Ví dụ sau vẫn dùng batch size `64` nếu file config có `batch_size: 64`:

```bash
uv run python main.py ./photos --batch-size 128 --config config.yaml
```

Có thể dùng file khác bằng `--config ./path/to/my-config.yaml`. Key không tồn tại, sai kiểu, sai lựa chọn hoặc thuộc workflow khác sẽ dừng với thông báo lỗi thay vì bị bỏ qua âm thầm.

## Cách dùng cơ bản

Quét một thư mục và tìm ảnh trùng:

```bash
uv run python main.py ./photos
```

Lệnh trên sẽ:

1. quét đệ quy các ảnh trong `./photos`;
2. tính SHA-256, pHash và embedding;
3. gom các ảnh trùng thành nhóm;
4. chọn ảnh có độ phân giải cao nhất để giữ lại;
5. ghi kết quả vào `./photos/dedup_report.json`.

Không có ảnh nào bị di chuyển hoặc xóa. Hãy mở `dedup_report.json` và kiểm tra hai phần:

- `groups`: các nhóm ảnh trùng; `keep` là ảnh được giữ, `duplicates` là các ảnh có thể loại bỏ;
- `review_only`: các cặp chỉ nên kiểm tra thủ công, **không bao giờ được tự động xóa**.

## Dọn trùng YOLO giữa train, val và test

Dùng `yolo-dedup` cho dataset có `data.yaml` và manifest `train.txt`, `test.txt`; manifest `val.txt` là tùy chọn. Lệnh chỉ xử lý các ảnh được khai báo trong manifest — không quét toàn bộ dataset root, nên thư mục như `_raw/` không bị đưa vào kết quả.

```bash
uv run python main.py yolo-dedup ./dataset \
  --output ./dataset-deduped
```

Lệnh luôn tạo dataset mới; **không có `--inplace`**. Mỗi ảnh được giữ hoặc loại cùng label YOLO tương ứng, rồi manifest và `data.yaml` được ghi lại trong output. Dùng `--copy-mode hardlink` nếu output cùng filesystem và muốn tiết kiệm dung lượng:

```bash
uv run python main.py yolo-dedup ./dataset \
  --output ./dataset-deduped \
  --copy-mode hardlink
```

YOLO dedup dùng cùng chính sách nhận diện của `dedup`: SHA-256, pHash và embedding với các ngưỡng `--cosine-*` và `--phash-*`. Mặc định ưu tiên `test > val > train`: trong mỗi nhóm trùng, tool chỉ chọn ảnh từ split ưu tiên cao nhất, rồi áp dụng `--keep-policy` nếu split đó có nhiều ảnh. Đổi thứ tự bằng `--split-priority`, ví dụ `--split-priority val,test,train`.

Mặc định cả các bản sao trong cùng một split cũng được xử lý; dùng `--cross-folder-only` để chỉ so sánh ảnh ở các thư mục cha khác nhau. Báo cáo `reports/yolo_dedup_report.json` ghi nhóm giữ/loại, các cặp cần review và `label_conflict` khi ảnh trong một nhóm có annotation khác nhau. Hãy kiểm tra các conflict này trước khi sử dụng dataset output.

## Các trường hợp sử dụng

<details>
<summary><strong>Quét nhiều thư mục</strong></summary>

```bash
uv run python main.py ./photos-2024 ./photos-2025
```

Ảnh ở tất cả thư mục được xét chung. Khi có nhiều thư mục:

- báo cáo mặc định là `./dedup_report.json` trong thư mục hiện tại;
- cache dùng chung nằm tại `./.imgdedup/`;
- nếu chuyển vào thùng rác, mỗi file được chuyển vào `.imgdedup/trash/` thuộc thư mục nguồn chứa nó.

</details>

<details>
<summary><strong>Chỉ so sánh ảnh giữa các thư mục</strong></summary>

Dùng khi các ảnh trong cùng một thư mục không được coi là trùng nhau:

```bash
uv run python main.py ./photos --cross-folder-only
```

Công cụ chỉ so sánh hai ảnh khi **thư mục cha trực tiếp** của chúng khác nhau.

</details>

<details>
<summary><strong>Đổi nơi lưu hoặc tắt báo cáo</strong></summary>

Chọn đường dẫn báo cáo:

```bash
uv run python main.py ./photos --report ./reports/dedup.json
```

Không ghi file JSON:

```bash
uv run python main.py ./photos --no-report
```

Không thể dùng đồng thời `--report` và `--no-report`.

</details>

<details>
<summary><strong>Chọn ảnh được giữ lại trong mỗi nhóm</strong></summary>

Mặc định công cụ giữ ảnh có độ phân giải cao nhất:

```bash
uv run python main.py ./photos --keep-policy highest-resolution
```

Các chính sách hỗ trợ:

| Giá trị | Ảnh được giữ |
| --- | --- |
| `highest-resolution` | Nhiều pixel nhất; nếu hòa thì giữ file lớn hơn |
| `best-quality` | Ưu tiên độ nét, độ sáng cân bằng, độ phân giải rồi kích thước file |
| `largest` | Kích thước file lớn nhất |
| `smallest` | Kích thước file nhỏ nhất |
| `newest` | Sửa đổi gần đây nhất |
| `oldest` | Sửa đổi lâu nhất |
| `lexi` | Đường dẫn đứng đầu theo thứ tự từ điển |

Chính sách này chỉ quyết định file nào nằm trong `keep`; nó không thay đổi cách nhận diện ảnh trùng.

</details>

<details>
<summary><strong>Siết chặt cách gom nhóm ảnh trùng</strong></summary>

Mặc định, công cụ dùng nhóm liên thông:

```text
A trùng B, B trùng C  →  A, B, C cùng một nhóm
```

Với bộ ảnh lớn hoặc có nhiều ảnh gần giống nhau, chuỗi liên kết này có thể tạo nhóm quá rộng. Dùng gom nhóm `agglomerative` để tách chặt hơn:

```bash
uv run python main.py ./photos \
  --grouping agglomerative \
  --agglomerative-linkage complete \
  --agglomerative-cosine-threshold 0.97
```

- `complete` (khuyến nghị): chặt hơn, mọi ảnh trong cụm phải duy trì ngưỡng đã chọn;
- `average`: ít chặt hơn, thường giữ được cụm lớn hơn.

Nếu không đặt `--agglomerative-cosine-threshold`, công cụ dùng giá trị của `--cosine-auto`.

</details>

<details>
<summary><strong>Chuyển ảnh trùng vào thùng rác</strong></summary>

Sau khi kiểm tra báo cáo, chạy lại với `--inplace`:

```bash
uv run python main.py ./photos --inplace
```

Ảnh trùng được chuyển tới:

```text
./photos/.imgdedup/trash/<run_id>/
```

Cấu trúc thư mục tương đối của ảnh được giữ nguyên. Manifest phục vụ khôi phục được ghi tại:

```text
./photos/.imgdedup/trash/<run_id>/restore_manifest.json
```

`--inplace` không xóa vĩnh viễn nếu không có `--hard-delete`.

</details>

<details>
<summary><strong>Xóa vĩnh viễn ảnh trùng</strong></summary>

> ⚠️ Chỉ thực hiện sau khi đã kiểm tra báo cáo. Chế độ này không có manifest khôi phục.

```bash
uv run python main.py ./photos --inplace --hard-delete --yes
```

Cần đủ cả ba cờ:

- `--inplace`: cho phép thay đổi file nguồn;
- `--hard-delete`: chọn xóa vĩnh viễn thay vì chuyển vào thùng rác;
- `--yes`: xác nhận thao tác phá hủy.

`--hard-delete` thiếu `--yes` sẽ bị từ chối. Nếu thiếu `--inplace`, lệnh vẫn là dry run và không xóa file.

</details>

<details>
<summary><strong>Tìm và loại ảnh giống một ảnh mẫu</strong></summary>

Dùng `remove-like` để so sánh từng ảnh trong thư mục với một ảnh tham chiếu:

```bash
uv run python main.py remove-like ./photos ./reference.jpg
```

Ảnh tham chiếu luôn được giữ. Báo cáo mặc định nằm tại:

```text
./photos/remove_like_report.json
```

Chuyển các ảnh khớp vào thùng rác:

```bash
uv run python main.py remove-like ./photos ./reference.jpg --inplace
```

Xóa vĩnh viễn các ảnh khớp:

```bash
uv run python main.py remove-like ./photos ./reference.jpg \
  --inplace --hard-delete --yes
```

`remove-like` dùng cùng ngưỡng nhận diện với lệnh quét thông thường, nhưng không dùng gom nhóm, `--keep-policy`, `--k`, `--cross-folder-only` hoặc FAISS index đã lưu.

</details>

<details>
<summary><strong>Chọn chính xác M ảnh đại diện để tạo dataset</strong></summary>

Lệnh `select` loại ảnh trùng, chọn đúng số ảnh yêu cầu rồi xuất sang thư mục mới mà không sửa dữ liệu nguồn:

```bash
uv run python main.py select ./photos \
  --output ./selected-dataset \
  --num 200 \
  --selection-method hybrid \
  --make-preview
```

Các phương pháp chọn:

| Phương pháp | Phù hợp khi |
| --- | --- |
| `hybrid` | Cần cân bằng giữa ảnh điển hình và độ đa dạng; đây là mặc định |
| `kmeans` | Muốn các ví dụ điển hình gần tâm cụm |
| `farthest` | Muốn độ phủ và sự khác biệt lớn nhất |

Kết quả có cấu trúc:

```text
selected-dataset/
├── images/       # ảnh đã chọn, giữ cấu trúc đường dẫn tương đối
├── reports/      # JSON, CSV, danh sách đường dẫn và thống kê
└── previews/     # contact sheet nếu dùng --make-preview
```

Mặc định ảnh được copy. Có thể dùng `--copy-mode hardlink` hoặc `--copy-mode symlink`. Thư mục output phải nằm ngoài cây thư mục input và không được tồn tại; dùng `--force` để thay thế an toàn một output directory đã có.

Kết quả có thể tái lập với cùng dữ liệu và `--seed` (mặc định `42`). Nếu sau khi lọc không còn đủ `M` ảnh hợp lệ, lệnh dừng với lỗi thay vì xuất ít ảnh hơn.

### Lọc ảnh chất lượng thấp trước khi chọn

Việc đo chất lượng luôn được dùng bởi `--keep-policy best-quality`, nhưng **loại bỏ** ảnh chất lượng thấp chỉ được bật khi có `--reject-low-quality`:

```bash
uv run python main.py select ./photos \
  --output ./selected-dataset \
  --num 200 \
  --reject-low-quality \
  --min-width 640 \
  --min-height 480 \
  --min-blur-score 20 \
  --min-brightness 15 \
  --max-brightness 240
```

Blur score là phương sai cạnh tính bằng Pillow, vì vậy không nên dùng trực tiếp ngưỡng lấy từ công cụ dựa trên OpenCV.

</details>

<details>
<summary><strong>Tinh chỉnh tốc độ và tài nguyên</strong></summary>

Ví dụ:

```bash
uv run python main.py ./photos \
  --batch-size 128 \
  --metadata-workers 8 \
  --loader-workers 0 \
  --gpus 1 \
  --gpu-memory-fraction 0.9 \
  --k 50
```

- Giảm `--batch-size` nếu GPU hết bộ nhớ.
- `--metadata-workers` điều khiển số worker tính SHA-256 và pHash.
- Giữ `--loader-workers 0` nếu chưa cần tối ưu; tăng thận trọng khi GPU phải chờ giải mã ảnh.
- Nếu không đặt `--gpus`, công cụ dùng tất cả GPU khả dụng và tự fallback về một GPU hoặc CPU.
- Tăng `--k` nếu mỗi ảnh có thể có rất nhiều bản gần trùng; giá trị lớn hơn làm tăng thời gian matching và kích thước báo cáo.
- `--save-faiss-index` lưu `.imgdedup/faiss.index`; mặc định tắt để tránh chi phí ghi file.

</details>

## Công cụ quyết định ảnh trùng như thế nào?

<details>
<summary><strong>Xem các quy tắc và ngưỡng mặc định</strong></summary>

Một cặp ảnh được tự động coi là trùng nếu thỏa **ít nhất một** điều kiện:

| Quy tắc | Điều kiện mặc định |
| --- | --- |
| Trùng tuyệt đối | SHA-256 giống nhau |
| pHash gần nhau | Khoảng cách Hamming ≤ `4` |
| Embedding rất giống | Cosine similarity ≥ `0.97` |
| Embedding + pHash xác minh | Cosine ≥ `0.90` **và** khoảng cách pHash ≤ `8` |

Cặp có cosine trong khoảng `0.85 ≤ cosine < 0.90` chỉ được đưa vào `review_only`, không được chuyển hoặc xóa tự động.

Có thể thay đổi ngưỡng:

```bash
uv run python main.py ./photos \
  --cosine-auto 0.97 \
  --cosine-verify 0.90 \
  --cosine-review 0.85 \
  --phash-auto-distance 4 \
  --phash-verify-distance 8
```

Ba ngưỡng cosine phải thỏa:

```text
cosine-review ≤ cosine-verify ≤ cosine-auto
```

Ngưỡng chặt hơn thường giảm false positive nhưng có thể bỏ sót ảnh trùng. Nên kiểm tra báo cáo trên dữ liệu thực tế trước khi dùng `--inplace`.

</details>

## Cache, định dạng ảnh và thư mục bị bỏ qua

<details>
<summary><strong>Xem chi tiết</strong></summary>

Các định dạng được quét:

```text
.jpg  .jpeg  .png  .bmp  .webp  .tif  .tiff
```

Các thư mục bị bỏ qua:

```text
.git  .imgdedup  __pycache__  node_modules
```

Cache nằm trong `.imgdedup/` cạnh thư mục được quét (hoặc trong thư mục hiện tại khi quét nhiều nguồn):

| File/thư mục | Nội dung |
| --- | --- |
| `db.sqlite` | Đường dẫn, hash, kích thước ảnh và vị trí embedding |
| `embeddings.npy` | Mảng embedding dùng lại giữa các lần chạy |
| `faiss.index` | Chỉ được tạo khi dùng `--save-faiss-index` |
| `trash/<run_id>/` | Ảnh đã chuyển và manifest khôi phục |

Chỉ file mới hoặc đã thay đổi cần xử lý lại.

</details>

## Tham chiếu nhanh các tùy chọn

<details>
<summary><strong>Lệnh quét ảnh trùng</strong></summary>

```text
uv run python main.py <folders...> [options]
```

| Tùy chọn | Mặc định | Ý nghĩa |
| --- | --- | --- |
| `--keep-policy` | `highest-resolution` | Chọn file giữ lại trong mỗi nhóm |
| `--grouping` | `connected` | `connected` hoặc `agglomerative` |
| `--cross-folder-only` | tắt | Chỉ so sánh khác thư mục cha trực tiếp |
| `--k` | `50` | Số láng giềng gần nhất FAISS |
| `--batch-size` | `128` | Batch inference embedding |
| `--metadata-workers` | `min(32, CPU)` | Worker tính metadata |
| `--loader-workers` | `0` | Worker nạp ảnh của PyTorch |
| `--model` | `facebook/dinov3-vitb16-pretrain-lvd1689m` | Model embedding Hugging Face |
| `--gpus` | tất cả | Số GPU sử dụng |
| `--gpu-memory-fraction` | `0.9` | Tỉ lệ bộ nhớ mỗi GPU, từ `0.1` đến `1.0` |
| `--inplace` | tắt | Áp dụng báo cáo lên file nguồn |
| `--hard-delete` | tắt | Xóa vĩnh viễn; cần `--yes` |
| `--report` | tự động | Đường dẫn báo cáo JSON |
| `--no-report` | tắt | Không ghi báo cáo JSON |

Chạy `uv run python main.py --help` để xem danh sách đầy đủ và giá trị hiện tại.

</details>

<details>
<summary><strong>Lệnh remove-like</strong></summary>

```text
uv run python main.py remove-like <folder> <image> [options]
```

Hỗ trợ các tùy chọn ngưỡng, model, GPU, batch/worker, report và chế độ xóa. Không hỗ trợ các tùy chọn chỉ dành cho gom nhóm ảnh trùng.

```bash
uv run python main.py remove-like --help
```

</details>

<details>
<summary><strong>Lệnh yolo-dedup</strong></summary>

```text
uv run python main.py yolo-dedup <dataset-root> --output <directory> [options]
```

| Tùy chọn | Mặc định | Ý nghĩa |
| --- | --- | --- |
| `--copy-mode` | `copy` | `copy`, `hardlink` hoặc `symlink` cho ảnh và label output |
| `--split-priority` | `test,val,train` | Split ưu tiên cao nhất đến thấp nhất khi chọn ảnh giữ lại |
| `--cosine-*`, `--phash-*`, `--k`, `--model`, `--batch-size`, `--gpus` | như `dedup` | Cùng cơ chế nhận diện trùng như workflow `dedup` |
| `--keep-policy`, `--cross-folder-only`, `--grouping` | như `dedup` | Chọn ảnh giữ lại và cách gom nhóm |
| `--force` | tắt | Thay output directory hiện có sau khi export thành công |

</details>

<details>
<summary><strong>Lệnh select</strong></summary>

```text
uv run python main.py select <folder> --output <directory> --num <M> [options]
```

Các tùy chọn riêng quan trọng gồm `--selection-method`, `--copy-mode`, `--seed`, `--force`, `--make-preview` và nhóm tùy chọn lọc chất lượng.

```bash
uv run python main.py select --help
```

</details>

## Ghi chú kỹ thuật

<details>
<summary><strong>Kiến trúc xử lý</strong></summary>

- SHA-256 phát hiện file giống hệt nhau.
- pHash và BK-tree phát hiện ảnh gần giống theo khoảng cách Hamming.
- Model mặc định `facebook/dinov3-vitb16-pretrain-lvd1689m` tạo embedding hình ảnh.
- Embedding được chuẩn hóa L2; inner product trong FAISS tương đương cosine similarity.
- FAISS tìm các ứng viên gần nhất trên GPU khi khả dụng, nếu không sẽ dùng CPU.
- Trích xuất embedding hỗ trợ nhiều GPU; Flash Attention 2 được dùng khi có và fallback về SDPA/eager attention.

</details>

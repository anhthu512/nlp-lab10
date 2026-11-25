# LAB 10 - Extractive Question Answering

Cài đặt và thực nghiệm Extractive Question Answering trên SQuAD 1.1 bằng PyTorch và Hugging Face Transformers. Mã nguồn gồm baseline keyword overlap, mô hình DistilBERT với linear head dự đoán vị trí bắt đầu/kết thúc, các thí nghiệm đánh giá và ứng dụng Document QA nhận trực tiếp câu hỏi cùng context.

Hai notebook đã được chạy và lưu output. Các bảng, biểu đồ, CSV, log dự đoán và checkpoint trong thư mục là kết quả của lượt chạy ngày **23/11/2026**.

## 1. Cấu trúc thư mục

```text
lab10/
├── README.md
├── tokenizer_alignment.ipynb
├── qa_model.py
├── qa_finetuning.ipynb
├── document_qa.py
├── results.csv
├── learning_curves.png
├── em_f1_analysis.png
├── answer_length_analysis.png
├── checkpoint/
│   ├── backbone/
│   ├── tokenizer/
│   ├── qa_head.pt
│   └── metadata.json
└── outputs/
    ├── reproducibility.json
    ├── split_ids.json
    ├── training_history.csv
    ├── training_steps.csv
    ├── context_length_results.csv
    ├── context_length_history.csv
    ├── em_f1_comparison.csv
    ├── matching_analysis.csv
    ├── answer_length_metrics.csv
    ├── error_counts.csv
    ├── boundary_totals.csv
    ├── partial_match_examples.csv
    ├── wrong_start_examples.csv
    ├── wrong_end_examples.csv
    ├── wrong_span_examples.csv
    └── ...
```

| Thành phần | Chức năng |
| --- | --- |
| [tokenizer_alignment.ipynb](tokenizer_alignment.ipynb) | Kiểm tra cấu trúc SQuAD, tokenizer, offset mapping và bốn trường hợp biên của alignment; gồm 9 cell code. |
| [qa_model.py](qa_model.py) | Baseline, mô hình QA, alignment, normalization, EM/F1, constrained decoding, huấn luyện, đánh giá và lưu/nạp checkpoint. |
| [qa_finetuning.ipynb](qa_finetuning.ipynb) | Chuẩn bị dữ liệu, chạy năm thí nghiệm, xuất bảng/biểu đồ, thống kê lỗi và đánh giá test; gồm 12 cell code. |
| [document_qa.py](document_qa.py) | Nạp checkpoint đã fine-tune, chạy demo và nhận câu hỏi từ terminal. |
| [results.csv](results.csv) | Tổng hợp 19 dòng kết quả của lượt chạy đã lưu. |
| `checkpoint/` | Backbone, tokenizer, QA head và metadata của checkpoint được chọn. |
| `outputs/` | CSV chi tiết, log JSONL, cấu hình và ID của các tập dữ liệu. |

## 2. Cài đặt môi trường

### 2.1. Windows và Jupyter

Môi trường đã kiểm chứng sử dụng **Python 3.10.11**. Cần kết nối Internet để tải thư viện, tokenizer, trọng số pretrained và SQuAD trong lần thiết lập đầu tiên.

Mở PowerShell tại thư mục `lab10`, tạo môi trường riêng:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
```

Chọn **một** cách cài PyTorch phù hợp với thiết bị.

Với GPU NVIDIA, lượt chạy đã lưu dùng bản CUDA 12.4:

```powershell
.\.venv\Scripts\python.exe -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```

Với CPU:

```powershell
.\.venv\Scripts\python.exe -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
```

Cài các thư viện còn lại và đăng ký kernel:

```powershell
.\.venv\Scripts\python.exe -m pip install transformers==4.57.1 datasets==4.4.1 numpy==1.26.4 pandas==2.2.3 matplotlib==3.9.4 jupyterlab ipykernel nbformat nbclient
.\.venv\Scripts\python.exe -m ipykernel install --user --name lab10 --display-name "Python (LAB 10)"
.\.venv\Scripts\python.exe -m jupyterlab
```

Trong Jupyter, chọn kernel **Python (LAB 10)**. Kiểm tra GPU bằng một cell:

```python
import torch

print(torch.__version__)
print(torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
```

### 2.2. Google Colab

1. Chọn runtime có GPU nếu muốn huấn luyện bằng CUDA.
2. Đặt các notebook và `qa_model.py` trong `/content/lab10/`, hoặc đặt trực tiếp các file trong `/content/`.
3. Chạy các cell theo thứ tự từ trên xuống. Cell đầu tự kiểm tra phiên bản và cài các dependency cần thiết.
4. Sau khi chạy, tải về notebook đã lưu output, các file PNG/CSV, thư mục `outputs/` và `checkpoint/` để giữ kết quả khi phiên Colab kết thúc.

Nếu chỉ dùng ứng dụng với checkpoint có sẵn, cần thêm `document_qa.py` và toàn bộ thư mục `checkpoint/`.

## 3. Thứ tự chạy

### Bước 1 - Kiểm tra tokenizer và alignment

Mở [tokenizer_alignment.ipynb](tokenizer_alignment.ipynb), chọn **Restart Kernel and Run All Cells**. Notebook in đầy đủ tensor đầu vào, offsets và kết quả kiểm thử:

- Answer bắt đầu giữa token.
- Answer kết thúc giữa token.
- Answer bị truncation hoàn toàn hoặc một phần.
- Answer nằm ngoài window đầu và được tìm thấy trong window sau.

`token_type_ids` chỉ được kiểm tra và sử dụng khi tokenizer/model hỗ trợ trường này.

### Bước 2 - Huấn luyện và thực nghiệm

Mở [qa_finetuning.ipynb](qa_finetuning.ipynb), chọn **Restart Kernel and Run All Cells**. Các cell dùng chung dữ liệu và biến được tạo ở những cell trước.

| Cell code | Nội dung và đầu ra chính |
| --- | --- |
| 1 | Import, seed, cấu hình và bảng reproducibility. |
| 2 | Chia dữ liệu, tạo sliding windows, alignment và lưu ID của từng split. |
| 3 | Đánh giá baseline trên validation. |
| 4 | Experiment 1: huấn luyện độc lập với `max_length` bằng 128, 256, 384. |
| 5 | Experiment 2: huấn luyện mô hình chính; lưu loss theo step/epoch, EM/F1 và checkpoint. |
| 6 | Vẽ và lưu `learning_curves.png`. |
| 7 | Experiment 3: so sánh EM/F1 của baseline và Transformer; lưu `em_f1_analysis.png`. |
| 8 | In và xuất các ví dụ có `EM = 0`, `F1 > 0`. |
| 9 | Experiment 4: đánh giá theo nhóm lexical overlap cao/thấp. |
| 10 | Experiment 5: đánh giá theo độ dài đáp án; lưu `answer_length_analysis.png`. |
| 11 | Thống kê và xuất các ví dụ wrong start, wrong end, wrong span. |
| 12 | Đánh giá test bằng checkpoint được chọn và tổng hợp `results.csv`. |

Thí nghiệm context length giữ `stride=32` cho cả ba cấu hình. Mô hình chính dùng `max_length=384, stride=128`. Giá trị stride phải nhỏ hơn số token context còn lại sau khi dành chỗ cho question và special tokens.

Notebook huấn luyện thực hiện **bốn lượt fine-tune độc lập**: ba lượt cho context length và một lượt cho mô hình chính. Thời gian huấn luyện được đo riêng cho mỗi lượt; thời gian tạo dữ liệu, đánh giá và lưu checkpoint không nằm trong `training_seconds`.

`wall_seconds` được đo từ trước epoch đầu đến hết epoch cuối, gồm huấn luyện, đánh giá validation và ghi checkpoint trong lượt đó. Giá trị này không gồm tải/khởi tạo model, tokenization hoặc thời gian chạy các lượt thí nghiệm khác.

### Bước 3 - Lưu notebook và dùng kết quả

Lưu notebook sau khi chạy để giữ bảng, log và biểu đồ trong các cell. Chạy lại notebook sẽ ghi đè các file output cùng tên; nên sao lưu kết quả nếu muốn giữ nhiều cấu hình thí nghiệm.

Các phần tính toán tay, prediction trước thí nghiệm, nhận xét kết quả, Individual Learning Check và reflection do người học tự hoàn thiện trong bài nộp.

## 4. Cấu hình và tái lập

Các giá trị sau được đối chiếu với [outputs/reproducibility.json](outputs/reproducibility.json):

| Tham số | Giá trị đã chạy |
| --- | --- |
| Model và tokenizer | `distilbert/distilbert-base-uncased` |
| Model revision đã ghi nhận | `12040accade4e8a0f71eabdb258fecc2e7e948be` |
| Dataset | `rajpurkar/squad`, SQuAD 1.1 |
| Train / validation / test | 2.000 / 1.000 / 500 examples |
| Max sequence length của mô hình chính | 384 |
| Stride của mô hình chính | 128 |
| Stride của Experiment 1 | 32 |
| Giới hạn question | 48 subword tokens |
| Giới hạn answer khi decode | 64 subword tokens |
| Batch size | 8 |
| Learning rate | `3e-5` |
| Epochs cho mỗi lượt fine-tune | 2 |
| Optimizer | AdamW |
| Weight decay | 0,01 |
| Gradient clipping | 1,0 |
| Random seed | 42 |
| Hardware | NVIDIA GeForce RTX 3050 Laptop GPU, 4 GB |
| Python | 3.10.11 |
| PyTorch | `2.6.0+cu124` |
| Transformers / Datasets | `4.57.1` / `4.4.1` |
| NumPy / pandas / Matplotlib | `1.26.4` / `2.2.3` / `3.9.4` |
| Chọn checkpoint | Validation F1 cao nhất; giữ epoch sớm hơn nếu bằng nhau |
| Best epoch của mô hình chính | 2 |
| Thời gian huấn luyện mô hình chính | 173,66 giây |
| Wall time của lượt huấn luyện chính | 206,26 giây |

### Cách chia dữ liệu

- Train lấy từ official training split sau khi shuffle với seed cố định.
- Các context duy nhất của official validation được sắp xếp, shuffle bằng seed 42 rồi chia theo nhóm context: khoảng 80% cho validation, 20% cho test. Tỷ lệ này áp dụng cho **số context**, không bắt buộc bằng tỷ lệ số câu hỏi.
- Lấy lần lượt `VAL_SIZE` và `TEST_SIZE` examples từ hai pool đã shuffle. Validation và test không trùng context.
- ID thực tế của các mẫu nằm trong [outputs/split_ids.json](outputs/split_ids.json). Số lượng examples có thể khác số lượng windows sau tokenization.
- Test được đánh giá ở cell cuối sau khi đã chọn checkpoint bằng validation.

Để thay đổi quy mô, sửa các biến ở cell code đầu của `qa_finetuning.ipynb`:

```python
TRAIN_SIZE = 2000
VAL_SIZE = 1000
TEST_SIZE = 500
BATCH_SIZE = 8
EPOCHS = 2
SEED = 42
```

Đặt một biến kích thước bằng `None` để dùng toàn bộ pool tương ứng. Kích thước số nguyên phải nằm trong phạm vi pool. Mã cố định seed cho Python, NumPy và PyTorch, bật deterministic algorithms và tắt cuDNN benchmark. Phiên bản thư viện, phần cứng và revision đã ghi nhận cần được giữ khi đối chiếu các lượt chạy; mã tải pretrained theo tên model và không khóa cứng revision trong lời gọi tải.

## 5. Số liệu đã xuất

Các bảng dưới đây chép số liệu từ CSV, làm tròn đến hai chữ số thập phân. EM và F1 dùng thang **0-100**; giá trị đầy đủ nằm trong file gốc.

### Context length - validation

Nguồn: [outputs/context_length_results.csv](outputs/context_length_results.csv). Mỗi cấu hình dùng stride 32 và được đánh giá trên 1.000 examples.

| Max length | EM | F1 | Training time (giây) |
| --- | ---: | ---: | ---: |
| 128 | 46,70 | 57,17 | 128,88 |
| 256 | 49,20 | 60,62 | 121,23 |
| 384 | 45,50 | 58,52 | 169,70 |

### Baseline và mô hình chính

Nguồn: [results.csv](results.csv). Mô hình chính dùng max length 384, stride 128 và checkpoint epoch 2.

| Split | Model | Số examples | EM | F1 |
| --- | --- | ---: | ---: | ---: |
| Validation | Baseline | 1.000 | 0,10 | 19,26 |
| Validation | Fine-tuned Transformer | 1.000 | 47,60 | 59,22 |
| Test | Baseline | 500 | 0,40 | 19,74 |
| Test | Fine-tuned Transformer | 500 | 51,40 | 62,66 |

## 6. File kết quả và quy ước đọc

| File hoặc nhóm file | Nội dung |
| --- | --- |
| [learning_curves.png](learning_curves.png) | Training loss, validation loss và validation EM/F1 theo epoch của mô hình chính. |
| [em_f1_analysis.png](em_f1_analysis.png) | Biểu đồ EM/F1 của baseline và Transformer trên validation. |
| [answer_length_analysis.png](answer_length_analysis.png) | Biểu đồ EM/F1 theo bốn nhóm độ dài đáp án. |
| [results.csv](results.csv) | Kết quả tổng hợp; cột `experiment`, `split`, `model` xác định từng loại dòng. Cột không áp dụng cho một loại dòng được để trống. |
| `outputs/training_history.csv`, `training_steps.csv` | Loss/metric theo epoch và training loss theo từng step của mô hình chính. |
| `outputs/context_length_results.csv`, `context_length_history.csv` | Kết quả và lịch sử theo epoch của ba cấu hình context length. |
| `outputs/matching_analysis.csv`, `matching_examples.csv` | Metric và từng example theo lexical overlap. |
| `outputs/answer_length_metrics.csv` | Metric và số examples của từng nhóm độ dài đáp án. |
| `outputs/baseline_validation.csv`, `transformer_validation.csv` | Kết quả theo example trên validation. |
| `outputs/baseline_test.csv`, `transformer_test.csv` | Kết quả theo example trên test. |
| `outputs/partial_match_examples.csv` | Các example có `EM = 0`, `F1 > 0`. |
| `outputs/wrong_start_examples.csv`, `wrong_end_examples.csv`, `wrong_span_examples.csv` | Các example được phân loại tự động theo vị trí biên. |
| `outputs/error_counts.csv`, `boundary_totals.csv` | Số lỗi theo nhóm và tổng số biên sai trên validation gốc. |
| `outputs/diagnostic_examples.csv` | Các mẫu dùng để lọc ví dụ; trường `source` ghi nguồn mẫu. |
| `outputs/*_predictions.jsonl` | Log từng example: câu hỏi, gold, answer, offsets, score và confidence. |
| [outputs/document_qa_demo.log](outputs/document_qa_demo.log) | Output thực thi ba câu hỏi demo bằng checkpoint đã fine-tune. |

### Quy ước metric và nhóm đánh giá

- Chuẩn hóa chuỗi theo SQuAD tiếng Anh: chuyển chữ thường, bỏ dấu câu ASCII, bỏ `a/an/the` và gộp khoảng trắng. Với nhiều gold answers, EM và F1 lấy giá trị cao nhất trên các đáp án tham chiếu cho từng metric.
- Lexical overlap là tỷ lệ từ nội dung duy nhất của question xuất hiện trong toàn bộ context sau chuẩn hóa và bỏ stop words. `high_overlap` có giá trị từ 0,75 trở lên; `low_overlap` nhỏ hơn 0,75. Nhóm thấp là proxy tự động, chưa được gán nhãn paraphrase thủ công.
- Độ dài đáp án dùng số subword tokens của **gold answer đầu tiên**, không gồm special tokens. Các nhóm là `1`, `2-3`, `4-6`, `7+`.
- `wrong_start`: start sai, end đúng; `wrong_end`: start đúng, end sai; `wrong_span`: cả hai biên sai. Biên được đối chiếu với các span đã gán nhãn, trong khi EM dùng chuỗi sau chuẩn hóa.
- Các cột `#wrong_start_including_wrong_span` và `#wrong_end_including_wrong_span` bao gồm lỗi của nhóm `wrong_span`.

Cell lọc ví dụ có thể đánh giá thêm các mẫu validation dự phòng nếu chưa đủ năm mẫu cho mỗi loại. Các mẫu bổ sung có `source=additional_validation_diagnostics`, không được cộng vào metric của thí nghiệm hoặc tập test. `DIAGNOSTIC_LIMIT=None` cho phép tìm trong toàn bộ validation pool còn lại. Notebook báo số mẫu tìm được nếu pool không có đủ ví dụ thật.

## 7. Sử dụng Document QA

Ứng dụng nhận trực tiếp `(question, context)` và nạp checkpoint trong `checkpoint/`. Chạy các lệnh dưới đây tại thư mục `lab10`, bằng môi trường đã cài dependency.

### Chạy demo

```powershell
.\.venv\Scripts\python.exe document_qa.py --no-interactive
```

### Demo rồi nhập câu hỏi từ terminal

```powershell
.\.venv\Scripts\python.exe document_qa.py
```

Các câu hỏi nhập thêm sử dụng cùng context của phiên demo. Nhập dòng trống để thoát.

### Dùng tài liệu riêng

Lưu context thành file văn bản UTF-8, ví dụ `context.txt`:

```powershell
.\.venv\Scripts\python.exe document_qa.py --context-file context.txt --question "What is the main topic?" --no-interactive
```

Có thể thêm `--device cpu`, `--device cuda` hoặc `--checkpoint <thu_muc_checkpoint>`. Khi không chỉ định thiết bị, ứng dụng tự chọn CUDA nếu khả dụng.

### Gọi từ Python

```python
from document_qa import DocumentQA

system = DocumentQA(checkpoint_dir="checkpoint", device="cpu")
context = "Paris is the capital of France."
result = system.answer("What is the capital of France?", context)
print(result)
assert context[result["start"]:result["end"]] == result["answer"]
```

Đầu ra gồm `answer`, `start`, `end`, `score`, `confidence`, `start_token`, `end_token`, `window_index` và `confidence_kind`. Character offsets dùng khoảng **`[start, end)`**; token end là chỉ số bao gồm token cuối trong window được chọn.

`score` là tổng start/end logits của span. `confidence` là tích xác suất biên trong window, chưa được hiệu chỉnh thành xác suất câu trả lời đúng. Decoder chỉ xét token context, ràng buộc start không vượt end, giới hạn độ dài đáp án và chọn span có score cao nhất qua các windows.

Checkpoint có định dạng riêng của `ExtractiveQAModel`: backbone, tokenizer và QA head được lưu riêng. Dùng `DocumentQA` hoặc `ExtractiveQAModel.load_checkpoint()` để nạp.

## 8. Giới hạn và xử lý lỗi thường gặp

| Tình huống | Cách xử lý |
| --- | --- |
| Không tìm thấy `qa_model.py` | Mở notebook trong `lab10`; trên Colab đặt module trong `/content/` hoặc `/content/lab10/`. |
| Không tìm thấy checkpoint | Chạy cell Experiment 2 trước, hoặc truyền đúng `--checkpoint`; giữ đầy đủ `backbone/`, `tokenizer/`, `qa_head.pt`, `metadata.json`. |
| `torch.cuda.is_available()` trả về `False` | Kiểm tra kernel, driver và bản PyTorch đã cài; có thể chạy bằng CPU. |
| CUDA hết bộ nhớ khi huấn luyện | Giảm `BATCH_SIZE` trong cell cấu hình rồi chạy lại notebook từ đầu. |
| Stride không hợp lệ | Giảm stride để nhỏ hơn phần context còn lại trong sequence; giữ cùng stride giữa ba cấu hình của Experiment 1. |
| Kích thước split vượt pool | Giảm `TRAIN_SIZE`, `VAL_SIZE`, `TEST_SIZE` hoặc đặt `None` để dùng toàn bộ pool tương ứng. |

Mô hình được fine-tune trên SQuAD 1.1 tiếng Anh và luôn cố chọn một span trong context. Ứng dụng chưa có cơ chế xác định no-answer đã được hiệu chỉnh. Context được cung cấp trực tiếp; ứng dụng không có bước truy xuất tài liệu. Question dài được giới hạn ở 48 subword tokens, còn span decode tối đa 64 subword tokens.

## 9. Kiểm chứng đã thực hiện

- Chạy 9/9 cell code của notebook alignment và 12/12 cell code của notebook fine-tuning; lưu output, không có cell báo lỗi thực thi.
- Kiểm tra normalization, metric, các trường hợp alignment, decoder và phân loại biên bằng các trường hợp kiểm thử.
- Kiểm tra forward, loss, gradient và lưu/nạp checkpoint với backbone BERT/DistilBERT trong lượt kiểm thử phần mềm bằng mô hình nhỏ.
- Chạy các thí nghiệm chính với trọng số pretrained DistilBERT thật trên GPU và cấu hình trong mục 4.
- Kiểm tra ba PNG đọc được và có ba biểu đồ nhúng trong notebook; CSV tổng hợp có 19 dòng.
- Kiểm tra dữ liệu xuất có ít nhất năm ví dụ partial match và ít nhất năm ví dụ cho mỗi nhóm wrong start, wrong end, wrong span.
- Chạy demo Document QA bằng checkpoint thật; kiểm tra định dạng mã nguồn bằng Ruff.

## 10. AI Usage

### What was generated

AI hỗ trợ tạo mã nguồn cho notebook và các module: baseline, character/token alignment, mô hình QA, loss, normalization, EM/F1, decoding, vòng lặp huấn luyện, các thí nghiệm, biểu đồ, xuất log/CSV và ứng dụng Document QA. README này cũng được biên soạn với hỗ trợ AI dựa trên mã nguồn và các file kết quả đã lưu.

### What was modified

Cấu hình thí nghiệm context length được đặt chung stride 32 để chạy được với cả max length 128, 256 và 384. Mô hình chính giữ stride 128. Mã được bổ sung log tiến độ, lưu output notebook, tách validation/test theo context và cơ chế tìm thêm ví dụ lỗi trong validation dự phòng.

### What was verified

Các bước kiểm chứng mã nguồn, thực thi notebook, checkpoint và file đầu ra được liệt kê tại mục 9. Bảng số liệu trong README được đối chiếu với CSV và metadata của lượt chạy đã lưu. Nội dung tính toán tay, dự đoán trước thực nghiệm, giải thích kết quả và reflection do người học tự viết.

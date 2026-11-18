# Báo cáo Day 09 — Trần Trọng Thái, 2A202604065

## 1. Kết luận trước

Pipeline kết luận **Release** với cấu hình đã calibration. Từ 224 ảnh raw, bước nhận dữ liệu giữ lại 193 ảnh và loại 31 ảnh hỏng/nhỏ/trùng/rò rỉ test. Sau QC và sửa tự động, còn 13/193 ảnh có cảnh báo chưa duyệt (6,7%), thấp hơn ngưỡng Hold 10%. Không còn lỗi bắt buộc trong vòng kiểm cuối, nên bản dataset được xuất tại `submission/export/clean_v1/` là bản phát hành.

## 2. Lỗi và phân việc

Trước khi sửa, lô raw có 36 lỗi bắt buộc thuộc nhóm nhãn và 108 cảnh báo sau khi áp dụng ngưỡng đã calibration. Trong báo cáo v1, cửa vào loại thêm 31 ảnh: 2 ảnh hỏng, 3 ảnh nhỏ, 8 ảnh trùng từng byte, 7 ảnh trùng gần đúng và 11 ảnh rò rỉ từ kho test/val. Sau sửa tự động còn 6 lỗi bắt buộc, đều là `MISSING_LABEL_FILE`; sau vòng người duyệt, danh sách v3 không còn lỗi bắt buộc.

| Mã lỗi | Số ảnh/dòng | Ai sửa (máy/người) | Vì sao |
|---|---:|---|---|
| `MISSING_LABEL_FILE` | 6 ảnh | Người | Máy không thể bịa nhãn; ảnh được chuyển vào hàng đợi với ưu tiên cao. |
| `BAD_FORMAT` | 3 dòng | Máy | Dòng không parse được nên xóa an toàn. |
| `BAD_CLASS_ID` | 6 dòng | Máy | ID ngoài danh sách lớp không thể dùng để huấn luyện. |
| `BOX_ZERO` | 4 dòng | Máy | Box không có diện tích nên phải xóa. |
| `DUP_BOX` | 7 dòng | Máy | Box lặp cùng lớp gây thiên lệch dữ liệu. |
| `BOX_OUT_OF_IMAGE` | 8 dòng | Máy | Cắt box về trong ảnh; một box nằm hẳn ngoài ảnh sẽ bị xóa. |
| `BAD_CLASS_NAME` | 2 tên trong `data.yaml` | Máy | Đồng bộ tên lớp với cấu hình chuẩn. |
| `EMPTY_LABEL` | 11 ảnh | Người | Không thể biết file rỗng là ảnh nền hay thiếu gán. |
| `BOX_TINY`, `BOX_HUGE` | 9 dòng | Người | Chỉ là cảnh báo, không sửa tự động để tránh mất đối tượng hợp lệ. |
| `POSSIBLE_*` từ checker | Nhiều cảnh báo | Người | Model lớn cũng có thể sai; chỉ dùng để xếp hàng đợi. |

`05_review_queue.csv` dùng trọng số theo mức nghiêm trọng trong `configs/pipeline.yaml`: `NEEDS_LABEL` và `POSSIBLE_WRONG_CLASS` được ưu tiên cao, sau đó đến box sai/đã sửa và cảnh báo box. Với ngân sách 80 ảnh, pipeline đã duyệt 80 ảnh đầu hàng đợi. Sau đó còn 13/193 ảnh có cảnh báo: 5 `EMPTY_LABEL`, 6 `BOX_TINY`, 3 `BOX_HUGE`, 11 `POSSIBLE_FALSE_POS`, 1 `POSSIBLE_MISSING` và 1 `POSSIBLE_WRONG_CLASS` (một ảnh có thể có nhiều mã).

## 3. Calibration (2 ngưỡng)

| Tham số | Khởi đầu | Đã chọn | Precision / Recall (trước → sau) | Lý do (đánh đổi) |
|---|---|---|---|---|
| `duplicates.phash_dist` | 2 | 16 | `DUP_IMAGE_EXACT`: 1,00/1,00 → 1,00/1,00; `DUP_IMAGE_NEAR`: 1,00/0,50 → 1,00/0,83 | Ưu tiên precision cao vì cờ sai sẽ xóa nhầm ảnh tốt. Ở 16, precision vẫn 1,00 và recall tăng từ 0,50 lên 0,83; ở 20, recall giảm. |
| `second_opinion.conf` | 0,25 | 0,80 | `WRONG_CLASS`: 0,56/0,83 → 1,00/0,83; `MISSING`: 0,50/1,00 → 0,67/0,75; `LOOSE_BOX`: 0,29/0,40 → 0,50/0,20; `FALSE_POS`: 0,31/1,00 → 0,31/1,00 | Ưu tiên giảm cảnh báo động giả và giữ hàng đợi dưới ngân sách 80. Đổi lại recall của `LOOSE_BOX` và `MISSING` giảm, nên đây là đánh đổi có ý thức. |

Bảng sweep đầy đủ nằm trong `submission/calibration/calibration_report.md`, dữ liệu máy đọc được nằm trong `submission/calibration/calibration.json`.

**Calibration có thể đánh lừa:** bộ calib chỉ có 39 ảnh và các ca trùng được tạo với biến đổi có kiểm soát. Trên raw, `phash_dist=16` loại thêm các cặp như `img_0008`/`img_0011`, `img_0074`/`img_0084` và `img_0113`/`img_0177`; việc mở xem các cặp này cho thấy phần lớn có cùng bố cục nhưng vẫn có thể khác biển số hoặc nền. Vì vậy recall 0,83 trên calib không bảo đảm cùng tỉ lệ trên raw. Ngược lại, `conf=0,80` làm recall `POSSIBLE_LOOSE_BOX` giảm xuống 0,20 trên calib, nên checker có thể bỏ sót box lỏng dù hàng đợi ngắn hơn. Hơn nữa, calib không chứa `LEAK_TEST` và lỗi của nó thiên về `person→car`, trong khi raw thiên về `car/truck/bus`.

## 4. Cửa chặn và điều pipeline KHÔNG bắt được

(a) Đã chạy `python -m pipeline.run --budget 0 --reports work/rep_nb --export-dir work/exp_nb`. Kết quả là `REWORK` với 6 lỗi bắt buộc trên 6 ảnh, mã thoát **20**; log ghi rõ “KHÔNG xuất: còn lỗi bắt buộc (Rework)”, nên không có bản dataset phát hành trong `work/exp_nb/`.

(b) Các điểm mù có bằng chứng từ báo cáo:

- `POSSIBLE_FALSE_POS` có 11 cảnh báo động giả trên calib ở mọi ngưỡng `conf`; đây là lệch tự nhiên giữa nhãn người và model, không phải lỗi chắc chắn.
- `POSSIBLE_LOOSE_BOX` chỉ đạt recall 0,20 sau calibration; checker bỏ sót box lỏng nhẹ, đặc biệt khi model cũng không chắc.
- Các cặp gần trùng với `phash` cách 16 có thể khác ở chi tiết nhỏ; phụ thuộc khoảng cách nhưng độ tương đồng cấu trúc có thể che giấu ảnh khác nội dung.

## 5. Train: raw vs clean (tùy chọn, chấm riêng)

Chưa train: phần train trên Colab là tùy chọn và chưa chạy, nên chưa có kết luận mAP50.

## 6. Quyết định và ghi chú AI

(a) Quyết định hiện tại là **Release** sau khi calibration hai ngưỡng. Để tăng độ tin cậy, nên xem lại 13 ảnh còn cảnh báo và đặc biệt các ảnh trùng gần đúng bị loại trước khi dùng bản xuất để huấn luyện.

(b) Ghi chú AI:

- Prompt chính: yêu cầu đọc docstring và test của 4 module, viết theo TDD, giữ nguyên cấu trúc `Issue`/cấu hình và chạy `python -m pytest -q`.
- Cách kiểm code AI: chạy toàn bộ 79 test, kiểm tra lỗi đầu tiên về gom issue theo file trong `autofix`, sau đó chạy lại pipeline và đối chiếu các báo cáo sinh ra.
- Một lần AI sai: bản đầu của `autofix` xử lý từng issue nên ghi file nhiều lần và làm mất các dòng hợp lệ; test `test_drop_several_lines_in_one_file_uses_original_numbers` đã bắt lỗi này và cách sửa là gom issue theo file rồi chỉ ghi lại một lần.

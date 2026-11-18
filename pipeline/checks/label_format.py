"""CHECK 2 — Dòng nhãn YOLO: sai định dạng, ID lớp lạ, box tràn ảnh, box rỗng, box lặp.   (Script 2 trong slide)   ★★

Test:  pytest tests/test_label_format.py -q

Mỗi dòng nhãn: `id cx cy w h` (5 số, toạ độ 0–1). Trả về list[Issue]; `Issue.line` = số dòng (đếm từ 1).

| code             | severity | khi nào                                                                  |
|------------------|----------|--------------------------------------------------------------------------|
| BAD_FORMAT       | ERROR    | không đúng 5 cột số (dùng `parse_yolo_line`, bắt `ValueError`)           |
| BAD_CLASS_ID     | ERROR    | id lớp ngoài 0..len(ds.classes)-1                                        |
| BOX_ZERO         | ERROR    | w <= 0 hoặc h <= 0                                                       |
| BOX_OUT_OF_IMAGE | ERROR    | mép box vượt [0,1] quá `cfg["label_format"]["overflow_tol"]`; extra={"overflow": mức tràn} |
| DUP_BOX          | ERROR    | cùng lớp với một dòng TRƯỚC ĐÓ và IoU > `dup_box_iou`; báo ở dòng SAU    |
| BOX_TINY         | WARN     | cạnh ngắn của box (px) < `min_box_px`  (cần `ds.image_size(stem)`)       |
| BOX_HUGE         | WARN     | w*h > `max_box_frac`                                                     |

Gợi ý
- Mép trái = x - w/2, mép phải = x + w/2 (xem câu hỏi "dòng YOLO nào lỗi?" trong slide).
- Một dòng hỏng (BAD_FORMAT, BAD_CLASS_ID, BOX_ZERO) thì `continue`, đừng kiểm tiếp hình học của dòng đó.
- Dòng thiếu cột làm script của slide dừng giữa chừng: đây là lý do phải `try/except`.
- `iou_xywh(a, b)` có sẵn trong common.py.
- Ngưỡng nằm trong `cfg["label_format"]` — đừng viết cứng số vào code.
"""
from ..common import ERROR, WARN, Issue, iou_xywh, parse_yolo_line  # noqa: F401


def run(ds, cfg):
    issues = []
    settings = cfg["label_format"]
    tolerance = settings["overflow_tol"]
    min_box_px = settings["min_box_px"]
    max_box_frac = settings["max_box_frac"]
    dup_box_iou = settings["dup_box_iou"]

    for label_path in ds.label_files():
        stem = label_path.stem
        previous_boxes = []
        image_size = ds.image_size(stem)

        for line_number, text in enumerate(
            label_path.read_text(encoding="utf-8", errors="replace").splitlines(), 1
        ):
            if not text.strip():
                continue

            try:
                class_id, x, y, width, height = parse_yolo_line(text)
            except ValueError as exc:
                issues.append(Issue(
                    "label_format", ERROR, "BAD_FORMAT", stem,
                    f"Dòng nhãn sai định dạng: {exc}", line_number,
                ))
                continue

            if not 0 <= class_id < len(ds.classes):
                issues.append(Issue(
                    "label_format", ERROR, "BAD_CLASS_ID", stem,
                    f"ID lớp {class_id} nằm ngoài 0..{len(ds.classes) - 1}",
                    line_number,
                ))
                continue

            if width <= 0 or height <= 0:
                issues.append(Issue(
                    "label_format", ERROR, "BOX_ZERO", stem,
                    "Box có chiều rộng hoặc chiều cao không dương",
                    line_number,
                ))
                continue

            left, top = x - width / 2, y - height / 2
            right, bottom = x + width / 2, y + height / 2
            overflow = max(0.0, -left, -top, right - 1.0, bottom - 1.0)
            if overflow > tolerance:
                issues.append(Issue(
                    "label_format", ERROR, "BOX_OUT_OF_IMAGE", stem,
                    f"Box tràn ảnh {overflow:.6f} (mức cho phép {tolerance:.6f})",
                    line_number, {"overflow": overflow},
                ))

            if any(
                previous_class == class_id and iou_xywh(box, (x, y, width, height)) > dup_box_iou
                for previous_class, box in previous_boxes
            ):
                issues.append(Issue(
                    "label_format", ERROR, "DUP_BOX", stem,
                    f"Box trùng box cùng lớp trước đó (IoU > {dup_box_iou})",
                    line_number,
                ))

            if image_size is not None:
                image_width, image_height = image_size
                short_side_px = min(width * image_width, height * image_height)
                if short_side_px < min_box_px:
                    issues.append(Issue(
                        "label_format", WARN, "BOX_TINY", stem,
                        f"Cạnh ngắn của box chỉ {short_side_px:.2f}px, nhỏ hơn {min_box_px}px",
                        line_number, {"short_side_px": short_side_px},
                    ))

            if width * height > max_box_frac:
                issues.append(Issue(
                    "label_format", WARN, "BOX_HUGE", stem,
                    f"Diện tích box {width * height:.6f} vượt ngưỡng {max_box_frac}",
                    line_number, {"box_fraction": width * height},
                ))

            previous_boxes.append((class_id, (x, y, width, height)))

    return issues

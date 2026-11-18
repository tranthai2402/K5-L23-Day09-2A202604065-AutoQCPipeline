"""BƯỚC 2/4 — Sửa tự động những lỗi bắt buộc mà máy sửa được AN TOÀN.   ★★

Test:  pytest tests/test_autofix.py -q

`fix(ds, issues, cfg) -> list[dict]`: duyệt `issues` (chỉ lỗi severity ERROR), sửa thẳng vào thư mục `ds`,
và trả về danh sách thay đổi; mỗi phần tử có khoá: image, line, action, before, after
(đây chính là file `03_fixes.csv` — bằng chứng pipeline đã làm gì, người có thể kiểm lại).

| code lỗi                                       | cách sửa                                    | action           |
|------------------------------------------------|---------------------------------------------|------------------|
| BAD_FORMAT, BAD_CLASS_ID, BOX_ZERO, DUP_BOX    | xoá dòng đó khỏi file nhãn                  | "drop_line"      |
| BOX_OUT_OF_IMAGE                               | cắt box về trong [0,1] (giữ nguyên lớp)     | "clip_box"       |
| MISSING_LABEL_FILE                             | KHÔNG sửa file (máy không thể bịa nhãn): ghi 1 dòng fix, before="(không có file)", after="chuyển người gán" | "needs_label" |
| BAD_CLASS_NAME                                 | ghi lại `names` trong data.yaml đúng `cfg["classes"]` | "rename_classes" |
| DUP_IMAGE_*, LEAK_TEST, CORRUPT_IMAGE, BAD_SIZE| xoá ảnh + nhãn khỏi `ds`                    | "remove_image"   |

Bẫy cần nghĩ tới (có test cho từng cái)
- Một file có NHIỀU dòng lỗi: số dòng trong `issues` là số dòng GỐC. Nếu xoá từng dòng một thì số dòng lệch.
  Gom theo file, rồi ghi lại MỘT lần.
- Sau khi cắt, box có thể không còn diện tích (nằm hẳn ngoài ảnh): lúc đó xoá dòng thay vì ghi box rỗng.
- Một dòng có 2 lỗi (vd. BOX_OUT_OF_IMAGE + DUP_BOX): xoá dòng thắng, chỉ ghi 1 fix `drop_line`, không ghi thêm `clip_box`. Dòng trống ở giữa file vẫn đếm vào số dòng gốc.
- Ảnh vừa bị xoá (DUP_IMAGE_*...) mà còn issue dòng của chính nó: bỏ qua êm, không được báo lỗi/chết.
- Không động tới lỗi WARN (cảnh báo là việc của người). Không sửa file ở `data/raw/` — chỉ sửa `ds`.
- Chạy lần 2 trên dữ liệu đã sửa phải không làm gì thêm (idempotent) — NGOẠI LỆ duy nhất: `needs_label` được ghi lại mỗi lần còn lỗi `MISSING_LABEL_FILE` (vì ảnh vẫn chưa có nhãn).

- `needs_label` KHÔNG tạo file nhãn rỗng (rỗng = "ảnh nền", khác với "chưa gán"). `review_queue` sẽ đẩy ảnh này lên đầu hàng đợi cho người.

Gợi ý: `yaml.safe_load / safe_dump`.
"""
from collections import defaultdict  # noqa: F401

import yaml  # noqa: F401

from ..common import ERROR, parse_yolo_line  # noqa: F401


DROP_CODES = {"BAD_FORMAT", "BAD_CLASS_ID", "BOX_ZERO", "DUP_BOX"}
REMOVE_CODES = {
    "DUP_IMAGE_EXACT",
    "DUP_IMAGE_NEAR",
    "DUP_IMAGE_PHASH",
    "LEAK_TEST",
    "CORRUPT_IMAGE",
    "BAD_SIZE",
}


def _change(image, line, action, before, after):
    return {
        "image": image,
        "line": "" if line is None else line,
        "action": action,
        "before": before,
        "after": after,
    }


def fix(ds, issues, cfg):
    errors = [issue for issue in issues if issue.severity == ERROR]
    fixes = []
    removed_images = set()
    processed_files = set()

    for issue in errors:
        if issue.code in REMOVE_CODES and issue.image not in {"<dataset>", "data.yaml"}:
            removed_images.add(issue.image)

    for issue in errors:
        image = issue.image
        line = issue.line
        code = issue.code

        if image in removed_images:
            continue

        if code == "BAD_CLASS_NAME":
            data = yaml.safe_load(ds.yaml_path.read_text(encoding="utf-8")) or {}
            before = data.get("names", {})
            data["names"] = {i: name for i, name in enumerate(cfg["classes"])}
            ds.yaml_path.write_text(
                yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
                encoding="utf-8",
            )
            fixes.append(_change(image, line, "rename_classes", before, data["names"]))
            continue

        if code == "MISSING_LABEL_FILE":
            fixes.append(_change(image, line, "needs_label", "(không có file)", "chuyển người gán"))
            continue

        if code not in DROP_CODES and code != "BOX_OUT_OF_IMAGE":
            continue

        label_path = ds.label_path(image)
        if not label_path.exists():
            continue
        if image in processed_files:
            continue
        processed_files.add(image)

        original_lines = label_path.read_text(encoding="utf-8", errors="replace").splitlines()
        grouped = defaultdict(list)
        for issue in errors:
            if issue.image == image and issue.line is not None:
                grouped[issue.line].append(issue.code)

        if not grouped:
            continue

        output_lines = []
        for number, text in enumerate(original_lines, 1):
            codes = grouped.get(number, [])
            if not codes:
                output_lines.append(text)
                continue

            drop_codes = [code for code in codes if code in DROP_CODES]
            if drop_codes:
                fixes.append(_change(image, number, "drop_line", text, ""))
                continue

            if "BOX_OUT_OF_IMAGE" in codes:
                try:
                    class_id, x, y, width, height = parse_yolo_line(text)
                except ValueError:
                    fixes.append(_change(image, number, "drop_line", text, ""))
                    continue

                left, top = x - width / 2, y - height / 2
                right, bottom = x + width / 2, y + height / 2
                clipped_left, clipped_top = max(0.0, left), max(0.0, top)
                clipped_right, clipped_bottom = min(1.0, right), min(1.0, bottom)
                clipped_width = clipped_right - clipped_left
                clipped_height = clipped_bottom - clipped_top

                if clipped_width <= 0 or clipped_height <= 0:
                    fixes.append(_change(image, number, "drop_line", text, ""))
                    continue

                new_x = (clipped_left + clipped_right) / 2
                new_y = (clipped_top + clipped_bottom) / 2
                after = f"{class_id} {new_x:.6f} {new_y:.6f} {clipped_width:.6f} {clipped_height:.6f}"
                output_lines.append(after)
                fixes.append(_change(image, number, "clip_box", text, after))

        label_path.write_text("\n".join(output_lines) + ("\n" if output_lines else ""), encoding="utf-8")

    for image in sorted(removed_images):
        image_path = ds.image_path(image)
        label_path = ds.label_path(image)
        before = ""
        if image_path is not None and image_path.exists():
            before = image_path.name
            image_path.unlink()
        if label_path.exists():
            label_path.unlink()
        fixes.append(_change(image, None, "remove_image", before, "đã xoá ảnh và nhãn"))

    return fixes

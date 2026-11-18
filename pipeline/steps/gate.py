"""BƯỚC 4 — Cửa chặn: quyết định Release / Hold / Rework.   ★

Test:  pytest tests/test_gate.py -q

Cây quyết định Day 07 (slide Day 09: "Lỗi bắt buộc chặn pipeline; lỗi cảnh báo chuyển người xem"):

  còn ≥ 1 lỗi bắt buộc (severity ERROR)                          -> "Rework"
  không còn lỗi bắt buộc, nhưng tỉ lệ ẢNH còn cảnh báo
      >= cfg["gate"]["warn_ratio_hold"]                          -> "Hold"
  còn lại                                                        -> "Release"

Tỉ lệ cảnh báo = (số ẢNH khác nhau có ≥ 1 cảnh báo) / (tổng số ảnh).  Chú ý:
- chỉ đếm issue có `image` nằm trong `image_stems` (bỏ "<dataset>", "data.yaml");
- bỏ qua ảnh nằm trong `reviewed` (người đã xem, cảnh báo của ảnh đó coi như đã xử lý) — nhưng LỖI BẮT BUỘC thì không bao giờ được bỏ qua;
- mẫu số là số ảnh, không phải số issue.

Trả về dict: decision, n_errors, n_warn_images, n_images, warn_ratio (làm tròn 4 chữ số), reasons (list[str], giải thích bằng tiếng Việt, có số liệu).
"""
from ..common import ERROR, WARN  # noqa: F401


def decide(issues, image_stems, cfg, reviewed=frozenset()):
    stems = set(image_stems)
    reviewed = set(reviewed)
    n_images = len(stems)
    errors = [issue for issue in issues if issue.severity == ERROR and issue.image in stems]
    n_errors = len(errors)

    warn_images = {
        issue.image
        for issue in issues
        if issue.severity == WARN
        and issue.image in stems
        and issue.image not in reviewed
    }
    n_warn_images = len(warn_images)
    warn_ratio = round(n_warn_images / n_images, 4) if n_images else 0.0
    threshold = cfg["gate"]["warn_ratio_hold"]

    if n_errors:
        decision = "Rework"
        reasons = [
            f"Còn {n_errors} lỗi bắt buộc trên {len({i.image for i in errors})} ảnh; phải sửa trước khi phát hành."
        ]
    elif n_warn_images and warn_ratio >= threshold:
        decision = "Hold"
        reasons = [
            f"{n_warn_images}/{n_images} ảnh còn cảnh báo ({warn_ratio:.1%}), "
            f"đạt ngưỡng Hold {threshold:.1%}; cần người duyệt."
        ]
    else:
        decision = "Release"
        if n_warn_images:
            reasons = [
                f"{n_warn_images}/{n_images} ảnh còn cảnh báo ({warn_ratio:.1%}), "
                f"thấp hơn ngưỡng Hold {threshold:.1%}."
            ]
        else:
            reasons = ["Không còn lỗi bắt buộc hoặc cảnh báo chưa xử lý; có thể phát hành."]

    return {
        "decision": decision,
        "n_errors": n_errors,
        "n_warn_images": n_warn_images,
        "n_images": n_images,
        "warn_ratio": warn_ratio,
        "reasons": reasons,
    }

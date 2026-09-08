# Agent Brief — QC (viết lại 2026-09-08)

## Đọc trước
1. `.claude/docs/topic1_implementation_plan.md` — bản mới, đặc biệt §8 (bài học phương pháp)
2. `docs/same_view_control_feasibility.md` — ghi lại một lỗi đã suýt thành kết luận sai

## Quyền hạn
**Phủ quyết.** Không con số nào vào bài nếu QC chưa ký. QC đo lại độc lập,
không tin log của AI-ENGINEER.

## Triết lý — đã được chứng minh bằng máu trong dự án này
Đề tài này suýt chết bốn lần vì **lỗi thầm lặng**, không phải crash. Mỗi lần,
training vẫn chạy, loss vẫn giảm, mọi chỉ số tổng hợp trông bình thường:

- bug `+100` camid → gap ảo +22.7% thay vì thật −1.59%
- probe granularity đo lại cùng một biểu diễn 4 lần (tương quan ≥0.96)
- ngưỡng slope cố định gán nhầm "FLAT" cho đường cong đang qua 0
- trích `−1.11%` (GSS) thay `−1.59%` (ma trận cặp camera) — khác trục

Nhiệm vụ QC là bắt loại lỗi đó trước khi nó thành kết luận.

---

## QC-N1 — Verify sampler (chặn AI-04, cấp bách nhất)
`tests/test_view_sampler.py`. Đo **trực tiếp trên batch thật**, không đọc log:
- mỗi batch có bao nhiêu % ID thực sự có cả aerial lẫn ground?
- số cặp cross-platform cùng ID trung bình/batch > 0?

Nếu = 0 thì `β(l, cross)` là no-op và toàn bộ đóng góp chết. Test phải fail rõ
ràng dưới ngưỡng đã thống nhất.

## QC-N2 — Unit test loss (chặn AI-06)
Cây giả + embedding giả:
- β áp **đúng tầng** (whole nhận 0.523 cho cross, 0.071 cho same ở `coarse_bias=2.0`)
- cross/same áp **đúng cặp** (dùng `binary_view_of_camera` của CARGO)
- `coarse_bias=0.0` cho **0.25 đều** — đây là nhánh uniform-β, phải đúng chính xác
- Σβ = 1 mỗi cặp
- gradient finite cả nhánh cross lẫn same

## QC-N3 — Công bằng giữa ba nhánh
Verify baseline / uniform-β / view-aware dùng **cùng**: backbone, batch, epoch,
LR schedule, augmentation, seed set. Lập bảng `config diff`; ô nào khác phải có
lý do ghi rõ.

⚠️ Cảnh giác đặc biệt: nếu view-aware train nhiều epoch hơn vì "cần hội tụ lâu
hơn", đó là so sánh không công bằng.

## QC-N4 — Đọc kết quả đúng thước đo
**Chặn mọi báo cáo chỉ có mAP.** Thước đo là `(mAP, gap ground-only)`. Đường
cong CARGO đã chứng minh 25 điểm mAP không co gap — nên mAP cao mà gap giữ
nguyên là **thất bại**, không phải thành công.

## QC-N5 — Guard đã có, giữ cho chúng chạy
Các guard này đã cứu dự án, đừng để chúng bị tắt:
- `MIN_SHARED_IDENTITIES = 100` trong ma trận CARGO
- cảnh báo dải mAP < 10 điểm trong `gap_vs_capability.py`
- `UnknownCameraError` khi camera lạ (thay vì mặc định về ground)
- bộ lọc cross-camera trong mọi same-view control

## QC-N6 — Seed
Mọi cấu hình chính ≥3 seed, report mean ± std. Không con số 1-seed trong bài.

## QC-N7 — Truy vết
Mỗi con số trong bài truy được về một log/JSON cụ thể. Lập bảng ánh xạ
`số trong bài → file`.

---

## Khi nào báo động toàn nhóm
- batch không có cặp cross-platform cùng ID
- view-aware không thắng uniform-β → kết cục B, **không nặn tiếp**
- std giữa seed > gain → gain không có ý nghĩa thống kê
- có con số không truy được về log

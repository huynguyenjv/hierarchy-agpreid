# BA-01 — AG-ReID.v2 view mapping

Module: `reid_advance/hierarchy/view_map.py` · Tests: `tests/test_view_map.py`, `tests/test_data_assumptions.py`

## Kết luận

| camera id | platform | binary view |
|---|---|---|
| C0 | aerial | 1 (aerial) |
| C2 | wearable | 0 (ground) |
| C3 | CCTV | 0 (ground) |

**Quy ước binary**: `ground = {wearable, cctv}`, `aerial = {aerial}`. Đây là quy ước dùng cho hệ số `β_cross` của CV-HWC — đóng góp C2 nói về khoảng cách *aerial–ground*, không phải phân biệt CCTV với wearable.

## Bằng chứng đối chiếu

Mapping có sẵn ở `pipelines/transreid.py:31` đã được verify độc lập bằng 4 file protocol chính thức. Mỗi file chỉ chứa đúng 2 loại camera, và camera của `query/` cho biết vế trái của tên protocol:

| protocol | camera xuất hiện | query camera | suy ra |
|---|---|---|---|
| `exp1_aerial_to_cctv` | {0, 3} | C0 | C0=aerial, C3=cctv |
| `exp2_aerial_to_wearable` | {0, 2} | C0 | C2=wearable |
| `exp4_cctv_to_aerial` | {0, 3} | C3 | xác nhận C3=cctv |
| `exp5_wearable_to_aerial` | {0, 2} | C2 | xác nhận C2=wearable |

Hai protocol sau xác nhận chéo hai protocol đầu: cùng một camera id giữ nguyên vai trò khi đổi chiều query/gallery.

## Độ phủ

Toàn bộ dataset chỉ chứa camera {0, 2, 3} — không có ảnh nào rơi vào `unknown`:

| split | ảnh | C0 (aerial) | C2 (wearable) | C3 (cctv) |
|---|---|---|---|---|
| `train_all` | 51,530 | 21,217 | 21,298 | 9,015 |
| `query` | 8,499 | 4,348 | 2,340 | 1,811 |
| `gallery` | 40,473 | 21,214 | 12,912 | 6,347 |

Tỉ lệ aerial/ground trong `train_all` là 21,217 / 30,313 ≈ **41% / 59%** — đủ cân bằng để sampler lấy K/2 mỗi view mà không phải bù nhiều.

## Quyết định thiết kế

**Camera id lạ thì raise, không mặc định về ground.** Nếu một camera chưa biết bị lặng lẽ gán thành ground, `β_cross` sẽ không kích hoạt cho các cặp liên quan mà không có dấu hiệu gì — training vẫn chạy, loss vẫn giảm, và đóng góp C2 âm thầm biến mất. `UnknownCameraError` bắt trường hợp đó ngay.

**Giữ song song hai API**: `platform_tensor()` (3 lớp) cho SIE giữ nguyên hành vi hiện có của `camera_to_view()`, `binary_view_tensor()` (2 lớp) cho CV-HWC. Test `test_platform_tensor_matches_the_existing_pipeline_helper` chốt hai hàm không lệch nhau.

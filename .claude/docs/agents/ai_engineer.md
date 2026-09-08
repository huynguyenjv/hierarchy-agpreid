# Agent Brief — AI-ENGINEER (viết lại 2026-09-08)

## Đọc trước
1. `.claude/docs/topic1_implementation_plan.md` — **bản mới**, thay thế hoàn toàn plan gốc
2. `docs/gap_vs_capability.md` — kết quả trung tâm, định nghĩa thước đo cho mọi thứ bạn làm
3. KB gốc chỉ là tham khảo lý thuyết, **không phải kế hoạch thi hành**

## Phạm vi
Sampler, loss, pipeline training, chạy thí nghiệm.

## Ba nguyên tắc bất di bất dịch

**1. Báo cáo `(mAP, gap)`, không bao giờ mAP đơn lẻ.** Đường cong CARGO đã chứng
minh 25 điểm mAP không co được gap dù 1 điểm. Nếu loss của bạn nâng mAP mà gap
giữ nguyên, bạn chỉ tái tạo cái capacity đã làm — đó là **thất bại**, dù mAP đẹp.

**2. Không đụng `transreid.py` và `cargo_transreid.py` theo cách phá baseline.**
Chúng là hai vế của kết quả đã công bố nội bộ. Code mới đi vào module riêng.

**3. Không bỏ `L_ID + L_triplet`.** Bảo hiểm không tụt dưới baseline.

---

## AI-04 — ViewBalancedPKSampler

Kế thừa `RandomIdentitySampler` (`data.py:330`). Mỗi ID lấy K/2 aerial + K/2
ground. Dùng `reid_advance.cargo.binary_view_of_camera` (Cam1–5 aerial,
Cam6–13 ground).

⚠️ **Verify trước khi train một epoch nào**: in tỉ lệ aerial/ground và **số cặp
cross-platform cùng ID** trong 3–5 batch đầu. Nếu batch không có cặp
cross-platform cùng ID thì `β(l, cross)` không bao giờ kích hoạt — loss thành
no-op, training vẫn chạy, loss vẫn giảm, và bạn mất ba ngày mới phát hiện.

CARGO thuận lợi: 2,351/2,500 gallery identity có cả hai platform. Nhưng vẫn phải
đo, không giả định.

---

## AI-06 — Multi-granularity loss

```
L = L_ID + L_triplet + α · L_MG
L_MG = Σ_l  β(l, cross_ij) · SupCon_l
```

`SupCon_l` tính trên feature tầng `l` từ `hierarchy/granularity.py`
(`pyramid_features` → `flatten_level`), đã L2-norm.

### β(l) khởi tạo — `level_weights_for_view(depth=4, coarse_bias=2.0)`

| | whole | half | quarter | stripe |
|---|---|---|---|---|
| cross-platform | **0.523** | 0.268 | 0.138 | 0.071 |
| same-platform | 0.071 | 0.138 | 0.268 | **0.523** |

Ablate `coarse_bias ∈ {1.0, 2.0, 3.0}` → tỉ lệ coarse/fine 2.7× / 7.4× / 20×.

`α = 1.0` khởi điểm; **in scale của `L_MG` vs `L_ID + L_tri` ở batch đầu** để
chỉnh trước khi train dài. Ablate {0.3, 1.0, 3.0}.

### Unit test bắt buộc trước khi train
Cây giả + embedding giả: β áp đúng tầng, cross/same áp đúng cặp, `coarse_bias=0`
cho 0.25 đều, gradient finite cả hai nhánh.

---

## Ba nhánh — chạy cùng một lượt, không thương lượng

| nhánh | β | vì sao cần |
|---|---|---|
| baseline | — | `L_ID + L_tri` thuần |
| **uniform-β** | `coarse_bias=0.0` → 0.25 đều | **phân biệt granularity thật với multi-scale** |
| view-aware | `coarse_bias=2.0` | đóng góp cần chứng minh |

Uniform-β **vẫn có đủ 4 tầng, cùng tham số, cùng compute, cùng schedule**. Chỉ
bỏ view-conditioning. Đây là baseline duy nhất chặn được phản biện "chỉ là
multi-scale đội tên hierarchy". Chạy sau là muộn.

---

## Sau khi train — cách đọc

Chạy `tools/cargo_camera_pair_matrix.py` trên checkpoint của cả ba nhánh, so
bảng `(mAP, gap ground-only)`:

- view-aware gap **thấp hơn** uniform-β, mAP tương đương → ✅
- view-aware mAP cao hơn, gap như nhau → ❌ capacity, không phải granularity
- cả hai ≈ baseline → ❌ multi-granularity vô nghĩa trên CARGO

Nếu ❌: **báo ngay, không nặn tiếp.** Kết cục B (analysis + negative result) là
hợp lệ và bài analysis đã chắc.

## Cấu hình
ViT-S/16, 256×128, batch 32, AMP, CARGO 30 epoch (trần của công thức — mAP
plateau 46.6% trong khi train loss vẫn giảm). Giữ batch 32 dù VRAM dư: đổi batch
là đổi công thức, mất tính so sánh với AG-ReID.v2.

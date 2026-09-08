# Implementation Plan — Topic 1 (bản viết lại, 2026-09-08)

**Repo**: `huynguyenjv/hierarchy-agpreid` (master) · **Phần cứng**: RTX 3060 12GB, 32GB RAM
**Chạy Python bằng `venv/Scripts/python.exe`** — bản global là torch CPU-only.

> Bản này thay thế hoàn toàn plan gốc. Hướng ban đầu (cây phân cấp ngữ nghĩa từ
> soft-biometric attributes, theo knowledge base) **đã bị dữ liệu bác bỏ**. Ba
> tiền đề nền của nó lần lượt đổ, và điều tra thay thế đã cho một kết quả mạnh
> hơn. KB gốc (`topic1_hierarchy_aware_agpreid_knowledge_base.md`) giữ lại làm
> tham khảo lý thuyết, **không còn là kế hoạch thi hành**.

---

## 0. Tóm tắt trạng thái

**Đã xong và chắc chắn** — đủ làm một bài analysis hoàn chỉnh:

| # | Kết quả | Bằng chứng |
|---|---|---|
| 1 | Attribute **không** suy giảm theo góc nhìn; chỉ `gender` dùng được làm tầng cây | `docs/vss_table.md` |
| 2 | Same-view control **không dựng được** từ protocol chính thức của AG-ReID.v2 | `docs/same_view_control_feasibility.md` |
| 3 | Aerial-ground **không** phải trục khó trên AG-ReID.v2 (gap −1.59%) | `docs/camera_pair_matrix.md` |
| 4 | Trên CARGO gap **có thật** và lớn (+17.20% trục ground-only) | `docs/cargo_camera_pair_matrix.md` |
| 5 | **Hai đường cong khác hình dạng**: AG-ReID.v2 hòa tan gap, CARGO giữ nguyên | `docs/gap_vs_capability.md` |
| 6 | Phân rã hai hiệu ứng: aerial-intrinsic vs platform-transfer (chỉ CARGO đo được) | `docs/cargo_camera_pair_matrix.md` |

**Chưa xong** — nhánh method, đang thử:

| # | Việc | Trạng thái |
|---|---|---|
| 7 | ViewBalancedPKSampler (AI-04) | chưa làm |
| 8 | Multi-granularity loss + view-conditioning (AI-06) | formulation đã nháp, chờ soi |
| 9 | Train 3 nhánh: baseline / uniform-β / view-aware | chưa chạy |

---

## 1. Vì sao hướng gốc đổ — chuỗi ba phát hiện

Ghi lại để không ai (kể cả tương lai) đi lại vòng này.

**(a) Attribute không suy giảm theo view.** KB §5.1 giả định "thuộc tính thô sống
sót khi bay lên cao, chi tiết tinh biến mất". Đo thật (AI-03): retention
(aerial ÷ ground balanced accuracy) là **0.86–1.07** cho gần hết 15 attribute —
lên cao gần như không mất gì. Nhưng ground-bal cũng chỉ ≤0.54 ngoài `gender`.
Các attribute này **khó nhận ngay từ ảnh ground**; đó là giới hạn nhãn/độ phân
giải, không phải hiện tượng aerial-ground. Chỉ 1/15 vượt ngưỡng VSS 0.65 → cây
3 tầng attribute không xây được.

**(b) Granularity không tự phân hóa.** Hướng thay thế: cây theo độ chi tiết
không gian (whole → half → quarter → stripe), lấy từ patch grid, không cần nhãn.
Probe trên checkpoint đóng băng (GSS) cho kết quả **INCONCLUSIVE** và được báo
đúng như vậy: các tầng tương quan **≥0.96** trong không gian khoảng cách, mAP
chênh nhau 0.13–0.46%. Mọi tầng pooling từ cùng tập token của một backbone
huấn luyện cho một mục tiêu global duy nhất — probe đo lại một biểu diễn 4 lần.
Nó **không bác bỏ** giả thuyết, chỉ cho thấy granularity chưa phân hóa sẵn.

**(c) Gap không nằm ở trục người ta tưởng.** Phân rã theo cặp camera trên
AG-ReID.v2: aerial–ground **73.18%** *cao hơn* ground–ground **71.59%**. Trục
aerial-ground không phải trục khó. Thứ nổi lên là *hướng*, không phải platform:
C0→C2 (76.56%) hơn C2→C0 (71.19%) 5.4 điểm trên cùng tập ID — hiệu ứng gallery,
không phải view gap.

---

## 2. Kết quả trung tâm — hai đường cong

Câu hỏi đóng biến: gap là nội tại dataset hay artifact của model yếu?

Không so được ở cùng mAP (CARGO trần 46.6%, AG-ReID.v2 trên 70% — cùng công
thức, hai trần khác nhau). Trả lời bằng **hình dạng đường cong** qua dải năng
lực của mỗi dataset, trục ground-only (trục duy nhất cả hai cùng có):

### AG-ReID.v2 — supervision hòa tan gap

| epoch | mAP | a–g | g–g | gap |
|---|---|---|---|---|
| 5 | 58.08% | 55.56% | 56.72% | **+1.16%** |
| 10 | 65.19% | 60.43% | 61.08% | **+0.65%** |
| 15 | 68.21% | 63.20% | 62.99% | **−0.21%** |
| 20 | 70.35% | 64.48% | 63.90% | **−0.59%** |
| *125 (độc lập)* | *75.47%* | *73.18%* | *71.59%* | ***−1.59%*** |

### CARGO — gap giữ nguyên

| epoch | mAP | a–g | g–g | gap |
|---|---|---|---|---|
| 5 | 21.61% | 33.63% | 51.90% | **+18.27%** |
| 10 | 42.06% | 54.67% | 72.65% | **+17.98%** |
| 15 | 41.88% | 56.72% | 75.12% | **+18.40%** |
| 20 | 45.50% | 60.59% | 77.70% | **+17.11%** |
| 25 | 46.88% | 61.42% | 78.58% | **+17.16%** |
| 30 | 46.56% | 61.47% | 78.67% | **+17.20%** |

**Dải mAP: 12.3 điểm (AG-ReID.v2) và 25.3 điểm (CARGO)** — cả hai vượt ngưỡng
10 nên đọc slope hợp lệ, không phải lùi về đọc mức.

**Kết luận**: gap cross-platform là **hàm của benchmark**, không phải hằng số
của bài toán. Trên một dataset supervision thường xóa sạch nó; trên dataset kia
mAP tăng gấp đôi mà gap không co 1 điểm. Khác nhau về **kiểu**, không chỉ về độ
khó — và kết luận này không phụ thuộc việc đặt hai model ở cùng mAP.

### Phân rã hai hiệu ứng (chỉ CARGO đo được)

| loại cặp | số cặp | mean mAP |
|---|---|---|
| aerial–ground | 80 | 61.47% |
| aerial–aerial | 20 | 67.14% |
| ground–ground | 56 | 78.67% |

`aerial–aerial` nằm **giữa** → hai hiệu ứng cộng dồn, không phải một: ảnh nhìn
từ trên cao tự nó khó hơn (67.1 vs 78.7) **ngay cả khi không đổi platform**, và
đổi platform tốn thêm nữa (61.5). Cộng đồng gộp hai thứ này làm một.
AG-ReID.v2 không thể tạo phân rã này (0 identity có ≥2 camera aerial).

---

## 3. Cây kết cục — nhánh nào dẫn về đâu

```
                    hai đường cong (ĐÃ CÓ)
                            │
              ┌─────────────┴─────────────┐
              │                           │
     method co được gap           method không co được gap
     ở fixed-mAP                  ở fixed-mAP
              │                           │
        A: analysis + method        B: analysis + negative result
        (bài mạnh nhất)             (vẫn là bài tốt)
```

**Ba trong bốn kết cục dẫn về analysis.** Chỉ một nhánh cho method. Không cột
danh dự vào nhánh hẹp — nếu loss không co được gap, lùi về B, không nặn tiếp.

**Đã chốt: A như hướng thử, không phải A như kết luận.**
- CARGO = dataset chính (nơi có gap thật để nhắm)
- AG-ReID.v2 = control-gap-vắng-mặt (chứng minh method không gây hại khi không có gap)

---

## 4. Rào mà đường cong vừa dựng cho method

Đường cong CARGO chứng minh **25 điểm mAP không co được gap dù 1 điểm**. Đó là
tin tốt cho "gap cấu trúc, đáng nghiên cứu" nhưng là gánh nặng trực tiếp lên
loss: **AI-06 phải làm cái mà 25 điểm mAP không làm được.**

### Thước đo — do đường cong định nghĩa, không phải tự đặt

| kết quả | đọc là |
|---|---|
| view-aware **gap thấp hơn** uniform-β, mAP tương đương | ✅ granularity thật |
| view-aware **mAP cao hơn** nhưng gap như nhau | ❌ chỉ là capacity — thất bại |
| cả hai bằng baseline | ❌ multi-granularity vô nghĩa trên CARGO |

**Luôn báo cáo cặp `(mAP, gap)`. Không bao giờ báo mAP đơn lẻ** — mAP tổng sẽ
ru ngủ.

---

## 5. Backlog còn lại

### AI-04 — ViewBalancedPKSampler
Kế thừa `RandomIdentitySampler` (`data.py:330`), mỗi ID lấy K/2 aerial + K/2
ground. Dùng `cargo.binary_view_of_camera` (1–5 aerial, 6–13 ground).

**Verify bắt buộc trước khi train một epoch nào**: in tỉ lệ aerial/ground và số
cặp cross-platform cùng ID trong vài batch đầu. Nếu batch không chứa cặp
cross-platform cùng ID thì view-aware loss là **no-op** và mất ba ngày mới biết.
QC đo trực tiếp trên batch thật, không tin log.

### AI-06 — Multi-granularity loss

```
L = L_ID + L_triplet + α · L_MG
L_MG = Σ_l  β(l, cross_ij) · SupCon_l
```

`L_ID + L_triplet` giữ nguyên trên CLS — bảo hiểm không tụt dưới baseline.

**β(l) khởi tạo** (`depth=4`, softmax qua tầng, `coarse_bias=2.0`):

| | whole | half | quarter | stripe | coarse/fine |
|---|---|---|---|---|---|
| cross-platform | **0.523** | 0.268 | 0.138 | 0.071 | 7.39× |
| same-platform | 0.071 | 0.138 | 0.268 | **0.523** | 0.14× |

Ablate `coarse_bias ∈ {1.0, 2.0, 3.0}` (tỉ lệ 2.7× / 7.4× / 20×).
`α = 1.0` khởi điểm, ablate {0.3, 1.0, 3.0}; in scale từng thành phần ở batch
đầu để chỉnh trước khi train dài.

**Unit test trên cây giả** trước khi train: β áp đúng tầng, cross/same áp đúng
cặp, gradient finite.

### Ba nhánh — chạy cùng một lượt (không thương lượng)

| nhánh | β | mục đích |
|---|---|---|
| baseline | — | `L_ID + L_tri` thuần |
| **uniform-β** | 0.25 đều mọi tầng, mọi cặp | **phân biệt granularity thật với multi-scale** |
| view-aware | bảng trên | đóng góp cần chứng minh |

Uniform-β **không phải** tắt multi-granularity — vẫn 4 tầng, cùng tham số, cùng
compute, cùng schedule. Chỉ bỏ **view-conditioning**. Nếu view-aware không thắng
nó, cái ta có chỉ là multi-scale features, thứ đã tồn tại từ lâu. Chạy sau là
muộn — reviewer sẽ buộc chạy.

---

## 6. Hai điểm formulation chưa chốt

**(1) Softmax qua tầng có đúng không?** Nó ép Σβ = 1, nên tăng coarse *phải*
giảm fine. Cách khác: β độc lập mỗi tầng (không chuẩn hóa), cho phép cross-platform
tăng coarse mà không bỏ fine. Softmax làm ablation sạch hơn (ngân sách cố định
→ gain không do loss lớn hơn) nhưng có thể quá cứng.

**(2) `α` khởi tạo** chưa có cơ sở. Nếu `L_MG` khác scale `L_ID + L_tri` thì 1.0
lệch.

---

## 7. Hạ tầng đã có

| công cụ | dùng để |
|---|---|
| `tools/camera_pair_matrix.py` | ma trận cặp camera AG-ReID.v2 |
| `tools/cargo_camera_pair_matrix.py` | ma trận CARGO + guard ID-chung + per-camera |
| `tools/gap_vs_capability.py` | hai đường cong, hai định nghĩa gap |
| `tools/compute_vss.py` | View Stability Score |
| `tools/compute_gss.py` | Granularity probe (kết quả inconclusive) |
| `tools/audit_attributes.py` | audit attribute |
| `tools/profile_memory.py` | VRAM vs batch |
| `reid_advance/cargo.py` | CARGO adapter |
| `reid_advance/hierarchy/` | view_map, granularity, vss |
| `tests/` (75 test) | giả định dữ liệu, view map, granularity, VSS, CARGO |

**Cấu hình đã chốt**: ViT-S/16, 256×128, batch 32 (giữ để so được với
AG-ReID.v2 — VRAM còn dư nhiều nhưng đổi batch là đổi công thức), AMP bật,
CARGO 30 epoch (trần của công thức).

---

## 8. Bài học phương pháp — áp cho mọi bước sau

Ghi lại vì mỗi cái đều suýt cho kết luận sai:

1. **Same-view control**: giữ bộ lọc cross-camera. Bug `+100` vào camid biến nó
   thành retrieval trong cùng camera → gap ảo +22.7% thay vì thật −1.59%.
2. **Ô ma trận mỏng**: đếm ID-chung, gạch ô dưới ngưỡng. Ô rỗng gallery trông
   giống ô khó.
3. **Một camera hỏng**: kiểm per-camera trước khi kết luận. Cam1 chiếm 7/8 cặp
   khó nhất CARGO — nhưng loại nó gap vẫn +11.6%.
4. **Hội tụ**: đo theo đại lượng đang dùng (hình dạng ma trận, rank-corr
   0.9973), không theo mAP.
5. **Ngưỡng cố định đọc sai ca biên**: verdict slope −0.15 gán nhầm "FLAT" cho
   AG-ReID.v2 — gap khởi điểm gần 0 không thể có slope dốc dù bị xóa sạch.
6. **Control vô dụng thì loại**: random-init cho 1.3% phẳng, không định vị được
   gì. Không giữ lại cho bảng trông đầy.
7. **Trích số đúng trục**: `−1.11%` (GSS) khác `−1.59%` (ma trận cặp camera).
   Tôi đã trích nhầm vài lượt.

# Agent Brief — QC (Quality Control)

## Đọc trước khi làm
1. `.claude/docs/topic1_hierarchy_aware_agpreid_knowledge_base.md` — đặc biệt §8.2 (ba chỗ dễ sai), §9.2 (random tree control), §12.2 (debug), §14 (checklist nộp bài).
2. `.claude/docs/topic1_implementation_plan.md`

## Phạm vi sở hữu
Unit test, reproducibility, kiểm chứng độc lập mọi số liệu, chống lỗi thầm lặng.

## Quyền hạn
**Quyền phủ quyết.** Không con số nào được đưa vào bảng bài báo nếu QC chưa ký duyệt.
QC là người duy nhất kiểm tra kết quả **độc lập** — không tin log của AI-ENGINEER, tự đo lại.

## Triết lý
Đề tài này chết vì **lỗi thầm lặng**, không phải vì crash. Loss vẫn giảm, training vẫn chạy, nhưng `ℓ` tính sai hoặc batch không có cross-view pair — và ba tháng sau mới phát hiện. Nhiệm vụ của QC là bắt những lỗi đó ở tuần thứ nhất.

---

## Task list

## Ràng buộc phần cứng
GPU **12GB**, 32GB RAM. Backbone **ViT-S/16**, không phải ViT-B như KB §9.4.

Hai điều chỉnh với vai trò QC:
- **Bỏ tiêu chí "khớp paper gốc ±0.5%"** (KB §11 T1). Không áp dụng được vì paper dùng ViT-B. Thay bằng: baseline **ổn định** (3 seed, std < 0.5% mAP) và mọi so sánh ở **cùng backbone, cùng batch**.
- **Thêm nhiệm vụ mới QC-00**: kiểm chứng tính công bằng của so sánh. Trên GPU eo hẹp, cám dỗ lớn nhất là chạy phương pháp mới ở cấu hình tốt hơn baseline. QC phải chặn.

### QC-00 — Kiểm chứng công bằng của so sánh (xuyên suốt)
Với **mọi** cặp số đưa vào bảng, verify baseline và phương pháp mới dùng **cùng**: backbone, batch size, số epoch, LR schedule, augmentation, image size, seed set.
- Lập bảng `config diff` giữa mọi dòng của bảng chính. Ô nào khác phải có lý do chính đáng ghi rõ.
- **Cảnh giác**: nếu phương pháp mới train nhiều epoch hơn baseline vì "cần hội tụ lâu hơn", đó là so sánh không công bằng — phải train baseline đủ số epoch tương ứng.
- Đây là điểm reviewer soi kỹ nhất khi thấy backbone nhỏ.

### QC-01 — Test infrastructure (làm ngay)
- Dựng `tests/` + `pytest.ini`. Repo hiện **chưa có test nào**.
- Seed-lock: mọi nguồn ngẫu nhiên (`random`, `numpy`, `torch`, sampler seed) được fix và log.
- Script chạy nhanh cho CI cục bộ.
- **DoD**: `pytest` xanh; baseline chạy 3 seed cho std < 0.5% mAP (nếu lớn hơn, training không ổn định — báo động).

### QC-02 — Test giả định dữ liệu (`tests/test_data_assumptions.py`)
Đây là các giả định mà **toàn bộ đề tài đứng trên**:
1. Mọi ảnh cùng một identity có **cùng vector attribute** (KB §2.2 khẳng định attribute ở mức identity). Test fail nếu bất kỳ ID nào có ≥2 vector khác nhau.
2. Mọi ảnh trong `train_all`/`query`/`gallery` được gán view — 0 ảnh `unknown`. Mapping đã có ở `pipelines/transreid.py:31` (`C0=aerial, C2=wearable, C3=CCTV`) — test phải bắt được nếu xuất hiện camera id ngoài {0,2,3}, và xác nhận quy ước binary `ground = {cctv, wearable}` được áp dụng nhất quán ở sampler, loss và metrics.
3. `track_key()` (`data.py:71`) map đúng sang `image_index` của MAT; đo tỉ lệ ID không tìm thấy entry.
4. Identity parsing dùng đúng composite key `P+T+A` (807 train identities) như `docs/project.md` quy định, không phải `P` đơn lẻ.

### QC-03 — Test ViewBalancedPKSampler (`tests/test_view_sampler.py`)
**Đo trực tiếp trên batch thật, không đọc log của AI.**
- Với P=16, K=8: mỗi batch có bao nhiêu % identity thực sự có cả aerial lẫn ground?
- Số cặp cross-view cùng ID trung bình mỗi batch > 0 (nếu = 0 thì `β_cross` là no-op và toàn bộ C2 chết).
- Test phải fail rõ ràng nếu tỉ lệ tụt dưới ngưỡng đã thống nhất.

### QC-04 — Test cv_hwc_loss (`tests/test_cv_hwc_loss.py`) — test quan trọng nhất
Dùng cây giả + embedding giả, tính tay kết quả kỳ vọng.

1. **cumprod vs sum** (KB §8.2 chỗ dễ sai #1): dựng hai mẫu **khác Gender ở L1** nhưng **cùng màu áo ở L3**. Đúng phải cho `ℓ = 0`. Nếu implement dùng `sum(match)` sẽ ra `ℓ = 1` → test phải bắt được.
2. **Trọng số**: `w = λ^(L−ℓ)` giảm đúng theo tầng; kiểm với λ=0.5, L=4 cho ℓ=0..4.
3. **Cross-view**: `β=1` khi cùng view, `β=β_cross` khi khác view. Kiểm ma trận β trực tiếp.
4. **Thoái hóa**: λ→0 phải hội tụ về SupCon chuẩn (so số với implement SupCon tham chiếu).
5. **Chuẩn hóa**: `Z_i = Σ β_ij w_ij` đúng; loss không phụ thuộc scale của coef.
6. **Gradient**: finite, không NaN, không inf. Kiểm cả nhánh Euclid lẫn hyperbolic.
7. **Đối xứng/mask**: đường chéo bị loại; `pos_mask = (ell>=1) & ~eye`.

### QC-05 — Test metrics (`tests/test_hierarchy_metrics.py`)
1. **H-mAP với L=1 phải bằng mAP chuẩn** của `evaluation.py:60 evaluate_rank`. Đây là test neo quan trọng nhất — nó chứng minh metric mới là mở rộng đúng đắn của metric cũ, không phải một thứ khác.
2. **MS = 0** khi mọi rank-1 đúng.
3. **AC@k** trên ranking tính tay được (k nhỏ, 5–10 mẫu).
4. Case biên: query không có match nào trong gallery; k > số gallery.

### QC-06 — Random tree control (Phase 2) — test khoa học quan trọng nhất
KB §9.2 Ablation B: **phải chứng minh cây random KHÔNG hiệu quả** (hoặc kém hẳn cây ngữ nghĩa).
- Chạy random tree ≥3 seed.
- **Nếu random tree cũng cải thiện mAP tương đương** → báo động cả nhóm ngay: đóng góp chỉ là regularization ngẫu nhiên, không phải hierarchy. Định vị bài phải viết lại.
- Reviewer khó tính chắc chắn hỏi câu này. QC là người trả lời trước khi reviewer hỏi.

### QC-07 — Không tụt trên view-homogeneous
Verify A→A và G→G không tụt so với baseline. Nếu tụt, phương pháp đánh đổi chứ không phải cải thiện — phải nói rõ trong Limitations.

### QC-08 — Numerical stability hyperbolic (`tests/test_hyperbolic.py`)
- Điểm gần biên ball (norm → 1): `dist` finite, gradient finite.
- Clip norm thực sự có hiệu lực ≤ 1−1e-5.
- float32 tối thiểu; test cả trường hợp curvature learnable.
- Chạy dài: không NaN sau 120 epoch (smoke test rút gọn cũng được, nhưng phải có).

### QC-09 — Chính sách seed
**Mọi cấu hình chính chạy ≥3 seed, report mean ± std** (KB §14).
Không con số nào trong bài báo được phép là 1-seed. QC duy trì bảng theo dõi: cấu hình nào đã đủ seed, cấu hình nào chưa.

### QC-10 — Chi phí tính toán
Đo #params và inference time của phương pháp ta vs VDT vs TransReID.
VDT nhấn mạnh "độ phức tạp cùng cấp" — ta phải chứng minh không nặng hơn, nếu không reviewer sẽ dùng điểm này đánh.

### QC-11 — Checklist nộp bài (Phase 5)
Chạy toàn bộ KB §14. Với mỗi mục, đính kèm bằng chứng (log path, số liệu):

**Kết quả**
- [ ] Baseline khớp paper gốc ±0.5%
- [ ] Thắng cả A→G và G→A trên AG-ReID.v2
- [ ] Có kết quả dataset thứ hai (CARGO hoặc AG-ReID.v1)
- [ ] `VDT + CV-HWC > VDT` (thí nghiệm sinh tử)
- [ ] Không tụt trên A→A, G→G
- [ ] Random tree control chứng minh gain đến từ ngữ nghĩa

**Kỹ thuật**
- [ ] Mỗi cấu hình chính ≥3 seed, mean ± std
- [ ] #params & inference time đã report
- [ ] Code sạch, script reproduce chạy được

**Truy vết**: mỗi con số trong bài phải truy được về một log file cụ thể. QC lập bảng ánh xạ `số trong bài → log path`.

### QC-12 — Fresh-clone reproduce
Clone repo sạch vào thư mục mới → chạy script reproduce → phải ra đúng số của bảng chính.
Đây là test cuối cùng trước khi nộp. Nếu fail, code release sẽ làm reviewer mất niềm tin.

---

## Lịch kiểm tra theo Gate

| Gate | Tuần | QC ký duyệt điều gì |
|---|---|---|
| **Gate 0** | 1 | Batch đã chốt bằng đo thực tế (AI-00), có headroom VRAM; AMP bật không đổi kết quả baseline |
| **Gate 1** | 4 | Baseline **ổn định** (3 seed, std<0.5%); view map 0 unknown; attribute ổn định trong ID; cây manual có bằng chứng VSS **và độ sâu L phù hợp batch** |
| **Gate 2** | 8 | Loss test xanh (đặc biệt cumprod); sampler có cross-view pair; **H-mAP tăng** |
| **Gate 3** | 12 | mAP ≥ baseline + 1.0% trên A→G, ≥3 seed; random tree không hiệu quả; QC-00 xác nhận so sánh công bằng |
| **Gate 4** | 16 | Hyperbolic không NaN; thắng Euclid ở d ≤ 64 (trần d = 384 cho ViT-S) |
| **Final** | 24 | Checklist §14 đầy đủ; fresh-clone reproduce |

## Khi nào phải báo động toàn nhóm
- Giả định "attribute ổn định trong 1 identity" sai.
- Batch không có cặp cross-view (C2 chết).
- Random tree cải thiện ngang cây ngữ nghĩa (đóng góp chỉ là regularization).
- std giữa các seed > gain → gain không có ý nghĩa thống kê.
- Có con số trong bài không truy được về log.

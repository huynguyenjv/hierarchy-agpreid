# Implementation Plan — Topic 1: Hierarchy-Aware AGPReID

**Nguồn**: `.claude/docs/topic1_hierarchy_aware_agpreid_knowledge_base.md`
**Codebase**: `D:\Source\AI\ReID_Advance` (branch `feature/huy`)
**Ngày lập**: 2026-09-06 · **Thời lượng**: 6 tháng / 24 tuần
**Phần cứng**: RTX 3060 12GB, 32GB RAM · torch 2.5.1+cu121 trong `venv/`

> ⚠️ **Dùng `venv/Scripts/python.exe`**, không phải `python` global — bản global là torch CPU-only.

---

## ✅ KẾT QUẢ PHASE 0 (đã chạy 2026-09-07)

### AI-00 — VRAM không phải ràng buộc (lật ngược giả định của plan)

Đo thực tế ViT-S/16, 256×128, K=8, JPM+SIE bật, AMP bật (`tools/profile_memory.py` → `docs/memory_profile.md`):

| batch | P | peak reserved | headroom / 12 GiB |
|---|---|---|---|
| 64 | 8 | 1.92 GiB | 10.1 |
| **128** | **16** | **3.35 GiB** | **8.65** |
| 192 | 24 | 4.77 GiB | 7.2 |
| 256 | 32 | 6.19 GiB | 5.8 |
| 384 | 48 | 9.10 GiB | 2.9 |

**Batch 128 chỉ dùng 3.35 GiB.** Ước lượng 6–8 GiB trước đó quá bảo thủ (tính theo ViT-B 86M thay vì ViT-S 22M). Overhead CV-HWC ở B=128, L=4 chỉ ~0.001 GiB — hoàn toàn không đáng kể.

**Hệ quả — ba ràng buộc trong plan được gỡ bỏ:**
- ✅ Batch 128 (P=16×K=8) chạy thoải mái → **L=4 khả thi**, không phải hạ xuống L=3.
- ✅ Còn dư ~8.6 GiB → **có thể tăng lên batch 256 (P=32)** nếu contrastive cần nhiều negative hơn. Đây là đòn bẩy tự do, dùng khi Gate 2 không đạt.
- ✅ Thứ tự fallback "ưu tiên P hơn K" giữ lại làm dự phòng nhưng **chưa cần dùng**.
- ⚠️ AMP **đã có sẵn** trong `transreid.py` từ trước (tôi nhận định nhầm là thiếu ở bản plan đầu). Chỉ thêm flag `use_amp` (mặc định `True`, giữ nguyên hành vi cũ) để tắt được khi profiling.

### BA-01 — View mapping ✅ verify xong
`C0=aerial, C2=wearable, C3=CCTV`, đối chiếu khớp cả 4 protocol. Toàn dataset chỉ có camera {0,2,3}, **0 ảnh unknown**. Đã refactor ra `reid_advance/hierarchy/view_map.py`. Chi tiết: `docs/view_mapping.md`.

### BA-02/BA-03 — Attribute audit ✅ (`docs/attribute_audit.md`)

**Ba giả định sống còn đều ĐÚNG:**
- Coverage **807/807 identity (100%)** resolve được sang MAT row, 0 ảnh thiếu.
- **0 identity có >1 attribute vector** → giả định "attribute ở mức identity" của KB §2.2 **được xác nhận**. Nhánh cây thật sự view-invariant.
- **807/807 identity (100%) có cả ảnh aerial lẫn ground** → sampler luôn tạo được cặp cross-view. Rủi ro "β_cross không kích hoạt" giảm mạnh.

**Phát hiện quan trọng cho thiết kế cây — cây nháp trong KB §6.2 KHÔNG dùng được nguyên trạng:**

| nhóm | entropy | class lớn nhất | đánh giá |
|---|---|---|---|
| `age` | 0.31 | **95.4%** | ❌ **Loại** — KB đề xuất làm L2, nhưng 768/807 ID cùng một nhóm tuổi. Vô dụng làm tầng |
| `head` | 0.56 | 90.7% | ❌ Loại — 85.1% unknown |
| `beard` / `moustache` | 0.33 / 0.38 | 94% / 92.7% | ❌ Loại — quá lệch |
| `gender` | 1.00 | 53.4% | ✅ Cân bằng đẹp (428/374) |
| `height` | 1.64 | 37.3% | ✅ Ứng viên tốt |
| `hairstyle` | 1.90 | 49.3% | ✅ Entropy cao nhất nhóm khả dụng |
| `lower` | 2.32 | 32.8% | ✅ Cân bằng nhất |
| `upper` | 2.10 | 58.2% | ✅ Tốt |
| `bag` | 1.98 | 47.4% | ⚠️ Cân bằng nhưng VSS khả nghi từ trên cao |

MI cao nhất: `upper × lower` 0.579, `gender × hairstyle` 0.555 → **không xếp hai cặp này thành hai tầng liên tiếp** (KB §6.1 tiêu chí 3).

→ **BA-04 phải thiết kế lại cây**, chờ VSS từ AI-03. Ứng viên thay `age`: `height` hoặc `hairstyle`.

### AI-03 — VSS ⚠️ **KẾT QUẢ BẤT LỢI, CẦN QUYẾT ĐỊNH** (`docs/vss_table.md`)

Setup: 1 ViT-S + 15 head, 12 epoch, train **chỉ trên ground** của 564 ID, test trên 243 ID **hoàn toàn tách biệt** (identity-disjoint — bắt buộc, vì attribute gán ở mức ID nên nếu trùng ID thì model chỉ cần nhận ra người rồi tra nhãn). VSS = **balanced accuracy** trên aerial (không dùng raw accuracy vì nhiều nhóm lệch tới 95%).

| attribute | VSS | ground-bal | retention | lift vs majority | dùng được? |
|---|---|---|---|---|---|
| **`gender`** | **0.742** | 0.836 | 0.89 | **+0.237** | ✅ |
| `moustache` | 0.502 | 0.503 | 1.00 | −0.006 | ❌ |
| `beard` | 0.500 | 0.508 | 0.98 | −0.002 | ❌ |
| `head` | 0.478 | 0.448 | 1.07 | +0.350 | ❌ |
| `lower` | 0.460 | 0.536 | 0.86 | +0.167 | ❌ |
| `hairstyle` | 0.412 | 0.424 | 0.97 | +0.051 | ❌ |
| `age` | 0.365 | 0.491 | 0.74 | −0.019 | ❌ |
| ... 8 nhóm còn lại | ≤ 0.40 | ≤ 0.42 | — | ≤ +0.12 | ❌ |

**Chỉ 1/15 attribute vượt ngưỡng 0.65.** Cây 3 tầng attribute của KB §6.2 không xây được.

**Chẩn đoán quan trọng — không phải lỗi của view:** `retention` (aerial ÷ ground) hầu hết là **0.86–1.07**, tức lên cao gần như không mất thêm gì. Nhưng `ground-bal` cũng chỉ ≤ 0.54 cho mọi nhóm ngoài gender. Nghĩa là các attribute này **khó nhận ngay từ ảnh ground**, chứ không phải bị độ cao phá hủy. Đây là giới hạn của nhãn/dữ liệu (ảnh người nhỏ, nhãn nhiễu), không phải hiện tượng aerial-ground.

> ⚠️ Điều này làm yếu chính giả thuyết nền của đề tài (KB §5.1: "thuộc tính thô sống sót khi bay lên cao"). Với AG-ReID.v2, thứ sống sót gần như chỉ có gender.

**Bốn hướng đi, cần chốt trước khi làm BA-04:**
1. **Cây nông L=2** (root → gender → ID). Trung thực với dữ liệu, nhưng ℓ chỉ nhận 3 giá trị {0,1,2} → tín hiệu phân cấp yếu, và C4 (H-mAP/MS) mất phần lớn độ phân giải.
2. **Cây từ tổ hợp attribute** thay vì từng attribute đơn lẻ — clustering trên vector 15 chiều (variant b). Không cần từng attribute phải nhận diện được; chỉ cần *cụm* có ý nghĩa. **Có thể là hướng cứu bài.**
3. **Feature-driven tree** (variant c) — bỏ hẳn attribute làm nguồn cây.
4. **Đổi khung sang analysis paper** — dùng chính bảng VSS làm đóng góp: *"soft-biometric attributes trong AGPReID không đủ tin cậy để làm hierarchy; đây là bằng chứng định lượng"*. KB §12.2 mục 6 đã dự phòng đường này.

→ **BA-04 bị chặn**, chờ quyết định. Ưu tiên thử hướng (2) trước vì rẻ và giữ nguyên được cấu trúc bài.

### QC-01/QC-02 ✅ — 34 test xanh (23 + 11 cho VSS)
`pytest.ini`, `tests/conftest.py` (seed-lock 42), `tests/test_view_map.py`, `tests/test_data_assumptions.py`. Chạy: `venv/Scripts/python.exe -m pytest -q`.

---

## Ràng buộc phần cứng & quyết định đã chốt

GPU 12GB (KB §9.4 giả định 24GB). **Đã đo thực tế — xem mục trên; VRAM không còn là ràng buộc.**

| Quyết định | Chốt | Lý do |
|---|---|---|
| Backbone chính | **ViT-S/16** (`vit_small_patch16_224.augreg_in21k_ft_in1k`, `embed_dim=384`) | Repo **vốn đã dùng ViT-S** (`TransReIDConfig`); không phải hạ cấp mà là giữ nguyên. ViT-B/16 batch 128 cần ~11GB activations + optimizer state → OOM chắc chắn |
| Batch | **128 view-balanced (P=16×K=8)** — phải nâng từ 32 hiện tại | Contrastive loss phụ thuộc số negative trong batch. Xem "Vì sao không giảm batch" bên dưới |
| AMP | **Bắt buộc bật** | `transreid.py` hiện **chưa dùng AMP**; chỉ `bnneck.py:105` có. Đây là task AI-00 |
| `grad_accum_steps` | **Bỏ cho nhánh hierarchy** | Xem bên dưới |
| Ưu tiên khi cắt scope | **Giữ C4 (metrics) + Ablation B (random tree)** | Rẻ về tính toán (eval-only), là trục thắng dự phòng khi mAP gain nhỏ |

### ⚠️ Vì sao KHÔNG giảm batch, và vì sao gradient accumulation không cứu được

`TransReIDConfig` hiện đặt `batch_size=32, instances_per_identity=4, grad_accum_steps=2` → P=8, K=4.

**Gradient accumulation vô dụng với contrastive loss.** Accumulation cộng gradient qua nhiều micro-batch, nhưng ma trận similarity `s_ij` chỉ tính được trong phạm vi **một** micro-batch. Loss vẫn chỉ "thấy" 32 mẫu. Với triplet/CE thì accumulation tương đương batch lớn; với CV-HWC thì **không**.

Hệ quả nếu giữ P=8, K=4 view-balanced: mỗi ID chỉ 2 aerial + 2 ground, và cây L=4 sẽ đói positive ở tầng trung gian — đúng lỗi KB §12.2 mục 5. Đây là vấn đề **khoa học**, không phải vấn đề tốc độ.

**→ Bắt buộc nâng batch lên 128 cho nhánh hierarchy.** Ước lượng ViT-S, 256×128, 129 token, AMP: ~6–8GB. Khả thi trên 12GB nhưng **phải đo thực tế trước** (AI-00).

Thứ tự fallback nếu 128 vẫn OOM (giảm theo thứ tự này, dừng ngay khi vừa):
1. P=16, K=8 = 128 ← mục tiêu
2. P=16, K=4 = 64 (giữ số ID, giảm ảnh/ID — ưu tiên hơn vì giữ được số negative)
3. P=12, K=8 = 96
4. P=8, K=8 = 64 + **giảm cây xuống L=3**

**Nguyên tắc khi buộc phải giảm: ưu tiên giữ P (số ID) hơn K.** Số negative trong contrastive tỉ lệ với P; K chỉ ảnh hưởng số positive cùng ID.

### Hệ quả lên định vị bài báo

ViT-S cho số tuyệt đối thấp hơn paper gốc (vốn dùng ViT-B). **Không đua SOTA tuyệt đối** — reframe thành *"so sánh công bằng ở cùng backbone và cùng ngân sách tính toán"*: mọi baseline (TransReID, VDT nếu chạy được) đều chạy lại trên ViT-S. Đây là cách trình bày hợp lệ và phổ biến.

Ba trục **không bị ảnh hưởng** bởi backbone nhỏ, và đó chính là lý do ưu tiên giữ chúng:
- **C4 (H-mAP, AC@k, MS)** — đo chất lượng ngữ nghĩa của lỗi, độc lập với sức mạnh backbone.
- **Ablation B (random tree control)** — câu hỏi khoa học, không phải cuộc đua số.
- **Ablation E (d thấp)** — GPU yếu và embedding chiều thấp là *cùng một câu chuyện*: ReID trên edge device/drone. Ràng buộc phần cứng ở đây biến thành một lập luận có lợi.

---

## 0. Đánh giá hiện trạng codebase (đã khảo sát thực tế)

Repo đã có sẵn khá nhiều mảnh ghép cần thiết.

| Thành phần cần cho đề tài | Trạng thái | File |
|---|---|---|
| Attribute schema 15 nhóm + parser `.mat` | ✅ Có sẵn | `reid_advance/attributes.py`, `data.py:86 load_track_attributes` |
| TransReID baseline (SIE + PK sampler) | ✅ Có sẵn | `reid_advance/pipelines/transreid.py` |
| PK sampler (P×K) | ⚠️ Có, **chưa view-balanced** | `data.py:330 RandomIdentitySampler` |
| Eval mAP/CMC | ✅ Có sẵn | `evaluation.py:60 evaluate_rank` |
| Camera id parsing | ✅ Có (regex `C\d+F`) | `data.py:64 parse_camera_id` |
| **View label (aerial/ground)** | ✅ **ĐÃ CÓ** — `C0=aerial, C2=wearable, C3=CCTV` | `pipelines/transreid.py:31 camera_to_view` |
| Backbone ViT-S + SIE + JPM | ✅ Có sẵn | `config.py:116`, `pipelines/transreid.py` |
| AMP trong pipeline TransReID | ❌ **Chưa có** (chỉ `bnneck.py:105` có) | — |
| **Attribute trả về trong `__getitem__`** | ❌ `TransReIDDataset` chỉ trả `(img, pid, camid)` | `data.py:195` |
| Semantic tree / `tree_paths` | ❌ Chưa có | — |
| CV-HWC loss | ❌ Chưa có (`losses.py` chỉ có Triplet + attr loss) | — |
| Hyperbolic head | ❌ Chưa có (chưa cài `geoopt`) | — |
| H-mAP / AC@k / MS | ❌ Chưa có | — |
| CARGO dataset + VDT | ❌ Chưa có | — |

**Ba rủi ro kỹ thuật cần chốt sớm (Week 1–2):**

1. **Batch hiện là 32 (P=8×K=4), cần nâng lên 128** — xem mục ràng buộc phần cứng. Rủi ro cao nhất còn lại, vì nó vừa là ràng buộc VRAM vừa quyết định loss có hoạt động hay không.
2. `RandomIdentitySampler` không đảm bảo mỗi ID có cả 2 view trong batch → `β_cross` sẽ không bao giờ kích hoạt. KB §7.3 cảnh báo đây là lỗi phổ biến số 1.
3. `load_track_attributes` chỉ đọc split `train` của MAT file — cần xác nhận có phủ hết ID train không, và `track_key()` (`P0000T02140A0` → `0000021400`) map đúng không.

> ✅ **Rủi ro "chưa có view label" đã được gỡ bỏ.** `pipelines/transreid.py:31` đã có `camera_to_view()` với mapping `C0=aerial, C2=wearable, C3=CCTV`. BA-01 chuyển từ "xây từ đầu" thành "verify + refactor ra module dùng chung". Tiết kiệm ~1 tuần trên critical path.

---

## 1. Kiến trúc module đề xuất (file mới)

```
reid_advance/
├── hierarchy/                       ← MỚI, toàn bộ đóng góp nằm ở đây
│   ├── __init__.py
│   ├── view_map.py                  # camera_id → {A, G}; VSS computation
│   ├── tree.py                      # build tree (manual / cluster / feature / random)
│   ├── tree_paths.py                # id → path tensor (B, L)
│   ├── losses.py                    # cv_hwc_loss, level-aware prototype loss
│   ├── hyperbolic.py                # PoincareBall wrapper (geoopt), exp_map, clip
│   └── metrics.py                   # H-mAP, AC@k, Mistake Severity
├── data.py                          # +ViewBalancedPKSampler, +HierarchyDataset
├── pipelines/
│   └── hierarchy_reid.py            # MỚI: pipeline chính (TransReID + CV-HWC)
└── ...
tools/
├── compute_vss.py                   # MỚI: sinh Table 1 của bài báo
├── build_tree.py                    # MỚI: sinh + dump cây ra JSON
└── plot_poincare.py                 # MỚI: Fig. 1 / Fig. 4
tests/                               ← MỚI
├── test_data_assumptions.py
├── test_tree_paths.py
├── test_cv_hwc_loss.py
├── test_view_sampler.py
├── test_hyperbolic.py
└── test_hierarchy_metrics.py
```

Entry point mới: `python run.py hierarchy --tree manual --geometry euclid`

Nguyên tắc: **không sửa phá vỡ** `transreid.py` — baseline phải chạy được nguyên trạng suốt dự án để so sánh công bằng.

---

## 2. Ba vai trò

Ba vai trò chạy **song song có checkpoint đồng bộ**, không tuần tự.

### 🧭 BA (Business / Research Analyst)
Sở hữu: dữ liệu, cây ngữ nghĩa, định vị nghiên cứu, metrics spec, viết bài.
Không viết code training; có viết code phân tích dữ liệu và visualization.

### 🤖 AI-ENGINEER
Sở hữu: loss, sampler, model head, pipeline training, chạy thí nghiệm.

### 🔍 QC (Quality Control)
Sở hữu: unit test, reproducibility, kiểm chứng số liệu, chống lỗi thầm lặng.
**Quyền phủ quyết**: không con số nào được đưa vào bảng bài báo nếu QC chưa ký.

---

## 3. Backlog chi tiết

### PHASE 0 — Foundation (Week 1–4)

| ID | Vai trò | Task | Deliverable | Definition of Done |
|---|---|---|---|---|
| **AI-00** | AI | **Memory profiling (làm trước tất cả)**. Bật AMP cho `transreid.py`; đo VRAM thực tế ở batch 32/64/96/128 với ViT-S 256×128 | Bảng VRAM vs batch | Chốt được batch lớn nhất chạy ổn định trên 12GB. Nếu 128 không vừa, áp thứ tự fallback (ưu tiên giữ P hơn K) |
| **BA-01** | BA | **Verify** `camera_to_view()` (`transreid.py:31`, `C0=aerial/C2=wearable/C3=CCTV`) rồi refactor ra `hierarchy/view_map.py` dùng chung | `docs/view_mapping.md` + `reid_advance/hierarchy/view_map.py` | Đối chiếu với 4 file `exp*.txt` xác nhận mapping đúng; mọi ảnh `train_all`/`query`/`gallery` có view; 0 `unknown`. Chốt quy ước binary: ground = {cctv, wearable} |
| **BA-02** | BA | Audit `qut_attribute_v8.mat`: in 15 nhóm, số class, phân bố, tỉ lệ `unknown` mỗi nhóm, độ phủ ID train | `docs/attribute_audit.md` | Xác nhận attribute ở **mức identity** (mọi ảnh cùng ID có cùng vector). Nếu sai giả định này → báo động đỏ ngay |
| **BA-03** | BA | Tính entropy mỗi nhóm + mutual information giữa 15 nhóm | Bảng MI 15×15 | Loại nhóm entropy quá thấp, cặp MI quá cao |
| **AI-01** | AI | Reproduce TransReID-S baseline trên AG-ReID.v2, exp1 (A→CCTV) + exp4 (CCTV→A), ở **batch đã chốt từ AI-00** | Log + checkpoint | Baseline ổn định, 3 seed std < 0.5% mAP. Đây là **baseline nội bộ trên ViT-S**, không kỳ vọng khớp số ViT-B của paper gốc — ghi rõ điều này trong bài |
| **AI-02** | AI | Mở rộng dataset → trả `(img, pid, camid, view, attr_vec)` | patch `data.py` | Pipeline baseline hiện có vẫn chạy không đổi |
| **AI-03** | AI | Train 15 attribute classifier riêng, đo acc trên aerial vs ground → **View Stability Score** | `tools/compute_vss.py` + VSS table | Có acc(aerial) và acc(ground) cho từng attribute |
| **QC-01** | QC | Dựng `pytest` scaffold + script CI; seed-lock toàn repo | `tests/`, `pytest.ini` | `pytest` xanh; baseline 3 seed cho std < 0.5% mAP |
| **QC-02** | QC | Test kiểm chứng giả định dữ liệu | `tests/test_data_assumptions.py` | Fail nếu bất kỳ ID nào có 2 vector attribute khác nhau, hoặc ảnh nào không có view |
| **BA-04** | BA | Từ VSS → thiết kế cây manual (variant a), ghi rõ lý do thứ tự tầng | `docs/tree_design.md` + `trees/manual.json` | Mọi tầng attribute có VSS ≥ 65%, entropy hợp lý, mỗi nhánh ≥ 20 ID |

> **Gate 1 (cuối Week 4)** — baseline khớp paper + view map đúng + cây manual có bằng chứng VSS. QC ký duyệt mới sang Phase 1.

---

### PHASE 1 — Core Loss (Week 5–8)

| ID | Vai trò | Task | Deliverable | DoD |
|---|---|---|---|---|
| **AI-04** | AI | `ViewBalancedPKSampler`: P ID × (K/2 aerial + K/2 ground), có fallback + log tỉ lệ | `data.py` | P=16, K=8; log tỉ lệ cross-view pair mỗi batch |
| **QC-03** | QC | Test sampler đo trực tiếp trên batch thật | `tests/test_view_sampler.py` | Đo trực tiếp tỉ lệ ID có đủ 2 view — **không tin log của AI** |
| **AI-05** | AI | `tree_paths.py`: identity → tensor `(L,)`; loader trả `(B, L)` | `hierarchy/tree_paths.py` | Path consistency: node con luôn kéo theo đúng node cha |
| **AI-06** | AI | `cv_hwc_loss` theo pseudo-code KB §8.2 | `hierarchy/losses.py` | Hỗ trợ `geometry ∈ {euclid, hyperbolic}` |
| **QC-04** | QC | Unit test loss trên cây giả + embedding giả | `tests/test_cv_hwc_loss.py` | Kiểm: (1) `ell` dùng **cumprod** không phải sum — hai mẫu khác Gender nhưng tình cờ cùng màu áo phải cho `ell=0`; (2) `w = λ^(L−ell)` giảm đúng theo tầng; (3) `β=1` khi cùng view, `β=β_cross` khi khác view; (4) λ→0 hội tụ về SupCon; (5) gradient finite, không NaN |
| **AI-07** | AI | Pipeline `hierarchy_reid.py`, loss tổng `L_ID + L_tri + α·L_CV-HWC` | `pipelines/hierarchy_reid.py` | `python run.py hierarchy` chạy end-to-end |
| **AI-08** | AI | Train HWC-only (β=1) so với baseline | Log + bảng | So sánh mAP, R-1, H-mAP |
| **BA-05** | BA | Định nghĩa + implement H-mAP, AC@k, Mistake Severity (C4) | `hierarchy/metrics.py` + `docs/metrics_spec.md` | Công thức viết ra spec trước khi code |
| **QC-05** | QC | Test metrics trên case biên tính tay được | `tests/test_hierarchy_metrics.py` | H-mAP với `L=1` phải bằng mAP thường; MS=0 khi rank-1 đúng hết |

> **Gate 2 (cuối Week 8)** — **H-mAP phải tăng**, kể cả nếu mAP chưa tăng. Nếu H-mAP không tăng → loss hoặc cây sai; chạy phác đồ debug KB §12.2 theo đúng thứ tự.

---

### PHASE 2 — Cross-View + Ablation (Week 9–12)

| ID | Vai trò | Task | DoD |
|---|---|---|---|
| **AI-09** | AI | Bật β_cross; grid tune λ∈{0.1,0.3,0.5,0.7,0.9}, β∈{1,1.5,2,3}, α∈{0.5,1,2} | Ablation D |
| **AI-10** | AI | Ablation A — từng thành phần cộng dồn | Bảng 4 dòng |
| **BA-06** | BA | Xây cây variant (b) hierarchical clustering trên vector attribute + (c) feature clustering trên ground feature + **(d) random tree** | `trees/*.json` |
| **AI-11** | AI | Ablation B (4 loại cây) + Ablation C (L = 2,3,4,5) | 2 bảng |
| **QC-06** | QC | **Kiểm chứng random tree KHÔNG hiệu quả**. Nếu random tree cũng cải thiện → báo động: đóng góp chỉ là regularization, không phải hierarchy | Chạy random tree ≥3 seed |
| **QC-07** | QC | Xác nhận không tụt trên protocol view-homogeneous (A→A, G→G) | Bảng phụ |

> **Gate 3 (cuối Week 12)** — mAP vượt baseline ≥ +1.0% trên A→G. Nếu không → phác đồ debug §12.2; cân nhắc đổi khung sang bài analysis (KB §12.2 mục 6).

---

### PHASE 3 — Hyperbolic (Week 13–16)

| ID | Vai trò | Task | DoD |
|---|---|---|---|
| **AI-12** | AI | Cài `geoopt`; `hyperbolic.py` với `PoincareBall`, exp_map, **clip norm ≤ 1−1e-5**, Riemannian Adam | Không NaN sau 120 epoch |
| **QC-08** | QC | Numerical stability test: điểm gần biên ball, float32, gradient finite | `tests/test_hyperbolic.py` |
| **AI-13** | AI | Pre-embed prototype cây vào Poincaré ball; `L_proto` với γ tăng dần theo depth; entailment cone constraint | Đo được distortion của tree embedding |
| **AI-14** | AI | Ablation E: d ∈ {32,64,128,256,768}, Euclid vs Hyperbolic | Figure đường cong mAP theo d |
| **BA-07** | BA | Poincaré disk visualization (Fig. 1 teaser + Fig. 4) | `tools/plot_poincare.py` |

> **Gate 4** — hyperbolic thắng Euclid rõ rệt ở d ≤ 64. Nếu không, hạ C3 xuống ablation phụ, không bỏ bài.

---

### PHASE 4 — Scale-out & thí nghiệm sinh tử (Week 17–20)

| ID | Vai trò | Task | DoD |
|---|---|---|---|
| **BA-08** | BA | Tải CARGO (repo `LinlyAC/VDT-AGPReID`); pseudo-attribute bằng CLIP hoặc PAR model, majority vote ở mức ID | Cây CARGO + doc phương pháp (→ contribution C5) |
| **AI-15** | AI | Train + eval trên CARGO A→G, G→A | Cột CARGO của bảng chính |
| **AI-16** | AI | **THÍ NGHIỆM SINH TỬ**: clone VDT repo, gắn CV-HWC vào → `VDT + CV-HWC` vs `VDT` | Dòng quyết định số phận bài báo |
| **AI-17** | AI | AG-ReID.v1 + cross-dataset generalization (train v2 → test CARGO) | Bảng phụ |
| **QC-09** | QC | Mọi cấu hình chính chạy **≥3 seed**, report mean±std | Không con số nào trong bài là 1-seed |
| **QC-10** | QC | Đo #params + inference time so với VDT/TransReID | Chứng minh không nặng hơn (VDT nhấn mạnh điểm này) |
| **BA-09** | BA | Qualitative retrieval figure (Fig. 5): baseline sai "thô bạo" vs ta sai "lịch sự" | Fig. 5 |

---

### PHASE 5 — Viết bài (Week 21–24)

| ID | Vai trò | Task |
|---|---|---|
| **BA-10** | BA | Viết Method + Experiments (phần dễ, viết trước) |
| **BA-11** | BA | Viết Intro + Related Work + Abstract (viết cuối) |
| **BA-12** | BA | Chuẩn bị bảng rebuttal theo KB §12.1 |
| **AI-18** | AI | Dọn code, README, script reproduce, chuẩn bị release |
| **QC-11** | QC | Chạy full checklist KB §14; verify mọi con số trong bài truy được về log gốc |
| **QC-12** | QC | Fresh-clone reproduce: clone sạch → chạy script → ra đúng số bảng chính |

---

## 4. Ma trận phụ thuộc (critical path)

```
AI-00 (batch/VRAM) ─┬─→ AI-01 (baseline) ────────────────────→ mọi so sánh
                    ├─→ AI-04 (sampler) ─→ AI-09 (β_cross) ──→ Gate 3
                    └─→ BA-04 (chốt độ sâu cây L) ─→ AI-05 ─→ AI-06 ─→ Gate 2
BA-01 (verify view map) ─→ AI-02, AI-04, QC-02
BA-02 (attr audit) ─→ BA-03 ─→ BA-04
AI-03 (VSS) ────────────────→ BA-04
                                 AI-06 ─→ AI-12 (hyperbolic) ─→ Gate 4
                                 AI-06 ─→ AI-16 (VDT + CV-HWC) ← quyết định số phận
```

**AI-00 là blocker số 1** (thay cho BA-01 ở bản trước). Batch khả dụng quyết định độ sâu cây tối đa, mà cây là nền của mọi thứ phía sau. Đo trước, thiết kế sau.

---

## 5. Rủi ro & giảm thiểu

| Rủi ro | Xác suất | Tác động | Giảm thiểu |
|---|---|---|---|
| **Batch 128 không vừa 12GB → phải giảm → cây đói positive** | **Cao** | **Cao** | AI-00 đo trước tiên. Fallback ưu tiên giữ P hơn K. Nếu buộc về batch 64, **giảm cây xuống L=3** (đừng cố giữ L=4) |
| Batch không đủ cross-view pair | **Cao** | Cao | AI-04 + QC-03 đo trực tiếp trên batch, không tin log |
| Attribute không ổn định trong 1 ID | Thấp | Chí mạng | QC-02 test tự động; nếu sai → majority vote ở mức ID |
| Hyperbolic NaN / instability | Cao | Trung bình | Clip norm, float32, QC-08; sẵn sàng hạ C3 xuống ablation |
| `VDT + CV-HWC` không cho gain | Trung bình | Cao | Vẫn còn C1/C2/C4; đổi framing sang analysis paper |
| mAP gain < 1% | Trung bình | Cao | C4 metrics + Ablation E (d thấp) là trục thắng dự phòng |
| VDT (fast-reid) không chạy nổi trên 12GB | Trung bình | Cao | Chạy VDT ở ViT-S + batch đã chốt; nếu vẫn không nổi, so gián tiếp và ghi rõ trong Limitations |
| Thời gian train dài do GPU đơn 12GB | **Cao** | Trung bình | Ưu tiên C4 + Ablation B (eval-only, rẻ). Cắt Ablation D xuống grid thưa hơn nếu cần |
| ~~Không map được camera → view~~ | — | — | ✅ **Đã gỡ** — `transreid.py:31` đã có |

---

## 6. Việc cần làm ngay (Week 1, ngày 1–3)

1. **AI-00 (trước tất cả)** — bật AMP cho `transreid.py`, đo VRAM ở batch 32/64/96/128 trên ViT-S. **Chốt batch trước khi ai làm gì khác**, vì nó quyết định độ sâu cây tối đa và do đó quyết định thiết kế của BA-04.
2. **BA-01** — verify `camera_to_view()` (`transreid.py:31`) đối chiếu 4 file `exp*.txt`, refactor ra `hierarchy/view_map.py`.
3. **BA-02** — mở rộng `tools/inspect_attributes.py` → audit `qut_attribute_v8.mat`.
4. **AI-01** — khởi động baseline TransReID-S ở batch đã chốt (chạy nền, mất vài ngày).
5. **QC-01** — dựng `tests/`, seed-lock.

> **AI-00 chặn BA-04**: nếu batch cuối cùng chỉ đạt 64, cây phải thiết kế L=3 chứ không phải L=4. BA đừng chốt cây trước khi có kết quả AI-00.

---

## 7. Prompt bàn giao cho agent

Ba file prompt riêng ở `.claude/docs/agents/` — mỗi agent đọc: KB + plan này + prompt vai trò của mình.

- `.claude/docs/agents/ba.md`
- `.claude/docs/agents/ai_engineer.md`
- `.claude/docs/agents/qc.md`

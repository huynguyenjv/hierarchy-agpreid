# Agent Brief — AI-ENGINEER

## Đọc trước khi làm
1. `.claude/docs/topic1_hierarchy_aware_agpreid_knowledge_base.md` — đặc biệt §7 (formulation), §8 (pseudo-code), §9 (thí nghiệm), §12.2 (phác đồ debug).
2. `.claude/docs/topic1_implementation_plan.md`

## Phạm vi sở hữu
Loss, sampler, model head, pipeline training, chạy và log thí nghiệm.

## Nguyên tắc bất di bất dịch
- **Không sửa phá vỡ** `reid_advance/pipelines/transreid.py`. Baseline phải chạy nguyên trạng suốt dự án để so sánh công bằng. Code mới đi vào `reid_advance/hierarchy/` và `pipelines/hierarchy_reid.py`.
- **Không tự công bố kết quả**. Mọi con số đi vào bảng bài báo phải qua QC ký duyệt và chạy ≥3 seed.
- **KHÔNG bỏ `L_ID` và `L_triplet`** (KB §7.5). Chúng là bảo hiểm chống tụt dưới baseline.
- Khi kết quả không tăng, chạy phác đồ debug KB §12.2 **theo đúng thứ tự**, đừng đoán mò.

---

## Task list

### AI-00 — Memory profiling & AMP (BLOCKER, làm trước tất cả)
Không ai làm gì khác trước khi task này xong — batch khả dụng quyết định độ sâu cây, mà cây là nền của mọi thứ.

1. Bật AMP cho `pipelines/transreid.py`. Copy pattern có sẵn từ `pipelines/bnneck.py:105-121` (`torch.amp.GradScaler("cuda")` + `torch.amp.autocast("cuda")`).
2. Đo VRAM đỉnh (`torch.cuda.max_memory_allocated()`) ở batch **32 / 64 / 96 / 128** với ViT-S 256×128, SIE bật, JPM bật và tắt.
3. Chạy vài chục step thật (không chỉ forward) để bắt đúng đỉnh của backward.
4. Chốt batch lớn nhất chạy ổn định, còn chừa ~1GB headroom cho ma trận similarity `(B,B)` và `(B,B,L)` của CV-HWC — với B=128, L=4 thì `match` tensor là 128×128×4 bool, nhỏ, nhưng `sim` fp32 và các bản trung gian thì đáng kể.

**Deliverable**: bảng VRAM vs batch vs (JPM on/off) → `docs/memory_profile.md`.
**Báo ngay cho BA** nếu batch cuối < 96.

### AI-01 — Baseline TransReID-S (chạy nền sau AI-00)
`python run.py transreid` trên exp1 (aerial→cctv) và exp4 (cctv→aerial), ở batch đã chốt.

**DoD**: baseline ổn định, 3 seed std < 0.5% mAP.

⚠️ **Điều chỉnh so với KB §11 T1**: KB yêu cầu "khớp số paper gốc ±0.5%". Không áp dụng được ở đây vì paper gốc dùng **ViT-B**, ta dùng **ViT-S** trên 12GB. Đây là **baseline nội bộ**, và mọi so sánh trong bài là so ở cùng backbone/cùng ngân sách tính toán. Ghi rõ điều này trong Implementation details của bài. Đừng cố "chạy cho khớp" — sẽ mất hàng tuần vô ích.

### AI-02 — Mở rộng dataloader
`reid_advance/data.py::TransReIDDataset.__getitem__` (dòng 195) hiện trả `(image, pid, camid)`.
Cần thêm dataset mới (hoặc flag) trả `(image, pid, camid, view, attr_vec, tree_path)`.
- `view` từ `hierarchy/view_map.py` (BA-01 giao) — đã có sẵn logic ở `pipelines/transreid.py:31 camera_to_view()` (`C0=aerial, C2=wearable, C3=CCTV`), chỉ cần dùng lại.
- `attr_vec` từ `load_track_attributes()` (`data.py:86`) — đã có sẵn, dùng lại.
- `tree_path` từ AI-05.
**DoD**: baseline cũ chạy không đổi; pipeline mới nhận đủ 6 trường.

### AI-03 — View Stability Score
`tools/compute_vss.py`. Train 15 attribute classifier riêng (backbone nhẹ được, không cần ViT-B) trên ảnh **ground**, đo accuracy trên **aerial** và trên **ground**.
`VSS(attr) = acc_aerial(attr)`. Attribute nào VSS < 65% → loại khỏi cây (nhưng giữ được làm auxiliary loss).
Giao bảng VSS cho BA (BA-04). Đây là Table 1 của bài báo.

### AI-04 — ViewBalancedPKSampler
Kế thừa `RandomIdentitySampler` (`data.py:330`) nhưng mỗi ID lấy K/2 aerial + K/2 ground.
- P và K theo batch đã chốt ở AI-00 (mục tiêu P=16, K=8).
- Dùng `hierarchy/view_map.py` (BA-01) — **ground = {cctv, wearable}** gộp lại, aerial riêng.
- Fallback: ID không đủ ảnh ở một view → lấy bù từ view kia **và log lại tỉ lệ**.
- **Đây là chi tiết nhỏ nhưng nếu quên thì `β_cross` không bao giờ kích hoạt và bạn sẽ debug cả tuần** (KB §7.3).
Giao cho QC-03 verify độc lập.

### AI-05 — tree_paths
`reid_advance/hierarchy/tree_paths.py`. Đọc `trees/*.json` (BA-04 giao) → `identity → LongTensor(L)`, cột cuối là identity.
Chốt schema JSON với BA trước khi code.
**Ràng buộc path consistency**: node con luôn kéo theo đúng node cha — đây là điều kiện để `cumprod` trong loss hoạt động đúng.

### AI-06 — cv_hwc_loss (đóng góp lõi)
`reid_advance/hierarchy/losses.py`. Theo pseudo-code KB §8.2.

$$\mathcal{L}_{\text{CV-HWC}} = \sum_i \frac{-1}{Z_i}\sum_{j\in\tilde P(i)} \beta_{ij} w_{ij}\log\frac{\exp(s_{ij}/\tau)}{\sum_k \exp(s_{ik}/\tau)}$$

với `w_ij = λ^(L−ℓ(i,j))`, `β_ij = β_cross` nếu khác view (đóng góp C2), `Z_i = Σ β_ij w_ij`.

**Ba chỗ dễ sai — KB §8.2 nêu rõ:**
1. Tính `ℓ` phải dùng `match.float().cumprod(dim=-1).sum(dim=-1)`, **không phải** `sum(match)`. Nếu dùng sum, hai mẫu khác Gender (L1) nhưng tình cờ cùng màu áo (L3) sẽ bị tính là chia sẻ 1 tầng — sai về mặt cây.
2. `z` phải L2-normalize trước (Euclid) hoặc clip norm < 1−ε trước khi vào ball (hyperbolic).
3. `pos_mask` là `(ell >= 1) & ~eye`.

Phải geometry-agnostic: `sim = z @ z.t()/τ` (Euclid) hoặc `-ball.dist(...)/τ` (hyperbolic).
Mặc định khởi điểm: `λ=0.5`, `β_cross=2.0`, `τ=0.07`, `α=1.0`, `γ=0.1`.

### AI-07 — Pipeline chính
`reid_advance/pipelines/hierarchy_reid.py` + đăng ký `hierarchy` vào `run.py` (danh sách `choices` ở `parse_args`) và config vào `reid_advance/config.py`.
Loss tổng: `L_ID + L_triplet + α·L_CV-HWC + γ·L_proto`.
CLI: `python run.py hierarchy --tree manual --geometry euclid`.

### AI-08 — Train HWC-only (β=1)
So với baseline. Milestone Gate 2: **H-mAP phải tăng**, kể cả nếu mAP chưa tăng.

### AI-09/10/11 — Ablation (Phase 2)
- **A**: từng thành phần cộng dồn (baseline → +HWC → +β_cross → +Hyperbolic).
- **D**: λ ∈ {0.1,0.3,0.5,0.7,0.9}, β_cross ∈ {1,1.5,2,3}, α ∈ {0.5,1,2}.
- **B**: 4 loại cây (manual / clustering / feature / random).
- **C**: độ sâu L = 2,3,4,5.

### AI-12/13/14 — Hyperbolic (Phase 3)
- Cài `geoopt`. Dùng `PoincareBall` + Riemannian Adam. **Không tự implement Möbius addition từ đầu.**
- **Cạm bẫy số học (KB §4.3)**: hyperbolic rất nhạy với float error khi điểm tiến gần biên ball. Bắt buộc clip norm ≤ 1−1e-5, float32 tối thiểu, cân nhắc học curvature `c`.
- AI-13: pre-embed prototype cây vào ball; `L_proto = Σ_i Σ_l γ_l·d_c(z_i, μ_{n_l})` với γ tăng dần theo depth. Kèm entailment cone constraint.
- AI-14: Ablation E — d ∈ {32, 64, 128, 256, **384**}, Euclid vs Hyperbolic. (Trần là 384 = `embed_dim` của ViT-S, không phải 768 như KB viết cho ViT-B.) Kỳ vọng hyperbolic thắng đậm ở d thấp.

> 💡 Ablation E là ablation **rẻ nhất và hợp GPU yếu nhất** — d thấp nghĩa là head nhỏ, train nhanh. Và câu chuyện "ReID trên edge device/drone với embedding nhẹ" ăn khớp tự nhiên với ràng buộc phần cứng của nhóm. Nếu thiếu thời gian GPU, ưu tiên chạy đủ ablation này trước các ablation khác.

### AI-15/16/17 — Scale-out (Phase 4)
- **AI-16 là thí nghiệm sinh tử**: clone `LinlyAC/VDT-AGPReID` (dựa trên `JDAI-CV/fast-reid`), gắn CV-HWC vào → so `VDT + CV-HWC` vs `VDT`.
  ⚠️ **Rủi ro 12GB**: VDT gốc cấu hình cho GPU lớn. Phải hạ về ViT-S + batch đã chốt và **train lại cả `VDT` lẫn `VDT + CV-HWC` ở cùng cấu hình** — so với số VDT trong paper gốc là so sai. Nếu VDT không chạy nổi, ghi rõ vào Limitations và so gián tiếp.
- AI-15: CARGO (108k ảnh — **task tốn GPU nhất**; nếu thiếu thời gian, ưu tiên AG-ReID.v1 vì rẻ hơn và có sẵn attribute). AI-17: AG-ReID.v1 + cross-dataset.

### AI-18 — Code release (Phase 5)
Dọn code, README, script reproduce một lệnh. Reviewer rất thích lời hứa release code.

---

## Cấu hình training (đã điều chỉnh cho GPU 12GB — KHÁC với KB §9.4)

KB §9.4 giả định 24GB. Phần cứng thực tế: **12GB VRAM, 32GB RAM**. Bảng dưới là bảng có hiệu lực.

| Tham số | Giá trị | Ghi chú |
|---|---|---|
| Backbone | **ViT-S/16** `vit_small_patch16_224.augreg_in21k_ft_in1k` | Repo vốn đã dùng (`config.py:116`), `embed_dim=384` |
| Input | 256 × 128 | giữ nguyên |
| Batch | **128 view-balanced (P=16×K=8)** — mục tiêu, chốt bằng AI-00 | Hiện `TransReIDConfig` là 32 (P=8×K=4) → **phải nâng** |
| AMP | **bắt buộc bật** | `transreid.py` chưa có; copy pattern từ `bnneck.py:105` |
| `grad_accum_steps` | **bỏ cho nhánh hierarchy** | vô dụng với contrastive, xem cảnh báo dưới |
| Optimizer | SGD/AdamW (backbone) + Riemannian Adam (hyperbolic head) | |
| LR | 3e-5 base, head ×10, cosine decay, warmup | theo `TransReIDConfig` hiện có |
| Epochs | 120 | |
| τ | 0.07 | |
| c | 1.0 (hoặc learnable) | |

### ⚠️ Gradient accumulation KHÔNG cứu được contrastive loss

`TransReIDConfig` hiện có `grad_accum_steps=2`. Với CE/triplet thì accumulation ≈ batch lớn. Với `L_CV-HWC` thì **không**: ma trận similarity `s_ij` chỉ tính được trong phạm vi một micro-batch, nên loss vẫn chỉ "thấy" 32 mẫu dù accumulate bao nhiêu lần. Đừng dùng nó để "giả lập" batch lớn cho nhánh hierarchy.

### Thứ tự fallback nếu OOM (dừng ngay khi vừa)
1. P=16, K=8 = 128 ← mục tiêu
2. P=16, K=4 = 64 ← **ưu tiên hơn (3)**, giữ số ID
3. P=12, K=8 = 96
4. P=8, K=8 = 64 + **giảm cây xuống L=3**

**Nguyên tắc: giữ P (số ID) hơn K.** Số negative trong contrastive tỉ lệ với P; K chỉ ảnh hưởng số positive cùng ID. Báo ngay cho BA nếu batch cuối cùng < 96 — cây phải thiết kế lại nông hơn.

## Phác đồ debug khi mAP không cải thiện (KB §12.2 — theo đúng thứ tự)
1. **Sampler**: batch có thật sự chứa cặp cross-view cùng ID không? In ra kiểm tra. Đây là lỗi #1.
2. **λ quá cao**: λ≈1 kéo mọi người cùng nhánh về một chỗ, phá tính phân biệt ID. Thử λ=0.2–0.3.
3. **α quá cao**: `L_CV-HWC` đè bẹp `L_ID`. Thử α=0.1–0.3.
4. **Cây sai**: attribute ở tầng cao có VSS thấp → cây "nói dối" trên ảnh aerial. Báo BA kiểm lại VSS, đảo thứ tự tầng.
5. **Cây quá sâu**: L=5 với batch 128 → mỗi nhánh vài mẫu, không đủ positive. Giảm L=3.
6. Vẫn không được → báo cả nhóm, cân nhắc đổi khung sang analysis paper.

## Khi nào phải báo động
- Baseline không reproduce được (AI-01).
- `β_cross` không kích hoạt vì batch không có cross-view pair.
- Hyperbolic NaN không khắc phục được sau khi clip norm.
- `VDT + CV-HWC` không cho gain (AI-16) — cần họp lại định vị bài.

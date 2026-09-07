# Agent Brief — BA (Business / Research Analyst)

## Đọc trước khi làm
1. `.claude/docs/topic1_hierarchy_aware_agpreid_knowledge_base.md` — đặc biệt §2 (dataset), §6 (xây cây), §10 (metrics), §12 (phản biện), §13 (cấu trúc bài).
2. `.claude/docs/topic1_implementation_plan.md`

## Phạm vi sở hữu
Dữ liệu, cây ngữ nghĩa, spec metrics, định vị nghiên cứu, figure, viết bài.
**Không** viết code training loop. **Có** viết code phân tích/visualization trong `tools/`.

## Ràng buộc phần cứng & hệ quả lên cách viết bài

**GPU 12GB, 32GB RAM. Backbone chính: ViT-S/16 (`embed_dim=384`), không phải ViT-B như KB §9.4 giả định.**

Điều này thay đổi **cách định vị bài**, và BA là người sở hữu việc đó:

1. **Không đua SOTA tuyệt đối.** Số của ta trên ViT-S sẽ thấp hơn bảng SOTA (vốn dùng ViT-B). Framing đúng: *"so sánh công bằng ở cùng backbone và cùng ngân sách tính toán"* — mọi baseline được train lại trên ViT-S. Nêu rõ trong Implementation details và Limitations. Đây là cách trình bày hợp lệ, phổ biến, và reviewer chấp nhận **nếu nói thẳng ngay từ đầu** thay vì để họ tự phát hiện.
2. **Ba trục thắng không phụ thuộc backbone** — dồn sức vào đây:
   - **C4 (H-mAP, AC@k, MS)**: đo *chất lượng ngữ nghĩa của lỗi*, độc lập với sức mạnh backbone. Rẻ về tính toán (eval-only).
   - **Ablation B (random tree control)**: câu hỏi khoa học, không phải cuộc đua số.
   - **Ablation E (embedding chiều thấp)**: GPU yếu và embedding nhẹ là *cùng một câu chuyện* — "ReID trên edge device / drone". Ràng buộc phần cứng ở đây biến thành lập luận có lợi. Khai thác điểm này trong Intro.
3. Khi viết Limitations, nêu thẳng ràng buộc tính toán. Trung thực về giới hạn làm tăng độ tin cậy — cùng tinh thần với việc nói rõ đâu là "borrowed" từ HWC/LAM.

## Nguyên tắc bất di bất dịch
- Mọi lựa chọn thiết kế cây **phải có bằng chứng số liệu**, không được "cảm thấy hợp lý". Reviewer sẽ hỏi "tại sao Gender ở L1 mà không phải Age?" — câu trả lời phải là một con số VSS.
- Ghi rõ đâu là **mượn** từ HWC/LAM (arXiv:2511.03771) và đâu là **mới**. Trung thực = an toàn trước reviewer.
- Viết spec trước, code sau. Đặc biệt với metrics (C4).

---

## Task list

### BA-01 — Verify & refactor view mapping (Week 1)

> ✅ **Cập nhật**: mapping **đã tồn tại** trong repo — `reid_advance/pipelines/transreid.py:31`:
> ```python
> def camera_to_view(camera_ids):
>     """AG-ReID.v2 cameras: C0 aerial, C2 wearable, C3 CCTV."""
> ```
> Task này chuyển từ "xây từ đầu" thành "verify + nhấc ra module dùng chung". Nhẹ hơn nhiều so với dự kiến ban đầu.

- **Verify**: đối chiếu mapping `C0=aerial, C2=wearable, C3=CCTV` với 4 file protocol `AG-ReID.v2/exp*.txt` (`exp1_aerial_to_cctv`, `exp2_aerial_to_wearable`, `exp4_cctv_to_aerial`, `exp5_wearable_to_aerial`). Suy ra tập camera của query/gallery mỗi protocol và kiểm chéo. Kiểm luôn có camera id nào ngoài {0,2,3} không.
- **Refactor** ra `reid_advance/hierarchy/view_map.py` để sampler, loss và metrics dùng chung — đừng để logic này nằm rải rác trong pipeline.
- API tối thiểu: `view_of_path(path) -> Literal["aerial","cctv","wearable"]` và `binary_view(path) -> 0|1`.
- **Chốt quy ước binary**: `ground = {cctv, wearable}`, `aerial = {aerial}`. Đây là quy ước dùng cho `β_cross` trong CV-HWC — ghi rõ vào doc vì cả AI lẫn QC đều phụ thuộc.
- **DoD**: 0 ảnh `unknown` trên `train_all`/`query`/`gallery`; `docs/view_mapping.md` có bảng camera_id → view + bằng chứng đối chiếu protocol.

### BA-02 — Attribute audit
Mở rộng `tools/inspect_attributes.py`, xuất `docs/attribute_audit.md`:
- 15 nhóm (đã khai báo ở `reid_advance/attributes.py::ATTRIBUTE_GROUPS`), số class mỗi nhóm, phân bố class, tỉ lệ `unknown`.
- Độ phủ: bao nhiêu % identity trong `train_all` tìm được entry trong MAT qua `track_key()`.
- **Kiểm chứng giả định sống còn**: mọi ảnh cùng một identity có cùng vector attribute không? KB §2.2 khẳng định attribute ở mức identity — nếu dữ liệu thật khác, báo cho cả nhóm ngay, vì toàn bộ lập luận "hierarchy là cầu nối view-invariant" phụ thuộc vào đây.
- Lưu ý `load_track_attributes` hiện chỉ đọc split `train` của MAT — kiểm tra xem có cần đọc thêm split khác cho query/gallery không.

### BA-03 — Entropy & Mutual Information
Bảng entropy mỗi nhóm + ma trận MI 15×15 giữa các nhóm (KB §6.1 tiêu chí 2, 3).
Đề xuất loại nhóm nào, giữ nhóm nào, kèm lý do.

### BA-04 — Thiết kế cây manual (variant a)

⛔ **Chờ AI-00 (memory profiling) trước khi chốt độ sâu cây.** Batch khả dụng trên GPU 12GB quyết định L tối đa:

| Batch chốt được | Độ sâu cây |
|---|---|
| 128 (P=16×K=8) | L=4 khả thi (3 tầng attribute + ID) |
| 96 (P=12×K=8) | L=4 nhưng rủi ro; cân nhắc L=3 |
| ≤ 64 | **L=3 bắt buộc** (2 tầng attribute + ID) |

Lý do: cây càng sâu, mỗi nhánh trong batch càng ít mẫu → không đủ positive ở tầng trung gian. Đây đúng là lỗi KB §12.2 mục 5, và với GPU 12GB nó là rủi ro thật chứ không phải lý thuyết.

Đầu vào: VSS từ AI-03 + entropy/MI từ BA-03 + batch từ AI-00.
- Nguyên tắc KB §6.2: **attribute ổn định nhất qua view nằm ở tầng CAO nhất (gần root)**.
- Cây nháp KB: root → Gender → Age → Upper-clothing-color-group → Identity (L=4).
  **Phải verify bằng VSS thật**, không copy mù. Nếu buộc L=3, bỏ tầng có VSS thấp nhất trong ba tầng attribute.
- Ràng buộc: mỗi tầng VSS ≥ 65%; mỗi nhánh ≥ 20 identity.
- Output: `trees/manual.json` (schema thống nhất với AI-05) + `docs/tree_design.md` giải thích thứ tự tầng **và lý do chọn L**.

### BA-05 — Spec + implement metrics hierarchy-aware (C4)
Viết `docs/metrics_spec.md` **trước**, rồi `reid_advance/hierarchy/metrics.py`.

Ba metric (KB §10.2):
1. **H-mAP** — relevance mềm `rel(q,g) = ℓ(q,g)/L` thay vì nhị phân. Chọn và ghi rõ công thức: nDCG-style hay AP mở rộng. Phải nêu rõ lựa chọn trong bài.
2. **AC@k** — `AC@k = (1/k)·Σ 1[ℓ(q,g_j) ≥ L−1]`.
3. **Mistake Severity** — với query sai ở rank-1: `MS = E[L − ℓ(q,g_1)]`. Càng thấp càng tốt.

Ràng buộc để QC test được: H-mAP với `L=1` phải trùng mAP chuẩn của `evaluation.py:60`.

### BA-06 — Ba cây còn lại (Phase 2)
- **(b) data-driven**: hierarchical clustering trên vector attribute 15 chiều của các ID, Jaccard/Hamming distance, cắt dendrogram thành L tầng.
- **(c) feature-driven**: clustering trên ground-view feature của baseline TransReID đã train.
- **(d) random tree** — *bắt buộc phải có*. Đây là control experiment chứng minh gain đến từ ngữ nghĩa chứ không phải regularization ngẫu nhiên. KB §9.2 Ablation B.

### BA-07 — Poincaré disk visualization (Phase 3)
`tools/plot_poincare.py`. Fig. 1 (teaser trang 1) + Fig. 4.
Kỳ vọng: coarse class gần tâm, fine class gần biên. So sánh cạnh nhau với t-SNE của baseline (phẳng, lộn xộn).

### BA-08 — CARGO pseudo-attribute (Phase 4)
CARGO không có attribute annotation (KB §6.4). Phương án ưu tiên: chạy CLIP hoặc PAR model pretrained (PA-100K/RAP) trên ảnh ground của CARGO → majority vote ở mức identity → xây cây.
Nếu chạy được, nâng thành **contribution C5**: "hierarchy sinh tự động bằng VLM, không cần annotate thủ công" — đây là đòn vô hiệu hóa phê phán mạnh nhất từ phe VDT.

### BA-09 — Qualitative retrieval figure (Fig. 5)
Chọn vài query aerial khó. So top-10 của baseline vs của ta.
Câu chuyện cần thấy được bằng mắt: baseline sai "thô bạo" (trả người hoàn toàn khác), ta sai "lịch sự" (trả người cùng giới, cùng tuổi, cùng màu áo).

### BA-10/11/12 — Viết bài (Phase 5)
Theo cấu trúc KB §13. Method + Experiments trước, Intro + Related Work + Abstract sau.
BA-12: chuẩn bị sẵn bảng rebuttal theo KB §12.1 — đặc biệt câu "VDT đạt SOTA mà không cần attribute, tại sao cần bài này?"

---

## Bàn giao cho ai
- BA-01 → AI-ENGINEER (AI-02, AI-04) và QC (QC-02)
- BA-04, BA-06 → AI-ENGINEER (AI-05, AI-11)
- BA-05 → QC (QC-05) để viết test

## Khi nào phải báo động
- Attribute không ổn định trong một identity (BA-02).
- Không nhóm attribute nào đạt VSS ≥ 65% (cây không xây được → phải đổi thiết kế).
- Không map được camera → view (BA-01 thất bại → C2 không tồn tại).

# Hierarchy-Aware Representation Learning for Aerial-Ground Person Re-Identification

### Knowledge Base & Research Guide — Topic 1

**Phiên bản**: 1.0 · **Ngày**: 13/07/2026
**Đối tượng**: nhóm nghiên cứu đang chạy TransReID trên AG-ReID.v2 (`nam0403/ReID_Advance`)
**Mục tiêu công bố**: Q1 (IEEE TCSVT / Pattern Recognition / IEEE TIFS) hoặc hội nghị hạng A/B (WACV, BMVC, ACCV, ICPR)

---

## MỤC LỤC

**PHẦN I — KNOWLEDGE BASE (nền tảng cần nắm)**
1. Bài toán AGPReID: định nghĩa, thách thức, ký hiệu
2. Dataset AG-ReID.v2: cấu trúc, protocol, 15 soft attributes
3. State of the Art: ai đã làm gì, còn thiếu gì
4. Nền tảng lý thuyết: SupCon, hierarchical contrastive, hyperbolic geometry

**PHẦN II — RESEARCH PROPOSAL (đóng góp của bạn)**
5. Insight cốt lõi & định vị novelty
6. Xây dựng Semantic Hierarchy từ attributes
7. Formulation: CV-HWC loss + Hyperbolic head
8. Kiến trúc tổng thể & pseudo-code

**PHẦN III — EXECUTION GUIDE (làm thế nào)**
9. Kế hoạch thí nghiệm & bảng ablation
10. Metrics: chuẩn + hierarchy-aware
11. Timeline 6 tháng theo tuần
12. Rủi ro, phản biện dự kiến & cách trả lời
13. Cấu trúc bài báo đề xuất
14. Checklist trước khi nộp

**PHỤ LỤC** — Tài liệu tham khảo, code repo, thuật ngữ

---
---

# PHẦN I — KNOWLEDGE BASE

## 1. Bài toán AGPReID

### 1.1. Định nghĩa

**Person Re-Identification (ReID)**: cho một ảnh truy vấn (query) của một người, tìm tất cả ảnh của cùng người đó trong một tập gallery thu từ các camera khác nhau, không chồng lấn (non-overlapping).

**Aerial-Ground Person ReID (AGPReID)**: trường hợp đặc biệt trong đó gallery và query đến từ hai loại nền tảng khác nhau — **aerial** (UAV/drone, độ cao 15–120m) và **ground** (CCTV cố định, camera đeo/wearable). Hai protocol chuẩn:

- **A→G** (Aerial to Ground): query là ảnh aerial, gallery là ảnh ground.
- **G→A** (Ground to Aerial): ngược lại.

### 1.2. Vì sao AGPReID khó hơn ReID thường?

Sự chênh lệch view (view discrepancy) là thách thức lớn nhất — nó phá vỡ tính phân biệt của biểu diễn identity. Cụ thể:

| Yếu tố | Ảnh ground | Ảnh aerial | Hệ quả |
|---|---|---|---|
| Góc nhìn | ngang, ~0–15° | từ trên xuống, 45–90° | tỷ lệ cơ thể biến dạng hoàn toàn |
| Độ phân giải | cao (chi tiết mặt, hoa văn áo) | thấp (đôi khi < 64×32 px) | mất texture, mất khuôn mặt |
| Pose | đứng/đi bình thường | bị nén theo trục dọc | pose prior không dùng được |
| Chiếu sáng | ổn định | thay đổi mạnh theo độ cao | color shift |

Điều này khiến các model ReID view-homogeneous (thiết kế cho ground-ground) **suy giảm nghiêm trọng** trên protocol view-heterogeneous.

### 1.3. Ký hiệu dùng xuyên suốt tài liệu

| Ký hiệu | Ý nghĩa |
|---|---|
| $x_i$ | ảnh thứ $i$ |
| $y_i \in \{1..N_{id}\}$ | nhãn identity |
| $v_i \in \{A, G\}$ | nhãn platform/view (aerial / ground) |
| $a_i \in \{0,1\}^{15}$ | vector 15 soft attributes |
| $f_i = F_\theta(x_i) \in \mathbb{R}^D$ | feature từ backbone |
| $z_i = g_\phi(f_i)$ | embedding sau projection head |
| $\mathcal{T}$ | cây phân cấp ngữ nghĩa (semantic hierarchy) |
| $L$ | độ sâu của cây $\mathcal{T}$ |
| $\text{lca}(i,j)$ | tổ tiên chung thấp nhất của $y_i, y_j$ trên $\mathcal{T}$ |
| $\ell(i,j)$ | độ sâu của $\text{lca}(i,j)$ (số tầng chia sẻ) |
| $\tau$ | nhiệt độ (temperature) của contrastive loss |

---

## 2. Dataset AG-ReID.v2 — mọi thứ bạn cần biết

### 2.1. Thống kê

AG-ReID.v2 gồm **100.502 ảnh** của **1.615 identity**, mỗi identity được gán nhãn matching ID và **15 soft attribute labels**. Dữ liệu thu từ ba nền tảng: **UAV**, **CCTV cố định**, và **camera tích hợp kính thông minh (wearable)**. Đây là một trong những dataset aerial-ground ReID lớn nhất hiện có.

Điểm quan trọng về pipeline annotation: dataset dùng YOLO detector + StrongSORT tracker để phát hiện và theo dõi người trong video, lưu 1 ảnh mỗi 30 frame, sau đó **hiệu chỉnh thủ công**. Dataset tập trung vào **short-term re-identification** — nghĩa là **không xét thay đổi trang phục/phụ kiện**. 

> ⚠️ **Đây là một fact cực kỳ quan trọng cho đề tài của bạn**: vì không có thay đổi trang phục, các thuộc tính về quần áo/màu sắc là **ổn định theo thời gian trong cùng một identity** → hoàn toàn hợp lệ để dùng chúng làm tầng trung gian trong cây phân cấp. Nếu dataset là long-term (đổi đồ), ý tưởng này sẽ sụp đổ. Hãy nêu rõ điểm này trong bài để phòng thủ trước reviewer.

### 2.2. 15 soft-biometric attributes

Các thuộc tính này bao phủ **đặc điểm thể chất, ngoại hình và phụ kiện** — bao gồm tuổi, giới tính, kiểu trang phục. Chúng được chọn dựa trên phân tích so sánh với các dataset ground/aerial/aerial-ground khác. So với các dataset ground truyền thống, AG-ReID.v2 có thêm **Age** và **Height** mà Market-1501/DukeMTMC-reID không có.

**Lưu ý về format**: trong AG-ReID.v1, attributes được annotate ở **mức identity** (không phải mức ảnh) — file `.mat` chứa ma trận 15 × số identity. AG-ReID.v2 kế thừa cấu trúc này.

> 💡 **Hệ quả kỹ thuật quan trọng**: attribute ở mức identity nghĩa là **mọi ảnh của cùng một ID đều có cùng vector attribute** → cây phân cấp bạn xây sẽ **nhất quán tuyệt đối giữa aerial và ground**. Đây là điều kiện lý tưởng: nhánh cha của một identity không đổi khi đổi view. Đây chính là chỗ để bạn lập luận "hierarchy là cầu nối view-invariant".

**Việc đầu tiên bạn phải làm**: mở file `.mat`, in ra danh sách 15 attribute và số class của mỗi attribute. Ghi lại thành bảng. Toàn bộ thiết kế cây phụ thuộc vào bảng này.

### 2.3. Protocol đánh giá

- **A→G** và **G→A**: hai protocol view-heterogeneous chính (đây là nơi bài của bạn phải thắng).
- Có thể thêm **A→A**, **G→G** (view-homogeneous) để chứng minh không tụt.
- **Metrics chuẩn**: mAP và CMC Rank-1/5/10.

### 2.4. Các dataset liên quan (dùng cho cross-dataset generalization)

| Dataset | Đặc điểm | Vai trò trong bài của bạn |
|---|---|---|
| **AG-ReID.v1** | 199 ID train / 189 ID test, có 15 attributes | Thí nghiệm phụ, chứng minh tính tổng quát |
| **CARGO** | synthetic (Unity3D), 13 camera (8 ground + 5 aerial), 5.000 ID, 108.563 ảnh, chia train/test ~1:1 (2.500/2.500 ID) | Dataset thứ 2 bắt buộc phải có để bài đủ mạnh. **Nhưng CARGO KHÔNG có attribute annotations** → xem mục 6.4 để biết cách xử lý |
| **AG-VPReID** | video-based, 3.027 ID, ~3.7 triệu frame | Ngoài scope, nhưng nên cite |

---

## 3. State of the Art — bản đồ đối thủ

### 3.1. Dòng 1: Attribute-based (Nguyen et al.)

- **AG-ReID.v1** (ICME 2023): đề xuất benchmark đầu tiên + mạng two-stream kết hợp ReID stream (transformer) với explainable stream, dùng **attribute supervision** để tăng tính phân biệt của feature.
- **AG-ReID.v2** (IEEE TIFS 2024): mở rộng thành kiến trúc **three-stream**, nhấn mạnh feature vùng đầu (head region) và soft-biometric attributes để tăng explainability. Có một **localization layer đơn giản hóa** trong luồng elevated-view attention, chuyển đổi động giữa feature toàn cục và feature vùng đầu.

> 🎯 **Đây là đối thủ trực tiếp gần nhất với bạn** — họ cũng dùng attributes. **Nhưng**: họ dùng attribute như **multi-task supervision phẳng** (thêm một classification head cho mỗi attribute, cộng loss). Họ **KHÔNG** xây cây phân cấp, **KHÔNG** dùng attribute để định hình cấu trúc metric space, và **KHÔNG** cho attribute tham gia vào contrastive/metric learning. Đó chính xác là khe hở của bạn.

Một điểm yếu đã được cộng đồng chỉ ra: các phương pháp này **phụ thuộc vào one-hot attribute labels**, có thể hạn chế khả năng tổng quát do lệ thuộc vào semantic cue thủ công.

### 3.2. Dòng 2: View-decoupling (Zhang et al., CVPR 2024 — VDT)

**View-decoupled Transformer (VDT)** — SOTA quan trọng nhất phải so sánh. Ý tưởng: tách feature liên quan đến view và không liên quan đến view bằng hai thành phần:
- **Hierarchical subtractive separation**: tách hai loại feature bên trong transformer.
- **Orthogonal loss**: ràng buộc meta token (chứa identity) và view token (chứa view attributes) trực giao với nhau, giảm nhiễu của thông tin view khi học identity.

Kết quả: vượt phương pháp trước đó tới **5.0% mAP / 2.7% Rank-1 trên CARGO** và **3.7% mAP / 5.2% Rank-1 trên AG-ReID**, với độ phức tạp tính toán cùng cấp. VDT dùng **ít prior label hơn** (chỉ cần nhãn view) mà vẫn tốt hơn Explainable-based methods.

> ⚠️ **Đây là mối đe dọa lớn nhất cho positioning của bạn.** Reviewer sẽ hỏi: *"VDT đạt SOTA mà không cần attribute; tại sao phải quay lại dùng attribute?"*
>
> **Câu trả lời chuẩn bị sẵn** (rất quan trọng):
> 1. VDT xử lý view discrepancy ở **mức feature disentanglement** — nó *loại bỏ* thông tin view. Bạn xử lý ở **mức cấu trúc metric space** — bạn *tổ chức lại* không gian embedding theo ngữ nghĩa. **Hai hướng trực giao (orthogonal), không loại trừ nhau.**
> 2. Chứng minh bằng thực nghiệm: **VDT + hierarchy của bạn > VDT đơn thuần**. Đây là thí nghiệm quan trọng nhất trong cả bài — nếu nó thắng, bài của bạn an toàn.
> 3. Đóng góp phụ: hierarchy cho **explainability** và **graceful failure** (khi model sai, nó sai vào người có cùng thuộc tính — hữu ích cho ứng dụng giám sát thực tế, nơi thu hẹp danh sách nghi phạm quan trọng hơn top-1 chính xác).

### 3.3. Dòng 3: Các hướng mới nổi (2025–2026)

- **SD-ReID**: framework 2 giai đoạn, dùng Stable Diffusion tổng hợp feature view-specific.
- **Text-based / prompt-tuning**: đưa tri thức văn bản về attribute vào training qua prompt-tuning để khai thác thông tin ngữ nghĩa hiệu quả hơn.
- **Perspective Driven Prototype Alignment** (ICIG 2025): dùng memory bank và text feature để tinh chỉnh mô tả xuyên góc nhìn, đạt SOTA trên CARGO và AG-ReID.v1.
- **View-Aware Semantic Alignment** (arXiv 2605.18192).

> 📌 Hướng "prototype alignment" đang nóng lên — điều này **thuận lợi** cho bạn vì hyperbolic prototype (thành phần 3 của bạn) nằm đúng mạch này, nhưng chưa ai làm với hyperbolic geometry trong AGPReID.

### 3.4. Bảng tổng kết gap

| Phương pháp | Dùng attribute? | Model hierarchy? | Tổ chức metric space theo ngữ nghĩa? | Hyperbolic? |
|---|---|---|---|---|
| TransReID (baseline) | ✗ | ✗ | ✗ | ✗ |
| AG-ReID v1/v2 (Nguyen) | ✓ (flat multi-task) | ✗ | ✗ | ✗ |
| VDT (CVPR'24) | ✗ | ✗ (chỉ decouple view) | ✗ | ✗ |
| Prompt/Text-based | ✓ (qua text) | ✗ | một phần | ✗ |
| **Đề xuất của bạn** | **✓ (làm cây)** | **✓** | **✓** | **✓** |

**→ Ô trống ở góc dưới bên phải chính là bài báo của bạn.**

---

## 4. Nền tảng lý thuyết cần nắm vững

### 4.1. Supervised Contrastive Learning (SupCon)

Loss cơ bản cho một anchor $i$, với $P(i)$ là tập positive (cùng nhãn):

$$\mathcal{L}_{\text{SupCon}} = \sum_{i \in I} \frac{-1}{|P(i)|} \sum_{p \in P(i)} \log \frac{\exp(z_i \cdot z_p / \tau)}{\sum_{k \in A(i)} \exp(z_i \cdot z_k / \tau)}$$

**Điểm yếu chí mạng với nhãn phân cấp**: SupCon coi **mọi negative là như nhau**. Một người khác ID nhưng cùng "nam, trẻ, áo đen" bị đẩy ra xa **y hệt** một người "nữ, lớn tuổi, váy đỏ". Đây là sự **mismatch giữa objective và cấu trúc taxonomy** — đúng như "Beyond Flat Labels" đã chỉ ra: contrastive supervision chuẩn trở nên lệch dưới nhãn phân cấp.

### 4.2. Hierarchy-aware contrastive — các biến thể đã có

**HiMulConE (CVPR 2022)** — dùng class hierarchy từ prior knowledge của dataset. Ràng buộc: nếu mẫu có nhãn node con thì bắt buộc phải có nhãn node cha tương ứng. ✅ AG-ReID.v2 thỏa điều kiện này một cách tự nhiên (mỗi ID thuộc đúng một nhánh attribute).

**HWC + LAM (arXiv:2511.03771, medical imaging)** — bài gần nhất về mặt kỹ thuật:
- **HWC (Hierarchy-Weighted Contrastive)**: scale cường độ positive/negative theo **số tổ tiên chung**, thúc đẩy sự gắn kết trong cùng node cha.
- **LAM (Level-Aware Margin)**: margin prototype tách các nhóm tổ tiên qua các tầng.
- **Geometry-agnostic**: áp dụng được cho cả Euclid và hyperbolic mà không đổi kiến trúc.
- Metrics đề xuất: **HF1** (hierarchical F1), **H-Acc** (accuracy có trọng số theo tree distance), **parent-distance violation rate**.

> 📌 Bạn sẽ **mượn khung HWC/LAM** nhưng đóng góp thêm 3 delta cho ReID (xem mục 5.2). Trong bài, hãy cite bài này rõ ràng và nêu chính xác bạn khác gì — trung thực về "borrowed" vs "novel" là cách tốt nhất để qua reviewer.

### 4.3. Hyperbolic geometry — hiểu ở mức trực giác

**Tại sao hyperbolic?** Thể tích của một hình cầu trong không gian hyperbolic tăng **theo hàm mũ** theo bán kính, trong khi số node của một cây cũng tăng theo hàm mũ theo độ sâu. → Cây nhúng vào hyperbolic space với **độ méo (distortion) gần bằng 0**, điều **bất khả thi** trong Euclid số chiều thấp.

Khrulkov et al. (CVPR 2020) chỉ ra **cả dữ liệu ảnh lẫn nhãn đều chứa cấu trúc phân cấp**; embedding từ mạng chuẩn được ánh xạ vào hyperbolic space, sau đó phân loại bằng hyperbolic logistic regression hoặc hyperbolic prototypical learning — cải thiện trực tiếp few-shot learning và uncertainty quantification.

**Ba công thức bạn cần** (Poincaré ball, độ cong $-c$):

*Exponential map* (đưa vector Euclid $v$ từ gốc vào ball):
$$\exp_0^c(v) = \tanh(\sqrt{c}\,\|v\|) \frac{v}{\sqrt{c}\,\|v\|}$$

*Khoảng cách Poincaré*:
$$d_c(u, v) = \frac{2}{\sqrt{c}} \,\text{arctanh}\!\left(\sqrt{c}\, \|(-u) \oplus_c v\|\right)$$

*Möbius addition*:
$$u \oplus_c v = \frac{(1 + 2c\langle u,v\rangle + c\|v\|^2)u + (1 - c\|u\|^2)v}{1 + 2c\langle u,v\rangle + c^2\|u\|^2\|v\|^2}$$

**Thực hành**: dùng thư viện **`geoopt`** (PyTorch) — có sẵn `PoincareBall`, Riemannian Adam optimizer. Không tự implement từ đầu.

⚠️ **Cạm bẫy số học**: hyperbolic rất nhạy với sai số floating point khi điểm tiến gần biên ball. Bắt buộc phải **clip norm** về $\le 1 - \epsilon$ (thường $\epsilon = 10^{-5}$), dùng float32 tối thiểu, và cân nhắc học $c$ như tham số. Chính các tác giả Hyperbolic Image Embeddings đã cảnh báo rằng số học fixed-precision phá vỡ tính tương đương giữa các model hyperbolic, cần cẩn trọng về hiệu ứng độ chính xác số.

---
---
# PHẦN II — RESEARCH PROPOSAL

## 5. Insight cốt lõi & định vị novelty

### 5.1. Câu chuyện một dòng (elevator pitch)

> *"Trong AGPReID, khi camera bay lên cao, mô hình mất dần chi tiết tinh (khuôn mặt, hoa văn) nhưng **các thuộc tính thô (giới tính, dáng, màu áo chủ đạo) vẫn sống sót**. Các mô hình hiện tại ép biểu diễn thành một không gian phẳng, nơi mọi identity cách đều nhau — nên khi chi tiết tinh biến mất, chúng không còn gì để bấu víu. Chúng tôi đề xuất tổ chức không gian embedding theo **cây ngữ nghĩa coarse-to-fine**, sao cho khi tín hiệu fine-grained suy giảm ở góc nhìn aerial, mô hình vẫn **rơi về (fall back) tầng thô** thay vì sụp đổ hoàn toàn."*

Đây là câu chuyện **hay** vì nó không chỉ là "thêm một loss" — nó là một **giả thuyết vật lý về cơ chế thất bại của AGPReID**, và loss là công cụ để kiểm chứng giả thuyết đó. Reviewer thích những bài có giả thuyết rõ ràng.

### 5.2. Bốn đóng góp (contributions) chính thức

Viết đúng như thế này trong abstract/intro:

> **(C1)** Chúng tôi là những người đầu tiên **định dạng lại (reframe) AGPReID như một bài toán học biểu diễn phân cấp**, chỉ ra rằng độ suy giảm thông tin theo góc nhìn tương ứng với các tầng khác nhau của một cây ngữ nghĩa xây từ soft-biometric attributes.
>
> **(C2)** Chúng tôi đề xuất **CV-HWC (Cross-View Hierarchy-Weighted Contrastive)** loss, mở rộng hierarchy-weighted contrastive bằng một **cơ chế tái cân bằng theo view (view-aware reweighting)**, ưu tiên các cặp positive xuyên nền tảng (aerial–ground) — thứ mà các hierarchical contrastive loss hiện tại (vốn thiết kế cho single-view classification) không có.
>
> **(C3)** Chúng tôi đưa **hyperbolic prototype head** vào AGPReID, nhúng cây ngữ nghĩa vào Poincaré ball với distortion thấp, và cho thấy hình học hyperbolic phù hợp hơn Euclid cho biểu diễn ReID phân cấp — đặc biệt ở **chiều embedding thấp**.
>
> **(C4)** Chúng tôi đề xuất **bộ metric hierarchy-aware cho ReID** (H-mAP, Attribute Consistency@k, Mistake Severity), đo lường *chất lượng ngữ nghĩa của lỗi* — một khía cạnh mà mAP/Rank-1 hoàn toàn mù, nhưng lại quan trọng bậc nhất trong ứng dụng giám sát thực tế.

> 💡 **Mẹo viết bài**: C4 là "contribution rẻ nhưng giá trị cao". Nếu C2/C3 chỉ cải thiện mAP khiêm tốn (+1–2%), C4 vẫn cứu bài bằng cách mở ra một trục đánh giá mới mà bạn thắng áp đảo. **Luôn có ít nhất một trục mà bạn thắng lớn.**

### 5.3. Định vị so với VDT (đọc kỹ — quyết định sinh tử của bài)

| | VDT (CVPR'24) | Đề xuất của bạn |
|---|---|---|
| **Triết lý** | *Loại bỏ* thông tin view khỏi feature | *Tổ chức lại* không gian theo ngữ nghĩa |
| **Cấp độ can thiệp** | Kiến trúc (token separation) | Objective + metric space geometry |
| **Nhãn cần** | ID + view | ID + view + attribute |
| **Quan hệ** | **Bổ trợ, không loại trừ** | |

**→ Thí nghiệm bắt buộc phải có**: `VDT + CV-HWC` vs `VDT`. Nếu kết hợp cho gain, bạn có một bài rất mạnh: *"phương pháp của chúng tôi trực giao với SOTA và cộng dồn được lợi ích"*. Đây là kiểu kết quả reviewer rất khó bác.

---

## 6. Xây dựng Semantic Hierarchy từ attributes

Đây là **bước quan trọng nhất và cũng dễ sai nhất** của cả đề tài. Làm cẩu thả ở đây thì mọi thứ phía sau vô nghĩa.

### 6.1. Nguyên tắc chọn thuộc tính làm tầng

Không phải cả 15 attributes đều dùng được. Chọn theo 3 tiêu chí, **theo đúng thứ tự ưu tiên**:

**Tiêu chí 1 — Độ ổn định xuyên view (View Stability).** Thuộc tính phải nhận diện được **từ cả aerial lẫn ground**. Ví dụ: "màu áo" ổn định; "có đeo kính không" thì aerial gần như không thấy → loại.

*Cách đo định lượng* (nên đưa vào bài như một phân tích):
- Train một attribute classifier riêng cho từng attribute trên ảnh ground và ảnh aerial.
- Tính accuracy trên aerial. Thuộc tính nào aerial-accuracy < 65% → **loại khỏi cây** (nhưng vẫn có thể giữ làm auxiliary loss).
- Đây chính là **View Stability Score (VSS)** — một con số bạn tự định nghĩa. Bảng VSS của 15 attributes là một Figure/Table đẹp cho bài báo.

**Tiêu chí 2 — Độ cân bằng (Balance).** Thuộc tính chia dữ liệu thành các nhánh quá lệch (99%/1%) là vô dụng → tính entropy, loại nếu entropy quá thấp.

**Tiêu chí 3 — Tính bổ trợ (Complementarity).** Hai thuộc tính tương quan cao (ví dụ "chiều cao" và "giới tính") không nên xếp thành hai tầng liên tiếp → tính mutual information giữa các cặp, chọn tập thuộc tính có MI thấp.

### 6.2. Thiết kế cây đề xuất (bản nháp — phải verify bằng dữ liệu thật)

```
Level 0:  root
Level 1:  Gender            (2 nhánh: male / female)
Level 2:  Age group         (3 nhánh: young / adult / elderly)
Level 3:  Upper-body clothing color group  (4–6 nhánh: dark / light / red-ish / ...)
Level 4:  Identity          (lá — 1.615 ID)
```

**Vì sao xếp thứ tự này?** Nguyên tắc: **thuộc tính ổn định nhất qua view nằm ở tầng CAO nhất (gần root)**. Gender/dáng người sống sót tốt nhất từ trên cao → L1. Màu áo cũng khá ổn nhưng bị ảnh hưởng bởi chiếu sáng/độ cao → L3.

**Số tầng**: 3 tầng attribute + 1 tầng ID là hợp lý. Nhiều hơn → mỗi nhánh quá ít mẫu, contrastive learning không đủ positive trong batch.

### 6.3. Ba biến thể xây cây — nên thử cả ba (thành ablation đẹp)

| Biến thể | Cách xây | Ưu | Nhược |
|---|---|---|---|
| **(a) Manual / Expert tree** | Xếp tay theo VSS như trên | Diễn giải được, dễ bảo vệ | Cần lý luận thuyết phục về thứ tự tầng |
| **(b) Data-driven tree** | Hierarchical clustering trên vector attribute 15 chiều của các ID (dùng Jaccard/Hamming distance), cắt dendrogram thành L tầng | Tự động, không cần chọn tay; giống cách MgRCL xây cây bằng clustering trên label space | Khó diễn giải, có thể ra nhánh vô nghĩa |
| **(c) Feature-driven tree** | Clustering trên **ground-view features** của baseline đã train, rồi gán ID vào cụm | Cây phản ánh cấu trúc mà model thực sự "thấy" | Vòng lặp phụ thuộc baseline |

> 🔬 **Ablation "How to build the tree?" là một section rất hay trong bài** — nó biến một lựa chọn thiết kế thành một câu hỏi khoa học. Reviewer đánh giá cao điều này.

### 6.4. Xử lý CARGO (không có attribute)

Vấn đề: CARGO là dataset thứ hai bắt buộc phải có, nhưng nó **không có attribute annotation**.

**Ba phương án, xếp theo độ khuyến nghị:**

1. **Pseudo-attribute từ VLM (khuyến nghị)**: chạy CLIP hoặc một Pedestrian Attribute Recognition model pretrained (ví dụ trên PA-100K/RAP) trên ảnh ground của CARGO để sinh attribute giả, lấy majority vote ở mức identity → xây cây từ đó. **Đây tự nó đã là một mini-contribution**: "phương pháp của chúng tôi không phụ thuộc annotation thủ công".
2. **Feature-driven tree (biến thể c ở trên)**: hoàn toàn không cần attribute.
3. Chỉ report CARGO cho biến thể (b)/(c) và nói rõ giới hạn.

> 📌 Nếu phương án 1 chạy được, hãy **nâng nó lên thành contribution C5**: *"Chúng tôi cho thấy hierarchy có thể được sinh tự động bằng vision-language model, loại bỏ nhu cầu annotate attribute thủ công — trực tiếp giải quyết phê phán rằng attribute-based methods phụ thuộc vào nhãn one-hot thủ công đắt đỏ."* Điều này **vô hiệu hóa** đòn tấn công mạnh nhất của phe VDT.

---

## 7. Formulation — công thức chi tiết

### 7.1. Định nghĩa mức chia sẻ phân cấp

Với hai mẫu $i, j$, gọi $\ell(i,j) \in \{0, 1, ..., L\}$ là **số tầng mà chúng chia sẻ tổ tiên**:

- $\ell = L$: cùng identity (lá).
- $\ell = L-1$: khác ID nhưng cùng nhánh màu áo, cùng tuổi, cùng giới.
- ...
- $\ell = 0$: không chia sẻ gì (khác giới).

### 7.2. Hierarchy-Weighted Contrastive (nền tảng)

Trọng số theo tổ tiên chung:

$$w_{ij} = \lambda^{\,L - \ell(i,j)}, \qquad \lambda \in (0, 1)$$

- $\lambda \to 0$: gần như bỏ qua hierarchy, thoái hóa về SupCon thường.
- $\lambda \to 1$: mọi mẫu cùng nhánh bị kéo lại như nhau → mất tính phân biệt ID.
- **Điểm khởi đầu đề xuất**: $\lambda = 0.5$, tune trên $\{0.1, 0.3, 0.5, 0.7, 0.9\}$.

Với anchor $i$, tập positive mở rộng $\tilde{P}(i) = \{j : \ell(i,j) \ge 1\}$ (tức chia sẻ ít nhất một tầng):

$$\mathcal{L}_{\text{HWC}} = \sum_{i \in I} \frac{-1}{\sum_{j \in \tilde{P}(i)} w_{ij}} \sum_{j \in \tilde{P}(i)} w_{ij} \cdot \log \frac{\exp(s_{ij}/\tau)}{\sum_{k \in A(i)} \exp(s_{ik}/\tau)}$$

trong đó $s_{ij} = \text{sim}(z_i, z_j)$ — **cosine** nếu Euclid, hoặc $-d_c(z_i, z_j)$ nếu hyperbolic. **Chính điểm này làm loss geometry-agnostic.**

### 7.3. ⭐ Cross-View Reweighting — đóng góp C2 của bạn

Đây là phần **mới hoàn toàn**, không có trong bài medical. Thêm hệ số theo view:

$$\beta_{ij} = \begin{cases} \beta_{\text{cross}} & \text{nếu } v_i \ne v_j \quad (\text{một aerial, một ground}) \\ 1 & \text{nếu } v_i = v_j \end{cases}$$

với $\beta_{\text{cross}} > 1$ (đề xuất khởi điểm: 2.0; tune trong $\{1.0, 1.5, 2.0, 3.0\}$).

**Loss cuối cùng:**

$$\boxed{\;\mathcal{L}_{\text{CV-HWC}} = \sum_{i \in I} \frac{-1}{Z_i} \sum_{j \in \tilde{P}(i)} \beta_{ij} \, w_{ij} \cdot \log \frac{\exp(s_{ij}/\tau)}{\sum_{k \in A(i)} \exp(s_{ik}/\tau)}\;}$$

với $Z_i = \sum_{j \in \tilde{P}(i)} \beta_{ij} w_{ij}$ là hằng số chuẩn hóa.

**Trực giác**: gradient được ưu tiên đổ vào chính các cặp khó nhất — cặp cùng nhánh ngữ nghĩa nhưng khác nền tảng. Đây là nơi mà view gap gây hại nhất, và cũng là nơi hierarchy có giá trị nhất.

**Lưu ý triển khai quan trọng**: sampler của bạn **phải đảm bảo mỗi batch có cả ảnh aerial lẫn ground của cùng ID**, nếu không $\beta_{\text{cross}}$ không bao giờ được kích hoạt. → Viết một **View-Balanced PK Sampler**: chọn P identity, mỗi ID lấy K/2 ảnh aerial + K/2 ảnh ground. Đây là chi tiết kỹ thuật nhỏ nhưng **nếu quên thì loss của bạn sẽ không hoạt động và bạn sẽ debug cả tuần**.

### 7.4. Level-Aware Margin / Prototype loss (tùy chọn, cho C3)

Đặt prototype $\mu_n$ cho mỗi node $n$ của cây trong Poincaré ball, nhúng trước (pre-embed) bằng thuật toán tree-embedding low-distortion. Với ảnh $i$ có đường đi trên cây $\pi(i) = (n_1, ..., n_L)$:

$$\mathcal{L}_{\text{proto}} = \sum_{i} \sum_{l=1}^{L} \gamma_l \cdot d_c\big(z_i, \mu_{n_l}\big)$$

với $\gamma_l$ tăng dần theo độ sâu (lá quan trọng nhất). Kèm ràng buộc **entailment cone**: node con phải nằm trong nón của node cha — giống cách các phương pháp hyperbolic tối ưu class theo entailment cone kèm distortion loss để có embedding phân cấp tốt hơn.

### 7.5. Loss tổng

$$\mathcal{L} = \underbrace{\mathcal{L}_{\text{ID}} + \mathcal{L}_{\text{triplet}}}_{\text{giữ nguyên từ TransReID}} + \alpha \cdot \mathcal{L}_{\text{CV-HWC}} + \gamma \cdot \mathcal{L}_{\text{proto}}$$

> ⚠️ **KHÔNG bỏ $\mathcal{L}_{ID}$ và $\mathcal{L}_{triplet}$.** Lý do: (1) đảm bảo không tụt dưới baseline — an toàn cho bạn; (2) reviewer sẽ hỏi "loss mới có thay thế được loss cũ không?" → bạn trả lời bằng một dòng trong ablation. Bắt đầu với $\alpha = 1.0$, $\gamma = 0.1$.

---

## 8. Kiến trúc & pseudo-code

### 8.1. Sơ đồ

```
        ảnh (aerial hoặc ground)
                 │
        ┌────────▼─────────┐
        │   ViT-B/16       │   ← TransReID backbone (SIE + JPM giữ nguyên)
        │   (+ side info:  │      SIE đã encode camera/view → tận dụng luôn
        │    camera, view) │
        └────────┬─────────┘
                 │  f ∈ R^768
        ┌────────┴──────────────────┬─────────────────────┐
        │                           │                     │
   ┌────▼─────┐            ┌────────▼────────┐   ┌────────▼────────┐
   │ ID head  │            │ Projection head │   │ (opt) Attribute │
   │ (BNNeck  │            │  g: R^768→R^d   │   │  aux head       │
   │  + FC)   │            │  + exp_map (hyp)│   │                 │
   └────┬─────┘            └────────┬────────┘   └────────┬────────┘
        │                           │                     │
    L_ID + L_triplet          L_CV-HWC + L_proto       L_attr (nhỏ)
```

### 8.2. Pseudo-code loss (PyTorch style)

```python
def cv_hwc_loss(z, ids, views, tree_paths, lam=0.5, beta_cross=2.0,
                tau=0.07, geometry='euclid', ball=None):
    """
    z          : (B, d) embeddings ĐÃ normalize (Euclid) hoặc đã map vào ball (hyp)
    ids        : (B,)   identity labels
    views      : (B,)   0 = ground, 1 = aerial
    tree_paths : (B, L) đường đi trên cây, cột l = node id ở tầng l
                        cột cuối cùng = identity
    """
    B, L = tree_paths.shape

    # --- 1. Ma trận tương đồng (điểm geometry-agnostic) ---
    if geometry == 'euclid':
        sim = z @ z.t() / tau                       # cosine (z đã L2-norm)
    else:                                            # hyperbolic
        sim = -ball.dist(z[:, None, :], z[None, :, :]) / tau

    # --- 2. Mức chia sẻ phân cấp ell(i,j) ---
    # so khớp từng tầng: match[b,i,j] = 1 nếu cùng node ở tầng l
    match = (tree_paths[:, None, :] == tree_paths[None, :, :])   # (B,B,L)
    ell   = match.float().cumprod(dim=-1).sum(dim=-1)            # (B,B) — cumprod:
    #        chỉ tính là "chia sẻ tầng l" nếu đã chia sẻ mọi tầng trước đó (tree consistency)

    # --- 3. Trọng số phân cấp w_ij = lam^(L - ell) ---
    w = lam ** (L - ell)                                          # (B,B)

    # --- 4. Cross-view reweighting (ĐÓNG GÓP C2) ---
    cross = (views[:, None] != views[None, :]).float()
    beta  = 1.0 + (beta_cross - 1.0) * cross                      # (B,B)

    # --- 5. Mask ---
    eye      = torch.eye(B, device=z.device).bool()
    pos_mask = (ell >= 1) & ~eye         # chia sẻ >= 1 tầng, bỏ chính nó
    coef     = w * beta * pos_mask.float()

    # --- 6. Log-softmax trên toàn bộ (trừ chính nó) ---
    logits   = sim.masked_fill(eye, -1e9)
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)

    # --- 7. Weighted NLL ---
    Z    = coef.sum(dim=1).clamp(min=1e-8)
    loss = -(coef * log_prob).sum(dim=1) / Z
    return loss.mean()
```

**Ba chỗ dễ sai — đọc kỹ:**
1. `cumprod` ở bước 2 là bắt buộc. Nếu chỉ `sum(match)`, bạn sẽ tính sai: hai mẫu khác Gender (L1) nhưng tình cờ cùng màu áo (L3) sẽ bị coi là "chia sẻ 1 tầng" — **sai về mặt cây**. `cumprod` đảm bảo tính nhất quán đường đi (path consistency).
2. `z` phải L2-normalize trước khi nhân ma trận (Euclid), hoặc clip norm < 1−ε trước khi vào ball (hyperbolic).
3. `pos_mask` với `ell >= 1` nghĩa là "chia sẻ ít nhất tầng Gender". Nếu batch nhỏ, có thể nới thành `ell >= 0` nhưng lúc đó $w$ rất nhỏ nên ảnh hưởng không đáng kể.

### 8.3. View-Balanced PK Sampler (bắt buộc)

```python
# Mỗi batch: P identity × K ảnh, trong đó mỗi ID cố gắng lấy K/2 aerial + K/2 ground
# Nếu ID không đủ ảnh ở một view → lấy bù từ view kia và log lại tỷ lệ
# Đề xuất: P = 16, K = 8  → batch = 128
```

---
---
# PHẦN III — EXECUTION GUIDE

## 9. Kế hoạch thí nghiệm

### 9.1. Bảng chính (Main Results) — bảng quan trọng nhất của bài

| Method | AG-ReID.v2 A→G |  | AG-ReID.v2 G→A |  | CARGO A→G |  |
|---|---|---|---|---|---|---|
| | mAP | R-1 | mAP | R-1 | mAP | R-1 |
| AGW (CNN) | | | | | | |
| TransReID (baseline) | | | | | | |
| Explainable (Nguyen'24, 3-stream) | | | | | | |
| VDT (CVPR'24) | | | | | | |
| **Ours (TransReID + CV-HWC)** | | | | | | |
| **Ours (TransReID + CV-HWC + Hyp)** | | | | | | |
| **⭐ VDT + CV-HWC (ours)** | | | | | | |

Dòng cuối là dòng **quyết định số phận bài báo**. Nếu nó là SOTA, bạn có bài Q1/hạng A.

### 9.2. Bảng ablation (5 bảng)

**Ablation A — Đóng góp từng thành phần**

| L_ID + L_tri | + HWC | + β cross-view | + Hyperbolic | mAP | R-1 | H-mAP |
|---|---|---|---|---|---|---|
| ✓ | | | | (baseline) | | |
| ✓ | ✓ | | | | | |
| ✓ | ✓ | ✓ | | | | |
| ✓ | ✓ | ✓ | ✓ | | | |

**Ablation B — Cách xây cây** (manual VSS / hierarchical clustering / feature clustering / random tree)
> 🔬 **Nhớ thêm dòng "Random tree"** — cây ngẫu nhiên. Nếu random tree cũng cải thiện thì đóng góp của bạn chỉ là regularization chứ không phải hierarchy. Bạn **phải** chứng minh random tree KHÔNG hiệu quả (hoặc kém hơn hẳn). Reviewer khó tính chắc chắn sẽ hỏi câu này — chủ động trả lời trước.

**Ablation C — Độ sâu cây**: L = 2 (Gender+ID), 3, 4, 5.

**Ablation D — Hyperparameters**: λ ∈ {0.1…0.9}, β_cross ∈ {1.0, 1.5, 2.0, 3.0}, α ∈ {0.5, 1, 2}.

**Ablation E — Chiều embedding (điểm bán hàng của hyperbolic)**
| d | Euclid mAP | Hyperbolic mAP |
|---|---|---|
| 32 | | |
| 64 | | |
| 128 | | |
| 256 | | |
| 768 | | |
> Kỳ vọng: hyperbolic **thắng đậm ở d thấp**, hòa ở d cao. Đây chính là claim của literature (hyperbolic cho hiệu năng tốt hơn ở không gian embedding chiều thấp) và là một **Figure đẹp** (đường cong mAP theo d, hai màu). Nếu kết quả đúng như vậy, nó vừa xác nhận lý thuyết vừa cho một ứng dụng thực tế: **ReID trên edge device / drone với embedding nhẹ**.

### 9.3. Phân tích định tính (Qualitative — đừng bỏ qua)

1. **Retrieval visualization**: với vài query aerial khó, hiển thị top-10 của baseline vs của bạn. Kỳ vọng: khi baseline sai, nó trả về người hoàn toàn khác; khi bạn sai, bạn trả về người **cùng giới, cùng tuổi, cùng màu áo**. → Hình này bán được cả bài.
2. **t-SNE / Poincaré disk visualization**: vẽ embedding space, tô màu theo tầng cây. Kỳ vọng thấy cấu trúc phân cấp rõ ràng trong bản của bạn, và cấu trúc "phẳng, lộn xộn" ở baseline. Với hyperbolic, vẽ trên đĩa Poincaré — coarse class gần tâm, fine class gần biên. **Đây là teaser figure của trang 1.**
3. **Failure case analysis**: khi nào hierarchy phản tác dụng? Trung thực về điều này làm tăng độ tin cậy.

### 9.4. Cấu hình training (điểm khởi đầu)

| Tham số | Giá trị |
|---|---|
| Backbone | ViT-B/16, pretrain ImageNet (theo TransReID) |
| Input size | 256 × 128 |
| Batch | P=16 IDs × K=8 imgs = 128 (view-balanced) |
| Optimizer | SGD (backbone) + Riemannian Adam (nếu hyperbolic) |
| LR | 0.008, cosine decay, warmup 10 epochs |
| Epochs | 120 |
| τ (temperature) | 0.07 |
| Curvature c | 1.0 (hoặc learnable) |
| GPU | 1 × 24GB (RTX 3090/4090/A5000) — batch 128 với ViT-B vừa đủ; nếu OOM giảm P=8, K=8 |

---

## 10. Metrics — bao gồm bộ metric mới (C4)

### 10.1. Metrics chuẩn (bắt buộc)
- **mAP**, **CMC Rank-1/5/10** trên A→G, G→A (chính) và A→A, G→G (phụ, chứng minh không tụt).

### 10.2. Metrics hierarchy-aware (đóng góp C4)

Mượn tinh thần từ HF1, H-Acc, parent-distance violation rate trong literature, nhưng **định nghĩa lại cho retrieval** (ReID là bài toán retrieval, không phải classification — nên không thể copy nguyên):

**(1) Hierarchical mAP (H-mAP)**: thay vì relevance nhị phân (đúng ID = 1, sai = 0), dùng relevance mềm theo tree:
$$\text{rel}(q, g) = \frac{\ell(q, g)}{L}$$
→ trả về người "gần đúng về ngữ nghĩa" vẫn được tính điểm một phần. Tính AP như bình thường với relevance mềm này (graded relevance → dùng công thức nDCG-style hoặc AP mở rộng).

**(2) Attribute Consistency @ k (AC@k)**: trong top-k trả về, tỷ lệ ảnh chia sẻ đầy đủ nhánh cha với query:
$$\text{AC@}k = \frac{1}{k}\sum_{j=1}^{k} \mathbb{1}[\ell(q, g_j) \ge L-1]$$

**(3) Mistake Severity (MS)**: với các query **bị sai** ở Rank-1, tính độ cao trung bình của LCA giữa query và kết quả sai:
$$\text{MS} = \mathbb{E}_{q \,:\, \text{rank1 sai}} \big[\, L - \ell(q, g_1) \,\big]$$
**Càng thấp càng tốt.** Đây là metric của tinh thần *Making Better Mistakes* áp dụng cho ReID — chưa ai làm.

> 💬 **Cách bán C4 cho reviewer**: *"Trong ứng dụng giám sát thực tế, một hệ thống trả về top-10 gồm toàn người cùng giới tính, cùng nhóm tuổi, cùng màu áo với nghi phạm — dù không có ID đúng — vẫn hữu ích hơn nhiều so với một hệ thống trả về 10 người ngẫu nhiên với cùng mAP. mAP hoàn toàn mù với sự khác biệt này."* Lập luận này rất khó bác.

---

## 11. Timeline 6 tháng (theo tuần)

### Tháng 1 — Nền móng
- **T1**: Reproduce TransReID trên AG-ReID.v2, khớp số với paper gốc (±0.5% mAP). **Không đi tiếp nếu chưa khớp.**
- **T2**: Parse file `.mat` attributes. Lập bảng 15 attributes + số class + phân bố. Viết dataloader trả về `(img, id, view, attr_vector)`.
- **T3**: Tính **View Stability Score** cho 15 attributes (train attribute classifier, đo accuracy trên aerial vs ground). → Table 1 của bài.
- **T4**: Xây cây (biến thể a, b, c). Viết code sinh `tree_paths`. Sanity-check: in ra vài nhánh, xem có hợp lý không.

### Tháng 2 — Loss chính
- **T5**: Implement View-Balanced PK Sampler. Verify bằng cách in tỷ lệ aerial/ground trong batch.
- **T6**: Implement `cv_hwc_loss`. **Unit test**: với cây giả và embedding giả, kiểm tra `ell` tính đúng, `w` giảm đúng theo tầng.
- **T7–T8**: Train với HWC (β=1). So sánh với baseline. **Milestone: phải thấy ít nhất H-mAP tăng, kể cả nếu mAP chưa tăng.**

### Tháng 3 — Cross-view + tuning
- **T9–T10**: Bật β_cross, tune λ, β, α. Chạy Ablation A và D.
- **T11**: Chạy Ablation B (cách xây cây) và C (độ sâu).
- **T12**: **Milestone quan trọng**: nếu mAP chưa vượt baseline ≥ +1.0%, dừng lại và debug (xem mục 12.2).

### Tháng 4 — Hyperbolic
- **T13**: Cài `geoopt`, implement hyperbolic projection head + Poincaré distance. Xử lý numerical stability.
- **T14**: Nhúng cây vào Poincaré ball (pre-embedding prototypes).
- **T15–T16**: Train hyperbolic, chạy Ablation E (chiều embedding). Vẽ Poincaré disk visualization.

### Tháng 5 — Mở rộng & bảng chính
- **T17**: Pseudo-attribute cho CARGO (CLIP/PAR model). Xây cây cho CARGO.
- **T18**: Chạy trên CARGO.
- **T19**: **Thí nghiệm sinh tử**: VDT + CV-HWC. (Clone repo `LinlyAC/VDT-AGPReID`, gắn loss của bạn vào.)
- **T20**: Cross-dataset generalization + qualitative figures.

### Tháng 6 — Viết bài
- **T21**: Viết Method + Experiments (viết phần dễ trước).
- **T22**: Viết Intro + Related Work (viết cuối, khi đã biết mình thực sự đóng góp gì).
- **T23**: Figures, polish, viết Abstract.
- **T24**: Đọc lại, checklist (mục 14), nộp.

---

## 12. Rủi ro, phản biện dự kiến & cách trả lời

### 12.1. Bảng phản biện (chuẩn bị sẵn cho rebuttal)

| Reviewer sẽ nói | Bạn trả lời |
|---|---|
| *"Chỉ là áp dụng HWC (arXiv 2511.03771) vào ReID. Incremental."* | Ba delta rõ ràng: (1) **cross-view reweighting** — không có trong bài gốc, và là thứ duy nhất giải quyết đúng vấn đề đặc thù của AGPReID; (2) chuyển từ **classification sang retrieval** — phải định nghĩa lại toàn bộ metric (H-mAP, AC@k, MS); (3) **quy trình xây cây từ soft-biometrics có kiểm chứng VSS** — bài gốc dùng taxonomy có sẵn. Cộng thêm C5 (pseudo-attribute từ VLM) nếu làm được. |
| *"VDT đạt SOTA mà không cần attribute. Tại sao cần bài này?"* | VDT hoạt động ở tầng **kiến trúc** (feature disentanglement), chúng tôi ở tầng **objective/geometry**. Bằng chứng: **VDT + CV-HWC > VDT** (Table X). Hai hướng bổ trợ. |
| *"Cải thiện mAP chỉ +1.5%, không đủ."* | (a) Nhấn vào H-mAP / MS / AC@k — nơi cải thiện lớn hơn nhiều; (b) nhấn vào Ablation E: ở d=64, hyperbolic vượt Euclid rõ rệt → giá trị thực tiễn cho edge deployment; (c) chỉ ra rằng gain **cộng dồn được với SOTA**, không cạnh tranh. |
| *"Cây do người thiết kế → có bias, không tổng quát."* | Ablation B: cây data-driven (clustering) cũng hiệu quả; cây random **thì không** → chứng minh chính **cấu trúc ngữ nghĩa** mới là thứ tạo ra gain, không phải regularization ngẫu nhiên. |
| *"Attribute annotation đắt, không scale."* | C5: pseudo-attribute từ VLM cho kết quả gần tương đương → không cần annotate thủ công. |
| *"Không thử trên dataset thứ ba."* | Có CARGO + AG-ReID.v1. Nếu có thời gian, thêm cross-dataset (train AG-ReID.v2 → test CARGO). |

### 12.2. Nếu mAP KHÔNG cải thiện — phác đồ debug

Kiểm tra theo thứ tự:

1. **Sampler**: batch có thực sự chứa cặp cross-view của cùng ID không? In ra kiểm tra. (Đây là lỗi #1 phổ biến nhất.)
2. **λ quá cao**: nếu λ ≈ 1, loss kéo mọi người cùng nhánh về một chỗ → **phá hủy** tính phân biệt ID. Thử λ = 0.2–0.3.
3. **α quá cao**: L_CV-HWC đè bẹp L_ID. Thử α = 0.1–0.3.
4. **Cây sai**: thuộc tính ở tầng cao có VSS thấp → cây "nói dối" trên ảnh aerial. Kiểm tra lại VSS, đảo thứ tự tầng.
5. **Cây quá sâu**: L=5 với batch 128 → mỗi nhánh chỉ vài mẫu, không đủ positive. Giảm L=3.
6. **Nếu vẫn không được**: **đổi khung bài báo**. Định vị lại thành *"An Analysis of Semantic Structure in Aerial-Ground ReID"* — một bài **phân tích/benchmark** với bộ metric mới (C4), chỉ ra rằng các model hiện tại mắc lỗi "nghiêm trọng về ngữ nghĩa" và đề xuất hierarchy như một hướng giảm thiểu. Bài analysis vẫn đăng được ở Q2 và các workshop hạng A. **Luôn có đường lui.**

---

## 13. Cấu trúc bài báo đề xuất

```
Title:  Climbing Down from the Sky: Hierarchy-Aware Representation Learning
        for Aerial-Ground Person Re-Identification
        (hoặc: "Semantic Hierarchies Bridge the Aerial-Ground Gap in Person ReID")

1. Introduction                                          (1.5 trang)
   - Hook: hình minh họa "khi bay lên cao, chi tiết tinh biến mất, thuộc tính thô còn lại"
   - Vấn đề: metric space phẳng → không có gì để fall back
   - Insight + 4 contributions

2. Related Work                                          (1 trang)
   2.1 Aerial-Ground Person ReID   (AG-ReID v1/v2, VDT, SD-ReID, prompt-based)
   2.2 Hierarchical Representation Learning  (HiMulConE, HWC/LAM, Making Better Mistakes)
   2.3 Hyperbolic Learning in Vision  (Khrulkov, Mettes survey, Poincaré ResNet)

3. Method                                                (2.5 trang)
   3.1 Problem formulation & notation
   3.2 Semantic Hierarchy Construction  (VSS + 3 biến thể)   ← Fig. 2
   3.3 Cross-View Hierarchy-Weighted Contrastive Loss        ← Fig. 3
   3.4 Hyperbolic Prototype Head
   3.5 Overall objective

4. Hierarchy-Aware Evaluation Metrics                    (0.5 trang)  ← C4
   H-mAP, AC@k, Mistake Severity

5. Experiments                                           (3.5 trang)
   5.1 Datasets & protocols
   5.2 Implementation details
   5.3 Comparison with SOTA          ← Table 2 (bảng chính)
   5.4 Ablation studies              ← Tables 3–7
   5.5 Analysis & visualization      ← Fig. 4 (Poincaré disk), Fig. 5 (retrieval)
   5.6 Limitations

6. Conclusion                                            (0.3 trang)
```

**Ba figure phải có:**
- **Fig. 1 (teaser)**: hai cột — trái là embedding space phẳng của baseline (điểm lộn xộn), phải là không gian phân cấp của bạn (cấu trúc cây rõ ràng trên đĩa Poincaré). Kèm ví dụ retrieval.
- **Fig. 2**: cây ngữ nghĩa xây từ attributes, kèm ảnh mẫu ở mỗi nhánh.
- **Fig. 5**: qualitative retrieval — baseline sai "thô bạo", bạn sai "lịch sự".

---

## 14. Checklist trước khi nộp

**Kết quả**
- [ ] Reproduce baseline khớp paper gốc (±0.5%)
- [ ] Thắng trên cả A→G và G→A của AG-ReID.v2
- [ ] Có kết quả trên dataset thứ hai (CARGO hoặc AG-ReID.v1)
- [ ] **VDT + CV-HWC > VDT** (thí nghiệm sinh tử)
- [ ] Không tụt trên protocol view-homogeneous (A→A, G→G)
- [ ] Ablation "random tree" chứng minh gain đến từ ngữ nghĩa

**Bài viết**
- [ ] 4 contributions viết rõ ràng trong Intro
- [ ] Nêu rõ điều gì mượn từ HWC/LAM và điều gì là mới (trung thực = an toàn)
- [ ] Nêu rõ giả định "short-term ReID, không đổi trang phục" và vì sao nó hợp lệ
- [ ] Có section Limitations thật (không phải làm cho có)
- [ ] Cite đầy đủ: TransReID, AG-ReID v1/v2, VDT, HiMulConE, HWC, Khrulkov, Mettes survey, Bertinetto

**Kỹ thuật**
- [ ] Code sạch, sẽ release (reviewer rất thích lời hứa release code)
- [ ] Chạy mỗi cấu hình chính ≥ 3 seed, report mean ± std
- [ ] Report số tham số & thời gian inference (VDT nhấn mạnh chi phí tính toán cùng cấp — bạn cũng phải chứng minh mình không nặng hơn)

---
---

## PHỤ LỤC

### A. Repositories

| Repo | Dùng để |
|---|---|
| `nam0403/ReID_Advance` | Codebase hiện tại của nhóm bạn |
| `huynguyen792/AG-ReID.v2` | Dataset chính thức + attributes (`.mat`) |
| `huynguyen792/AG-ReID` | AG-ReID.v1, file `qut_attribute_v4_88_attributes.mat` |
| `LinlyAC/VDT-AGPReID` | VDT (CVPR'24) + dataset CARGO — cần cho thí nghiệm sinh tử |
| `geoopt/geoopt` | Hyperbolic operations trong PyTorch |
| `JDAI-CV/fast-reid` | Codebase mà VDT dựa trên |

### B. Đọc theo thứ tự (10 bài cốt lõi)

1. **TransReID** (He et al., ICCV 2021) — backbone của bạn.
2. **AG-ReID.v2** (Nguyen et al., IEEE TIFS 2024, arXiv:2401.02634) — dataset + đối thủ attribute-based.
3. **VDT** (Zhang et al., CVPR 2024, arXiv:2403.14513) — SOTA, đối thủ chính.
4. **Supervised Contrastive Learning** (Khosla et al., NeurIPS 2020) — nền tảng loss.
5. **HiMulConE / Use All The Labels** (Zhang et al., CVPR 2022) — hierarchical contrastive đầu tiên trong vision.
6. **Climbing the Label Tree** (arXiv:2511.03771) — HWC/LAM, nguồn cảm hứng trực tiếp nhất.
7. **Making Better Mistakes** (Bertinetto et al., CVPR 2020) — triết lý của metric Mistake Severity.
8. **Hyperbolic Image Embeddings** (Khrulkov et al., CVPR 2020) — hyperbolic vào CV.
9. **Hyperbolic Deep Learning in CV: A Survey** (Mettes et al., IJCV 2024, arXiv:2305.06611) — bản đồ toàn cảnh.
10. **Beyond Flat Labels** (arXiv:2606.21838) — phân tích vì sao contrastive chuẩn lệch dưới nhãn phân cấp.

### C. Thuật ngữ

| Thuật ngữ | Nghĩa |
|---|---|
| **AGPReID** | Aerial-Ground Person Re-Identification |
| **View-heterogeneous** | Protocol query và gallery khác nền tảng (A→G, G→A) |
| **Soft biometrics** | Thuộc tính mềm: giới tính, tuổi, màu áo... (không định danh duy nhất) |
| **LCA** | Lowest Common Ancestor — tổ tiên chung thấp nhất trên cây |
| **Distortion** | Độ méo khi nhúng cây vào không gian metric |
| **Entailment cone** | Nón bao hàm — vùng trong hyperbolic space mà mọi con của một node phải nằm trong |
| **VSS** | View Stability Score (metric bạn tự định nghĩa) |
| **PK sampler** | Sampler chọn P identity × K ảnh mỗi identity |

---

*Tài liệu này là knowledge base nội bộ cho đề tài, tổng hợp từ literature tính đến 07/2026. Các con số SOTA và trạng thái venue nên được kiểm tra lại trước khi nộp.*

# SYSTEM PROMPT & PROJECT SPECIFICATION CHO CLAUDE 3 OPUS
**Role:** Bạn là một Senior AI Engineer và PyTorch Expert chuyên về Computer Vision, đặc biệt là bài toán Person Re-Identification (Re-ID).
**Task:** Viết toàn bộ mã nguồn (Full Codebase) cho đồ án Thạc sĩ: "Unsupervised Aerial Person Re-Identification". 
**Constraint:** Viết code Clean, Modular, có Type Hinting, sử dụng `PyTorch`, `timm`, và logging bằng `TensorBoard`. Code phải chạy được ngay (Ready-to-train).

---

## 1. Yêu cầu Kiến trúc (Architecture Requirements)

Hệ thống phải hỗ trợ "cắm và chạy" (plug-and-play) giữa 2 Backbone thông qua một tham số cấu hình:
1.  **ResNet50:** Tích hợp thêm khối Attention **CBAM** (Convolutional Block Attention Module) ở cuối các layer để tập trung vào đối tượng, loại bỏ nhiễu background từ trên cao.
2.  **ViT-B/32:** Vision Transformer Base với patch size 32 (sử dụng thư viện `timm`: `vit_base_patch32_224`). Không cần CBAM vì ViT đã có Self-Attention.

*Đầu ra của Backbone:* Vector đặc trưng liên tục (Float32 Embedding) đã được normalize (L2-norm).

## 2. Yêu cầu Tiền xử lý Dữ liệu (Data-Centric Tweak)

Vì dữ liệu là ảnh từ Drone (Aerial view, từ trên đỉnh đầu xuống), hướng đầu/chân không còn quan trọng. Trong class `Transform` hoặc Dataloader, bắt buộc phải implement 2 hàm Augmentation tùy chỉnh (Custom Augmentation) sau vào luồng Training:
1.  **Random Rotation:** Xoay ảnh ngẫu nhiên các góc $0^\circ, 90^\circ, 180^\circ, 270^\circ$.
2.  **Random Patch Permutation:** Cắt ảnh thành lưới $2 \times 2$ (hoặc $4 \times 4$) và xáo trộn ngẫu nhiên vị trí các patch với xác suất $p=0.3$.

## 3. Yêu cầu Luồng Huấn luyện (Unsupervised Pipeline)

Áp dụng chiến lược **Clustering-based Self-training** theo chuẩn các bài báo SOTA (như SpCL, Cluster-Contrast). Luồng này bao gồm:

* **Tính Jaccard Distance:** Sử dụng k-reciprocal nearest neighbors để tính ma trận khoảng cách trên không gian đặc trưng.
* **Phân cụm (Clustering):** Sử dụng `DBSCAN` (từ `scikit-learn`) để tạo Pseudo-labels (Nhãn giả) vào đầu mỗi epoch.
* **Hàm Loss (Cluster-Contrast Loss):** * Tạo một `Memory Bank` chứa vector trung tâm của các cụm.
    * Cập nhật Memory Bank theo cơ chế Momentum update.
    * Sử dụng InfoNCE Loss (Contrastive Loss) để kéo đặc trưng của ảnh về gần vector trung tâm của cụm tương ứng, và đẩy ra xa các cụm khác.

## 4. Cấu trúc Source Code Yêu cầu bạn (Opus) sinh ra:

Hãy sinh ra mã nguồn chi tiết cho các file sau:

### File 1: `transforms.py`
* Chứa logic chuẩn bị dữ liệu. Đảm bảo code 2 class custom augmentation: `RandomRotationAerial` và `RandomPatchPermutation`.

### File 2: `models.py`
* Định nghĩa module `CBAM`.
* Định nghĩa class `AerialReIDNet(backbone_type='resnet50_cbam' | 'vit_b_32')`. Sử dụng thư viện `timm` để load pretrained weights.

### File 3: `loss.py`
* Định nghĩa `ClusterContrastLoss` kết hợp `MemoryBank`.
* Input: Features, Pseudo-labels. Output: Contrastive loss value.

### File 4: `utils/clustering.py`
* Viết hàm `compute_jaccard_distance(features)`.
* Viết hàm `get_pseudo_labels(features, eps)` sử dụng DBSCAN. Bỏ qua các nhãn nhiễu (outliers label = -1).

### File 5: `dataset.py`
* Định nghĩa Custom Dataset đọc dữ liệu ảnh. 
* Viết logic Dataloader để mỗi batch phải chứa $P$ person IDs, mỗi ID có $K$ ảnh (PK-Sampler) để tối ưu hóa cho Contrastive Loss.

### File 6: `main.py` (Trainer script)
* Khởi tạo Model, Optimizer (AdamW), Dataloader.
* Vòng lặp Training (Vòng lặp lồng: Extract features $\rightarrow$ DBSCAN $\rightarrow$ Khởi tạo lại Memory Bank $\rightarrow$ Train 1 epoch với Cluster-Contrast Loss).
* Lưu Checkpoint sau mỗi $N$ epochs.
* Tích hợp `SummaryWriter` (TensorBoard) để log Loss và số lượng Clusters sinh ra sau mỗi epoch.

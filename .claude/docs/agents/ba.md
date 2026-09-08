# Agent Brief — BA (viết lại 2026-09-08)

## Đọc trước
1. `.claude/docs/topic1_implementation_plan.md` — bản mới
2. `docs/gap_vs_capability.md`, `docs/same_view_control_feasibility.md`,
   `docs/cargo_camera_pair_matrix.md`, `docs/vss_table.md`

## Phạm vi
Định vị nghiên cứu, cấu trúc bài, figure, viết bài, rebuttal.

> **Vai trò này đã đổi hẳn.** Không còn thiết kế cây ngữ nghĩa hay tính VSS —
> hướng đó đã bị dữ liệu bác bỏ. Việc bây giờ là biến chuỗi điều tra đã có thành
> một bài báo.

---

## Luận điểm trung tâm (đã có bằng chứng, chắc bất kể method)

> Gap cross-platform trong AGPReID là **hàm của benchmark**, không phải hằng số
> của bài toán. Trên AG-ReID.v2 supervision thường xóa sạch nó; trên CARGO mAP
> tăng gấp đôi mà gap không co 1 điểm. Cách báo mAP tổng hợp theo protocol chính
> thức che giấu điều này.

Bốn trụ:

**(T1) Hai đường cong khác hình dạng.** AG-ReID.v2 +1.16% → −0.59% (qua 0);
CARGO +18.27% → +17.20% (phẳng). Dải mAP 12.3 và 25.3 điểm.

**(T2) Phân rã hai hiệu ứng.** aerial–aerial (67.1%) nằm giữa aerial–ground
(61.5%) và ground–ground (78.7%) → "khó do nhìn từ trên cao" **khác** "khó do
đổi platform". Cộng đồng gộp làm một. **Chỉ CARGO đo được** — AG-ReID.v2 có 0
identity với ≥2 camera aerial.

**(T3) Same-view control không dựng được trên AG-ReID.v2.** Mỗi split mỗi
protocol chỉ một camera. Ai báo "same-view baseline" trên dataset đó đã làm gì
đó phi chuẩn.

**(T4) Attribute không suy giảm theo view.** Retention 0.86–1.07; chỉ 1/15
attribute vượt ngưỡng. Bác bỏ giả định phổ biến rằng thuộc tính thô "sống sót"
khi bay lên cao.

---

## Nhiệm vụ

### BA-N1 — Figure trung tâm
Hai đường cong gap-vs-mAP trên cùng một trục. Đây là hình bán cả bài: hai đường
**khác hình dạng**, không chỉ khác mức. Dữ liệu ở `docs/gap_vs_capability.json`.

### BA-N2 — Figure phân rã hai hiệu ứng
Ba cột aerial–ground / aerial–aerial / ground–ground trên CARGO, làm nổi việc
aerial–aerial nằm giữa. Kèm chú thích AG-ReID.v2 không tạo được cột giữa.

### BA-N2b — Mục "What we ruled out and why" (bắt buộc, không phải phụ lục)

Đây là phần khó-bác nhất của bài và cũng là phần dễ bị bỏ nhất. Bốn negative
control đã chạy, mỗi cái giết một giả thuyết — kể cả giả thuyết của chính nhóm:

| control | giết cái gì | bằng chứng |
|---|---|---|
| **VSS** | attribute không làm được hierarchy | retention 0.86–1.07; 1/15 vượt ngưỡng |
| **GSS** | granularity chưa phân hóa sẵn | tầng tương quan ≥0.96, mAP chênh 0.13–0.46% |
| **random-init** | không định vị được gap | 1.3% phẳng — control vô dụng, và ta nói thế |
| **same-view khả thi** | control chuẩn bất khả thi trên AG-ReID.v2 | 0 identity ≥2 camera aerial |

Chèn giữa phần thí nghiệm và phần hàm ý. Nó chứng minh nhóm **không cherry-pick
con đường tới kết luận** — một analysis paper sống bằng đúng điều đó.

Nêu cả bug `+100` đã tự bắt: nó là minh chứng rẻ nhất rằng lỗi loại này **vô
hình trong mọi chỉ số tổng hợp** (loss, training curve, mAP protocol đều bình
thường suốt thời gian nó xảy ra).

### BA-N3 — Khuyến nghị phương pháp luận
Phần bài mà cộng đồng dùng được ngay:
- phân rã theo cặp camera, đừng chỉ báo mAP tổng hợp
- dựng same-view control đúng (giữ bộ lọc cross-camera), kiểm nó có dựng được không
- đếm ID-chung trước khi diễn giải ô ma trận
- kiểm per-camera trước khi kết luận về platform

### BA-N4 — Cấu trúc bài
Viết Method + Experiments trước, Intro + Related Work sau. Nếu nhánh method
thất bại, bài vẫn đứng bằng T1–T4 cộng negative result trung thực.

### BA-N5 — Rebuttal
Chuẩn bị trước ba câu chắc chắn bị hỏi:
- *"Chỉ là một dataset có gap, một không — có gì mới?"* → khác **hình dạng
  đường cong**, không phải khác mức tại một điểm; và điều đó nói supervision
  chuẩn xử lý được gap trên benchmark này chứ không phải trên benchmark kia.
- *"Sao không dùng SOTA backbone / mAP thấp?"* → so ở cùng backbone, cùng ngân
  sách; câu hỏi là gap, không phải mAP tuyệt đối.
- *"AG-ReID.v2 chỉ có 6 cặp camera"* → thừa nhận thẳng, xem giới hạn dưới đây.

---

## Hai ràng buộc lên cách phát biểu (xem plan §2)

**(R1) Phân rã hai hiệu ứng chỉ dựa CARGO.** AG-ReID.v2 có 0 cặp aerial-aerial
nên không thể tham gia. Vai trò của nó là mệnh đề về **dấu** ("gap không dương
đáng kể"), thứ chịu được 2 cặp. Đừng bắt nó gánh phân rã ba thành phần.

**(R2) Claim "gap tan theo supervision" chỉ trên dải tự train**: +1.16% → −0.59%
(mAP 58.1→70.4), đã qua 0 — đủ cho claim. Checkpoint 125-epoch có config khớp
hoàn toàn nhưng không do run này sinh; nhắc nó như **xác nhận độc lập**, không
phải mắt xích lập luận.

## Giới hạn phải nêu trong bài, không giấu

1. **AG-ReID.v2 có 3 camera → 4 cặp a–g, 2 cặp g–g.** Con số ~1% mong manh hơn
   +17% dựng trên 80/56 cặp của CARGO. Điều cứu nó: chỉ dùng cho claim về dấu
   (R1), và hướng nhất quán qua 4 checkpoint tự train.
2. **CARGO là synthetic (Unity3D).** Gap có thể phản ánh đặc tính render.
3. **Backbone ViT-S, batch 32** — không đua SOTA tuyệt đối.
4. **Không so được ở cùng mAP** (hai trần khác nhau) — đó chính là lý do dùng
   đường cong; nói rõ thay vì để reviewer phát hiện.

---

## Nguyên tắc viết
Trung thực về cái không biết là tài sản, không phải điểm yếu. Cả chuỗi điều tra
này mạnh chính vì mỗi bước đều có một control giết được giả thuyết trước đó —
kể cả giả thuyết của chính nhóm. Viết đúng tinh thần đó: nêu cả bug `+100` đã
tự bắt (`docs/same_view_control_feasibility.md`) như minh chứng rằng lỗi này vô
hình trong mọi chỉ số tổng hợp.

# Báo cáo Lab Day 1 — VuongVietHoang — 2A202602528

## 1. Thiết lập
- Môi trường: Google Colab; PyTorch 2.11.0+cpu; thiết bị: CPU.
- Chia train/eval cố định; validation phân tầng 20% train, seed 42. Chuẩn hoá 10 cột số theo thống kê train; giữ nguyên 44 cột nhị phân.
- M-base có 47,879 tham số; baseline dùng CE, SGD+momentum, lr=0.06, batch=512, 20 epoch, He.
- Validation majority-class accuracy: 0.4876.

## 2. Kiểm tra ban đầu và độ nhiễu
- Loss bước 0 (baseline): 2.26906; tham chiếu ln(7)=1.94591.
- Overfit 20 mẫu: loss=0.000002, accuracy=1.000.
- Gradient khác 0 ở mọi tham số: True.
- Baseline validation accuracy trung bình ± SD (2 seed): 0.9005 ± 0.0015.
- Baseline validation macro-F1 trung bình ± SD (2 seed): 0.8461 ± 0.0015; 2σ=0.0029390355992336714.

## 3. Kết quả validation theo thí nghiệm
LR pilot chạy 4 epoch chỉ dùng chọn learning rate; không xem là so sánh hội tụ tương đương với thí nghiệm chính 20 epoch.
| exp_id | nhóm | thay đổi | best epoch | val accuracy | val macro-F1 | giây/epoch | biểu đồ |
|---|---|---|---:|---:|---:|---:|---|
| lr-sgd-0.01 | hparam | SGD momentum learning-rate pilot lr=0.01; 4 epochs | 4 | 0.7975 | 0.6314 | 2.23 | [ảnh](figures/lr-sgd-0.01.png) |
| lr-sgd-0.03 | hparam | SGD momentum learning-rate pilot lr=0.03; 4 epochs | 4 | 0.8304 | 0.7133 | 2.17 | [ảnh](figures/lr-sgd-0.03.png) |
| lr-sgd-0.06 | hparam | SGD momentum learning-rate pilot lr=0.06; 4 epochs | 4 | 0.8428 | 0.7397 | 2.29 | [ảnh](figures/lr-sgd-0.06.png) |
| lr-adam-0.001 | optimizer | Adam learning-rate pilot lr=0.001; 4 epochs | 4 | 0.8454 | 0.7491 | 3.03 | [ảnh](figures/lr-adam-0.001.png) |
| lr-adam-0.003 | optimizer | Adam learning-rate pilot lr=0.003; 4 epochs | 4 | 0.8689 | 0.7913 | 2.96 | [ảnh](figures/lr-adam-0.003.png) |
| base-s1 | baseline | M-base; CE; SGD momentum 0.9; He | 19 | 0.9016 | 0.8451 | 2.70 | [ảnh](figures/base-s1.png) |
| base-s2 | baseline | M-base baseline, seed 2 | 20 | 0.8995 | 0.8472 | 2.31 | [ảnh](figures/base-s2.png) |
| loss-mse | loss | MSE loss vs baseline CE | 20 | 0.8612 | 0.7097 | 2.19 | [ảnh](figures/loss-mse.png) |
| opt-adam | optimizer | Adam; learning rate selected with validation pilot | 20 | 0.9157 | 0.8752 | 2.43 | [ảnh](figures/opt-adam.png) |
| hparam-wide | hparam | M-wide (512-256) vs M-base | 18 | 0.9127 | 0.8621 | 5.56 | [ảnh](figures/hparam-wide.png) |
| dropout-03 | dropout | Dropout p=0.3 after hidden ReLUs | 19 | 0.8614 | 0.7655 | 4.21 | [ảnh](figures/dropout-03.png) |
| clip-1 | clipping | Global gradient clipping at norm 1.0 | 18 | 0.9030 | 0.8440 | 2.23 | [ảnh](figures/clip-1.png) |
| init-xavier | init | Xavier-normal initialization vs He | 20 | 0.9004 | 0.8451 | 2.26 | [ảnh](figures/init-xavier.png) |

### Dự đoán trước, kết quả sau và giải thích

**loss-mse (loss).** Dự đoán: CE có thể phân loại tốt hơn MSE vì gradient CE phản ánh trực tiếp sai số xác suất lớp.
Quan sát: val macro-F1=0.7097, chênh lệch so với base-s1=-0.1354; vượt ngưỡng 2σ. Giải thích: CE tối ưu từ logits với softmax-cross-entropy; MSE tối ưu khoảng cách với one-hot. So macro-F1/accuracy, không so độ lớn hai loss.
Biểu đồ: [thí nghiệm](figures/loss-mse.png).

**opt-adam (optimizer).** Dự đoán: Adam có thể giảm loss nhanh hơn nhờ bước cập nhật thích nghi theo từng tham số.
Quan sát: val macro-F1=0.8752, chênh lệch so với base-s1=+0.0301; vượt ngưỡng 2σ. Giải thích: SGD+momentum tích luỹ hướng gradient; Adam chuẩn hoá theo moment. Các learning rate được sàng lọc trên validation, nhưng pilot ngắn nên đây chưa phải so sánh hội tụ hoàn toàn.
Biểu đồ: [thí nghiệm](figures/opt-adam.png).

**hparam-wide (hparam).** Dự đoán: learning rate tốt hơn có thể tăng tốc hội tụ; M-wide có thể biểu diễn ranh giới phức tạp hơn nhưng tăng chi phí.
Quan sát: val macro-F1=0.8621, chênh lệch so với base-s1=+0.0171; vượt ngưỡng 2σ. Giải thích: đối chiếu val macro-F1, best epoch và giây/epoch; số bước cập nhật mỗi epoch giữ nguyên với batch 512.
Biểu đồ: [thí nghiệm](figures/hparam-wide.png).

**dropout-03 (dropout).** Dự đoán: dropout chỉ hữu ích nếu baseline có dấu hiệu overfit.
Quan sát: val macro-F1=0.7655, chênh lệch so với base-s1=-0.0796; vượt ngưỡng 2σ. Giải thích: đối chiếu khoảng cách train-val loss và val macro-F1; dropout giảm đồng thích nghi nhưng cũng làm khó tối ưu hơn.
Biểu đồ: [thí nghiệm](figures/dropout-03.png).

**clip-1 (clipping).** Dự đoán: clipping norm 1.0 chỉ ảnh hưởng nếu grad norm trước clip đủ lớn.
Quan sát: val macro-F1=0.8440, chênh lệch so với base-s1=-0.0011; không vượt ngưỡng 2σ. Giải thích: đường ghi nhận là norm trung bình trước clip theo epoch; nếu giá trị thấp, cần thận trọng khi kết luận clipping đã có tác dụng.
Biểu đồ: [thí nghiệm](figures/clip-1.png).

**init-xavier (init).** Dự đoán: He phù hợp với ReLU; Xavier có thể giữ phương sai cân bằng hơn giữa fan-in/fan-out.
Quan sát: val macro-F1=0.8451, chênh lệch so với base-s1=-0.0000; không vượt ngưỡng 2σ. Giải thích: so loss bước 0, độ lệch chuẩn kích hoạt và validation sau huấn luyện.
Biểu đồ: [thí nghiệm](figures/init-xavier.png).

## 4. Đánh giá cuối trên eval
| cấu hình | val macro-F1 | eval macro-F1 | eval accuracy |
|---|---:|---:|---:|
| Baseline (base-s1) | 0.8451 | 0.8473 | 0.9010 |
| Cuối (opt-adam) | 0.8752 | 0.8770 | 0.9147 |
- Chênh lệch val macro-F1 cuối so với baseline: +0.0301; vượt 2σ_seed: True.

### F1 theo lớp và phân tích nhầm lẫn
| lớp | support | precision | recall | F1 |
|---:|---:|---:|---:|---:|
| 0 | 42368 | 0.9163 | 0.9026 | 0.9094 |
| 1 | 56661 | 0.9207 | 0.9354 | 0.9280 |
| 2 | 7151 | 0.9126 | 0.9055 | 0.9090 |
| 3 | 549 | 0.8530 | 0.8033 | 0.8274 |
| 4 | 1899 | 0.8252 | 0.7930 | 0.8088 |
| 5 | 3473 | 0.8335 | 0.8318 | 0.8327 |
| 6 | 4102 | 0.9355 | 0.9118 | 0.9235 |

Lớp F1 thấp nhất là **4** (F1=0.8088, support=1899); trong các dự đoán sai, lớp bị nhầm nhiều nhất là **1** (323 mẫu).

Ma trận nhầm lẫn (hàng là nhãn thật, cột là dự đoán):
| thật\dự đoán | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 38241 | 3844 | 8 | 0 | 34 | 12 | 229 |
| 1 | 3141 | 53000 | 113 | 0 | 245 | 133 | 29 |
| 2 | 1 | 199 | 6475 | 59 | 33 | 384 | 0 |
| 3 | 0 | 0 | 71 | 441 | 0 | 37 | 0 |
| 4 | 42 | 323 | 17 | 0 | 1506 | 11 | 0 |
| 5 | 4 | 146 | 411 | 17 | 6 | 2889 | 0 |
| 6 | 306 | 55 | 0 | 0 | 1 | 0 | 3740 |

## 5. Câu hỏi dẫn dắt và hạn chế
- Dropout: dùng kết quả train-val gap và validation macro-F1 phía trên để kết luận có dấu hiệu overfit hay không.
- Clipping: xem grad norm trước clip; thí nghiệm này dùng learning rate baseline, không khẳng định xử lý được mọi trường hợp gradient nổ.
- Mixed precision: kết luận nhanh/chậm chỉ dựa trên số đo của runtime này; T4/GPU khác có thể cho kết quả khác.
- Nếu loss không giảm: (1) kiểm tra nhãn/dữ liệu, chuẩn hoá và loss bước 0; (2) thử overfit 20 mẫu; (3) kiểm tra gradient từng tham số và grad_norm, sau đó rà soát learning rate/zero_grad.
- Hạn chế: hai seed baseline cho ước lượng nhiễu còn thô; learning-rate pilots chỉ 4 epoch; kết quả chỉ đại diện cho phần cứng Colab đã chạy. Mọi lựa chọn cấu hình dựa trên validation; eval chỉ được chấm sau khi chốt.

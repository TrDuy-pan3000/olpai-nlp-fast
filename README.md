# OlpAI NLP — Fast Contest Transformer

[![Mở bằng Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TrDuy-pan3000/olpai-nlp-fast/blob/main/OlpAI_NLP_Fast_Contest.ipynb)

Bộ khung dịch **tiếng Hoa → tiếng Việt** theo hướng Top-5 nhưng được viết lại bằng các module cơ bản để dễ học, dễ sửa và chạy nhanh trên Google Colab T4. Attention, feed-forward, encoder, decoder và model đều hiện trực tiếp trong notebook; không giấu kiến trúc trong `nn.Transformer`.

## Chạy nhanh nhất

1. Bấm nút **Mở bằng Google Colab** ở trên.
2. Chọn **Runtime → Change runtime type → T4 GPU**.
3. Chọn **Runtime → Run all**.
4. Mặc định `SMOKE_TEST=False`: notebook chạy luôn bản train thật trên toàn bộ dữ liệu.
5. Chỉ đổi thành `True` khi cần kiểm tra pipeline trong khoảng 1–3 phút.

Notebook tự tải code và `dataset.zip` từ Google Drive đã cung cấp, tự giải nén, huấn luyện, tạo hai CSV và đóng gói `submission.zip`. Cell cuối tự bật hộp tải file kết quả.

## Cấu hình mặc định

- Vocab BPE: 8.000
- Transformer basic viết tường minh: 4 encoder + 4 decoder
- `d_model=256`, 8 attention heads, FFN 1.024
- Batch size 128, tối đa 15 epoch
- AMP + TF32 trên GPU, early stopping theo BLEU
- Greedy decoding theo batch; beam-3 chỉ là tùy chọn
- Không dùng RMSNorm, RoPE, GQA hay SwiGLU
- Nhánh hai chiều và contrastive có code basic để học nhưng tắt mặc định
- Khoảng 9,42 triệu tham số

Ước tính trên Colab T4: 8–18 phút train và 3–7 phút validation + tạo submission. Tổng thường 12–25 phút, tùy tải của Colab.

## Cấu trúc repo

```text
.
├── OlpAI_NLP_Fast_Contest.ipynb  # kiến trúc hiện trực tiếp, bấm Run all
├── nlp_basic.py                  # bản module hóa để test và tái sử dụng
├── requirements.txt
└── tests/
    └── test_nlp_basic.py
```

## Chạy test local

```bash
python -m pip install -r requirements.txt
python -m pytest -q
```

## Lưu ý thi

Greedy là lựa chọn mặc định vì nhanh và chắc chắn tạo kịp file nộp. Hãy tải `submission.zip`, `best_model.pt` và tokenizer về máy trước khi đóng runtime. Không bật beam search cho đến khi đã có một submission greedy hợp lệ.

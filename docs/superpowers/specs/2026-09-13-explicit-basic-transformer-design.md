# Explicit Basic Transformer Design

## Goal

Biến notebook thành tài liệu học có thể chạy một nút trên Colab T4: toàn bộ kiến trúc Transformer Hoa → Việt được viết rõ theo bộ khung lời giải Top-5, nhưng chỉ dùng các khối PyTorch cơ bản.

## Architecture

Giữ nguyên pipeline dữ liệu, joint BPE, lọc độ dài, label smoothing, AdamW, warmup, AMP, gradient clipping, BLEU, checkpoint, greedy/beam và đóng gói submission. Thay `nn.Transformer` nguyên khối bằng `BasicMultiHeadAttention`, `BasicFeedForward`, `EncoderLayer`, `Encoder`, `DecoderLayer`, `Decoder`, rồi ghép trong `Seq2SeqTransformer`.

Mỗi lớp attention dùng `nn.MultiheadAttention`; feed-forward dùng `Linear → ReLU → Dropout → Linear`; chuẩn hóa dùng `LayerNorm`; vị trí dùng sinusoidal encoding. Không dùng RMSNorm, RoPE, GQA, SwiGLU, contrastive learning hoặc huấn luyện hai chiều.

## Notebook presentation

Notebook hiển thị trực tiếp các class kiến trúc theo thứ tự dữ liệu chảy qua model. Các helper dữ liệu/đánh giá vẫn được import tường minh từ `nlp_basic.py`; không dùng wildcard import. Mỗi cell có giải thích tiếng Việt ngắn và sơ đồ shape để người học theo dõi.

## Verification

- Unit test kiểm tra module kiến trúc tồn tại và forward đúng shape.
- Contract test kiểm tra notebook chứa code attention/encoder/decoder trực tiếp và không chứa module cao cấp.
- Toàn bộ cell Python phải compile.
- Colab T4 phải hoàn thành smoke forward và bắt đầu full training với progress bar.

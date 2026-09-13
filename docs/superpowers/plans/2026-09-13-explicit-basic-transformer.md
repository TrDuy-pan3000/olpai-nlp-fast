# Explicit Basic Transformer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Viết rõ từng module Transformer cơ bản trong code và notebook, giữ nguyên pipeline thi đang hoạt động.

**Architecture:** Tách `nn.Transformer` thành attention, feed-forward, encoder và decoder dùng module chuẩn PyTorch. Notebook chứa trực tiếp định nghĩa kiến trúc để người học đọc được mà không mở file phụ.

**Tech Stack:** Python, PyTorch, Jupyter/Colab, pytest.

## Global Constraints

- Chạy được trên Colab T4 với cấu hình full hiện tại.
- Không dùng RMSNorm, RoPE, GQA, SwiGLU hoặc contrastive learning.
- Không dừng runtime Colab hiện tại trước khi bản mới sẵn sàng.

---

### Task 1: Explicit architecture modules

**Files:**
- Modify: `nlp_basic.py`
- Test: `tests/test_nlp_basic.py`

- [ ] Thêm test yêu cầu các class basic và kiểm tra forward shape.
- [ ] Chạy test để thấy thất bại vì class chưa tồn tại.
- [ ] Cài đặt attention, feed-forward, encoder và decoder cơ bản.
- [ ] Chạy toàn bộ test.

### Task 2: Educational notebook

**Files:**
- Modify: `OlpAI_NLP_Fast_Contest.ipynb`
- Test: `tests/test_notebook_contract.py`

- [ ] Thêm contract test yêu cầu định nghĩa kiến trúc xuất hiện trực tiếp trong notebook.
- [ ] Chạy test để thấy thất bại với notebook cũ.
- [ ] Chia kiến trúc thành các cell có chú thích và bỏ wildcard import.
- [ ] Chạy compile contract và toàn bộ test.

### Task 3: Publish and Colab verification

**Files:**
- Modify: `README.md`

- [ ] Cập nhật hướng dẫn về notebook tự chứa kiến trúc.
- [ ] Kiểm tra diff, chạy toàn bộ test và commit.
- [ ] Push lên GitHub, mở notebook cache-busted và Run all trên T4.
- [ ] Theo dõi đến khi tạo xong `submission.zip` hoặc có lỗi cụ thể cần sửa.

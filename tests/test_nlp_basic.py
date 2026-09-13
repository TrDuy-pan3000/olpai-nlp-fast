import sys
from pathlib import Path

import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nlp_basic import (  # noqa: E402
    ModelConfig,
    Seq2SeqTransformer,
    TranslationDataset,
    WarmupInverseSqrtScheduler,
    build_translation_memory,
    causal_mask,
    collate_batch,
    deduplicate_pairs,
    filter_pairs_by_token_length,
    group_split,
    normalize_text,
    package_submission,
    write_submission_csv,
)


class FakeTokenizer:
    """Tokenizer nhỏ để test Dataset mà không phải train SentencePiece."""

    def encode(self, text, out_type=int):
        return [4 + (len(word) % 10) for word in text.split()]


def test_filter_pairs_by_token_length_keeps_only_complete_examples():
    pairs = [("a bb", "c"), ("a bb ccc dddd", "e")]
    assert filter_pairs_by_token_length(pairs, FakeTokenizer(), max_len=5) == [("a bb", "c")]


def test_default_config_is_basic_but_contest_sized():
    config = ModelConfig()
    assert config.vocab_size == 8000
    assert config.d_model == 256
    assert config.encoder_layers == 4
    assert config.decoder_layers == 4


def test_normalize_text_preserves_underscore_and_punctuation():
    assert normalize_text("  Tôi  yêu_học_máy  ! ") == "Tôi yêu_học_máy !"


def test_translation_memory_uses_majority_then_shorter_then_lexicographic():
    src = ["甲", "甲", "甲", "乙", "乙", "丙", "丙"]
    tgt = ["xin chào", "xin chào", "chào", "rất tốt", "tốt", "b", "a"]
    memory = build_translation_memory(src, tgt)
    assert memory["甲"] == "xin chào"  # majority
    assert memory["乙"] == "tốt"       # cùng tần suất -> ngắn hơn
    assert memory["丙"] == "a"         # vẫn hòa -> thứ tự từ điển


def test_deduplicate_and_group_split_have_no_source_leakage():
    pairs = [("a", "x"), ("a", "x"), ("a", "y"), ("b", "z"), ("c", "q")]
    unique = deduplicate_pairs(pairs)
    train, valid = group_split(unique, valid_ratio=0.34, seed=42)
    assert len(unique) == 4
    assert {s for s, _ in train}.isdisjoint({s for s, _ in valid})
    assert sorted(train + valid) == sorted(unique)


def test_dataset_collate_and_causal_mask():
    ds = TranslationDataset([("a bb", "ccc d")], FakeTokenizer(), max_len=5)
    src, tgt = ds[0]
    assert src.tolist()[0] == 2 and src.tolist()[-1] == 3
    assert tgt.tolist()[0] == 2 and tgt.tolist()[-1] == 3
    src_batch, tgt_batch = collate_batch([(src, tgt), (src[:2], tgt[:3])])
    assert src_batch.shape == (2, len(src))
    assert tgt_batch[1, -1].item() == 0
    mask = causal_mask(4, torch.device("cpu"))
    assert mask.dtype == torch.bool
    assert mask[0, 1] and not mask[1, 0]


def test_transformer_forward_shape():
    cfg = ModelConfig(vocab_size=32, d_model=24, nhead=4, encoder_layers=1,
                      decoder_layers=1, dim_feedforward=48, dropout=0.0, max_len=8)
    model = Seq2SeqTransformer(cfg)
    src = torch.tensor([[2, 5, 3, 0], [2, 6, 7, 3]])
    tgt = torch.tensor([[2, 8, 3, 0], [2, 9, 10, 3]])
    logits = model(src, tgt[:, :-1])
    assert logits.shape == (2, 3, 32)
    assert model.output.weight.data_ptr() == model.embedding.weight.data_ptr()


def test_transformer_initial_logits_are_not_exploded():
    cfg = ModelConfig(vocab_size=128, d_model=24, nhead=4, encoder_layers=1,
                      decoder_layers=1, dim_feedforward=48, dropout=0.0, max_len=8)
    model = Seq2SeqTransformer(cfg)
    src = torch.tensor([[2, 5, 6, 3]])
    tgt = torch.tensor([[2, 7, 8]])
    logits = model(src, tgt)
    assert model.embedding.weight.std().item() < 0.3
    assert logits.std().item() < 3.0


def test_write_csv_and_package_zip(tmp_path):
    public = tmp_path / "public_test.csv"
    private = tmp_path / "private_test.csv"
    write_submission_csv(["甲", "乙"], ["a", "b"], public)
    write_submission_csv(["丙"], ["c"], private)
    frame = pd.read_csv(public)
    assert list(frame.columns) == ["tieng_trung", "tieng_viet"]
    archive = package_submission(public, private, tmp_path / "submission.zip")
    import zipfile
    with zipfile.ZipFile(archive) as zf:
        assert sorted(zf.namelist()) == ["private_test.csv", "public_test.csv"]


def test_write_csv_rejects_wrong_row_count(tmp_path):
    with pytest.raises(ValueError, match="số câu"):
        write_submission_csv(["甲"], [], tmp_path / "bad.csv")


def test_scheduler_resume_restores_current_learning_rate():
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.AdamW([parameter], lr=0.0)
    scheduler = WarmupInverseSqrtScheduler(optimizer, peak_lr=0.01, warmup_steps=2)
    scheduler.step(); expected_lr = scheduler.step()

    new_optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.tensor(1.0))], lr=0.0)
    resumed = WarmupInverseSqrtScheduler(new_optimizer)
    resumed.load_state_dict(scheduler.state_dict())
    assert resumed.step_num == 2
    assert new_optimizer.param_groups[0]["lr"] == pytest.approx(expected_lr)

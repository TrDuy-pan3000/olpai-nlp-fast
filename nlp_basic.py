"""Pipeline dịch Hoa -> Việt đơn giản cho OlpAI.

Module chỉ dùng các khối chuẩn của PyTorch. Notebook Colab gọi các hàm ở đây
để phần trình bày ngắn, còn người học vẫn có thể mở file này để đọc toàn bộ.
"""

from __future__ import annotations

import math
import random
import unicodedata
import zipfile
from copy import deepcopy
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm


PAD_ID, UNK_ID, BOS_ID, EOS_ID = 0, 1, 2, 3


@dataclass
class ModelConfig:
    """Cấu hình Fast Contest; giảm layer nếu chỉ chạy smoke test."""

    vocab_size: int = 8000
    d_model: int = 256
    nhead: int = 8
    encoder_layers: int = 4
    decoder_layers: int = 4
    dim_feedforward: int = 1024
    dropout: float = 0.1
    max_len: int = 40


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_text(text: str) -> str:
    """Chuẩn hóa rất nhẹ; không phá dấu gạch dưới của dữ liệu tiếng Việt."""

    return " ".join(unicodedata.normalize("NFC", text).strip().split())


def read_lines(path: str | Path) -> list[str]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy file: {path}")
    return [normalize_text(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_parallel_data(src_path: str | Path, tgt_path: str | Path) -> list[tuple[str, str]]:
    src, tgt = read_lines(src_path), read_lines(tgt_path)
    if len(src) != len(tgt):
        raise ValueError(f"Số dòng nguồn ({len(src)}) khác số dòng đích ({len(tgt)}).")
    return list(zip(src, tgt))


def deduplicate_pairs(pairs: Sequence[tuple[str, str]]) -> list[tuple[str, str]]:
    """Xóa cặp trùng hoàn toàn nhưng giữ nhiều bản dịch khác nhau."""

    return list(dict.fromkeys((normalize_text(s), normalize_text(t)) for s, t in pairs))


def group_split(
    pairs: Sequence[tuple[str, str]], valid_ratio: float = 0.05, seed: int = 42
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Chia theo câu nguồn để cùng một câu không lọt sang cả train và valid."""

    if not 0 < valid_ratio < 1:
        raise ValueError("valid_ratio phải nằm trong khoảng (0, 1).")
    groups: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for src, tgt in pairs:
        groups[src].append((src, tgt))
    sources = list(groups)
    random.Random(seed).shuffle(sources)
    n_valid = max(1, round(len(sources) * valid_ratio))
    valid_sources = set(sources[:n_valid])
    train = [pair for src in sources if src not in valid_sources for pair in groups[src]]
    valid = [pair for src in sources if src in valid_sources for pair in groups[src]]
    return train, valid


def build_translation_memory(src_lines: Sequence[str], tgt_lines: Sequence[str]) -> dict[str, str]:
    if len(src_lines) != len(tgt_lines):
        raise ValueError("Không thể tạo memory vì số câu nguồn và đích khác nhau.")
    choices: dict[str, Counter[str]] = defaultdict(Counter)
    for src, tgt in zip(src_lines, tgt_lines):
        choices[normalize_text(src)][normalize_text(tgt)] += 1
    memory = {}
    for src, counts in choices.items():
        # Ưu tiên tần suất cao, rồi câu ít token, cuối cùng thứ tự từ điển.
        memory[src] = min(counts, key=lambda text: (-counts[text], len(text.split()), text))
    return memory


def train_sentencepiece(
    train_pairs: Sequence[tuple[str, str]], output_dir: str | Path, vocab_size: int = 8000
):
    """Huấn luyện joint BPE chỉ từ train; trả về SentencePieceProcessor."""

    import sentencepiece as spm

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    corpus = output_dir / "tokenizer_corpus.txt"
    corpus.write_text("\n".join(text for pair in train_pairs for text in pair), encoding="utf-8")
    prefix = output_dir / "spm_zh_vi"
    spm.SentencePieceTrainer.train(
        input=str(corpus), model_prefix=str(prefix), vocab_size=vocab_size,
        model_type="bpe", character_coverage=1.0,
        pad_id=PAD_ID, unk_id=UNK_ID, bos_id=BOS_ID, eos_id=EOS_ID,
        hard_vocab_limit=False,
    )
    sp = spm.SentencePieceProcessor(model_file=str(prefix) + ".model")
    return sp


def filter_pairs_by_token_length(pairs, tokenizer, max_len: int = 40):
    """Giữ các cặp vừa trọn max_len để không cắt mất phần cuối câu."""

    kept = []
    for src, tgt in tqdm(pairs, desc="Lọc độ dài", leave=False):
        src_len = len(tokenizer.encode(src, out_type=int)) + 2  # BOS + EOS
        tgt_len = len(tokenizer.encode(tgt, out_type=int)) + 2
        if src_len <= max_len and tgt_len <= max_len:
            kept.append((src, tgt))
    return kept


class TranslationDataset(Dataset):
    """Tokenize một lần khi khởi tạo để DataLoader trong mỗi epoch chạy nhanh."""

    def __init__(self, pairs: Sequence[tuple[str, str]], tokenizer, max_len: int = 40):
        self.items = []
        for src, tgt in tqdm(pairs, desc="Tokenizing", leave=False):
            src_ids = [BOS_ID] + tokenizer.encode(src, out_type=int)[: max_len - 2] + [EOS_ID]
            tgt_ids = [BOS_ID] + tokenizer.encode(tgt, out_type=int)[: max_len - 2] + [EOS_ID]
            self.items.append((torch.tensor(src_ids), torch.tensor(tgt_ids)))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


class BidirectionalTranslationDataset(TranslationDataset):
    """Bản basic của dataset hai chiều trong lời giải mẫu.

    Nhánh này chỉ để học/thử nghiệm và mặc định tắt trong notebook vì bài thi
    cần Hoa -> Việt. Khi bật, mỗi cặp được thêm một bản đảo Việt -> Hoa.
    """

    def __init__(self, pairs, tokenizer, max_len: int = 40, include_reverse: bool = False):
        expanded = list(pairs)
        if include_reverse:
            expanded += [(tgt, src) for src, tgt in pairs]
        super().__init__(expanded, tokenizer, max_len)


def collate_batch(batch):
    src_list, tgt_list = zip(*batch)
    src = nn.utils.rnn.pad_sequence(src_list, batch_first=True, padding_value=PAD_ID)
    tgt = nn.utils.rnn.pad_sequence(tgt_list, batch_first=True, padding_value=PAD_ID)
    return src, tgt


def causal_mask(length: int, device: torch.device) -> torch.Tensor:
    """True ở phía trên đường chéo: decoder không được nhìn token tương lai."""

    return torch.triu(torch.ones(length, length, dtype=torch.bool, device=device), diagonal=1)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int, dropout: float):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(x + self.pe[:, : x.size(1)])


class BasicMultiHeadAttention(nn.Module):
    """Attention chuẩn: mọi head có cùng kích thước, không GQA và không RoPE."""

    def __init__(self, d_model: int, nhead: int, dropout: float):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )

    def forward(self, query, key, value, key_padding_mask=None, attn_mask=None):
        output, _ = self.attention(
            query, key, value,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
            need_weights=False,
        )
        return output


class BasicFeedForward(nn.Module):
    """FFN cơ bản thay SwiGLU: Linear -> ReLU -> Dropout -> Linear."""

    def __init__(self, d_model: int, dim_feedforward: int, dropout: float):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )

    def forward(self, x):
        return self.layers(x)


class EncoderLayer(nn.Module):
    """Self-attention rồi FFN; mỗi nhánh có residual và LayerNorm."""

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float):
        super().__init__()
        self.self_attention = BasicMultiHeadAttention(d_model, nhead, dropout)
        self.feed_forward = BasicFeedForward(d_model, dim_feedforward, dropout)
        self.norm1, self.norm2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, padding_mask=None):
        attended = self.self_attention(x, x, x, key_padding_mask=padding_mask)
        x = self.norm1(x + self.dropout(attended))
        transformed = self.feed_forward(x)
        return self.norm2(x + self.dropout(transformed))


class Encoder(nn.Module):
    """Chồng nhiều EncoderLayer giống bộ khung lời giải mẫu."""

    def __init__(self, layer: EncoderLayer, num_layers: int, d_model: int):
        super().__init__()
        self.layers = nn.ModuleList(deepcopy(layer) for _ in range(num_layers))
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x, padding_mask=None):
        for layer in self.layers:
            x = layer(x, padding_mask)
        return self.final_norm(x)


class DecoderLayer(nn.Module):
    """Masked self-attention, cross-attention với encoder, rồi FFN."""

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float):
        super().__init__()
        self.self_attention = BasicMultiHeadAttention(d_model, nhead, dropout)
        self.cross_attention = BasicMultiHeadAttention(d_model, nhead, dropout)
        self.feed_forward = BasicFeedForward(d_model, dim_feedforward, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memory, tgt_padding_mask=None, memory_padding_mask=None):
        mask = causal_mask(x.size(1), x.device)
        attended = self.self_attention(
            x, x, x, key_padding_mask=tgt_padding_mask, attn_mask=mask
        )
        x = self.norm1(x + self.dropout(attended))
        crossed = self.cross_attention(
            x, memory, memory, key_padding_mask=memory_padding_mask
        )
        x = self.norm2(x + self.dropout(crossed))
        transformed = self.feed_forward(x)
        return self.norm3(x + self.dropout(transformed))


class Decoder(nn.Module):
    """Chồng nhiều DecoderLayer giống bộ khung lời giải mẫu."""

    def __init__(self, layer: DecoderLayer, num_layers: int, d_model: int):
        super().__init__()
        self.layers = nn.ModuleList(deepcopy(layer) for _ in range(num_layers))
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x, memory, tgt_padding_mask=None, memory_padding_mask=None):
        for layer in self.layers:
            x = layer(x, memory, tgt_padding_mask, memory_padding_mask)
        return self.final_norm(x)


class Seq2SeqTransformer(nn.Module):
    """Transformer viết tường minh để có thể đọc và thay từng khối."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.d_model, padding_idx=PAD_ID)
        self.position = PositionalEncoding(config.d_model, config.max_len, config.dropout)
        self.encoder = Encoder(
            EncoderLayer(config.d_model, config.nhead, config.dim_feedforward, config.dropout),
            config.encoder_layers, config.d_model,
        )
        self.decoder = Decoder(
            DecoderLayer(config.d_model, config.nhead, config.dim_feedforward, config.dropout),
            config.decoder_layers, config.d_model,
        )
        self.output = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.output.weight = self.embedding.weight  # weight tying: ít tham số hơn
        self.scale = math.sqrt(config.d_model)
        # Embedding mặc định của PyTorch có std≈1; khi weight tying sẽ làm logits
        # quá lớn. Scale chuẩn này giữ loss ban đầu gần log(vocab).
        nn.init.normal_(self.embedding.weight, mean=0.0, std=config.d_model ** -0.5)
        with torch.no_grad():
            self.embedding.weight[PAD_ID].zero_()

    def encode(self, src):
        src_pad = src.eq(PAD_ID)
        x = self.position(self.embedding(src) * self.scale)
        return self.encoder(x, src_pad), src_pad

    def decode(self, tgt, memory, src_pad):
        tgt_pad = tgt.eq(PAD_ID)
        y = self.position(self.embedding(tgt) * self.scale)
        hidden = self.decoder(y, memory, tgt_pad, src_pad)
        return self.output(hidden)

    def forward(self, src, tgt_input):
        memory, src_pad = self.encode(src)
        return self.decode(tgt_input, memory, src_pad)


class LabelSmoothedCrossEntropyLoss(nn.Module):
    """Tên khối giống lời giải mẫu, bên trong dùng API chuẩn của PyTorch."""

    def __init__(self, ignore_index: int = PAD_ID, smoothing: float = 0.1):
        super().__init__()
        self.ignore_index = ignore_index
        self.smoothing = smoothing

    def forward(self, logits, targets):
        return F.cross_entropy(
            logits, targets,
            ignore_index=self.ignore_index,
            label_smoothing=self.smoothing,
        )


@dataclass
class ContrastiveConfig:
    """Cấu hình nhánh contrastive tùy chọn; mặc định không chạy để tiết kiệm giờ thi."""

    d_model: int = 256
    projection_dim: int = 128
    temperature: float = 0.1
    weight: float = 0.0


class ProjectionHead(nn.Module):
    """Projection head tối giản thay cho nhánh contrastive phức tạp."""

    def __init__(self, d_model: int, projection_dim: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, projection_dim),
        )

    def forward(self, x):
        return F.normalize(self.layers(x), dim=-1)


def mean_pool(hidden, padding_mask):
    """Lấy trung bình các token thật, bỏ qua PAD."""

    valid = (~padding_mask).unsqueeze(-1).to(hidden.dtype)
    return (hidden * valid).sum(1) / valid.sum(1).clamp_min(1.0)


def contrastive_loss(source_vectors, target_vectors, temperature: float = 0.1):
    """InfoNCE đối xứng cơ bản; chỉ dùng khi CONTRASTIVE_WEIGHT > 0."""

    source_vectors = F.normalize(source_vectors, dim=-1)
    target_vectors = F.normalize(target_vectors, dim=-1)
    logits = source_vectors @ target_vectors.T / temperature
    labels = torch.arange(logits.size(0), device=logits.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2


def compute_crosslingual_loss(model, projection, src, tgt, temperature: float = 0.1):
    """Mã hóa hai ngôn ngữ bằng cùng encoder rồi kéo đúng cặp lại gần nhau."""

    source_hidden, source_pad = model.encode(src)
    target_hidden, target_pad = model.encode(tgt)
    source_vectors = projection(mean_pool(source_hidden, source_pad))
    target_vectors = projection(mean_pool(target_hidden, target_pad))
    return contrastive_loss(source_vectors, target_vectors, temperature)


def select_vi2zh_window(epoch: int, start_epoch: int = 3, end_epoch: int = 5):
    """Cửa sổ epoch đơn giản cho nhánh đảo chiều tùy chọn."""

    return start_epoch <= epoch <= end_epoch


def contrastive_train_epoch(model, projection, loader, optimizer, device, temperature=0.1):
    """Một epoch contrastive tối giản, tách riêng và không bật trong cấu hình thi."""

    model.train(); projection.train()
    total = 0.0
    for src, tgt in tqdm(loader, desc="Contrastive", leave=False):
        src, tgt = src.to(device), tgt.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = compute_crosslingual_loss(model, projection, src, tgt, temperature)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(projection.parameters()), 1.0)
        optimizer.step()
        total += loss.item()
    return total / max(1, len(loader))


class WarmupInverseSqrtScheduler:
    def __init__(self, optimizer, peak_lr: float = 5e-4, warmup_steps: int = 300):
        self.optimizer, self.peak_lr = optimizer, peak_lr
        self.warmup_steps, self.step_num = warmup_steps, 0

    def step(self):
        self.step_num += 1
        if self.step_num <= self.warmup_steps:
            lr = self.peak_lr * self.step_num / self.warmup_steps
        else:
            lr = self.peak_lr * math.sqrt(self.warmup_steps / self.step_num)
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        return lr

    def state_dict(self):
        return {"peak_lr": self.peak_lr, "warmup_steps": self.warmup_steps, "step_num": self.step_num}

    def load_state_dict(self, state):
        self.peak_lr, self.warmup_steps, self.step_num = state["peak_lr"], state["warmup_steps"], state["step_num"]
        if self.step_num <= self.warmup_steps:
            lr = self.peak_lr * self.step_num / self.warmup_steps
        else:
            lr = self.peak_lr * math.sqrt(self.warmup_steps / self.step_num)
        for group in self.optimizer.param_groups:
            group["lr"] = lr


def make_optimizer(model, lr: float = 5e-4, weight_decay: float = 0.01):
    kwargs = dict(lr=lr, betas=(0.9, 0.98), weight_decay=weight_decay)
    if torch.cuda.is_available():
        try:
            return torch.optim.AdamW(model.parameters(), fused=True, **kwargs)
        except TypeError:
            pass
    return torch.optim.AdamW(model.parameters(), **kwargs)


def train_one_epoch(model, loader, optimizer, scheduler, criterion, device, scaler=None, epoch=1):
    model.train()
    total = 0.0
    bar = tqdm(loader, desc=f"Train {epoch:02d}")
    use_amp = device.type == "cuda"
    for step, (src, tgt) in enumerate(bar, 1):
        src, tgt = src.to(device), tgt.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(src, tgt[:, :-1])
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt[:, 1:].reshape(-1))
        if scaler is not None and use_amp:
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update()
        else:
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        lr = scheduler.step()
        total += loss.item()
        gpu = f"{torch.cuda.memory_allocated()/2**30:.1f}GB" if use_amp else "CPU"
        bar.set_postfix(loss=f"{loss.item():.3f}", avg=f"{total/step:.3f}", lr=f"{lr:.2e}", gpu=gpu)
    return total / max(1, len(loader))


@torch.inference_mode()
def evaluate_loss(model, loader, criterion, device):
    model.eval(); total = 0.0
    for src, tgt in tqdm(loader, desc="Valid loss", leave=False):
        src, tgt = src.to(device), tgt.to(device)
        logits = model(src, tgt[:, :-1])
        total += criterion(logits.reshape(-1, logits.size(-1)), tgt[:, 1:].reshape(-1)).item()
    return total / max(1, len(loader))


@torch.inference_mode()
def greedy_decode_batch(model, src, max_len: int = 40):
    model.eval()
    memory, src_pad = model.encode(src)
    generated = torch.full((src.size(0), 1), BOS_ID, dtype=torch.long, device=src.device)
    finished = torch.zeros(src.size(0), dtype=torch.bool, device=src.device)
    for _ in range(max_len - 1):
        next_id = model.decode(generated, memory, src_pad)[:, -1].argmax(-1)
        next_id = torch.where(finished, torch.full_like(next_id, EOS_ID), next_id)
        generated = torch.cat([generated, next_id[:, None]], dim=1)
        finished |= next_id.eq(EOS_ID)
        if finished.all():
            break
    return generated


def decode_token_rows(rows: torch.Tensor, tokenizer) -> list[str]:
    texts = []
    for row in rows.tolist():
        ids = row[1:]
        if EOS_ID in ids:
            ids = ids[: ids.index(EOS_ID)]
        texts.append(normalize_text(tokenizer.decode([i for i in ids if i > EOS_ID])))
    return texts


@torch.inference_mode()
def evaluate_bleu(model, loader, tokenizer, device, max_len=40):
    hypotheses, references = [], []
    for src, tgt in tqdm(loader, desc="Valid BLEU", leave=False):
        pred = greedy_decode_batch(model, src.to(device), max_len).cpu()
        hypotheses.extend(decode_token_rows(pred, tokenizer))
        references.extend(decode_token_rows(tgt, tokenizer))
    import sacrebleu
    return sacrebleu.corpus_bleu(hypotheses, [references]).score


def save_checkpoint(path, model, optimizer, scheduler, epoch, best_bleu, config, scaler=None):
    payload = {
        "model": model.state_dict(), "optimizer": optimizer.state_dict() if optimizer else None,
        "scheduler": scheduler.state_dict() if scheduler else None,
        "scaler": scaler.state_dict() if scaler else None, "epoch": epoch,
        "best_bleu": best_bleu, "config": asdict(config),
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path, model, optimizer=None, scheduler=None, scaler=None, device="cpu"):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    if optimizer is not None and checkpoint.get("optimizer"):
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and checkpoint.get("scheduler"):
        scheduler.load_state_dict(checkpoint["scheduler"])
    if scaler is not None and checkpoint.get("scaler"):
        scaler.load_state_dict(checkpoint["scaler"])
    return checkpoint


def encode_source_batch(sentences, tokenizer, max_len=40):
    rows = [torch.tensor([BOS_ID] + tokenizer.encode(s, out_type=int)[: max_len - 2] + [EOS_ID]) for s in sentences]
    return nn.utils.rnn.pad_sequence(rows, batch_first=True, padding_value=PAD_ID)


@torch.inference_mode()
def translate_sentences(model, sentences, tokenizer, device, memory=None, batch_size=128, max_len=40):
    """Dịch nhanh bằng greedy theo batch; memory được ưu tiên khi khớp exact."""

    memory = memory or {}
    results: list[str | None] = [None] * len(sentences)
    pending = []
    for i, sentence in enumerate(sentences):
        key = normalize_text(sentence)
        if key in memory:
            results[i] = memory[key]
        else:
            pending.append((i, key))
    for start in tqdm(range(0, len(pending), batch_size), desc="Greedy inference"):
        chunk = pending[start : start + batch_size]
        src = encode_source_batch([s for _, s in chunk], tokenizer, max_len).to(device)
        texts = decode_token_rows(greedy_decode_batch(model, src, max_len).cpu(), tokenizer)
        for (index, _), text in zip(chunk, texts):
            results[index] = text
    return [text or "" for text in results]


@dataclass
class BeamSearchHypothesis:
    """Một ứng viên trong beam, tách thành class như lời giải mẫu."""

    tokens: list[int]
    score: float


@torch.inference_mode()
def beam_search_decode_sentence(model, sentence, tokenizer, device, beam_size=3, max_len=40, length_penalty=0.6):
    """Beam search dễ đọc; dùng tùy chọn khi còn thời gian, không phải mặc định."""

    src = encode_source_batch([normalize_text(sentence)], tokenizer, max_len).to(device)
    memory, src_pad = model.encode(src)
    beams = [([BOS_ID], 0.0)]
    finished = []
    for _ in range(max_len - 1):
        active = [(tokens, score) for tokens, score in beams if tokens[-1] != EOS_ID]
        finished.extend((tokens, score) for tokens, score in beams if tokens[-1] == EOS_ID)
        if not active:
            break
        tgt = nn.utils.rnn.pad_sequence(
            [torch.tensor(tokens, device=device) for tokens, _ in active], batch_first=True, padding_value=PAD_ID
        )
        repeated_memory = memory.expand(len(active), -1, -1)
        repeated_mask = src_pad.expand(len(active), -1)
        log_probs = model.decode(tgt, repeated_memory, repeated_mask)[:, -1].log_softmax(-1)
        candidates = []
        for row, (tokens, score) in enumerate(active):
            values, indices = log_probs[row].topk(beam_size)
            candidates.extend((tokens + [idx.item()], score + value.item()) for value, idx in zip(values, indices))
        beams = sorted(candidates, key=lambda item: item[1] / (len(item[0]) ** length_penalty), reverse=True)[:beam_size]
    finished.extend(beams)
    best, _ = max(finished, key=lambda item: item[1] / (len(item[0]) ** length_penalty))
    return decode_token_rows(torch.tensor([best]), tokenizer)[0]


def write_submission_csv(src_lines, translations, output_path):
    if len(src_lines) != len(translations):
        raise ValueError("Số câu nguồn và số câu dịch không bằng nhau.")
    frame = pd.DataFrame({"tieng_trung": list(src_lines), "tieng_viet": list(translations)})
    output_path = Path(output_path); output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False, encoding="utf-8-sig")
    return output_path


def package_tokenizer(model_path, vocab_path, output_zip):
    """Đóng gói tokenizer để mang checkpoint sang runtime khác."""

    model_path, vocab_path, output_zip = map(Path, (model_path, vocab_path, output_zip))
    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.write(model_path, model_path.name)
        archive.write(vocab_path, vocab_path.name)
    return output_zip


def package_submission(public_csv, private_csv, output_zip):
    public_csv, private_csv, output_zip = map(Path, (public_csv, private_csv, output_zip))
    for path in (public_csv, private_csv):
        if not path.is_file():
            raise FileNotFoundError(path)
        if list(pd.read_csv(path, nrows=1).columns) != ["tieng_trung", "tieng_viet"]:
            raise ValueError(f"Sai schema CSV: {path}")
    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.write(public_csv, public_csv.name)
        archive.write(private_csv, private_csv.name)
    return output_zip


def make_loader(pairs, tokenizer, batch_size=128, max_len=40, shuffle=True, num_workers=0):
    dataset = TranslationDataset(pairs, tokenizer, max_len)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_batch,
                      num_workers=num_workers, pin_memory=torch.cuda.is_available())

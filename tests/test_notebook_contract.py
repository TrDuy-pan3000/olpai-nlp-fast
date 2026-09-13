import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "OlpAI_NLP_Fast_Contest.ipynb"


def load_notebook():
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def test_notebook_is_one_click_colab_ready():
    notebook = load_notebook()
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )
    assert "TrDuy-pan3000/olpai-nlp-fast.git" in source
    assert "190wIN301_X2Z7dqtPmLA7Z4Cggn3twl4" in source
    assert "files.download(str(submission))" in source
    assert "sys.modules.pop('nlp_basic', None)" in source
    assert "drive.mount" not in source
    assert "D:\\OlpAI" not in source


def test_notebook_defaults_to_real_basic_transformer_training():
    notebook = load_notebook()
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    assert "SMOKE_TEST = False" in source
    assert "VOCAB_SIZE = 2000 if SMOKE_TEST else 8000" in source
    assert "d_model=256" in source
    assert "encoder_layers=4, decoder_layers=4" in source
    for forbidden_code in (
        "class RMSNorm",
        "class RoPE",
        "class GroupedQueryAttentionRoPE",
        "class FFN_SwiGLU",
    ):
        assert forbidden_code not in source


def test_notebook_shows_transformer_architecture_instead_of_hiding_it():
    notebook = load_notebook()
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    for class_name in (
        "class BasicMultiHeadAttention",
        "class BasicFeedForward",
        "class EncoderLayer",
        "class DecoderLayer",
        "class Encoder",
        "class Decoder",
        "class Seq2SeqTransformer",
    ):
        assert class_name in source
    assert "from nlp_basic import *" not in source
    assert "self.transformer = nn.Transformer" not in source
    assert "class LabelSmoothedCrossEntropyLoss" in source
    assert "class BeamSearchHypothesis" in source
    assert "class BidirectionalTranslationDataset" in source
    assert "class ContrastiveConfig" in source
    assert "class ProjectionHead" in source
    assert "class WarmupInverseSqrtScheduler" in source
    assert "def compute_crosslingual_loss" in source
    assert "def contrastive_train_epoch" in source
    assert "def select_vi2zh_window" in source


def test_notebook_avoids_colab_multiprocessing_loader_shutdown_warning():
    notebook = load_notebook()
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    assert "config.max_len, True, 0" in source
    assert "config.max_len, False, 0" in source


def test_all_python_cells_compile():
    notebook = load_notebook()
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        lines = [line for line in cell.get("source", []) if not line.lstrip().startswith("%")]
        try:
            ast.parse("".join(lines))
        except SyntaxError as error:
            raise AssertionError(f"Cell {index} không compile được: {error}") from error

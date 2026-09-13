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
    for advanced_name in ("RMSNorm", "RoPE", "GroupedQueryAttention", "SwiGLU", "contrastive"):
        assert advanced_name not in source


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

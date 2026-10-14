"""Integration tests for Typer CLI commands."""

from typer.testing import CliRunner

from tinymodel.cli import app

runner = CliRunner()


def test_cli_list_models():
    """Verify tinymodel list-models command."""
    result = runner.invoke(app, ["list-models"])
    assert result.exit_code == 0
    assert "resnet18" in result.stdout
    assert "tiny_convnet" in result.stdout


def test_cli_hardware():
    """Verify tinymodel hardware command."""
    result = runner.invoke(app, ["hardware"])
    assert result.exit_code == 0
    assert "Operating System" in result.stdout
    assert "Python Version" in result.stdout


def test_cli_profile():
    """Verify tinymodel profile command."""
    result = runner.invoke(app, ["profile", "--model", "tiny_convnet"])
    assert result.exit_code == 0
    assert "Profile Summary: tiny_convnet" in result.stdout
    assert "Total Parameters" in result.stdout


def test_cli_export_onnx(tmp_path):
    """Verify tinymodel export-onnx command."""
    out_file = tmp_path / "tiny.onnx"
    result = runner.invoke(app, ["export-onnx", "--model", "tiny_convnet", "--output", str(out_file)])
    assert result.exit_code == 0
    assert "ONNX Export Summary" in result.stdout
    assert out_file.exists()


def test_cli_benchmark(tmp_path):
    """Verify tinymodel benchmark command."""
    json_out = tmp_path / "bench.json"
    result = runner.invoke(app, [
        "benchmark",
        "--model", "tiny_convnet",
        "--warmup", "2",
        "--runs", "5",
        "--json-out", str(json_out),
    ])
    assert result.exit_code == 0
    assert "Benchmark Report" in result.stdout
    assert json_out.exists()


def test_cli_compare(tmp_path):
    """Verify tinymodel compare command."""
    rep1 = tmp_path / "rep1.json"
    rep2 = tmp_path / "rep2.json"

    # Generate two reports
    runner.invoke(app, ["benchmark", "--model", "tiny_convnet", "--warmup", "1", "--runs", "2", "--json-out", str(rep1)])
    runner.invoke(app, ["benchmark", "--model", "tiny_convnet", "--warmup", "1", "--runs", "2", "--json-out", str(rep2)])

    result = runner.invoke(app, ["compare", "--baseline", str(rep1), "--optimized", str(rep2)])
    assert result.exit_code == 0
    assert "Comparison:" in result.stdout
    assert "Compression Ratio" in result.stdout

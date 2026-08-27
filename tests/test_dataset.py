"""Dataset aggregation and web-replay tests."""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request

import pytest
from mock_openai import start_mock_server

from rubikbench.aggregate import aggregate_files, read_run, write_dataset
from rubikbench.benchmark import BenchmarkRunner, export_jsonl
from rubikbench.cli import main
from rubikbench.config import BenchmarkConfig
from rubikbench.llm import OpenAICompatibleClient


@pytest.fixture()
def mock():
    server, url = start_mock_server()
    yield server, url
    server.shutdown()


def _make_run(url: str, path, num_solves: int = 2) -> None:
    cfg = BenchmarkConfig(base_url=url, model="mock", num_solves=num_solves, max_turns=12, seed=7)
    result = BenchmarkRunner(cfg, OpenAICompatibleClient(
        base_url=url, api_key="k", model="mock", max_retries=0,
    )).run()
    export_jsonl(result, path)


# --------------------------------------------------------------------------- aggregation

def test_aggregate_files_merges_runs_and_denormalizes(mock, tmp_path):
    _, url = mock
    run_a = tmp_path / "a.jsonl"
    run_b = tmp_path / "b.jsonl"
    _make_run(url, run_a, num_solves=2)
    _make_run(url, run_b, num_solves=1)

    ds = aggregate_files([run_a, run_b])
    assert ds["event"] == "dataset"
    assert ds["runs"] == 2
    assert ds["total_solves"] == 3
    assert ds["models"] == ["mock"]
    assert len(ds["runs_detail"]) == 2
    assert all(s["run"] in (str(run_a), str(run_b)) for s in ds["solves"])
    assert all(s["model"] == "mock" for s in ds["solves"])
    # solves carry their timeline so the viewer can replay them
    assert all(s["timeline"] for s in ds["solves"])

    t = ds["totals"]
    assert t["solves"] == 3
    assert t["solved"] == 3
    assert t["prompt_tokens"] > 0
    assert t["cached_tokens"] > 0  # the mock always reports cached tokens
    assert t["truncated"] == 0
    assert t["retries"] == 0


def test_read_run_rejects_invalid_file(tmp_path):
    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps({"event": "solve", "index": 0}) + "\n")
    with pytest.raises(ValueError, match="no benchmark header"):
        read_run(bad)

    garbage = tmp_path / "garbage.json"
    garbage.write_text("not json at all\n")
    with pytest.raises(json.JSONDecodeError):
        read_run(garbage)


def test_write_dataset_round_trip(mock, tmp_path):
    _, url = mock
    run = tmp_path / "run.jsonl"
    _make_run(url, run)
    ds = aggregate_files([run])
    out = write_dataset(ds, tmp_path / "ds" / "dataset.json")
    loaded = json.loads(out.read_text())
    assert loaded["event"] == "dataset"
    assert loaded["total_solves"] == ds["total_solves"]
    assert loaded["solves"][0]["timeline"] == ds["solves"][0]["timeline"]


def test_export_aggregate_replay_keep_request_config_but_never_the_key(mock, tmp_path):
    _, url = mock
    secret = "sk-super-secret-not-for-export"
    cfg = BenchmarkConfig(
        base_url=url, model="acme/bench-model-1", api_key=secret,
        num_solves=1, max_turns=12, seed=7, temperature=0.3,
        context_window_tokens=16384, extra_body={"reasoning_effort": "high"},
        cache_retention=3600,
    )
    run = tmp_path / "run.jsonl"
    result = BenchmarkRunner(cfg, OpenAICompatibleClient(
        base_url=url, api_key=secret, model=cfg.model, max_retries=0,
    )).run()
    export_jsonl(result, run)

    # JSONL header carries model/endpoint/sampling/ctx/extra body, no key.
    config = read_run(run)[0]["config"]
    assert config["model"] == "acme/bench-model-1"
    assert config["base_url"] == url
    assert config["temperature"] == 0.3
    assert config["context_window_tokens"] == 16384
    assert config["extra_body"] == {"reasoning_effort": "high"}
    assert config["cache_retention"] == 3600
    assert "api_key" not in config
    assert secret not in run.read_text()

    # Dataset JSON denormalizes the same metadata and stays key-free.
    ds = aggregate_files([run])
    assert ds["models"] == ["acme/bench-model-1"]
    detail = ds["runs_detail"][0]
    assert detail["base_url"] == url
    assert detail["config"]["temperature"] == 0.3
    assert detail["config"]["cache_retention"] == 3600
    assert "api_key" not in detail["config"]
    dataset_bytes = write_dataset(ds, tmp_path / "dataset.json").read_text()
    assert "acme/bench-model-1" in dataset_bytes
    assert secret not in dataset_bytes

    # Web replay embeds the dataset: metadata visible, key never rendered.
    from rubikbench.webui.server import build_replay_document

    html = build_replay_document(run)
    assert "acme/bench-model-1" in html
    assert "api_key" not in html
    assert secret not in html


def test_cli_aggregate_command(mock, tmp_path, capsys):
    _, url = mock
    run = tmp_path / "run.jsonl"
    _make_run(url, run)
    out = tmp_path / "dataset.json"
    rc = main(["aggregate", str(run), "-o", str(out)])
    assert rc == 0
    assert out.exists()
    loaded = json.loads(out.read_text())
    assert loaded["total_solves"] == 2
    assert "cached_tokens" in loaded["totals"]
    captured = capsys.readouterr().out
    assert "solves across" in captured


def test_cli_aggregate_rejects_bad_input(tmp_path, capsys):
    rc = main(["aggregate", str(tmp_path / "missing.jsonl"), "-o", str(tmp_path / "x.json")])
    assert rc == 1
    assert "error:" in capsys.readouterr().err


# --------------------------------------------------------------------------- web replay

def test_build_replay_document_embeds_run(mock, tmp_path):
    _, url = mock
    run = tmp_path / "run.jsonl"
    _make_run(url, run, num_solves=1)
    from rubikbench.webui.server import build_replay_document

    html = build_replay_document(run)
    assert "RubikBench replay" in html
    assert "/*__RUN_DATA__*/" not in html
    start = html.find("window.RUN = ")
    end = html.find("</script>", start)
    payload = html[start + len("window.RUN = "):end].strip().rstrip(";")
    data = json.loads(payload)
    assert data["total_solves"] == 1
    assert data["solves"][0]["timeline"][0]["action"] == "start"


def test_view_handler_serves_page_and_assets(mock, tmp_path):
    _, url = mock
    run = tmp_path / "run.jsonl"
    _make_run(url, run, num_solves=1)
    from http.server import ThreadingHTTPServer

    from rubikbench.webui.server import _Handler

    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)

    server.run_path = str(run)  # type: ignore[attr-defined]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        page = urllib.request.urlopen(base + "/", timeout=5).read().decode()
        assert "window.RUN = " in page and "RubikBench replay" in page
        # query strings and /index.html must reach the same page (the embedded
        # dataset regenerates created_at per request, so compare without it)
        import re as _re

        def payload(html: str) -> dict:
            match = _re.search(r"window\.RUN = (\{.*?\})\s*</script>", html, _re.DOTALL)
            data = json.loads(match.group(1))
            data.pop("created_at", None)
            return data

        assert payload(urllib.request.urlopen(base + "/?x=1", timeout=5).read().decode()) == payload(page)
        assert payload(urllib.request.urlopen(base + "/index.html", timeout=5).read().decode()) == payload(page)
        js = urllib.request.urlopen(base + "/app.js?v=2", timeout=5).read().decode()
        assert "three" in js
        with pytest.raises(urllib.error.HTTPError):
            urllib.request.urlopen(base + "/missing", timeout=5)
    finally:
        server.shutdown()
        server.server_close()


def test_cli_view_missing_file(tmp_path, capsys):
    rc = main(["view", str(tmp_path / "nope.jsonl"), "--no-open"])
    assert rc == 1
    assert "error:" in capsys.readouterr().err

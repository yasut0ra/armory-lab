from __future__ import annotations

import pytest

flask = pytest.importorskip("flask")

from armory_lab.web import create_app


def test_web_index_get() -> None:
    app = create_app()
    client = app.test_client()
    resp = client.get("/")
    assert resp.status_code == 200
    body = resp.get_data(as_text=True)
    assert "Armory Lab BAI Console" in body


def test_web_index_post_runs() -> None:
    app = create_app()
    client = app.test_client()
    resp = client.post(
        "/",
        data={
            "algo": "lucb",
            "k": "8",
            "delta": "0.1",
            "means": "topgap:0.12",
            "seed": "1",
            "max_pulls": "30000",
        },
    )
    assert resp.status_code == 200
    body = resp.get_data(as_text=True)
    assert "推奨腕" in body
    assert "停止時の総試行数" in body

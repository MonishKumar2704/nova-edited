def test_health_endpoint_returns_ok(client):
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["success"] is True
    assert body["status"] == "ok"
    assert "llm" in body
    assert body["llm"]["provider"] == "none"
    assert body["llm"]["available"] is False

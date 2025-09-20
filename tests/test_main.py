from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_sentiment_analysis_valid_text():
    data = {"text": "This is a great day!"}
    response = client.post("/analize", json=data)
    assert response.status_code == 200
    assert isinstance(response.json(), list)
    for item in response.json():
        assert "label" in item
        assert "score" in item
        assert isinstance(item["label"], str)
        assert isinstance(item["score"], float)


def test_sentiment_analysis_empty_text():
    data = {"text": ""}
    response = client.post("/analize", json=data)
    assert response.status_code == 200
    assert isinstance(response.json(), list)
    # Make an assumption based on model behavior for empty input
    assert len(response.json()) > 0


def test_sentiment_analysis_special_characters_text():
    data = {"text": "!@#$%^&*()"}
    response = client.post("/analize", json=data)
    assert response.status_code == 200
    assert isinstance(response.json(), list)
    for item in response.json():
        assert "label" in item
        assert "score" in item
        assert isinstance(item["label"], str)
        assert isinstance(item["score"], float)

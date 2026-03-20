from app.schemas.response import PredictResponse


def test_predict_response_schema() -> None:
    payload = {
        "image_width": 640,
        "image_height": 480,
        "model": "yolov8n.pt",
        "detections": [
            {
                "class_id": 0,
                "class_name": "person",
                "confidence": 0.95,
                "bbox_xyxy": [10.0, 20.0, 100.0, 200.0],
            }
        ],
        "count": 1,
    }

    parsed = PredictResponse(**payload)
    assert parsed.count == 1
    assert parsed.detections[0].class_name == "person"

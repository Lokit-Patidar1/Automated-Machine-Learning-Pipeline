import io
import pickle
from typing import Any


def save_model(model: Any, scaler: Any, model_name: str) -> bytes:
    """Save model and scaler to bytes for download."""
    model_data = {
        'model': model,
        'scaler': scaler,
        'model_name': model_name,
    }
    buffer = io.BytesIO()
    pickle.dump(model_data, buffer)
    buffer.seek(0)
    return buffer.getvalue()

import io
import pickle
from typing import Any


def serialize_model(model: Any, scaler: Any, model_name: str) -> bytes:
	data = {'model': model, 'scaler': scaler, 'model_name': model_name}
	buffer = io.BytesIO()
	pickle.dump(data, buffer)
	buffer.seek(0)
	return buffer.getvalue()

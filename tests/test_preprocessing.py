from PIL import Image
import numpy as np

from signature_verifier.config import AppSettings
from signature_verifier.preprocessing import SignatureTransform


def test_model_assets_are_project_relative_and_present() -> None:
    AppSettings.from_env().validate(require_detectors=True)


def test_inference_transform_shape_and_finite_values() -> None:
    image = Image.new("RGB", (400, 180), "white")
    tensor = SignatureTransform()(image)
    assert tensor.shape == (3, 224, 224)
    assert tensor.dtype == np.float32
    assert np.isfinite(tensor).all()


def test_dark_ink_is_encoded_as_bright_foreground_on_dark_canvas() -> None:
    image = Image.new("RGB", (320, 160), "white")
    pixels = np.asarray(image).copy()
    pixels[70:90, 60:260] = 0
    tensor = SignatureTransform()(Image.fromarray(pixels))
    restored = np.clip(
        tensor * np.asarray((0.229, 0.224, 0.225), dtype=np.float32)[:, None, None]
        + np.asarray((0.485, 0.456, 0.406), dtype=np.float32)[:, None, None],
        0,
        1,
    )
    assert restored.max() > 0.9
    assert np.median(restored) < 0.1

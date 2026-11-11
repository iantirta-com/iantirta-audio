import librosa
import numpy as np
import pytest

from iantirta.audio import feature

# @pytest.mark.parametrize(
#     "x",
#     [
#         np.array([0.0, 1.0, -2.0, 3.5]),
#         np.array([1 + 2j, 3 - 4j, -2 + 0.5j]),
#         np.array([], dtype=np.float64),
#     ],
# )
# def test_abs2(x):
#     expected = np.abs(x) ** 2
#     actual = audio_utils.abs2(x)

#     np.testing.assert_allclose(actual, expected)


# @pytest.mark.parametrize(
#     ("dtype", "expected_dtype"),
#     [
#         (None, np.float32),
#         (np.float64, np.float64),
#     ],
# )
# def test_abs2_dtype(dtype, expected_dtype):
#     x = np.asarray([1.0, 2.0, 3.0], dtype=np.float32)

#     actual = audio_utils.abs2(x, dtype=dtype)

#     assert actual.dtype == expected_dtype


# def test_abs2_complex():
#     x = np.asarray(
#         [1 + 2j, 3 - 4j],
#         dtype=np.complex64,
#     )

#     actual = audio_utils.abs2(x)

#     expected = np.abs(x) ** 2

#     np.testing.assert_allclose(actual, expected)


# @pytest.mark.parametrize(
#     ("x", "frame_length", "hop_length", "expected"),
#     [
#         (
#             np.arange(5),
#             3,
#             1,
#             np.array([
#                 [0, 1, 2],
#                 [1, 2, 3],
#                 [2, 3, 4],
#             ]),
#         ),
#         (
#             np.arange(5),
#             3,
#             2,
#             np.array([
#                 [0, 1, 2],
#                 [2, 3, 4],
#             ]),
#         ),
#         (
#             np.arange(6),
#             2,
#             2,
#             np.array([
#                 [0, 1],
#                 [2, 3],
#                 [4, 5],
#             ]),
#         ),
#     ],
# )
# def test_frame(
#     x,
#     frame_length,
#     hop_length,
#     expected,
# ):
#     actual = audio_utils.frame(
#         x,
#         frame_length=frame_length,
#         hop_length=hop_length,
#     )

#     np.testing.assert_array_equal(actual, expected)


# def test_frame_too_short():
#     x = np.arange(2)

#     with pytest.raises(ValueError, match="Input is too short"):
#         audio_utils.frame(
#             x,
#             frame_length=3,
#             hop_length=1,
#         )


# @pytest.mark.parametrize("hop_length", [0, -1])
# def test_frame_invalid_hop_length(hop_length):
#     x = np.arange(10)

#     with pytest.raises(ValueError, match="Invalid hop_length"):
#         audio_utils.frame(
#             x,
#             frame_length=3,
#             hop_length=hop_length,
#         )

@pytest.mark.parametrize(
    ("frame_length", "hop_length"),
    [
        (4, 1),
        (4, 2),
        (8, 2),
        (16, 4),
    ],
)
def test_rms_against_librosa(frame_length, hop_length):
    rng = np.random.default_rng(1234)
    y = rng.normal(size=100).astype(np.float32)

    expected = librosa.feature.rms(
        y=y,
        frame_length=frame_length,
        hop_length=hop_length,
        center=True,
        pad_mode="constant",
    )

    actual = feature.rms(
        y=y,
        frame_length=frame_length,
        hop_length=hop_length,
        center=True,
        pad_mode="constant",
    )

    np.testing.assert_allclose(
        actual,
        expected,
        rtol=1e-5,
        atol=1e-6,
    )

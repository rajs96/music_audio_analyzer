"""
Test instrument detector parsing and error handling WITHOUT GPU.

Tests:
1. _parse_instruments with various model outputs
2. Error field handling (no nan from PyArrow serialization)
3. Full pipeline flow with mocked inference

Run with: python test/test_detector_parsing.py
"""

import json
import numpy as np
import pyarrow as pa
from loguru import logger


def test_parse_instruments():
    """Test the CoT _parse_instruments logic directly."""

    # Import the parsing logic
    def parse_instruments(prediction: str) -> dict:
        """Copy of the parsing logic from InstrumentDetectorCoTAgent."""
        try:
            start = prediction.find("{")
            end = prediction.rfind("}") + 1
            if start != -1 and end > start:
                json_str = prediction[start:end]
                parsed = json.loads(json_str)

                background = parsed.get("background", [])
                middle_ground = parsed.get("middle_ground", [])
                foreground = parsed.get("foreground", [])

                all_instruments = list(
                    set(background) | set(middle_ground) | set(foreground)
                )

                return {
                    "background": background,
                    "middle_ground": middle_ground,
                    "foreground": foreground,
                    "instruments": all_instruments,
                }
            else:
                logger.warning(
                    f"No JSON object found in prediction: {prediction[:100]}..."
                )
        except Exception as e:
            logger.warning(f"Failed to parse: {e}. Raw: {prediction[:100]}...")

        return {
            "background": [],
            "middle_ground": [],
            "foreground": [],
            "instruments": [],
        }

    # Test cases
    test_cases = [
        # Valid JSON
        (
            '{"background": ["pad", "strings"], "middle_ground": ["piano"], "foreground": ["vocals"]}',
            {
                "background": ["pad", "strings"],
                "middle_ground": ["piano"],
                "foreground": ["vocals"],
            },
        ),
        # JSON with extra text before/after
        (
            'Here is the analysis:\n{"background": ["drums"], "middle_ground": [], "foreground": ["guitar"]}\nDone.',
            {"background": ["drums"], "middle_ground": [], "foreground": ["guitar"]},
        ),
        # Empty lists
        (
            '{"background": [], "middle_ground": [], "foreground": []}',
            {"background": [], "middle_ground": [], "foreground": []},
        ),
        # Missing keys (should default to empty lists)
        (
            '{"background": ["synth"]}',
            {"background": ["synth"], "middle_ground": [], "foreground": []},
        ),
        # No JSON at all
        (
            "I hear piano and drums in this track.",
            {
                "background": [],
                "middle_ground": [],
                "foreground": [],
                "instruments": [],
            },
        ),
        # Invalid JSON
        (
            '{"background": ["piano"',
            {
                "background": [],
                "middle_ground": [],
                "foreground": [],
                "instruments": [],
            },
        ),
    ]

    print("=" * 60)
    print("TEST: _parse_instruments")
    print("=" * 60)

    passed = 0
    failed = 0

    for i, (input_str, expected) in enumerate(test_cases):
        result = parse_instruments(input_str)

        # Check each expected key
        success = True
        for key in ["background", "middle_ground", "foreground"]:
            if key in expected and result.get(key) != expected[key]:
                success = False
                break

        if success:
            print(f"  [{i+1}] PASS: {input_str[:50]}...")
            passed += 1
        else:
            print(f"  [{i+1}] FAIL: {input_str[:50]}...")
            print(f"       Expected: {expected}")
            print(f"       Got: {result}")
            failed += 1

    print(f"\nResults: {passed}/{passed+failed} passed")
    return failed == 0


def test_error_field_pyarrow():
    """Test that error field doesn't become 'nan' through PyArrow."""

    print("\n" + "=" * 60)
    print("TEST: Error field through PyArrow serialization")
    print("=" * 60)

    # Simulate success result with error: ""
    success_result = {
        "job_id": "test_123",
        "filename": "test.mp3",
        "instruments": ["piano", "drums"],
        "error": "",  # Empty string, not None
    }

    # Simulate error result with actual error
    error_result = {
        "job_id": "test_456",
        "filename": "test2.mp3",
        "instruments": [],
        "error": "Failed to process audio",
    }

    # Old behavior (None) - should become nan
    old_success_result = {
        "job_id": "test_789",
        "filename": "test3.mp3",
        "instruments": ["guitar"],
        "error": None,  # This becomes nan!
    }

    # Put through PyArrow round-trip
    results = [success_result, error_result, old_success_result]
    table = pa.Table.from_pylist(results)
    recovered = table.to_pylist()

    print(
        f"\n  Original success error field: '{success_result['error']}' (type: {type(success_result['error'])})"
    )
    print(
        f"  Recovered success error field: '{recovered[0]['error']}' (type: {type(recovered[0]['error'])})"
    )

    print(
        f"\n  Original error error field: '{error_result['error']}' (type: {type(error_result['error'])})"
    )
    print(
        f"  Recovered error error field: '{recovered[1]['error']}' (type: {type(recovered[1]['error'])})"
    )

    print(
        f"\n  Original None error field: '{old_success_result['error']}' (type: {type(old_success_result['error'])})"
    )
    print(
        f"  Recovered None error field: '{recovered[2]['error']}' (type: {type(recovered[2]['error'])})"
    )

    # Check results
    passed = True

    # Success case: empty string should stay empty string
    if recovered[0]["error"] != "":
        print(
            f"\n  FAIL: Success error field changed from '' to '{recovered[0]['error']}'"
        )
        passed = False
    else:
        print(f"\n  PASS: Success error field preserved as empty string")

    # Error case: error message should be preserved
    if recovered[1]["error"] != "Failed to process audio":
        print(f"  FAIL: Error message not preserved")
        passed = False
    else:
        print(f"  PASS: Error message preserved")

    # None case: should become nan (demonstrating the bug we fixed)
    none_recovered = recovered[2]["error"]
    is_nan = none_recovered is None or (
        isinstance(none_recovered, float) and str(none_recovered) == "nan"
    )
    print(f"  INFO: None became '{none_recovered}' (is nan-like: {is_nan})")
    print(f"        This is why we use '' instead of None!")

    return passed


def test_error_checking_logic():
    """Test the error checking logic in _get_waveforms_from_items."""

    print("\n" + "=" * 60)
    print("TEST: Error checking logic")
    print("=" * 60)

    def has_real_error(error_value) -> bool:
        """Copy of the error checking logic."""
        return (
            error_value is not None
            and error_value != "nan"
            and not (isinstance(error_value, float) and str(error_value) == "nan")
            and str(error_value).strip() != ""
        )

    test_cases = [
        # (input, expected_has_error, description)
        ("", False, "Empty string (success)"),
        ("Preprocessing failed: invalid format", True, "Real error message"),
        (None, False, "None value"),
        ("nan", False, "String 'nan'"),
        (float("nan"), False, "Float nan"),
        ("  ", False, "Whitespace only"),
        ("Error: file not found", True, "Another real error"),
    ]

    passed = 0
    failed = 0

    for error_val, expected, desc in test_cases:
        result = has_real_error(error_val)
        if result == expected:
            print(f"  PASS: {desc} -> has_error={result}")
            passed += 1
        else:
            print(f"  FAIL: {desc} -> expected {expected}, got {result}")
            failed += 1

    print(f"\nResults: {passed}/{passed+failed} passed")
    return failed == 0


def test_mock_pipeline_flow():
    """Test full pipeline flow with mock inference."""

    print("\n" + "=" * 60)
    print("TEST: Mock pipeline flow (no GPU)")
    print("=" * 60)

    # Simulate preprocessor output
    waveform = np.random.randn(16000).astype(np.float32)
    preprocessor_output = {
        "job_id": "job_001",
        "song_id": "song_001",
        "song_hash": "abc123",
        "filename": "test_song.mp3",
        "waveform_bytes": waveform.tobytes(),
        "sample_rate": 16000,
        "duration_seconds": 1.0,
        "error": "",  # Success
    }

    # Simulate PyArrow serialization between stages
    table = pa.Table.from_pylist([preprocessor_output])
    recovered = table.to_pylist()[0]

    print(f"\n  Preprocessor output keys: {list(preprocessor_output.keys())}")
    print(f"  Recovered keys: {list(recovered.keys())}")

    # Check waveform_bytes survived
    waveform_bytes = recovered["waveform_bytes"]

    # Handle PyArrow binary types
    if hasattr(waveform_bytes, "as_py"):
        waveform_bytes = waveform_bytes.as_py()
    elif hasattr(waveform_bytes, "tobytes"):
        waveform_bytes = waveform_bytes.tobytes()

    recovered_waveform = np.frombuffer(waveform_bytes, dtype=np.float32)

    print(f"  Original waveform shape: {waveform.shape}")
    print(f"  Recovered waveform shape: {recovered_waveform.shape}")
    print(f"  Waveforms match: {np.allclose(waveform, recovered_waveform)}")

    # Check error field
    print(f"  Error field after PyArrow: '{recovered['error']}'")

    # Simulate detector output (mock inference)
    mock_model_response = '{"background": ["synth pad"], "middle_ground": ["bass", "drums"], "foreground": ["vocals"]}'

    # Parse (copy of logic)
    parsed = json.loads(mock_model_response)

    detector_output = {
        "job_id": recovered["job_id"],
        "song_id": recovered["song_id"],
        "song_hash": recovered["song_hash"],
        "filename": recovered["filename"],
        "background": parsed.get("background", []),
        "middle_ground": parsed.get("middle_ground", []),
        "foreground": parsed.get("foreground", []),
        "instruments": list(
            set(parsed.get("background", []))
            | set(parsed.get("middle_ground", []))
            | set(parsed.get("foreground", []))
        ),
        "planning_response": "Mock planning response...",
        "detected_at": 1234567890,
        "error": "",  # Success
    }

    print(f"\n  Mock detector output:")
    print(f"    background: {detector_output['background']}")
    print(f"    middle_ground: {detector_output['middle_ground']}")
    print(f"    foreground: {detector_output['foreground']}")
    print(f"    instruments: {detector_output['instruments']}")
    print(f"    error: '{detector_output['error']}'")

    # Final PyArrow round-trip
    final_table = pa.Table.from_pylist([detector_output])
    final_recovered = final_table.to_pylist()[0]

    print(f"\n  After final PyArrow serialization:")
    print(f"    error field: '{final_recovered['error']}'")
    print(
        f"    instruments preserved: {final_recovered['instruments'] == detector_output['instruments']}"
    )

    # All checks
    success = (
        np.allclose(waveform, recovered_waveform)
        and recovered["error"] == ""
        and final_recovered["error"] == ""
        and final_recovered["instruments"] == detector_output["instruments"]
    )

    if success:
        print("\n  PASS: Full mock pipeline flow works correctly!")
    else:
        print("\n  FAIL: Something went wrong in the pipeline flow")

    return success


def main():
    print("\n" + "=" * 60)
    print("INSTRUMENT DETECTOR UNIT TESTS (NO GPU REQUIRED)")
    print("=" * 60)

    results = []

    results.append(("_parse_instruments", test_parse_instruments()))
    results.append(("Error field PyArrow", test_error_field_pyarrow()))
    results.append(("Error checking logic", test_error_checking_logic()))
    results.append(("Mock pipeline flow", test_mock_pipeline_flow()))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    all_passed = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False

    if all_passed:
        print("\nALL TESTS PASSED!")
        return 0
    else:
        print("\nSOME TESTS FAILED!")
        return 1


if __name__ == "__main__":
    import sys

    sys.exit(main())

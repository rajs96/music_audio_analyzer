"""
Minimal test to debug PyArrow serialization of ObjectRefs.
"""

import ray
import pickle
import numpy as np
import pyarrow as pa

ray.init(ignore_reinit_error=True)

# Create a waveform and put it in object store
waveform = np.random.randn(16000).astype(np.float32)
waveform_ref = ray.put(waveform)

print(f"Original waveform_ref: {waveform_ref}")
print(f"Original type: {type(waveform_ref)}")

# Verify we can get it back
retrieved = ray.get(waveform_ref)
print(f"Direct ray.get works: {retrieved.shape}")

# Now simulate what happens in the pipeline
# Step 1: Serialize (what _items_to_batch does)
serialized = b"__PICKLED_OBJREF__" + pickle.dumps(waveform_ref)
print(f"\nSerialized bytes: {serialized[:50]}...")
print(f"Serialized type: {type(serialized)}")

# Step 2: Put in PyArrow table (what Ray Data does)
row = {
    "filename": "test.mp3",
    "waveform_ref": serialized,
    "error": "",
}
table = pa.Table.from_pylist([row])
print(f"\nPyArrow table created")
print(f"Table schema: {table.schema}")

# Step 3: Read back from PyArrow (what Ray Data does)
# Convert back to Python dict
read_back = table.to_pylist()[0]
print(f"\nRead back from PyArrow:")
print(f"  waveform_ref type: {type(read_back['waveform_ref'])}")
print(f"  waveform_ref value: {read_back['waveform_ref'][:50]}...")

# Step 4: Deserialize (what _batch_to_items should do)
wf_bytes = read_back["waveform_ref"]

# Check various ways to handle the bytes
print(f"\nTesting deserialization:")
print(f"  isinstance(wf_bytes, bytes): {isinstance(wf_bytes, bytes)}")
print(f"  hasattr 'as_py': {hasattr(wf_bytes, 'as_py')}")
print(f"  hasattr 'tobytes': {hasattr(wf_bytes, 'tobytes')}")

# Try to deserialize
if isinstance(wf_bytes, bytes):
    if wf_bytes.startswith(b"__PICKLED_OBJREF__"):
        deserialized_ref = pickle.loads(wf_bytes[len(b"__PICKLED_OBJREF__") :])
        print(f"\nDeserialized ObjectRef: {deserialized_ref}")
        print(f"Deserialized type: {type(deserialized_ref)}")

        # Try to get the waveform
        final_waveform = ray.get(deserialized_ref)
        print(f"Final ray.get works: {final_waveform.shape}")
        print(f"Waveforms match: {np.allclose(waveform, final_waveform)}")
    else:
        print("ERROR: Bytes don't start with marker!")
else:
    print(f"ERROR: wf_bytes is not bytes, it's {type(wf_bytes)}")

    # Try as_py if available
    if hasattr(wf_bytes, "as_py"):
        py_val = wf_bytes.as_py()
        print(f"  as_py() result type: {type(py_val)}")
        if isinstance(py_val, bytes) and py_val.startswith(b"__PICKLED_OBJREF__"):
            deserialized_ref = pickle.loads(py_val[len(b"__PICKLED_OBJREF__") :])
            print(f"  Deserialized via as_py: {deserialized_ref}")
            final_waveform = ray.get(deserialized_ref)
            print(f"  Final ray.get works: {final_waveform.shape}")

ray.shutdown()
print("\n=== TEST COMPLETE ===")

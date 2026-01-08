"""
JWST core functionality test
Run with: python jwst_core_test.py
"""

import sys
import numpy as np
from astropy.io import fits
from jwst.datamodels import ImageModel
from jwst.stpipe import Step
from jwst.pipeline import Detector1Pipeline
import crds

def fail(msg):
    print(f"[FAIL] {msg}")
    sys.exit(1)

def ok(msg):
    print(f"[OK]   {msg}")

# -------------------------------
# 1. CRDS context resolution
# -------------------------------
try:
    context = crds.get_context_name()
    ok(f"CRDS context resolved: {context}")
except Exception as e:
    fail(f"CRDS failed: {e}")

# -------------------------------
# 2. Create a valid JWST datamodel
# -------------------------------
try:
    data = np.random.rand(32, 32)
    model = ImageModel(data=data)
    model.meta.instrument.name = "NIRCAM"
    model.meta.instrument.detector = "NRCA1"
    model.meta.exposure.type = "NRC_IMAGE"
    model.meta.exposure.readpatt = "RAPID"
    model.meta.exposure.nframes = 1
    model.meta.exposure.ngroups = 2
    model.meta.exposure.groupgap = 0
    model.meta.subarray.name = "FULL"
    ok("JWST ImageModel created and populated")
except Exception as e:
    fail(f"Datamodel creation failed: {e}")

# -------------------------------
# 3. Serialize + deserialize model
# -------------------------------
try:
    model.save("test_jwst_model.fits")
    reloaded = ImageModel("test_jwst_model.fits")
    assert reloaded.data.shape == (32, 32)
    ok("Datamodel FITS serialization works")
except Exception as e:
    fail(f"Datamodel I/O failed: {e}")

# -------------------------------
# 4. Run an individual pipeline step
# -------------------------------
try:
    from jwst.assign_wcs import AssignWcsStep
    step = AssignWcsStep()
    result = step.run(reloaded)
    ok("Individual pipeline step executed")
except Exception as e:
    fail(f"Pipeline step execution failed: {e}")

# -------------------------------
# 5. Pipeline graph construction
# -------------------------------
try:
    pipe = Detector1Pipeline()
    step_names = list(pipe.step_defs.keys())
    assert len(step_names) > 0
    ok(f"Detector1Pipeline constructed with steps: {step_names}")
except Exception as e:
    fail(f"Pipeline construction failed: {e}")

# -------------------------------
# 6. Pipeline dry-run (no refs)
# -------------------------------
try:
    pipe.assign_wcs.skip = True
    pipe.flat_field.skip = True
    pipe.jump.skip = True
    pipe.ramp_fit.skip = True
    pipe.dark_current.skip = True
    pipe.gain_scale.skip = True
    pipe.refpix.skip = True
    pipe.linearity.skip = True
    pipe.saturation.skip = True

    output = pipe.run(reloaded)
    ok("Detector1Pipeline executed (core infrastructure test)")
except Exception as e:
    fail(f"Pipeline execution failed: {e}")

# -------------------------------
# 7. Metadata propagation
# -------------------------------
try:
    assert output.meta.instrument.name == "NIRCAM"
    ok("Metadata propagated correctly through pipeline")
except Exception as e:
    fail(f"Metadata propagation failed: {e}")

print("\nAll JWST core functionality tests PASSED.")

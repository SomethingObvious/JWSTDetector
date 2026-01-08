import gzip
import os
import shutil

base_dir = r"mosaic_miri_f770w_COSMOS-Web+PRIMER_60mas_all_v1.0_wht"  # change this

for name in os.listdir(base_dir):
    if name.lower().endswith(".gz"):
        gz_path = os.path.join(base_dir, name)
        out_path = os.path.join(base_dir, os.path.splitext(name)[0])

        with gzip.open(gz_path, "rb") as f_in:
            with open(out_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)

        os.remove(gz_path)

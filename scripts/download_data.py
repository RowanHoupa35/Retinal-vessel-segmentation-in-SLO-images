"""Download the 10-image IOSTAR subset distributed with the ENSTA course.

The images are not committed: the IOSTAR licence could not be checked. The
full dataset (30 images, 1024 x 1024) is available from RetinaCheck:
https://www.retinacheck.org/download-iostar-retinal-vessel-segmentation-dataset
"""

import hashlib
import io
import urllib.request
import zipfile
from pathlib import Path

URL = "https://perso.ensta.fr/~manzaner/Cours/MI206/images_projet3.zip"
SHA256 = "c9a875d6676962a7294459c1fcc2dec359780ae1bf2aa70a3c9eb97390a22b37"
ROOT = Path(__file__).resolve().parents[1]


def main():
    target = ROOT / "images_IOSTAR"
    if target.is_dir() and any(target.iterdir()):
        print(f"{target} already exists, nothing to do.")
        return
    with urllib.request.urlopen(URL, timeout=60) as response:
        payload = response.read()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != SHA256:
        raise RuntimeError(f"Unexpected checksum {digest}, the archive has changed.")
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        archive.extractall(ROOT)
    print(f"Extracted {len(list(target.iterdir()))} files to {target}")


if __name__ == "__main__":
    main()

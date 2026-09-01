# 1. PA1 Data Provenance

## 1.1. Generated assets

Run the following command from the repository root:

```bash
python data/setup_assets.py
```

The script creates `data/generated/` with a flat image, white-square
sanity image, corner-rich synthetic texture, easy similarity-warp pair, and a
short synthetic smoke-test video. All assets are generated locally and contain
no personal data; synthetic content uses fixed seed `16720`. Image contents are
reproducible, while MP4 container bytes may vary with the platform encoder.
Generated files are intentionally untracked and may be deleted and recreated.

It also exports `photo_texture.png` from scikit-image's bundled `coffee()` sample.
That photograph is credited to Rachel Michetti and released under CC0 by its
source, as recorded in the scikit-image data-module documentation. It is used
only for the required two-radius photographic NMS example. The setup script
does not download it at runtime.

## 1.2. Intended use

The images support the required B2/B4 tests and the video exercises decoding,
matching, transform estimation, warping, and encoding without a camera. The
synthetic video is visibly artificial and is not evidence of physical capture.

The standard D1 requirement is a student's own safe capture. Course staff may
approve the supplied synthetic clip or a separately distributed staff-owned
clip only for access, privacy, safety, or accessibility reasons. It is not a
general substitute for the capture requirement.

## 1.3. Student-captured data

Students retain raw captures locally and submit only the evidence required by
the handout. Captures must avoid non-consenting people, private documents,
screens with personal information, vehicle operation, restricted spaces, and
other unsafe situations. Students retain ownership of their own media and may
decline optional public showcase participation.

## 1.4. External data and dependencies

Asset generation uses NumPy, OpenCV, and scikit-image versions allowed by the
root `requirements-colab.txt`. There are no network downloads, pretrained
weights, API keys, or restrictive third-party data licenses. The one bundled
photograph is CC0; every other asset is course-generated.

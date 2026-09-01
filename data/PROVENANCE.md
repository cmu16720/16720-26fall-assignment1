# 1. PA1 Data Provenance

## 1.1. Generated assets

Run the following command from the repository root:

```bash
python data/setup_assets.py
```

The script creates `data/generated/` with a flat image, white-square
sanity image, corner-rich synthetic texture, easy similarity-warp pair,
difficult perspective/blur/occlusion pair, a short synthetic smoke-test video,
and a SHA-256 manifest. All of those assets are generated locally from fixed
seed `16720` and contain no external or personal data. The generated files may
be deleted and recreated at any time.

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

## 1.5. Staff-only reference media

Sections 1.1-1.4 fully describe the student release. The staff reference
notebook additionally downloads four public research/test assets into
`data/downloads/` and `data/vgg_affine/`. The release builder excludes both
directories, so none of this media enters the student release, and the
repository does not redistribute the files. Every download is SHA-256 verified
before use:

- Oxford VGG affine-covariance sequences `leuven` (one scene under a real
  lighting series) and `boat` (one scene under real camera zoom and rotation),
  from <https://www.robots.ox.ac.uk/~vgg/research/affine/>, downloaded from the
  publisher for instructional evaluation only, under the same source and terms
  as the PA3 `wall`/`graf` data:
  - `leuven.tar.gz`: `9b1ade6e64e5b27eee43463ff8f9ad2bd2f9c6911666e93de2634f3a5c0fe5dc`
  - `boat.tar.gz`: `6e5721f37fbf9c3e974fd05935a42c1670f3e71aa3e5b218807ea046a068c603`
- Xiph.Org derf-collection test clips `mobile_cif.y4m` ("mobile & calendar", a
  real camera pan/zoom across a wall calendar) and `flower_cif.y4m` ("flower
  garden", a real translating camera with strong parallax and a foreground
  occluder), from <https://media.xiph.org/video/derf/>, long-standing standard
  sequences hosted for research and testing:
  - `mobile_cif.y4m`: `8b32924ea00ef3bd52f7ddafd3ceb55d2bb331610b59957c5c3cc9628f9e0d90`
  - `flower_cif.y4m`: `26c7934b77186836a39db113e8011e841bd593e37295efb98a38915b24aa2b8e`

These files are staff exemplars for the C1/C3/D pipelines. They are not
distributed to students and do not replace any student capture requirement.

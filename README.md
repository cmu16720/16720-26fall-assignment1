# 1. PA1: Local Features and Object-Locked Video

| Item | Value |
|---|---|
| Release | September 8, 2026 |
| Due | September 29, 2026 |
| Base credit | 100 points |
| Extra credit | Up to 10 points total, including showcase participation |
| Expected effort | 8–10 hours |
| Platform | Google Colab; CPU is sufficient |
| Final artifact | A 10–20 second object-locked H.264 MP4, at most 50 MB |

In this assignment you will derive, implement, test, and apply a complete classical local-feature pipeline. You will build Harris corner detection, a compact gradient descriptor, and ratio-test matching from basic array operations. You will then use those correspondences with supplied geometric infrastructure to make a short video in which a chosen object appears fixed while the surrounding scene moves.

Read the assignment handout (the distributed PA1 specification PDF) before editing code. It is the authoritative student specification; this README is a setup and submission guide.

## 2. Start here

The recommended workflow is Google Colab. Download the complete course-supplied
`pa1_student.zip`; the notebook by itself does not contain the Python package,
tests, data, or shared requirements.

1. From the ZIP, locate `notebooks/pa1_starter.ipynb` and upload that notebook to a fresh Colab runtime.
2. Run its first environment cell and choose the **complete, unopened `pa1_student.zip`** when the upload picker appears. The cell checks every ZIP path, extracts the release beneath `/content`, changes to the release root, and installs `requirements-colab.txt`.
3. If Colab asks to restart after installation, restart and rerun the environment cell; it will reuse the extracted release.
4. Run the convention, seed, and data-setup cells before editing student code.
5. Complete the required functions in `student/pa1.py` in problem order.
6. Run the public checks after each function, then execute the notebook from top to bottom.
7. Complete the controlled experiments before recording the final video.
8. Compile `answer_template.tex` and perform the submission audit in the notebook.

Python 3.10 or newer is required for local work. From the repository root:

```bash
uv venv .venv
uv pip install --python .venv/bin/python -e '.[dev]'
.venv/bin/python data/setup_assets.py
.venv/bin/python -m pytest tests/test_pa1_public.py -q
```

If `uv` is unavailable, create and activate a standard virtual environment and
run `python -m pip install -e '.[dev]'` before the final two commands.

The notebook and public tests require no paid compute, credentials, API keys, or private service.

## 3. Student release map

| Path | Purpose |
|---|---|
| `assignment.md` | Source text of the assignment handout PDF |
| `expected_outputs.md` | Non-revealing sanity checks and diagnostics |
| `README.md` | Setup, workflow, and submission checklist |
| `notebooks/pa1_starter.ipynb` | Colab workflow and required evidence |
| `student/pa1.py` | Functions you implement |
| `common/` | Supplied plotting, image, experiment, geometry, and video helpers |
| `data/` | Small deterministic examples, approved fallback media, and provenance |
| `tests/test_pa1_public.py` | Public interface and sanity checks |
| `answer_template.tex` | Stable written-response and Gradescope template |
| `requirements-colab.txt` | Shared runtime requirements |

The staff solution, private tests, rubric details, and grader notes are not part of the student release.

## 4. Academic-integrity and generative-AI boundaries

These boundaries apply even when a tool is used only to “check” or rewrite work.

| Work | Generative AI policy |
|---|---|
| Part A | Not permitted to generate, explain, solve, check, or rewrite any response |
| Part B required implementations in `student/pa1.py` | Not permitted to generate, complete, translate, explain, debug, or rewrite the required implementation |
| Part C | Permitted, with disclosure; you remain responsible for the analysis |
| Part D | Permitted and encouraged, with disclosure; it may not replace your Part B implementation |
| Part E | Permitted, with disclosure |

For Part A and the required Part B implementation, you may use course notes, textbooks, papers, conventional web resources, NumPy/PyTorch/OpenCV documentation, and course staff as allowed by the course collaboration policy.

Generative AI is optional everywhere it is permitted. If you do not use it, D4 provides a non-AI reflection alternative. A permitted AI tool may help with Part D infrastructure, but it may not create or repair the graded detector, descriptor, or matcher from Part B. Disclose collaborators, outside resources, and every permitted use of generative AI in the written PDF.

## 5. Fixed technical conventions

- Course-loaded images are two-dimensional `float32` NumPy arrays normalized to `[0, 1]`; graded functions never clip or renormalize their numeric inputs.
- Arrays use `(row, column)` indexing; points and keypoints use Cartesian `(x, y)` order.
- The shared random seed is `16720`.
- Gradients use centered differences; `Ix` is zero in the first/last columns and `Iy` is zero in the first/last rows.
- Harris uses `sigma=1.5`, `k=0.04`, and deterministic greedy Euclidean-disk NMS by default.
- Descriptors use a `16 x 16` patch, a `4 x 4` cell grid, and 8 orientation bins, for 128 values.
- Matching uses Euclidean distance and a strict nearest-neighbor ratio test with default ratio `0.80`.
- The supplied estimator maps current-frame points to reference-frame points. Warping uses that mapping to align the current frame to the reference.

The exact contracts, border rules, sorting rules, empty-output shapes, and validation requirements are in the assignment handout. Treat those details as part of the problem.

## 6. Capture, privacy, and fallback policy

Record in landscape orientation at 720p and 30 fps when possible. Choose a textured, matte, approximately rigid target that occupies about 25–60% of the frame. Keep it visible while introducing moderate translation, rotation, scale change, and limited viewpoint change. Avoid rapid motion, severe blur, glossy or featureless surfaces, and highly repetitive targets.

Do not record non-consenting people, private documents, personal information on screens, or unsafe/prohibited locations. A course-provided capture alternative is available only with staff approval for access, privacy, safety, or accessibility reasons. Request approval before relying on it. The alternative receives identical grading and is not a general substitute for the controlled capture work.

## 7. Required submission

Submit the following through the course submission system:

1. `pa1.py`, containing your required Part B implementations;
2. the completed starter notebook, with the required evidence outputs retained;
3. one LaTeX-generated PDF built from `answer_template.tex`;
4. one 10–20 second labeled original/object-locked comparison video as H.264 MP4, yuv420p, at most 720p and 30 fps, and at most 50 MB; and
5. acknowledgments and disclosure of collaborators, outside resources, and permitted generative-AI use.

Every graded figure and table must appear both in the executable notebook and in the corresponding labeled section of the PDF. The comparison video supplies the original-capture evidence, so a separate raw capture is not required. Keep raw captures locally until grades are final.

Use the answer template without changing its stable labels. Begin each labeled problem on the page already assigned to it, upload one PDF, assign every answer page to the matching Gradescope item, and inspect the rendered preview.

## 8. Verification workflow

Before submission:

1. run the public tests from a clean process;
2. restart the notebook runtime and execute all cells in order;
3. confirm that Part B uses your own detector, descriptor, and matcher rather than a prohibited black box;
4. confirm that every experiment uses the required metric columns and has a readable visualization;
5. play the exported MP4 outside Colab and verify its duration, codec, pixel format, dimensions, frame rate, and size;
6. compile the answer template and verify every Gradescope page assignment; and
7. complete the acknowledgment, AI-disclosure, and file checklist.

## 9. Common debugging checks

- If detections appear transposed, check every conversion between array indexing `(y, x)` and keypoint order `(x, y)`.
- If a flat image produces points, inspect derivative borders, response threshold semantics, and handling of nonpositive maxima.
- If descriptors have the wrong length, check cell traversal and make the orientation bin the fastest-changing index.
- If all matches pass or all fail, inspect zero descriptors, the two-neighbor requirement, and the strict ratio comparison.
- If stabilization moves in the wrong direction, verify that the estimated transform maps current points to reference points.
- If a few failed frames corrupt the entire video, use the supplied bounded failure policy rather than chaining frame-to-frame transforms.

## 10. Final checklist

- [ ] Part A was completed without generative AI and typeset in LaTeX.
- [ ] The required Part B implementation was completed without generative AI.
- [ ] Public tests pass, and the notebook runs top to bottom from a clean runtime.
- [ ] B1–B4 figures and explanations appear in both the notebook and PDF.
- [ ] C1 contains exactly the four required conditions and the shared metric table.
- [ ] C2 analyzes two conditions, including one descriptor-invariance analysis.
- [ ] C3 documents a clear failure and a technically grounded diagnosis.
- [ ] D1 shows the first-frame target box and compliant capture evidence.
- [ ] D2 uses Part B correspondences and the supplied current-to-reference geometry.
- [ ] The hero result is labeled, playable, 10–20 seconds, and at most 50 MB.
- [ ] D4 is 150–250 words and follows either the AI-use or no-AI path.
- [ ] The PDF page mapping, acknowledgments, and AI disclosure are complete.
- [ ] Any extension/showcase credit stays within the 10-point combined cap.

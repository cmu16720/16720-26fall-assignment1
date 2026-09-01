# 16-720 Computer Vision — Assignment 1

## Local Features and Object-Locked Video

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

## 1. Start here

The recommended workflow is Google Colab. The notebook depends on the rest of
this repository, so keep a repository ZIP available when starting a new runtime.

1. On GitHub, choose **Code → Download ZIP** and keep the downloaded repository archive unopened.
2. Download the root-level `pa1_starter.ipynb` separately and upload it to a fresh Colab runtime.
3. Run its first environment cell and choose the repository ZIP when prompted. The cell validates the archive paths, extracts the repository beneath `/content`, changes to its root, and installs `requirements-colab.txt`.
4. If Colab asks to restart after installation, restart and rerun the environment cell; it will reuse the extracted repository.
5. Run the convention, seed, and data-setup cells before editing student code.
6. Complete the required functions in `student/pa1.py` in problem order.
7. Run the public checks after each function, then execute the notebook from top to bottom.
8. Complete the controlled experiments before recording the final video.
9. Compile `answer_template.tex` and perform the submission audit in the notebook.

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

## 2. Academic-integrity and generative-AI boundaries

These boundaries apply even when a tool is used only to “check” or rewrite work.

| Work | Generative AI policy |
|---|---|
| Part A | Not permitted to generate, explain, solve, check, or rewrite any response |
| Part B required implementations in `student/pa1.py` | Not permitted to generate, complete, translate, explain, debug, or rewrite the required implementation |
| Part C | Permitted, with disclosure; you remain responsible for the analysis |
| Part D | Permitted and encouraged, with disclosure; it may not replace your Part B implementation |
| Part E | Permitted, with disclosure |

For Part A and the required Part B implementation, you may use course notes, textbooks, papers, conventional web resources, NumPy/OpenCV documentation, and course staff as allowed by the course collaboration policy.

Generative AI is optional everywhere it is permitted. If you do not use it, D4 provides a non-AI reflection alternative. A permitted AI tool may help with Part D infrastructure, but it may not create or repair the graded detector, descriptor, or matcher from Part B. Disclose collaborators, outside resources, and every permitted use of generative AI in the written PDF.

## 3. Capture, privacy, and fallback policy

Record in landscape orientation at 720p and 30 fps when possible. Choose a textured, matte, approximately rigid target that occupies about 25–60% of the frame. Keep it visible while introducing moderate translation, rotation, scale change, and limited viewpoint change. Avoid rapid motion, severe blur, glossy or featureless surfaces, and highly repetitive targets.

Do not record non-consenting people, private documents, personal information on screens, or unsafe/prohibited locations. A course-provided capture alternative is available only with staff approval for access, privacy, safety, or accessibility reasons. Request approval before relying on it. The alternative receives identical grading and is not a general substitute for the controlled capture work.

## 4. Required submission

Submit the following through the course submission system:

1. `pa1.py`, containing your required Part B implementations;
2. the completed starter notebook, with the required evidence outputs retained;
3. one LaTeX-generated PDF built from `answer_template.tex`;
4. one 10–20 second labeled original/object-locked comparison video as H.264 MP4, yuv420p, at most 720p and 30 fps, and at most 50 MB; and
5. acknowledgments and disclosure of collaborators, outside resources, and permitted generative-AI use.

Every graded figure and table must appear both in the executable notebook and in the corresponding labeled section of the PDF. The comparison video supplies the original-capture evidence, so a separate raw capture is not required. Keep raw captures locally until grades are final.

Use the answer template without changing its stable labels. Begin each labeled problem on the page already assigned to it, upload one PDF, assign every answer page to the matching Gradescope item, and inspect the rendered preview.

## 5. Verification workflow

Before submission:

1. run the public tests from a clean process;
2. restart the notebook runtime and execute all cells in order;
3. confirm that Part B uses your own detector, descriptor, and matcher rather than a prohibited black box;
4. confirm that every experiment uses the required metric columns and has a readable visualization;
5. play the exported MP4 outside Colab and verify its duration, codec, pixel format, dimensions, frame rate, and size;
6. compile the answer template and verify every Gradescope page assignment; and
7. complete the acknowledgment, AI-disclosure, and file checklist.

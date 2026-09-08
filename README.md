# 16-720 Computer Vision — Assignment 1

**Local Features and Object-Locked Video**

| | |
|---|---|
| Release | September 8, 2026 |
| Due | September 29, 2026 |
| Credit | 100 points, plus up to 10 optional points |
| Expected effort | 8–10 hours |
| Compute | Google Colab; CPU is sufficient |
| Final artifact | 10–20 second object-locked H.264 MP4, at most 50 MB |

Build a classical local-feature pipeline—Harris corners, a compact gradient
descriptor, and ratio-test matching—then use it to create a video in which a
chosen object remains fixed while the surrounding scene moves.

## Start here

Read the [assignment instructions (PDF)](assignment_instructions.pdf) before editing code.
It is the authoritative source for all requirements, numerical contracts, grading,
and policy. The notebook is its executable companion.

- Work through the [starter notebook](pa1_starter.ipynb).
- Implement the required functions in [`student/pa1.py`](student/pa1.py).
- Write responses in [`answer_template.tex`](answer_template.tex).
- Use [`tests/test_pa1_public.py`](tests/test_pa1_public.py) for public checks.
- See [`data/PROVENANCE.md`](data/PROVENANCE.md) for data sources and privacy notes.

## Local setup

Python 3.10 or newer is required.

```bash
git clone https://github.com/cmu16720/16720-26fall-assignment1.git
cd 16720-26fall-assignment1
uv venv .venv
uv pip install --python .venv/bin/python -e '.[dev]'
.venv/bin/python data/setup_assets.py
# Run after implementing B2–B4; the untouched starter is expected to fail.
.venv/bin/python -m pytest tests/test_pa1_public.py -q
```

If `uv` is unavailable, create a standard virtual environment and install the
project with `python -m pip install -e '.[dev]'`.

## Academic integrity and generative AI

| Work | Policy |
|---|---|
| Part A | Generative AI may not generate, explain, solve, check, or rewrite responses. |
| Part B required implementation | Generative AI may not generate, complete, translate, explain, debug, or rewrite the required functions. |
| Parts C and E | Generative AI is permitted with disclosure. |
| Part D | Generative AI is encouraged with disclosure, but may not replace the Part B implementation. |

Disclose collaborators, outside resources, and every permitted use of
generative AI in the written PDF. See the
[assignment PDF, Section 4](assignment_instructions.pdf)
for the complete policy.

## Submission

Submit through the course submission system:

1. `pa1.py` with your Part B implementation;
2. the completed notebook with required outputs retained;
3. one PDF compiled from `answer_template.tex`;
4. one labeled H.264/yuv420p comparison MP4, 10–20 seconds and at most 50 MB;
5. all required acknowledgments and disclosures.

Before submitting, run the public tests in a clean process, execute the notebook
from a fresh runtime, inspect the compiled PDF, and play the exported video
outside Colab. Verify every detailed requirement against the
[assignment PDF, Section 12](assignment_instructions.pdf).

## Colab editing and PDF evidence

Upload `pa1_starter.ipynb` to Colab, then upload the complete repository ZIP when
the first cell prompts for it. In the Files pane, open `student/pa1.py`, save your
edits, and run the notebook's reload cell before rerunning dependent cells.
Download edited code regularly; saving the notebook does not back up separate
runtime files. The final backup cell downloads code and artifacts. Download the
executed notebook separately with File → Download → Download .ipynb.

The notebook saves supplied figures to `artifacts/`. Save your added figures there
too, upload them alongside `answer_template.tex` in Overleaf, and replace each
relevant evidence box with `\includegraphics[width=\linewidth]{your_figure.png}`.
Compile with pdfLaTeX and inspect every answer page. The notebook includes a
worked save example and full target-box/tracking-diagnostics plots.

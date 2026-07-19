# sar-optical-gnn

Research project: SAR-to-optical image translation using a superpixel Region-Adjacency-Graph (RAG)
+ Graph Neural Network (GNN) as a structural prior, hybridized with a CNN encoder/decoder, trained
on paired Sentinel-1/Sentinel-2 imagery and evaluated against GAN/diffusion baselines plus a
downstream land-cover-classification-fidelity check.

**Status**: M0 (repo hygiene), M1 (data pipeline), and M2 (graph construction at scale) done. The
full M0-M2 pipeline has been validated against a real downloaded dataset (SARptical), including
finding and fixing a real segmentation bug (a hardcoded SLIC parameter that produced a rigid grid
instead of real regions on actual SAR data) by rendering the pipeline's output with
`scripts/visualize_sample.py` rather than trusting test counts alone. M3 (baseline models:
pix2pix, CycleGAN) is next. See `docs/` locally for the full research plan, background, literature
review, and a build-by-build log of what was done and why — that folder is intentionally
git-ignored (it's local working material, not meant to ship in the repo), so if you're reading this
on GitHub without local access to it, ask whoever's running the project for the docs directly.

## The one-paragraph version

Sentinel-1 SAR sees through cloud cover; Sentinel-2 optical is easier to interpret but often
unavailable exactly when needed. This project tests whether adding a graph-based structural prior
— built by segmenting the SAR image into superpixels and reasoning over the resulting adjacency
graph with a GNN — improves *thematic* fidelity of SAR-to-optical translation (does the generated
image classify correctly as the right land-cover type), not just raw pixel similarity, compared to
standard pixel-wise GAN baselines (pix2pix, CycleGAN).

## Repo layout

```
sar-optical-gnn/
├── src/
│   ├── graph_builder.py    # superpixel segmentation + Region Adjacency Graph construction
│   ├── graph/
│   │   ├── pooling.py      # pixel<->node scatter pool (mean/max) and unpool
│   │   └── features.py     # regionprops geometric features + channel mean/std per node
│   └── datasets/
│       ├── common.py       # shared CHW->HWC array conversion used by every loader
│       ├── bigearthnet.py  # primary dataset: paired S1/S2 + real CORINE land-cover labels
│       ├── sen1_2.py       # validation harness: reproduce literature baseline numbers on this
│       ├── sen12ms.py      # secondary: superpixel-granularity ablation, generalization check
│       ├── sarptical.py    # real, downloaded stretch dataset (10,108 pairs) used to validate
│       │                   #   the whole M0-M2 pipeline against actual data
│       └── delhi_gee.py    # Earth Engine fetch/export for the Delhi ROI qualitative demo
├── scripts/
│   ├── build_graphs_offline.py  # precompute/cache graphs to .npz; --benchmark mode
│   └── visualize_sample.py      # render SAR/optical/segmentation/graph for one real sample
├── tests/                  # mirrors src/ and scripts/ layout
├── requirements.txt
├── pyproject.toml          # pytest config
└── docs/                   # (git-ignored, local only) research plan, literature review, build log
```

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

A couple of evaluation-only dependencies (`lpips`, `torch-fidelity`) are pinned but commented out
in `requirements.txt` until M5 actually needs them — see that file for which milestone pulls in
what.

## Running tests

```bash
pytest
```

## Where this is going

The build follows a staged roadmap (data pipeline → graph construction at scale → GAN baselines →
the GNN-hybrid model → downstream evaluation → ablations → generalization check → writeup). Full
detail, including why each stage is ordered the way it is, is in the local `docs/` folder.

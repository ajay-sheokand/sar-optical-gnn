# sar-optical-gnn

Research project: SAR-to-optical image translation using a superpixel Region-Adjacency-Graph (RAG)
+ Graph Neural Network (GNN) as a structural prior, hybridized with a CNN encoder/decoder, trained
on paired Sentinel-1/Sentinel-2 imagery and evaluated against GAN/diffusion baselines plus a
downstream land-cover-classification-fidelity check.

**Status**: early build-out, milestone M0 (repo hygiene) in progress. See `docs/` locally for the
full research plan, background, literature review, and build log — that folder is intentionally
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
├── data_loader.py       # Google Earth Engine fetch for the Delhi ROI (qualitative demo target)
├── src/
│   └── graph_builder.py # superpixel segmentation + Region Adjacency Graph construction
├── tests/
│   └── test_graph_builder.py
├── requirements.txt
├── pyproject.toml       # pytest config
└── docs/                # (git-ignored, local only) research plan, literature review, build log
```

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Some heavier dependencies (`torchgeo`, `lpips`, `torch-fidelity`, `torchmetrics`) are pinned but
commented out in `requirements.txt` until the milestones that actually need them — see that file
for which milestone pulls in what.

## Running tests

```bash
pytest
```

## Where this is going

The build follows a staged roadmap (data pipeline → graph construction at scale → GAN baselines →
the GNN-hybrid model → downstream evaluation → ablations → generalization check → writeup). Full
detail, including why each stage is ordered the way it is, is in the local `docs/` folder.

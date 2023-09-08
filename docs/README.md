# ARCLE - ARC Learning Environment

ARCLE is a lightweight Gymnasium (previously OpenAI Gym) environment for training on ARC(Abstraction and Reasoning Corpus) and ARC-like datasets.

## Installation
`pip install arcle`

## Changelogs

#### 0.2.0
- ArcEnv-v0 & MiniArcEnv-v0 has renamed to ARCEnv-v0 & MiniARCEnv-v0.
- Changed internal `current_grid`` variable to `grid` & `grid_dim`
- **Added O2ARCv2Env-v0 environment.**
- Added O2ARC Actions
    - Uses `selection` and `selected`. Some internal variables should be initialized.
    - Color, Flood Fill(DFS Color)
    - Move, Rotate, Flip
    - Copy, Paste and Clipboard Features (For O2ARC)

#### 0.1.1
- Change Images
- Fix minor issues

#### 0.1.0 
- Initial Build.
- **ArcEnv-v0 & MiniArcEnv-v0 Launched.**
# ARCLE - ARC Learning Environment

ARCLE is a lightweight Gymnasium (previously OpenAI Gym) environment for training on ARC(Abstraction and Reasoning Corpus) and ARC-like datasets.

## Installation
`pip install arcle`

## Changelogs

#### 0.2.2
- Bug fix
    - Default all-or-none reward now gives only when submitted. It affects to existing all environments.

#### 0.2.1
- O2ARCv2Env-v0
    - Changed `ResizeGrid` action to `CropGrid` action
    - Forced `FloodFill` action to select only one pixel (otherwise, it is NoOP)
    - Now `CopyI`, `CopyO`, `Paste` regards black pixels (pixel value 0) as a background. It copies pixels where the value is nonzero and the pixel is selected.
    - `action['selection']` can handle int (automatically casts into `np.bool_`)
- Several exceptions handling

#### 0.2.0
- ArcEnv-v0 & MiniArcEnv-v0 has renamed to ARCEnv-v0 & MiniARCEnv-v0.
- Changed internal `current_grid` variable to `grid` & `grid_dim`
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

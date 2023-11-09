# ARCLE - ARC Learning Environment

ARCLE is a lightweight Gymnasium (previously OpenAI Gym) environment for training on ARC(Abstraction and Reasoning Corpus) and ARC-like datasets.

## Requirements
Python >= 3.8

## Installation
`pip install arcle`

## Changelogs

#### 0.2.5
- Python 3.8 Support (minimum requirements of Gymnasium)
- Env Changes
    - **Rename `ARCEnv` into `RawARCEnv` in this version.**
    - **Removed `MiniARCEnv`. Please use `RawARCEnv` with `loader=MiniARCLoader()` instead.**
    - **New `ARCEnv` added, consisting action space of ARC testing interface given along with the ARC Dataset.**
    - States in every environments are fully observable. All state-related instance variables are now in the `current_state` dictionary.
        - All operations receives `state` and `action`, and it changes `state` in-place.
        - You can deepcopy the state and call `env.transition(state_copied, action)` to get next state without changing original state on `O2ARCv2Env`. `env.transition` will be replaced as separated utility function in the future.
    - n-trial mode added. You canset maximum trials when you call `gym.make()` by putting argument `max_trial=num`. Unlimited trial mode is available when it is set by -1 (default).
    - Customizable `Submit` operation. It is defined in each env class as a method, not in a separated module.
        - You can specify boolean option `reset_on_submit` in `env.reset` (default=`False`)
    
- Bug fix
    - `FloodFill` operation without selection case fixed
    - `Paste` operation out-of-bound case fixed
    - `CopyI` operation out-of-bound case fixed
    - Apply patch exception handling

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

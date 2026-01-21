# openPMD Visualization Script

This script visualizes particle momenta and kinetic energy from openPMD simulation data.

## Features
- Loads openPMD data and reads particle momenta
- Plots average radial and tangential momentum, and kinetic energy per iteration
- Supports global or local normalization
- Outputs images for each iteration and species

## Requirements
- Python 3.x
- openPMD-api
- numpy
- matplotlib
- opencv-python
- torch (PyTorch) (cuda)
- scipy

## Usage
```sh
python visualize_openPMD.py --series <path_to_simulation_directory> [--out <output_dir>] [-s <shrink_factor>] [--norm] [--debug]
```

- `--series` (required): Path to the simulation directory (where `simOutput` is located)
- `--out`: Output directory for plots (default: `<series>/simOutput/visualization`)
- `-s`, `--shrink`: Shrink factor for canvas size (default: 1 - size of the grid)
- `--norm`: Normalize colors to global min/max across all iterations
- `--debug`: Enable verbose debug output

## Example
```sh
python visualize_openPMD.py --series ./electrons_2d_B0_largeGrid_full --norm --debug
```

## Output
- Images are saved in the output directory, organized by quantity and species.
- To create videos from the images, run:
  ```sh
  python create_videos.py --series <path_to_simulation_directory>
  ```

## Example Output

A sample video generated from the output images:

[Energy_ions_B20_Large_Grid.mp4](https://github.com/FilipO28555/openPMD_visualization_torch/raw/refs/heads/main/Energy_ions_B20_Large_Grid.mp4)

You can open this file in your video player to preview the results.

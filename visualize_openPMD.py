#!/usr/bin/env python3
"""
Minimal script: load openPMD, read particle momenta, plot average momentum per iteration.

IMPORTANT NOTE ON PARTICLE POSITIONS:
In openPMD, particle positions are stored in a special format:
- The raw position values are cell-relative coordinates (typically 0-1 within a cell)
- You MUST add the 'positionOffset' to get actual grid cell indices
- Then multiply by unitSI to convert to physical units (e.g., meters)
Without positionOffset, you'll get random-looking distributions!
"""
import os
import re
import argparse
from pathlib import Path
from collections import Counter

import numpy as np
import openpmd_api as opmd
import matplotlib.pyplot as plt
import cv2
import torch
from scipy import constants

from torch_draw import *


def format_value_with_unit(value, unit_type='momentum'):
    """
    Format a value with appropriate SI prefix for readability.
    
    Args:
        value: The value to format
        unit_type: 'momentum' or 'energy'
    
    Returns:
        Tuple of (formatted_value, unit_string)
    """
    if unit_type == 'energy':
        # Convert Joules to eV
        value_eV = value / constants.e  # 1 eV = 1.602e-19 J
        
        if abs(value_eV) >= 1e9:
            return value_eV / 1e9, 'GeV'
        elif abs(value_eV) >= 1e6:
            return value_eV / 1e6, 'MeV'
        elif abs(value_eV) >= 1e3:
            return value_eV / 1e3, 'keV'
        else:
            return value_eV, 'eV'
    
    elif unit_type == 'momentum':
        # Keep in kg*m/s but use SI prefixes
        if abs(value) >= 1e-15:
            return value * 1e18, 'atto kg*m/s (x10^-18)'
        elif abs(value) >= 1e-18:
            return value * 1e21, 'zepto kg*m/s (x10^-21)'
        elif abs(value) >= 1e-21:
            return value * 1e24, 'yocto kg*m/s (x10^-24)'
        else:
            return value, 'kg*m/s'
    
    return value, ''


def to_numpy(component, series):
    """Read a RecordComponent into a numpy array using slice + flush (compat)."""
    arr = component[:]
    series.flush()
    return np.array(arr)


def add_colorbar_and_scalebar(canvas_img, value_min, value_max, value_name, grid_spacing_si, grid_size):
    """
    Add colorbar and scale bar to the canvas image.
    
    Args:
        canvas_img: numpy array (H, W, 3) with the canvas
        value_min: minimum value for colorbar
        value_max: maximum value for colorbar
        value_name: name of the quantity being plotted
        grid_spacing_si: grid spacing in SI units (meters)
        grid_size: tuple of (nx, ny) grid dimensions
    
    Returns:
        Extended image with colorbar and scalebar
    """
    h, w = canvas_img.shape[:2]
    
    # Scale all dimensions based on width (reference: 1000px width is perfect)
    scale_factor = w / 1000.0
    
    # Scaled dimensions
    extension_width = int(300 * scale_factor)
    colorbar_width = int(30 * scale_factor)
    colorbar_margin = int(20 * scale_factor)
    colorbar_y_margin = int(50 * scale_factor)
    colorbar_bottom_margin = int(200 * scale_factor)
    label_offset = int(5 * scale_factor)
    title_offset = int(20 * scale_factor)
    
    # Scaled font parameters
    font_scale = 0.5 * scale_factor
    font_thickness = max(1, int(1 * scale_factor))
    title_font_scale = 0.6 * scale_factor

    # Create extended canvas
    extended_w = w + extension_width
    extended_img = np.zeros((h, extended_w, 3), dtype=np.uint8)
    extended_img[:, :w] = canvas_img
    
    # Create colorbar
    colorbar_x = w + colorbar_margin
    colorbar_y_start = colorbar_y_margin
    colorbar_height = h - colorbar_bottom_margin
    
    # Generate colorbar gradient
    colorbar = np.linspace(255, 0, colorbar_height).astype(np.uint8)
    colorbar = np.tile(colorbar[:, np.newaxis], (1, colorbar_width))
    
    # Place colorbar
    extended_img[colorbar_y_start:colorbar_y_start+colorbar_height, 
                 colorbar_x:colorbar_x+colorbar_width] = colorbar[:, :, np.newaxis]
    
    # Add colorbar labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    color_white = (255, 255, 255)
    
    # Determine format based on range (use decimal if less than 3 orders of magnitude)
    if value_max > 0 and value_min > 0:
        orders_of_magnitude = np.log10(value_max / value_min) if value_min > 0 else float('inf')
    elif value_max != value_min:
        orders_of_magnitude = np.log10(abs(value_max - value_min))
    else:
        orders_of_magnitude = 0
    
    use_decimal = orders_of_magnitude < 3
    
    if use_decimal:
        # Use decimal notation
        # Determine appropriate number of decimal places
        if abs(value_max) >= 100:
            label_max = f"{value_max:.1f}"
            label_min = f"{value_min:.1f}"
            label_mid = f"{(value_max + value_min) / 2:.1f}"
        elif abs(value_max) >= 10:
            label_max = f"{value_max:.2f}"
            label_min = f"{value_min:.2f}"
            label_mid = f"{(value_max + value_min) / 2:.2f}"
        elif abs(value_max) >= 1:
            label_max = f"{value_max:.3f}"
            label_min = f"{value_min:.3f}"
            label_mid = f"{(value_max + value_min) / 2:.3f}"
        else:
            label_max = f"{value_max:.4f}"
            label_min = f"{value_min:.4f}"
            label_mid = f"{(value_max + value_min) / 2:.4f}"
    else:
        # Use exponential notation
        label_max = f"{value_max:.2e}"
        label_min = f"{value_min:.2e}"
        label_mid = f"{(value_max + value_min) / 2:.2e}"
    
    # Top label (max value)
    cv2.putText(extended_img, label_max, 
                (colorbar_x + colorbar_width + label_offset, colorbar_y_start + int(10 * scale_factor)),
                font, font_scale, color_white, font_thickness)
    
    # Bottom label (min value)
    cv2.putText(extended_img, label_min,
                (colorbar_x + colorbar_width + label_offset, colorbar_y_start + colorbar_height),
                font, font_scale, color_white, font_thickness)
    
    # Middle label
    cv2.putText(extended_img, label_mid,
                (colorbar_x + colorbar_width + label_offset, colorbar_y_start + colorbar_height // 2),
                font, font_scale, color_white, font_thickness)
    
    # Add quantity name
    cv2.putText(extended_img, value_name,
                (colorbar_x, colorbar_y_start - title_offset),
                font, title_font_scale, color_white, font_thickness)
    
    # Add scale bar (1/20th of grid in bottom right corner)
    scale_length_cells = grid_size[0] / 20  # 1/20th of grid
    scale_length_si = scale_length_cells * grid_spacing_si  # in meters
    scale_length_pixels = int(w / 20)  # pixels on canvas
    
    scale_bar_y = h - int(30 * scale_factor)
    scale_bar_x_start = w - scale_length_pixels - int(20 * scale_factor)
    scale_bar_x_end = w - int(20 * scale_factor)
    scale_bar_thickness = max(2, int(2 * scale_factor))
    scale_bar_tick_height = int(5 * scale_factor)
    
    # Draw scale bar
    cv2.line(extended_img, (scale_bar_x_start, scale_bar_y),
             (scale_bar_x_end, scale_bar_y), color_white, scale_bar_thickness)
    cv2.line(extended_img, (scale_bar_x_start, scale_bar_y - scale_bar_tick_height),
             (scale_bar_x_start, scale_bar_y + scale_bar_tick_height), color_white, scale_bar_thickness)
    cv2.line(extended_img, (scale_bar_x_end, scale_bar_y - scale_bar_tick_height),
             (scale_bar_x_end, scale_bar_y + scale_bar_tick_height), color_white, scale_bar_thickness)
    
    # Add scale bar label
    scale_label = f"{scale_length_si:.2e} m"
    cv2.putText(extended_img, scale_label,
                (scale_bar_x_start, scale_bar_y - int(10 * scale_factor)),
                font, font_scale, color_white, font_thickness)
    
    return extended_img


def get_grid_info(iteration, debug=False):
    """
    Extract grid information from openPMD iteration.
    
    Returns:
        grid_spacing_si: grid spacing in SI units (meters)
        grid_size: tuple of (nx, ny) grid dimensions
    """
    # Try to get grid info from meshes
    if len(iteration.meshes) > 0:
        # Get the first mesh - iterate directly over the container
        mesh_name = None
        for name in iteration.meshes:
            mesh_name = name
            break
        
        mesh = iteration.meshes[mesh_name]
        
        # Get grid spacing
        grid_spacing = mesh.grid_spacing
        grid_unit_si = mesh.grid_unit_SI
        
        # Calculate actual SI spacing
        grid_spacing_si = grid_spacing[0] * grid_unit_si  # assuming uniform spacing
        
        # Get grid size from first component
        component_name = None
        for name in mesh:
            component_name = name
            break
        
        component = mesh[component_name]
        grid_size = component.shape
        
        if debug:
            print(f"    Grid info: mesh='{mesh_name}', size={grid_size}, spacing={grid_spacing_si:.2e} m")
        
        return grid_spacing_si, grid_size
    else:
        if debug:
            print("    Warning: No mesh found, using default grid info")
        return 1e-6, (512, 512)  # Default values


def plot_momenta(series, iteration, it, species_name, out_dir, shrink_factor=1, global_ranges=None, debug=False):
    species = iteration.particles[species_name]
    if debug:
        print(f"  Processing species: {species_name}")
    
    if "momentum" not in species:
        if debug:
            print(f"  Species {species_name} has no 'momentum' record; skipping")
        return
    
    # Get grid information
    grid_spacing_si, grid_size = get_grid_info(iteration, debug)
    # Apply shrink factor to canvas size
    canvas_size_x = (grid_size[0] if len(grid_size) > 0 else 512) // shrink_factor
    canvas_size_y = (grid_size[1] if len(grid_size) > 1 else canvas_size_x) // shrink_factor
    if debug:
        print(f"    Canvas size after shrink factor {shrink_factor}: {canvas_size_x}x{canvas_size_y}")
    
    # Read momentum
    record = species["momentum"]
    px = to_numpy(record["x"], series)
    py = to_numpy(record["y"], series)
    pz = to_numpy(record["z"], series)
    
    # Apply unit conversion if needed
    momentum_unit_si = record["x"].unit_SI
    
    # Convert to PyTorch CUDA tensors for faster computation
    px_torch = torch.from_numpy(px).cuda() * momentum_unit_si
    py_torch = torch.from_numpy(py).cuda() * momentum_unit_si
    pz_torch = torch.from_numpy(pz).cuda() * momentum_unit_si
    
    if debug:
        print(f"    momentum px: min={px_torch.min().item():.2e}, max={px_torch.max().item():.2e} kg*m/s")
        print(f"    momentum py: min={py_torch.min().item():.2e}, max={py_torch.max().item():.2e} kg*m/s")
    
    # Read weights
    record_weight = species["weighting"]
    weights = to_numpy(record_weight, series)
    weights_torch = torch.from_numpy(weights).cuda()
    if debug:
        print(f"    weights: min={weights_torch.min().item()}, max={weights_torch.max().item()}")
    
    # Read positions
    record_pos = species["position"]
    x = to_numpy(record_pos["x"], series) 
    y = to_numpy(record_pos["y"], series)
    
    # Apply positionOffset
    if "positionOffset" in species:
        record_offset = species["positionOffset"]
        offset_x = to_numpy(record_offset["x"], series)
        offset_y = to_numpy(record_offset["y"], series)
        x = x + offset_x
        y = y + offset_y
    
    # Apply unit conversion to SI and convert to CUDA tensors
    x_torch = torch.from_numpy(x).cuda() * record_pos["x"].unit_SI
    y_torch = torch.from_numpy(y).cuda() * record_pos["y"].unit_SI
    
    # Convert positions to cell indices (integer grid positions)
    x_cell_torch = (x_torch / grid_spacing_si).long()
    y_cell_torch = (y_torch / grid_spacing_si).long()
    
    # Apply shrink factor to cell positions
    x_cell_torch = x_cell_torch // shrink_factor
    y_cell_torch = y_cell_torch // shrink_factor
    
    # Clamp to grid boundaries
    x_cell_torch = torch.clamp(x_cell_torch, 0, canvas_size_x - 1)
    y_cell_torch = torch.clamp(y_cell_torch, 0, canvas_size_y - 1)
    
    if debug:
        print(f"    Cell positions: x=[{x_cell_torch.min().item()}, {x_cell_torch.max().item()}], y=[{y_cell_torch.min().item()}, {y_cell_torch.max().item()}]")
    
    # Calculate center based on SI positions (using PyTorch)
    center_x = 0.5 * (x_torch.max() + x_torch.min())
    center_y = 0.5 * (y_torch.max() + y_torch.min())
    
    # Vectors from center to each particle
    vec_x = x_torch - center_x
    vec_y = y_torch - center_y
    
    # Radial distance
    r = torch.sqrt(vec_x**2 + vec_y**2)
    r[r == 0] = 1.0  # avoid division by zero
    
    # Normalized radial vectors
    vec_x_norm = vec_x / r
    vec_y_norm = vec_y / r
    
    # Radial momentum component (projection onto radial vector)
    p_radial = px_torch * vec_x_norm + py_torch * vec_y_norm
    
    # Angular momentum component (cross product in 2D: tangential component)
    # L = r × p, in 2D: L_z = x*p_y - y*p_x, but we want tangential momentum
    # Tangential momentum: perpendicular to radial
    p_tangential = -px_torch * vec_y_norm + py_torch * vec_x_norm
    
    # Kinetic energy: E_k = p²/(2m)
    # Get particle mass from the species (should be in the mass record)
    if "mass" in species:
        record_mass = species["mass"]
        mass = to_numpy(record_mass, series)
        # Mass might be a single value or per-particle
        if mass.size == 1:
            particle_mass = mass.item() * record_mass.unit_SI
        else:
            mass_torch = torch.from_numpy(mass).cuda() * record_mass.unit_SI
            particle_mass = mass_torch
        if debug:
            print(f"    Particle mass: {particle_mass if isinstance(particle_mass, float) else particle_mass[0].item():.2e} kg")
    else:
        # Default to electron mass if not specified
        particle_mass = constants.m_e
        if debug:
            print(f"    Warning: No mass record found, assuming electron mass: {particle_mass:.2e} kg")
    
    # Calculate kinetic energy: E_k = p²/(2m)
    p_magnitude = torch.sqrt(px_torch**2 + py_torch**2 + pz_torch**2)
    kinetic_energy = (p_magnitude**2) / (2.0 * particle_mass)  # In Joules
    
    if debug:
        print(f"    Radial momentum: min={p_radial.min().item():.2e}, max={p_radial.max().item():.2e} kg*m/s")
        print(f"    Tangential momentum: min={p_tangential.min().item():.2e}, max={p_tangential.max().item():.2e} kg*m/s")
        
        # Convert kinetic energy to eV for display
        kinetic_energy_eV = kinetic_energy / constants.e
        print(f"    Kinetic energy: min={kinetic_energy_eV.min().item():.2e}, max={kinetic_energy_eV.max().item():.2e} eV")
    
    # Create plots for each quantity
    quantities = [
        ("radial_momentum", p_radial, "Radial Momentum", 'momentum'),
        ("tangential_momentum", p_tangential, "Tangential Momentum", 'momentum'),
        ("kinetic_energy", kinetic_energy, "Kinetic Energy", 'energy')
    ]
    
    for qty_name, qty_values, qty_label, unit_type in quantities:
        if debug:
            print(f"    Creating plot for {qty_name}...")
        
        # Create output folder for this quantity
        qty_out_dir = os.path.join(out_dir, qty_name)
        Path(qty_out_dir).mkdir(parents=True, exist_ok=True)
        
        # Get min/max for colorbar and normalization
        if global_ranges and species_name in global_ranges and qty_name in global_ranges[species_name]:
            # Use global ranges for this specific species
            value_min = global_ranges[species_name][qty_name]['min']
            value_max = global_ranges[species_name][qty_name]['max']
            if debug:
                print(f"      Using global range for species '{species_name}': [{value_min:.2e}, {value_max:.2e}]")
        else:
            # Use local ranges from this iteration
            value_min = qty_values.min().item()
            value_max = qty_values.max().item()
        
        # Create accumulation grids on GPU
        sum_grid = torch.zeros((canvas_size_y, canvas_size_x), dtype=torch.float64, device='cuda')
        count_grid = torch.zeros((canvas_size_y, canvas_size_x), dtype=torch.float64, device='cuda')
        
        # Accumulate weighted values and weights separately for mean calculation
        weighted_values = (qty_values * weights_torch).double()
        weights_double = weights_torch.double()
        
        # Convert to linear indices for faster accumulation
        linear_indices = y_cell_torch * canvas_size_x + x_cell_torch
        
        # Accumulate sum of weighted values
        sum_grid_flat = sum_grid.view(-1)
        sum_grid_flat.scatter_add_(0, linear_indices, weighted_values)
        
        # Accumulate sum of weights (count)
        count_grid_flat = count_grid.view(-1)
        count_grid_flat.scatter_add_(0, linear_indices, weights_double)
        
        # Calculate mean: sum / count (avoid division by zero)
        mean_grid = torch.zeros_like(sum_grid)
        mask = count_grid > 0
        mean_grid[mask] = sum_grid[mask] / count_grid[mask]
        
        # Normalize mean values to [0, 1] range based on global/local min/max
        if value_max != value_min:
            normalized_grid = (mean_grid - value_min) / (value_max - value_min)
            normalized_grid = torch.clamp(normalized_grid, 0.0, 1.0)
        else:
            normalized_grid = torch.zeros_like(mean_grid)
        
        # Convert to 8-bit grayscale
        canvas_img = (normalized_grid.cpu().numpy() * 255).astype(np.uint8)
        
        # Format values with appropriate units for colorbar
        qty_min_formatted, _ = format_value_with_unit(value_min, unit_type)
        qty_max_formatted, _ = format_value_with_unit(value_max, unit_type)
        _, unit_str = format_value_with_unit(value_max, unit_type)
        
        # Create label with unit
        qty_label_with_unit = f"{qty_label} ({unit_str})"
        
        # Convert to BGR for OpenCV
        canvas_img_bgr = cv2.cvtColor(canvas_img, cv2.COLOR_GRAY2BGR)
        
        # Add colorbar and scalebar
        final_img = add_colorbar_and_scalebar(
            canvas_img_bgr, 
            qty_min_formatted, 
            qty_max_formatted, 
            qty_label_with_unit,
            grid_spacing_si,
            grid_size
        )
        
        # Save image
        out_path = os.path.join(qty_out_dir, f'species_{species_name}')
        Path(out_path).mkdir(parents=True, exist_ok=True)
        out_path = os.path.join(out_path, f'{species_name}_it_{it:05d}.png')
        cv2.imwrite(out_path, final_img)
        if debug:
            print(f"    Saved {qty_name} plot to {out_path}")


def scan_global_ranges(series, iter_ids, debug=False):
    """
    Scan all iterations to find global min/max values for each quantity per species.
    
    Returns:
        Dictionary with min/max for each quantity type per species
    """
    if debug:
        print("Scanning all iterations for global min/max values per species...")
    
    # Structure: global_ranges[species_name][quantity_name] = {'min': ..., 'max': ...}
    global_ranges = {}
    
    for idx, it in enumerate(iter_ids):
        if debug:
            print(f"  Scanning iteration {it} ({idx+1}/{len(iter_ids)})...")
        iteration = series.iterations[it]
        species_names = list(iteration.particles)
        
        for species_name in species_names:
            # Initialize species entry if not exists
            if species_name not in global_ranges:
                global_ranges[species_name] = {
                    'radial_momentum': {'min': float('inf'), 'max': float('-inf')},
                    'tangential_momentum': {'min': float('inf'), 'max': float('-inf')},
                    'kinetic_energy': {'min': float('inf'), 'max': float('-inf')}
                }
            
            species = iteration.particles[species_name]
            
            if "momentum" not in species:
                continue
            
            # Read momentum
            record = species["momentum"]
            px = to_numpy(record["x"], series)
            py = to_numpy(record["y"], series)
            pz = to_numpy(record["z"], series)
            
            momentum_unit_si = record["x"].unit_SI
            px_torch = torch.from_numpy(px).cuda() * momentum_unit_si
            py_torch = torch.from_numpy(py).cuda() * momentum_unit_si
            pz_torch = torch.from_numpy(pz).cuda() * momentum_unit_si
            
            # Read positions
            record_pos = species["position"]
            x = to_numpy(record_pos["x"], series)
            y = to_numpy(record_pos["y"], series)
            
            if "positionOffset" in species:
                record_offset = species["positionOffset"]
                offset_x = to_numpy(record_offset["x"], series)
                offset_y = to_numpy(record_offset["y"], series)
                x = x + offset_x
                y = y + offset_y
            
            x_torch = torch.from_numpy(x).cuda() * record_pos["x"].unit_SI
            y_torch = torch.from_numpy(y).cuda() * record_pos["y"].unit_SI
            
            # Calculate center
            center_x = 0.5 * (x_torch.max() + x_torch.min())
            center_y = 0.5 * (y_torch.max() + y_torch.min())
            
            vec_x = x_torch - center_x
            vec_y = y_torch - center_y
            r = torch.sqrt(vec_x**2 + vec_y**2)
            r[r == 0] = 1.0
            
            vec_x_norm = vec_x / r
            vec_y_norm = vec_y / r
            
            # Calculate quantities
            p_radial = px_torch * vec_x_norm + py_torch * vec_y_norm
            p_tangential = -px_torch * vec_y_norm + py_torch * vec_x_norm
            
            # Get mass for kinetic energy
            if "mass" in species:
                record_mass = species["mass"]
                mass = to_numpy(record_mass, series)
                if mass.size == 1:
                    particle_mass = mass.item() * record_mass.unit_SI
                else:
                    mass_torch = torch.from_numpy(mass).cuda() * record_mass.unit_SI
                    particle_mass = mass_torch
            else:
                particle_mass = constants.m_e
            
            p_magnitude = torch.sqrt(px_torch**2 + py_torch**2 + pz_torch**2)
            kinetic_energy = (p_magnitude**2) / (2.0 * particle_mass)
            
            # Update global ranges for this species
            global_ranges[species_name]['radial_momentum']['min'] = min(
                global_ranges[species_name]['radial_momentum']['min'], p_radial.min().item())
            global_ranges[species_name]['radial_momentum']['max'] = max(
                global_ranges[species_name]['radial_momentum']['max'], p_radial.max().item())
            
            global_ranges[species_name]['tangential_momentum']['min'] = min(
                global_ranges[species_name]['tangential_momentum']['min'], p_tangential.min().item())
            global_ranges[species_name]['tangential_momentum']['max'] = max(
                global_ranges[species_name]['tangential_momentum']['max'], p_tangential.max().item())
            
            global_ranges[species_name]['kinetic_energy']['min'] = min(
                global_ranges[species_name]['kinetic_energy']['min'], kinetic_energy.min().item())
            global_ranges[species_name]['kinetic_energy']['max'] = max(
                global_ranges[species_name]['kinetic_energy']['max'], kinetic_energy.max().item())
    
    if debug:
        print("\nGlobal ranges found:")
        for species_name, species_ranges in global_ranges.items():
            print(f"  Species '{species_name}':")
            for qty_name, ranges in species_ranges.items():
                print(f"    {qty_name}: [{ranges['min']:.2e}, {ranges['max']:.2e}]")
    
    return global_ranges


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--series", required=True, help="Path to simulation directory (e.g., /path/to/electrons_2d_B0_largeGrid_full)"
    )
    parser.add_argument(
        "--out", default=None, help="Output directory for plots (default: <series>/simOutput/visualization)"
    )
    parser.add_argument(
        "-s", "--shrink", type=int, default=1, 
        help="Shrink factor for canvas size (e.g., 2 means canvas is half the grid size)"
    )
    parser.add_argument(
        "--norm", action="store_true",
        help="Normalize colors to global min/max across all iterations"
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Enable verbose debug output"
    )
    args = parser.parse_args()
    base_series_path = args.series
    shrink_factor = args.shrink
    normalize_global = args.norm
    debug = args.debug
    
    # Automatically append /simOutput/openPMD to the series path
    series_path = os.path.join(base_series_path, "simOutput", "openPMD")
    
    # Set default output directory to simOutput/visualization if not specified
    if args.out is None:
        out_dir = os.path.join(base_series_path, "simOutput", "visualization")
    else:
        out_dir = args.out
    
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    
    if shrink_factor < 1:
        print(f"Error: shrink factor must be >= 1, got {shrink_factor}")
        return

    # Construct the full openPMD series path
    fullPath = os.path.join(series_path, "simOutput_%T.bp5")
    series = opmd.Series(fullPath, opmd.Access.read_only)

    try:
        iter_ids = sorted(series.iterations)
    except TypeError:
        iter_ids = sorted(series.iterations.keys())
    
    # Get grid info from first iteration for info table
    first_iteration = series.iterations[iter_ids[0]]
    grid_spacing_si, grid_size = get_grid_info(first_iteration, debug=False)
    species_names = list(first_iteration.particles)
    canvas_size_x = grid_size[0] // shrink_factor
    canvas_size_y = grid_size[1] // shrink_factor
    
    # Print information table
    print("=" * 70)
    print("SIMULATION VISUALIZATION PARAMETERS")
    print("=" * 70)
    print(f"Base series path:      {base_series_path}")
    print(f"OpenPMD data path:     {series_path}")
    print(f"Output directory:      {out_dir}")
    print(f"Grid size:             {grid_size[0]} x {grid_size[1]} cells")
    print(f"Grid spacing (dx):     {grid_spacing_si:.2e} m")
    print(f"Shrink factor:         {shrink_factor}")
    print(f"Canvas size:           {canvas_size_x} x {canvas_size_y} pixels")
    print(f"Number of iterations:  {len(iter_ids)}")
    print(f"Iteration range:       {iter_ids[0]} - {iter_ids[-1]}")
    print(f"Number of species:     {len(species_names)}")
    print(f"Species names:         {', '.join(species_names)}")
    print(f"Normalization mode:    {'Global (across all iterations)' if normalize_global else 'Local (per iteration)'}")
    print(f"Debug mode:            {'Enabled' if debug else 'Disabled'}")
    print("=" * 70)
    print()

    # Scan for global ranges if normalization is enabled
    global_ranges = None
    if normalize_global:
        global_ranges = scan_global_ranges(series, iter_ids, debug)

    for idx, it in enumerate(iter_ids):
        iteration = series.iterations[it]
        species_names = list(iteration.particles)
        print(f"Processing iteration {it} ({idx+1}/{len(iter_ids)})...")

        for species_name in species_names:
            plot_momenta(series, iteration, it, species_name, out_dir, shrink_factor, global_ranges, debug)
            
            
            

    series.close()
    print("\nVisualization complete!")
    print("\nTo create videos from these images, run:")
    print(f"  python create_videos.py --series {base_series_path}")


if __name__ == "__main__":
    main()

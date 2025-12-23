# Ensemble-Based Event Camera Place Recognition Under Varying Illumination

This repository contains the official codebase for the paper:

**Ensemble-Based Event Camera Place Recognition Under Varying Illumination** *Therese Joseph, Tobias Fischer, Michael Milford*

### NSAVP night

<table>
  <tr>
    <td align="center"><b>Event Count</b></td>
    <td align="center"><b>No Polarity</b></td>
    <td align="center"><b>Time Surface</b></td>
    <td align="center"><b>E2VID</b></td>
  </tr>
  <tr>
    <td><img src="./plots/gifs_cropped/R0_RN0_eventCount_0.2_last10.gif" width="220"/></td>
    <td><img src="./plots/gifs_cropped/R0_RN0_eventCount_noPolarity_0.2_last10.gif" width="220"/></td>
    <td><img src="./plots/gifs_cropped/R0_RN0_timeSurface_0.2_last10.gif" width="220"/></td>
    <td><img src="./plots/gifs_cropped/R0_RN0_e2vid_0.2_last10.gif" width="220"/></td>
  </tr>
</table>

---

### NSAVP afternoon

<table>
  <tr>
    <td align="center"><b>Event Count</b></td>
    <td align="center"><b>No Polarity</b></td>
    <td align="center"><b>Time Surface</b></td>
    <td align="center"><b>E2VID</b></td>
  </tr>
  <tr>
    <td><img src="./plots/gifs_cropped/R0_FA0_eventCount_0.2_last10.gif" width="220"/></td>
    <td><img src="./plots/gifs_cropped/R0_FA0_eventCount_noPolarity_0.2_last10.gif" width="220"/></td>
    <td><img src="./plots/gifs_cropped/R0_FA0_timeSurface_0.2_last10.gif" width="220"/></td>
    <td><img src="./plots/gifs_cropped/R0_FA0_e2vid_0.2_last10.gif" width="220"/></td>
  </tr>
</table>

---

### Brisbane daytime

<table>
  <tr>
    <td align="center"><b>Event Count</b></td>
    <td align="center"><b>No Polarity</b></td>
    <td align="center"><b>Time Surface</b></td>
    <td align="center"><b>E2VID</b></td>
  </tr>
  <tr>
    <td><img src="./plots/gifs_cropped/daytime_eventCount_0.2_last10.gif" width="220"/></td>
    <td><img src="./plots/gifs_cropped/daytime_eventCount_noPolarity_0.2_last10.gif" width="220"/></td>
    <td><img src="./plots/gifs_cropped/daytime_timeSurface_0.2_last10.gif" width="220"/></td>
    <td><img src="./plots/gifs_cropped/daytime_e2vid_0.2_last10.gif" width="220"/></td>
  </tr>
</table>

---

### Abstract

Compared to conventional cameras, event cameras provide a high dynamic range and low latency, offering greater robustness to rapid motion and challenging lighting conditions. Although the potential of event cameras for visual place recognition (VPR) has been established, developing robust VPR frameworks under severe illumination changes remains an open research problem.

In this paper, we introduce an **ensemble-based approach** to event camera place recognition that combines sequence-matched results from multiple event-to-frame reconstructions, VPR feature extractors, and temporal resolutions. Our broader fusion strategy delivers significantly improved robustness under varied lighting conditions, achieving a **57% relative improvement in Recall@1** across day-night transitions.

---

## Project Structure

| File | Description |
| --- | --- |
| `load_and_save.py` | Reconstructs event streams into frames using methods like `eventCount`, `timeSurface`, or `e2vid`. |
| `testing.py` | Runs the VPR evaluation pipeline for individual methods and binnings. |
| `ablate_ensembles.py` | Performs the ensemble fusion of multiple VPR methods and reconstructions. |
| `evaluate.ipynb` | Jupyter notebook for result aggregation, Recall@N calculation, and plotting. |
| `parser_config.py` | Central configuration for dataset paths and default hyperparameters. |
| `submit_jobs_*.py` | Automation scripts for batch processing on HPC clusters (PBS). |

---

## Workflow

### 1. Data Reconstruction

Convert raw event data into visual frames. You can specify different binning resolution and reconstruction methods.

```bash
python load_and_save.py --dataset_type Brisbane --reconstruct_method_name eventCount --time_res 1.0

```

### 2. Individual VPR Evaluation

Run a specific VPR method (e.g., MixVPR, NetVLAD, CosPlace) on the reconstructed frames.

```bash
python testing.py --method mixvpr --dataset_type NSAVP --reconstruct_method_name e2vid --seq_len 10

```

### 3. Running Ensembles (`ablate_ensembles.py`)

This script implements the core contribution of the paper: fusing results from different configurations to improve robustness. It applies the **modified sequence matching** framework and calculates the combined Recall@1.

To run an ensemble over multiple VPR methods and reconstructions:

```bash
python ablate_ensembles.py \
    --dataset_name Brisbane \
    --ref_seq daytime --qry_seq night \
    --ensemble_over vpr_methods \
    --vpr_methods mixvpr,netvlad,cosplace \
    --recon_methods eventCount\
    --time_strs 0.1 \
    --seq_len 10

```

* `--ensemble_over`: Specify whether to ensemble over VPR methods, reconstructions, or temporal resolutions.
* `--vpr_methods` / `--recon_methods`: Comma-separated lists of methods to include in the fusion.

### 4. Result Analysis & Visualization (`evaluate.ipynb`)

The `evaluate.ipynb` notebook is used to process the CSV outputs generated by the scripts above.

* **Aggregation:** Automatically loads results from the `./results/` directory.
* **Performance Metrics:** Generates Recall@N curves and compares different binning strategies.
* **Visual Debugging:** Visualizes similarity matrices alongside ground truth masks to identify where the localization succeeds or fails across different traverses.

---

## HPC Implementation

For large-scale experiments, use the `submit_jobs_*.py` scripts. These are designed to generate and submit PBS scripts to a cluster, handling the combinations of dataset sequences and hyperparameters automatically.

```bash
# Example: Batch submit evaluation jobs
python submit_jobs_vpr_eval.py

```

---

## Citation

If you find this code useful in your research, please cite our paper:

```bibtex
@ARTICLE{11283034,
  author={Joseph, Therese and Fischer, Tobias and Milford, Michael},
  journal={IEEE Robotics and Automation Letters}, 
  title={Ensemble-Based Event Camera Place Recognition Under Varying Illumination}, 
  year={2026},
  volume={11},
  number={2},
  pages={1290-1297},
  keywords={Feature extraction;Visual place recognition;Cameras;Image reconstruction;Lighting;Reconstruction algorithms;Event detection;Robustness;Pipelines;Surface reconstruction;Localization;computer vision for transportation},
  doi={10.1109/LRA.2025.3641119}}


```
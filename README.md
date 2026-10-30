# Communicating Robot Perceptual Uncertainty in Collaborative Manipulation

**Official Companion Repository for ICCII 2026 Conference Submission**

[![IEEE Conference](https://img.shields.io/badge/Conference-ICCII%202026-00629B.svg)](https://iccii2026.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Simulation: Gazebo Harmonic](https://img.shields.io/badge/Gazebo-Harmonic%208.15-orange.svg)](https://gazebosim.org)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)

---

## Authors
- **Syed Ali Turab Kazmi** — *Department of Robotics and Intelligent Systems, Bahria School of Engineering and Applied Sciences (BSEAS), Bahria University, H-11 Campus, Islamabad, Pakistan* (`syedaliturab072@gmail.com`)
- **Dr. Muhammad Hasan Danish Khan** (*Corresponding Author*) — *Department of Electrical Engineering, Bahria School of Engineering and Applied Sciences (BSEAS), Bahria University, H-11 Campus, Islamabad, Pakistan* (`hasandanish.buic@bahria.edu.pk`)

---

## Overview
This repository contains the official code, simulation environments, evaluation pipelines, and 6-page camera-ready manuscript submitted to the **2026 International Conference on Computational Intelligence and Informatics (ICCII 2026)**.

The research investigates the fundamental trade-off between **implicit kinematic hesitation** (velocity scaling, dwell pauses, micro-retreats) and **explicit visual-verbal eHMI** (chromatic status bars, percentages, acoustic earcons) for communicating robot perceptual uncertainty during close-proximity human-robot collaborative tabletop manipulation.

---

## Repository Structure
```
├── README.md                           # This document
├── LICENSE                             # MIT Open Access License
├── paper/
│   ├── main.tex                        # Condensed 6-page IEEEtran conference LaTeX source
│   ├── paper.pdf                       # Compiled 6-page IEEE Conference Paper
│   ├── references.bib                  # Peer-reviewed BibTeX references
│   └── figures/                        # High-resolution vector/raster figures
├── figures/                            # 300 DPI Publication figures (Fig 1–4)
└── sim/
    ├── generate_hri_sim_and_telemetry.py # 50 Hz Gazebo kinematics & DDM observer simulator
    ├── run_statistical_analysis.py       # Repeated-measures ANOVA & Welch t-tests
    ├── render_sim_video.py               # 3D keyframe & simulation video generator
    ├── sim_metrics.csv                   # High-frequency trajectory telemetry
    ├── hri_trials.csv                    # Participant trial log (576 trials, N=24)
    ├── hri_surveys.csv                   # NASA-TLX & Trust Perception surveys
    ├── statistical_results.json          # Complete inferential statistics & p-values
    ├── sim_video.mp4                     # Full composite simulation demo video
    ├── trial_c0_baseline.mp4             # Condition C0 trial recording
    ├── trial_c1_implicit.mp4             # Condition C1 trial recording
    ├── trial_c2_explicit.mp4             # Condition C2 trial recording
    ├── trial_c3_hybrid.mp4               # Condition C3 trial recording
    └── worlds/
        └── collaborative_tabletop.sdf    # High-fidelity Franka Panda Gazebo Harmonic SDF world
```

---

## Quickstart & Replication

### 1. Requirements
- Python 3.10+
- NumPy, SciPy, Pandas, Statsmodels, Matplotlib, Seaborn
- Gazebo Harmonic 8.15+ (for physics simulation execution)

```bash
git clone https://github.com/T-Storm-072/02_hri_uncertainty_iccii2026.git
cd 02_hri_uncertainty_iccii2026
pip install numpy scipy pandas statsmodels matplotlib seaborn
```

### 2. Run Inferential Statistical Verification
```bash
python sim/run_statistical_analysis.py
```

### 3. Regenerate Simulation Telemetry
```bash
python sim/generate_hri_sim_and_telemetry.py
```

---

## Citation
```bibtex
@inproceedings{kazmi2026communicating,
  author    = {Kazmi, Syed Ali Turab and Khan, Muhammad Hasan Danish},
  title     = {Communicating Robot Perceptual Uncertainty in Collaborative Manipulation: Comparing Implicit Kinematic Hesitation and Explicit Visual-Verbal Cues on Human Trust Calibration and Task Performance},
  booktitle = {Proceedings of the International Conference on Computational Intelligence and Informatics (ICCII)},
  year      = {2026}
}
```

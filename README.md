# 🧠 2P Imaging Analysis Pipeline

A modular, plug-and-play pipeline for **2-photon calcium imaging** data using [Suite2p](https://github.com/MouseLand/suite2p). Designed for analyzing multiple experiments with minimal setup, across different imaging planes and acquisition sessions.

---

## 🔧 Features

- ✅ **Suite2p integration** with multi-experiment compatibility  
- ✅ **Plane-aware support** (Z-stack / multiplane data)  
- ✅ **Minimal configuration** required per experiment  
- ✅ **Run hashing** to ensure reproducibility of analyses  
- ✅ **Trace extraction** (raw, dF/F, z-score)  
- ✅ **Postprocessing and visualization** with customizable outputs  

---

## ✍️ Authorship & Credits

- **Johannes Kappel** – Core pipeline foundation  
- **Enrico Kohn** & **Katja Slangewal** – Classifier design  
- **Inbal Shainer** – Cellpose model  
- **Joseph Donovan** – BiDiOffset utility  

---

## Installation

```bash
git clone https://github.com/lisakatobauer/2p_imaging_analysis_runner.git
cd 2p_imaging_analysis_runner
pip install -e .
````

---

## 🏗️ Project Structure

| Module               | Description                                                                 |
|----------------------|-----------------------------------------------------------------------------|
| `Suite2pProcessor`   | Prepares and runs Suite2p on multi-session data                             |
| `Suite2pLoader`      | Loads processed Suite2p output                                              |
| `Suite2pTraces`      | Extracts, filters, and normalizes fluorescence traces                       |
| `Suite2pVisualiser`  | Visualizes ROI activity, top traces, and heatmaps                          |

---

## 🚀 Getting Started

Each experiment requires a Python config file (e.g., `fish_125.py`) with information on the experiments.

Example file can be found in /config/configlist. 

Example runs can be found in /usage_examples

---

## 🔐 License
MIT License. See [LICENSE](LICENSE) for details.

---

## Contact

Created by [Lisa Bauer](https://github.com/lisakatobauer). Feel free to reach out!


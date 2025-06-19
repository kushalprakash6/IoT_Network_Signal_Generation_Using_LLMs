# IoT‑Network‑Signal‑Generation using LLMs

<p align="center">
  <img src="Images/banner.png" alt="IoT banner" width="600"/>
</p>

> **Synthetic, yet realistic.** This repository shows how Large Language Models (LLMs) can be coaxed into **speaking the language of a network packet‑sniffer**, producing traffic that mirrors real‑world IoT deployments.  The resulting traces can be used to stress‑test Intrusion‑Detection Systems (IDS), benchmark anomaly‑detectors, or augment scarce training data.

---

\## Table of Contents

1. [Why Synthetic Traffic?](#why-synthetic-traffic)
2. [Quick Start](#quick-start)
3. [Repository Layout](#repository-layout)
4. [Installation & Requirements](#installation--requirements)
5. [End‑to‑End Workflow](#end-to-end-workflow)
6. [Examples](#examples)
7. [Results & Benchmarks](#results--benchmarks)
8. [Contributing](#contributing)
9. [License](#license)
10. [Contact](#contact)

---

\## Why Synthetic Traffic?

* **Privacy** – real network captures often contain sensitive data that cannot be shared.
* **Class balance** – recreate rare attack vectors to balance training sets.
* **Cost & Time** – emulate an entire fleet of IoT devices without owning them.
* **Continuous testing** – generate fresh scenarios on‑demand for CI pipelines.

---

\## Quick Start

```bash
# Clone the repository
$ git clone https://github.com/kushalprakash6/IoT_Network_Signal_Generation_Using_LLMs.git
$ cd IoT_Network_Signal_Generation_Using_LLMs

# Install Python dependencies
$ pip install -r requirements.txt

# 1 Pre‑process raw pcaps (or CSV‑like log files)
$ python Pre-processing/clean_and_extract.py \
        --input data/raw \
        --output data/processed

# 2 Fine‑tune / train the language model
$ python Training_LLMs/train_model.py \
        --dataset data/processed \
        --epochs 3 \
        --out_dir models/iot_llm

# 3 Generate synthetic traces (PCAP, NetFlow or CSV)
$ python Generating_Signals/generate_signals.py \
        --model models/iot_llm \
        --config Generating_Signals/config.yaml \
        --out synthetic/signal_001.pcap

# 4 Evaluate the traffic with an IDS baseline
$ python IDS/evaluate.py \
        --traffic synthetic/signal_001.pcap \
        --ids_model IDS/signature_baseline.pkl
```

> Tip  Run `python <module>.py ‑‑help` for the full CLI on any script.

---

\## Repository Layout

```text
IoT_Network_Signal_Generation_Using_LLMs/
├── Generating_Signals/       # Signal synthesis & device‑profile prompts
│   ├── generate_signals.py
│   └── config.yaml
├── IDS/                      # Benchmark & scoring notebooks / scripts
│   ├── evaluate.py
│   └── models/              # Pre‑trained IDS baselines
├── Images/                   # Figures used in the README / papers
├── Pre-processing/           # Cleaning + feature‑extraction pipeline
│   ├── clean_and_extract.py
│   └── mapping.yaml         # Port / protocol look‑ups, etc.
├── Training_LLMs/            # Fine‑tuning routines & hyper‑param grids
│   ├── train_model.py
│   └── configs/
├── LICENSE                   # BSD 3‑Clause
└── README.md                 # You are here
```

---

\## Installation & Requirements

| Requirement          | Version tested | Notes                                               |
| -------------------- | -------------- | --------------------------------------------------- |
| Python               |  >= 3.8        | [pyenv](https://github.com/pyenv/pyenv) recommended |
| PyTorch / CUDA       |  2.2 / 11.8    | HuggingFace Transformers backend                    |
| scapy (optional)     |  2.5           | Writing PCAP files                                  |
| pandas / Polars      |  2.2 / 0.20    | Fast CSV+Parquet handling                           |
| matplotlib / seaborn |  3.9 / 0.14    | Plotting results                                    |

Install everything in one go:

```bash
pip install -r requirements.txt  # coming soon
```

*GPU* support is automatically detected; pass `--device cuda` when available.

---

\## End‑to‑End Workflow

```mermaid
graph LR
    A[Raw traffic captures (.pcap / logs)] --> B[Pre‑processing\nFeature extraction]
    B --> C[Fine‑tune LLM]
    C --> D[Generate synthetic traffic]
    D --> E[IDS / ML evaluation]
    E -->|metrics, feedback| C
```

1. **Pre‑processing** – parse pcap or CSV, anonymise IPs/MACs, derive flow‑level tokens.
2. **LLM Fine‑tuning** – auto‑regressive or seq‑to‑seq training with causal masking.
3. **Generation** – device‑prompt templates steer behaviour (e.g., *“Smart‑bulb → status report every 100 s”*).
4. **Evaluation** – compare IDS metrics (TPR, FPR, F1) between real vs. synthetic blends.

---

\## Examples

* **Notebooks** – see `examples/` (coming soon) for step‑by‑step Jupyter flows.
* **Docker** – run `docker compose up` for a reproducible sandbox with Suricata.
* **CI Demo** – GitHub Actions badge builds a mini‑dataset on every push.

---

\## Results & Benchmarks

| Dataset   | Real TPR | Synthetic TPR | Δ (pts) |
| --------- | -------: | ------------: | ------: |
| UNSW‑NB15 |    0.94  |         0.92  |   ‑0.02 |
| TON‑IoT   |    0.88  |         0.87  |   ‑0.01 |

> The synthetic traces retain >97 % detection efficacy while offering complete anonymisation.  See `Images/results.pdf` for full plots.

---

\## Contributing

1. Fork ▷ create branch ▷ commit ▷ PR.
2. Run `pre-commit run --all-files` before pushing.
3. For large model weights › open an issue first to discuss storage options.

All contributions – bug‑reports, documentation fixes, ideas – are welcome! 

---

\## License

Distributed under the **BSD 3‑Clause** license.  See the [LICENSE](LICENSE) file.

---

\## Contact

**Kushal Prakash**
[kushal.prakash@yahoo.in](mailto:kushal.prakash@yahoo.in)
Project link: [https://github.com/kushalprakash6/IoT\_Network\_Signal\_Generation\_Using\_LLMs](https://github.com/kushalprakash6/IoT_Network_Signal_Generation_Using_LLMs)

Feel free to open a discussion or an issue for questions and suggestions!

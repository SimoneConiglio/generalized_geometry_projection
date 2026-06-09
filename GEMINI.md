# SAMO Project: Simulation and Analysis of Manufacturing Operations

This project is a research and development environment focused on **Additive Manufacturing (ALM)** process modeling and **Multidisciplinary Design Analysis and Optimization (MDAO)**. It integrates computational mechanics with Generative AI to automate literature research and simulation prototyping.

## Project Overview

- **Domain:** Computational Mechanics, Additive Manufacturing, MDAO.
- **Core Technologies:**
  - **Simulation:** [FEniCS](https://fenicsproject.org/) (`dolfin`, `dolfin_adjoint`) for finite element analysis and adjoint-based optimization.
  - **AI Orchestration:** Google Gemini API (`google-genai`) for agentic workflows.
  - **Automation:** `typer` for CLI interfaces, `requests` for API interactions.

## Key Components

- **`orchestrator.py`**: A CLI tool that uses Gemini to generate and iteratively debug FEniCS simulation code. It targets structural mechanics problems (e.g., cantilever beams) with a focus on gradient-based optimization of process parameters.
- **`literature_orchestrator.py`**: A multi-agent "Research Swarm" that automates scientific literature reviews. It queries arXiv and Semantic Scholar, evaluates papers for differentiability/compatibility with MDAO, and synthesizes reports.
- **`sandbox/`**: Contains generated simulation prototypes (e.g., `fenics_prototype.py`).
- **`poisson_results/` & `poisson_solution/`**: Output directories for simulation results (PVD, VTU, PNG).

## Getting Started

### Prerequisites

- **Python 3.x**
- **FEniCS Stack**: It is recommended to use a Linux-based environment (or WSL/Docker) as FEniCS has better support there. The `Miniconda3-latest-Linux-x86_64.sh` script is available for environment setup.
- **Dependencies**:
  ```bash
  pip install google-genai typer requests
  ```
- **API Keys**: Ensure `GOOGLE_API_KEY` is set in your environment variables for Gemini access.

### Running the Orchestrators

- **Build/Debug FEniCS Prototype**:
  ```bash
  python orchestrator.py build-prototype --max-iterations 3
  ```
- **Run Literature Research Swarm**:
  ```bash
  python literature_orchestrator.py
  ```

## Development Conventions

- **Agentic Workflow**: Logic is often delegated to Gemini agents with specific system instructions (e.g., Documentalist, Evaluator, Synthesizer).
- **Prototyping**: New simulation scripts should be generated or tested within the `sandbox/` directory.
- **Mathematical Integrity**: Preference for differentiable models (e.g., PINNs, analytical solutions like Rosenthal) to support gradient-based optimization in MDAO frameworks like GEMSEO.

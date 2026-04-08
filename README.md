# Causality_NSA

Code and scripts to reproduce the core results and figures from the paper
"The causal structure of galactic astrophysics" (Desmond and Ramsey, 2026; <code>https://arxiv.org/abs/2510.01112</code>),
using Nasa Sloan Atlas (NSA) data and the Fast Causal Inference with Targeted Testing (FCIT) causal discovery algorithm.

## Configuration

This repository comes configured to match the paper setup:
<ul>
	<li>FCIT <code>p-value threshold = 0.01</code></li>
	<li>FCIT <code>truncation_limit = 14</code></li>
	<li>FCIT <code>penalty_discount = 50</code> (real-data FCIT run)</li>
	<li>mock data generation: sweep of 30 <code>penalty_discount</code> values (1-200) at <code>truncation_limit = 14</code></li>
</ul>

## Dependencies

Python packages:
<ul>
	<li><code>numpy</code></li>
	<li><code>pandas</code></li>
	<li><code>matplotlib</code></li>
	<li><code>astropy</code></li>
	<li><code>corner</code></li>
	<li><code>seaborn</code></li>
	<li><code>graphviz</code> (Python package and Graphviz executable)</li>
	<li><code>torch</code></li>
	<li><code>mpi4py</code></li>
	<li><code>causaldag</code></li>
	<li><code>pytetrad</code> (see version pinning note below)</li>
	<li><code>JPype1</code> (required by py-tetrad bridge)</li>
</ul>

### py-tetrad version pinning

The results in this paper (in particular the PAG of Fig. 4) were produced with
<code>py-tetrad</code> at commit
<a href="https://github.com/cmu-phil/py-tetrad/commit/097783348ce46d97efeb4c2e4ad8ae506c95806d"><code>0977833</code></a>
(March 9, 2026). Later versions ship a newer <code>tetrad-current.jar</code>
whose refactored BOSS/PermutationSearch and CPDAG-transformation logic can
produce a different PAG from the same data and hyperparameters.

To install the pinned version:

<pre>
pip install git+https://github.com/cmu-phil/py-tetrad@0977833
</pre>

System/runtime requirements:
<ul>
	<li>Java JDK 21+ (for <code>pytetrad</code>/Tetrad)</li>
	<li>MPI runtime (<code>mpiexec</code>)</li>
</ul>

Included module used by the pipeline: <code>cpn.py</code>

## Data Requirement

Required input file in working directory: <code>nsa_v1_0_1.fits</code>

Download source: SDSS DR17 NSA page: <code>https://www.sdss4.org/dr17/manga/manga-target-selection/nsa/</code>

## Reproducibility Chain

Run from repository root <code>Causality_NSA</code>.

### `make_example_plots.py`

Purpose: Generate the conceptual causal-discovery illustration.

Inputs: None

Outputs: <code>Plots_paper/example_plots.pdf</code> (paper fig. 2)

Command: <code>python3 make_example_plots.py</code>

### `prepare_nsa.py`

Purpose: Load and filter NSA data, apply paper transforms, write FCIT input table, generate corner plot.

Inputs: <code>nsa_v1_0_1.fits</code>

Outputs:
<ul>
	<li>Data: <code>nsa.pkl</code></li>
	<li>Plot: <code>Plots_paper/data_corner.pdf</code> (paper fig. 1)</li>
</ul>

Command: <code>python3 prepare_nsa.py</code>

### `learn_causality_nsa.py`

Purpose: Run FCIT on prepared real NSA data to generate PAG.

Inputs: <code>nsa.pkl</code>

Outputs: <code>Plots_paper/data.pdf</code> (paper fig. 4)

Command: <code>python3 learn_causality_nsa.py</code>

### `make_mocks.py`

Purpose: Generate NSA-like synthetic datasets from random DAGs using CPN.

Inputs: CLI argument: <code>&lt;num_mocks&gt;</code> (total number of mocks to generate)

Outputs:
<ul>
	<li>Data file per mock: <code>simulated_data/data_&lt;mock_id&gt;.npy</code></li>
	<li>Data header/text per mock: <code>simulated_data/data_&lt;mock_id&gt;.txt</code></li>
	<li>Ground-truth CPDAG dictionary list: <code>simulated_data/cpdag_dicts_by_rank.txt</code></li>
	<li>Diagnostic plot per mock: <code>Plots_paper/pairwise_&lt;mock_id&gt;.pdf</code></li>
	<li>DAG plot per mock: <code>Plots_paper/graph_&lt;mock_id&gt;.pdf</code></li>
	<li>PAG plot per mock: <code>Plots_paper/graph_pag_&lt;mock_id&gt;.pdf</code></li>
</ul>

Command example: <code>mpiexec -n 20 python3 make_mocks.py 200</code>

Any MPI process count can be used. Work is distributed by striding over mock IDs; choosing a factor of <code>num_mocks</code> is optional and only helps load balance.

Resource note: this stage is the heaviest step (large sample count per mock plus diagnostic plots) and can require substantial RAM, runtime, and disk.

### `analyse_mocks.py`

Purpose: Run FCIT on all available mock datasets and compute recovery precision, recall, and F1 score across <code>penalty_discount</code> values.

Inputs:
<ul>
	<li><code>simulated_data/data_&lt;mock_id&gt;.npy</code></li>
	<li><code>simulated_data/data_&lt;mock_id&gt;.txt</code></li>
	<li><code>simulated_data/cpdag_dicts_by_rank.txt</code></li>
</ul>

Outputs: Metrics table <code>penalty_results_trunc&lt;truncation_limit&gt;.txt</code> (default paper filename: <code>penalty_results_trunc14.txt</code>)

Default: <code>&#45;&#45;truncation_limit</code> is optional and defaults to <code>14</code> (paper setting).

Command example: <code>mpiexec -n 20 python3 analyse_mocks.py</code>

Any MPI process count can be used, as in <code>make_mocks.py</code>.

Important: <code>simulated_data/cpdag_dicts_by_rank.txt</code> must contain one valid dictionary per line in mock-id order. If analysis prints malformed-line warnings, regenerate mocks before trusting metrics.

### `plot_mock_results.py`

Purpose: Convert the mock metrics table into the penalty-vs-performance plot.

Inputs: <code>penalty_results_trunc&lt;truncation_limit&gt;.txt</code>

Outputs: <code>Plots_paper/mocks.pdf</code> when <code>truncation_limit=14</code> (paper fig. 3), otherwise <code>Plots_paper/mocks_trunc&lt;truncation_limit&gt;.pdf</code>

Default: <code>&#45;&#45;truncation_limit</code> is optional and defaults to <code>14</code> (paper setting).

Command: <code>python3 plot_mock_results.py</code>


## `Contact`

If you have questions or comments, email Harry Desmond (harry.desmond@port.ac.uk)

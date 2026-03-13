"""Generate NSA-like mock datasets and diagnostic plots for the paper setup."""

from mpi4py import MPI
import os
import time as tm
import numpy as np
import torch.nn as nn
import graphviz as gviz
import sys
import warnings
import seaborn as sns
from pathlib import Path

try:
    import edu.cmu.tetrad.graph as tg
except Exception as exc:
    raise SystemExit(
        "Failed to import Tetrad Java graph bindings (edu.cmu.tetrad.graph). "
        "Check Java/JPype/py-tetrad environment setup."
    ) from exc

try:
    import cpn as cpn
except Exception as exc:
    raise SystemExit("Failed to import local module 'cpn.py'.") from exc

try:
    import causaldag as cd
except Exception as exc:
    raise SystemExit("Failed to import 'causaldag'.") from exc

warnings.filterwarnings("ignore")

# MPI setup
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if len(sys.argv) != 2:
    raise SystemExit("Usage: python make_mocks.py <num_mocks>")

try:
    num_mocks = int(sys.argv[1])
except ValueError as exc:
    raise SystemExit("num_mocks must be an integer") from exc

if num_mocks <= 0:
    raise SystemExit("num_mocks must be > 0")


def to_py_dict(d):
    return {str(k): [str(vv) for vv in v] for k, v in d.items()}


def construct_dot(graph):
    dot_str = "digraph G {\n"
    for node in graph.getNodes():
        dot_str += f'  {node.getName()};\n'
    for edge in graph.getEdges():
        node1 = edge.getNode1().getName()
        node2 = edge.getNode2().getName()
        endpoint1 = edge.getEndpoint1()
        endpoint2 = edge.getEndpoint2()
        if endpoint1 == tg.Endpoint.TAIL and endpoint2 == tg.Endpoint.ARROW:
            dot_str += f'  {node1} -> {node2};\n'
        elif endpoint1 == tg.Endpoint.ARROW and endpoint2 == tg.Endpoint.TAIL:
            dot_str += f'  {node2} -> {node1};\n'
        elif endpoint1 == tg.Endpoint.CIRCLE or endpoint2 == tg.Endpoint.CIRCLE:
            dot_str += f'  {node1} -> {node2} [dir=both, arrowtail=odot, arrowhead=odot];\n'
        elif endpoint1 == tg.Endpoint.TAIL and endpoint2 == tg.Endpoint.CIRCLE:
            dot_str += f'  {node1} -> {node2} [arrowhead=odot];\n'
        elif endpoint1 == tg.Endpoint.CIRCLE and endpoint2 == tg.Endpoint.TAIL:
            dot_str += f'  {node2} -> {node1} [arrowhead=odot];\n'
        elif endpoint1 == tg.Endpoint.ARROW and endpoint2 == tg.Endpoint.CIRCLE:
            dot_str += f'  {node1} -> {node2} [arrowhead=odot];\n'
        elif endpoint1 == tg.Endpoint.CIRCLE and endpoint2 == tg.Endpoint.ARROW:
            dot_str += f'  {node2} -> {node1} [arrowhead=odot];\n'
        elif endpoint1 == tg.Endpoint.ARROW and endpoint2 == tg.Endpoint.ARROW:
            dot_str += f'  {node1} -> {node2} [dir=both];\n'
        elif endpoint1 == tg.Endpoint.TAIL and endpoint2 == tg.Endpoint.TAIL:
            dot_str += f'  {node1} -- {node2};\n'
    dot_str += "}"
    return dot_str


def generate_one_mock(mock_id: int, plots_dir: Path):
    seed = mock_id
    np.random.seed(seed)
    seed_ = seed
    num_edges = np.random.randint(12, 17)  # Random number of edges between 12 and 16 (inclusive)

    # Generate a random DAG
    true_dag = tg.RandomGraph.randomGraph(num_nodes, 0, num_edges, 100, 100, 100, False, seed_)
    print(f"RANK {rank}: MOCK {mock_id}: TRUE DAG")
    print(true_dag)

    true_pag = tg.GraphTransforms.dagToPag(true_dag)

    # Define the DAG by specifying its directed edges
    edges = [(edge.getNode1().getName(), edge.getNode2().getName()) for edge in true_dag.getEdges()]
    dag = cd.DAG(arcs=edges)

    # Convert the DAG to its CPDAG
    cpdag = dag.cpdag()

    # Create dictionary for CPDAG edges
    cpdag_dict = {node: [] for node in cpdag.nodes}
    for parent, child in cpdag.arcs:
        cpdag_dict[parent].append(child)
    for node1, node2 in cpdag.edges:
        cpdag_dict[node1].append(node2)
        cpdag_dict[node2].append(node1)

    # Set up noise distributions for each node
    noise_distributions = {}
    for node in true_dag.getNodes():
        noise_distributions[node] = cpn.NoiseDistribution(distribution_type="beta", alpha=2, beta=5)

    cpn_net = cpn.CausalPerceptronNetwork(
        graph=true_dag,
        num_samples=num_samples,
        noise_distributions=noise_distributions,
        hidden_dimensions=[50, 50, 50, 50],
        input_scale=input_scale,
        activation_module=nn.ReLU(),
        nonlinearity='relu',
        discrete_prob=0,
        seed=seed
    )

    start = tm.time()
    df = cpn_net.generate_data()
    shuffled_columns = np.random.permutation(df.columns)
    df = df[shuffled_columns]

    # Keep data-generation behavior unchanged; gracefully skip only the heavy diagnostic plot if it fails.
    try:
        g = sns.pairplot(df, kind='hist', diag_kind="hist", corner=True)
        g.figure.set_size_inches(8, 8)
        g.figure.savefig(plots_dir / f"pairwise_{mock_id}.pdf", dpi=300, bbox_inches="tight")
    except Exception as exc:
        print(
            f"RANK {rank}: MOCK {mock_id}: pairplot failed ({exc}); continuing without pairplot",
            flush=True,
        )

    gdot = construct_dot(true_dag)
    graph = gviz.Source(gdot)
    graph.render(filename=str(plots_dir / f"graph_{mock_id}"), format="pdf", cleanup=True)

    gdot = construct_dot(true_pag)
    graph = gviz.Source(gdot)
    graph.render(filename=str(plots_dir / f"graph_pag_{mock_id}"), format="pdf", cleanup=True)

    df.to_csv(f"simulated_data/data_{mock_id}.txt", sep="\t", index=False)
    np.save(f"simulated_data/data_{mock_id}", df.to_numpy())
    print(f"RANK {rank}: MOCK {mock_id}: completed in {tm.time() - start:.1f}s", flush=True)
    return mock_id, to_py_dict(cpdag_dict)

num_samples = 445763
num_nodes = 7
input_scale = 1

# Create output directories on every rank to avoid race conditions.
os.makedirs("simulated_data", exist_ok=True)

plots_dir = Path("Plots_paper")
plots_dir.mkdir(parents=True, exist_ok=True)

mock_ids = list(range(num_mocks))
my_mock_ids = mock_ids[rank::size]
print(f"RANK {rank}: assigned {len(my_mock_ids)} mocks out of {num_mocks}", flush=True)

my_dicts = []
for mock_id in my_mock_ids:
    my_dicts.append(generate_one_mock(mock_id, plots_dir))

# Write per-rank CPDAG dictionaries to disk to avoid large gather payloads on rank 0.
tmp_dir = Path("simulated_data") / ".tmp_cpdag_dicts"
tmp_dir.mkdir(parents=True, exist_ok=True)
rank_tmp_path = tmp_dir / f"rank_{rank}.txt"
with open(rank_tmp_path, "w") as f:
    for mock_id, d in my_dicts:
        f.write(f"{mock_id}\t{d}\n")

comm.Barrier()

if rank == 0:
    flat = []
    for r in range(size):
        p = tmp_dir / f"rank_{r}.txt"
        if not p.exists():
            continue
        with open(p, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                mock_id_str, d_str = line.split("\t", 1)
                flat.append((int(mock_id_str), d_str))

    flat.sort(key=lambda x: x[0])
    out_path = "simulated_data/cpdag_dicts_by_rank.txt"
    with open(out_path, "w") as f:
        for _, d_str in flat:
            f.write(f"{d_str}\n")
    print(f"Wrote {len(flat)} CPDAG dictionaries to {out_path}", flush=True)

    # Best-effort cleanup of temporary shard files.
    for r in range(size):
        p = tmp_dir / f"rank_{r}.txt"
        try:
            if p.exists():
                p.unlink()
        except OSError:
            pass
    try:
        tmp_dir.rmdir()
    except OSError:
        pass

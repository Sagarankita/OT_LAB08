"""Setup-only smoke checks; not assignment benchmark results."""
import json, os, platform, shutil, sqlite3, sys
from pathlib import Path
os.environ.setdefault("MPLCONFIGDIR", str(Path(".mplconfig").resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(".cache").resolve()))
import numpy as np
import scipy
from scipy.optimize import milp, Bounds
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest
import torch
import pennylane as qml
import coverage

out = Path("results/setup")
out.mkdir(parents=True, exist_ok=True)
r = milp(c=[-1.0], integrality=[1], bounds=Bounds([0], [1]))
assert r.success and abs(r.x[0]-1) < 1e-8
x = torch.tensor(2.0, requires_grad=True)
(x*x).backward()
assert x.grad.item() == 4.0
@qml.qnode(qml.device("default.qubit", wires=1))
def circuit():
    qml.Hadamard(wires=0)
    return qml.probs(wires=0)
assert np.allclose(circuit(), [.5,.5])
with sqlite3.connect(":memory:") as db:
    assert db.execute("select 2 + 2").fetchone()[0] == 4
plt.plot([0,1],[0,1])
plt.savefig(out / "smoke.png")
plt.close()
record = {"purpose":"environment smoke only", "python":sys.version,
"platform":platform.platform(), "architecture":platform.machine(), "logical_cpus":os.cpu_count(),
"versions":{m.__name__:m.__version__ for m in [np,scipy,matplotlib,pytest,torch,qml,coverage]},
"sqlite":sqlite3.sqlite_version,
"tools":{t:shutil.which(t) for t in ["git","clang","make","qemu-system-riscv64","riscv64-unknown-elf-gcc","riscv64-linux-gnu-gcc"]},
"checks":{"milp":"pass","torch_cpu_autograd":"pass","quantum_simulator":"pass","sqlite":"pass","plot":"pass"}}
(out / "environment.json").write_text(json.dumps(record,indent=2)+"\n")
print(json.dumps(record,indent=2))

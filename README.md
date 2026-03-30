# SE4SC-LLM

**SE4SC-LLM: An LLM-augmented Symbolic Execution Framework for Smart Contracts**

Smart contracts are immutable Blockchain programs, where a single vulnerability can lead to significant financial losses. Symbolic execution is a cornerstone technique for detecting such flaws by systematically exploring execution paths. However, it suffers from path explosion problem, where the exponential growth of candidate states limits Control Flow Graph (CFG) coverage. This limitation results in overlooking critical paths with potential security flaws, reducing vulnerability detection accuracy.

To address this, we propose SE4SC-LLM, an LLM-augmented symbolic execution framework for smart contracts. It operates in a learning-inference paradigm, anchoring state selection on guidance from a Large Language Model (LLM). In SE4SC-LLM, LLM directly provides the dominant control-flow semantics by encoding bytecode sequences, introducing a feature modality beyond numeric runtime statistics. A coverage-driven attention mechanism fuses the semantic embeddings with numeric features, and a regression model trained on the fused representation iteratively learns a high-reward state selection strategy. This strategy guides symbolic execution toward more effective path exploration.

Evaluated on two public datasets, SE4SC-LLM achieves 95.1% average CFG coverage, a 6.5 percentage point improvement over the strongest baseline, and detects 11.2% more vulnerabilities, particularly in reentrancy and unchecked calls.

## Requirements

- Python 3.8.9
- Solcx 0.4.24
- Pyevmasm 0.2.3
- Transformers 4.52.4
- Scikit-learn 1.0.2
- PyTorch 2.5.1
- SciPy 1.12.0

## Usage

```bash
cd scripts
python machine_learning_for_se.py
```

## License

This project is open-sourced for academic research purposes.

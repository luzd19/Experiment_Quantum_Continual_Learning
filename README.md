# Experiment_Quantum_Continual_Learning

We report an experimental demonstration of quantum continual learning on a fully programmable superconducting processor. In particular, we sequentially train a quantum classifier with three tasks, two about identifying real-life images and the other on classifying quantum states, and demonstrate its catastrophic forgetting through experimentally observed rapid performance drops for prior tasks. To overcome this dilemma, we exploit the elastic weight consolidation strategy and show that the quantum classifier can incrementally learn and retain knowledge across the three distinct tasks.

Here, we provide the codes for numerical simulations, data for experimental results and numerical results.

## Contents

- [Numerical Simulations](Numerical_Simulations)
- [Experimental Results](Experimental_Results)

## The numerical simulations are built With

* [Yao Quantum](https://yaoquantum.org/) - An open-source quantum simulation framework in Julia language

Detailed installation instructions and tutorials of Julia and Yao.jl can be found at [julialang.org](https://julialang.org/) and [yaoquantum.org/documentation](https://docs.yaoquantum.org/dev/).


## License

Released under [MIT License](https://github.com/luzd19/Experiment_Quantum_Continual_Learning/blob/main/LICENSE) 

# Energy-Accuracy-Security Trade-offs in Resistive In-Memory Computing Architectures
This repository provides the code accompanying our paper, "Energy-Accuracy-Security Trade-offs in Resistive In-Memory Computing Architectures", authored by Saion K. Roy and Naresh R. Shanbhag, submitted IEEE TCAD 2025. 

## About
The security vulnerabilities of embedded non-volatile memory (eNVM)-based in-memory computing (IMC) architectures remain overlooked today. Although their inherently low computational accuracy might suggest greater resilience to model extraction attacks (MEAs), our findings indicate otherwise. We experimentally evaluate the fundamental trade-off between energy, accuracy, and security in eNVM-IMCs, with measured data from a 22nm MRAM-based IMC prototype. Our experiments reveal that operating in low signal-to-noise-and-distortion ratio (SNDR) regimes provides increased resistance to MEAs compared to high-SNDR regimes. Building on this, we employ a chip-validated behavioral model to probe the attack surface targeted for model extraction in case of MRAM, ReRAM, and FeFET-based IMCs. We show that devices with higher conductance contrast are more susceptible to MEAs. Interestingly, the MEA-resilient low-SNDR regimes are also more energy-efficient, but they necessitate accuracy-enhancing techniques such as statistical error compensation (SEC) and noise-aware training (NAT) for achieving high natural accuracy. These observations highlight the need for energy-efficient defenses that simultaneously achieve high inference accuracy and robustness against model extraction.

## Environment
The following Python 3 packages are required to run the program
* numpy
* matplotlib
* math

## Acknowledgements
This research was supported by SRC and DARPA funded JUMP 2.0 centers, COCOSYS and CUBIC.

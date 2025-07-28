# Comprehending the Energy-Accuracy-Security Trade-offs in Embedded Non-Volatile In-Memory Computing Architectures
This repository provides the code accompanying our paper, "Comprehending the Energy-Accuracy-Security Trade-offs in Embedded Non-Volatile In-Memory Computing Architectures", authored by Saion K. Roy and Naresh R. Shanbhag, submitted to IEEE TCAD 2025. 

## About
The security vulnerabilities of embedded non-volatile memory (eNVM)-based in-memory computing (IMC) architectures remain overlooked today. This paper proposes a statistical framework to construct model extraction attacks (MEAs) for eNVM-based IMCs and employs it to construct three different MEAs - basis vector (BV), least squares (LS), and stochastic gradient descent (SGD) attacks. Efficacy of these attacks in model parameter retrieval is demonstrated experimentally by applying them to a 22nm integrated circuit (IC) prototype of an MRAM-based IMC to quantify the fundamental trade-off between energy, accuracy, and security (EAS) vulnerability. This study indicates that the strongest MEA is able to retrieve weights with sufficient fidelity to achieve an inference accuracy within 0.1% of a fixed-point baseline digital architecture, in spite of a bank-level signal-to-noise-and-distortion ratio (SNDR) as low as 4.18dB. Resilience to MEAs is observed to increase with a reduction in SNDR accompanied by a reduction in energy. We extend these observations to study the EAS trade-off for MRAM, ReRAM, and FeFET-based IMCs by employing a silicon-validated behavioral model of resistive IMCs to probe the attack surface targeted by MEAs. We find that technologies with higher conductance contrast, e.g., ReRAM and FeFET, are more susceptible to MEAs. Finally, we note the potential for algorithmic methods, such as statistical error compensation (SEC) and noise-aware training (NAT), to simultaneously achieve high accuracy, low energy, and resilience to MEAs.

## Environment
The following Python 3 packages are required to run the program
* numpy
* matplotlib
* math

## Acknowledgements
This research was supported by SRC and DARPA funded JUMP 2.0 centers, COCOSYS and CUBIC.

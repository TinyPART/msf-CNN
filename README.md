# msf-CNN: Patch-based Multi-Stage Fusion with Convolutional Neural Networks for TinyML

For more information on this work, please read our preprint on [alphaXiv](https://www.alphaxiv.org/abs/2505.11483) or [arXiv](https://www.arxiv.org/abs/2505.11483).

## Reproduce analytical results

1. install python packages using requirement.txt
2. Run `python analysis_optimization.py`. Please check the following parameters:

```bash
usage: analysis_optimization.py [-h] [-m M] [-p P] [-c1 C1] [-c2 C2] [--log LOG]

options:
  -h, --help  show this help message and exit
  -m M        Model: MBV2-w0.35; MN2-vww5; MN2-320K
  -p P        Problem: [P1, P2, vanilla, heuristic]
  -c1 C1      specify the constraint set for the optimization problem P1: F_max. format: n1, n2...n_N
  -c2 C2      specify the constraint set for the optimization problem P2: P_max in kB. format: n1, n2...n_N
  --log LOG   Log Level
```

## Reproduce Experimental Results on MCU

The experiments were conducted via [RIOT-ML](https://github.com/TinyPART/RIOT-ML/). To reproduce the results it is required to get RIOT-ML ready.

0. Go get a IoT board e.g. [STM32 nucleo-f767zi](https://www.st.com/en/evaluation-tools/nucleo-f767zi.html). Connect the IoT board to your PC.
1. Clone the RIOT-ML from https://github.com/TinyPART/RIOT-ML/.
2. Follow the [Prequisites](https://github.com/TinyPART/RIOT-ML/tree/main?tab=readme-ov-file#prequisites) section, install all necessary packages and toolchains.
3. Copy `msf_cnn_eval_hardware.py`, `main.c` and `msf_CNN_eval_models` to the RIOT-ML directory.
4. Run `python msf_cnn_eval_hardware.py` under the RIOT-ML directory.
5. Grab some coffee and wait for the results written in `msf_cnn_HIL_eval_result_{board}.json` :).

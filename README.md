# [MAgICoRe: A Multi-Agent Coarse-to-Fine Refinement Framework for Reasoning](https://arxiv.org/abs/2409.12147)

[Justin Chih-Yao Chen](https://dinobby.github.io/), [Archiki Prasad](https://archiki.github.io/), [Swarnadeep Saha](https://swarnahub.github.io/), [Elias Stengel-Eskin](https://esteng.github.io/), and [Mohit Bansal](https://www.cs.unc.edu/~mbansal/)

<img width="1043" alt="image" src="https://i.imgur.com/hxX0XEo.png">

# Installation
This repository is tested on Python 3.10.12. All dependencies can be installed as follows:

```
pip install -r requirements.txt
```

# Run Experiment
Step 1: use the following script to generate $k$ reasoning chains for each problem.
```
CUDA_VISIBLE_DEVICES=0,1 python generate.py --k 40
```

Step 2: use the following script to get RM scores for the first iteration.
```
CUDA_VISIBLE_DEVICES=0,1 python annotate.py --k 40
```

Step 3: use the following script to refine hard instances.
```
CUDA_VISIBLE_DEVICES=0,1 python refine.py --k 40
```

`k` corresponds to the number of generations for each problem.

# Citation
```
@article{chen2024magicore,
  title={MAgICoRe: A Multi-Agent Coarse-to-Fine Refinement Framework for Reasoning},
  author={Chen, Justin Chih-Yao and Prasad, Archiki and Saha, Swarnadeep and Stengel-Eskin, Elias and Bansal, Mohit},
  journal={arXiv preprint arXiv:2409.12147},
  year={2024}
}
```

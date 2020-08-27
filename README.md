# Intro
Code accompanying graduation thesis "Automated Lane Change Strategy in Highway Environment Based on Deep Reinforcement Learning".
Discrete action space version of published paper on IEEE IV 2020, [link](https://arxiv.org/pdf/2002.02667.pdf).

# Dependencies
- tensorflow-1.14
- OpenAI baselines
- SUMO-1.6
- mpi
- mpi4py

# Usage
- to start a training:
```shell script
cd ppo_new
mpirun -n 8 python run.py --log trial2 --cont trial1
```
This will continue from ``trial1`` and save tensorboard data and model files
to ``trial2`` using 8 threads for parallelization.
- to see latest training result:
```shell script
python restore.py
```
- to evaluate current policy:
```shell script
python evaluate.py
```

# Cite
```
@article{ye2020automated,
  title={Automated Lane Change Strategy using Proximal Policy Optimization-based Deep Reinforcement Learning},
  author={Ye, Fei and Cheng, Xuxin and Wang, Pin and Chan, Ching-Yao},
  journal={arXiv preprint arXiv:2002.02667},
  year={2020}
}
```


#!/bin/bash

# # afk
# python main.py --T 0.1 --cuda 0 --bs 256 --data afk --n_worker 16 --lr 1e-4 --l2 5e0 --seed 1234

# # abe
python main.py --T 0.1 --cuda 0 --bs 256 --data abe --n_worker 16 --lr 1e-4 --l2 5e0 --seed 1234

# # amb
# python main.py --T 0.1 --cuda 0 --bs 256 --data amb --n_worker 8 --lr 1e-4 --l2 5e0 --seed 1234

# # T = [0, 0.1, 0.5, 1, 2, 5, 10, 1e-9] 0 means full cross (Cross variant), 1e-9 means no cross (Self variant)
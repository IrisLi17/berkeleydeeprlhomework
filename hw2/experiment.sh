#!/usr/bin/env bash
conda activate py35
python train_pg.py InvertedPendulum-v2 -n 100 -b 1000 -e 1 -rtg -s 64 --seed 1 --exp_name sb_rtg_na_s64
python train_pg.py InvertedPendulum-v2 -n 100 -b 1000 -e 1 -rtg -s 64 --seed 11 --exp_name sb_rtg_na_s64
python train_pg.py InvertedPendulum-v2 -n 100 -b 1000 -e 1 -rtg -s 64 --seed 21 --exp_name sb_rtg_na_s64
python train_pg.py InvertedPendulum-v2 -n 100 -b 1000 -e 1 -rtg -s 64 --seed 31 --exp_name sb_rtg_na_s64
python train_pg.py InvertedPendulum-v2 -n 100 -b 1000 -e 1 -rtg -s 64 --seed 41 --exp_name sb_rtg_na_s64

python train_pg.py InvertedPendulum-v2 -n 100 -b 1000 -e 1 -rtg --seed 1 --exp_name sb_rtg_na
python train_pg.py InvertedPendulum-v2 -n 100 -b 1000 -e 1 -rtg --seed 11 --exp_name sb_rtg_na
python train_pg.py InvertedPendulum-v2 -n 100 -b 1000 -e 1 -rtg --seed 21 --exp_name sb_rtg_na
python train_pg.py InvertedPendulum-v2 -n 100 -b 1000 -e 1 -rtg --seed 31 --exp_name sb_rtg_na
python train_pg.py InvertedPendulum-v2 -n 100 -b 1000 -e 1 -rtg --seed 41 --exp_name sb_rtg_na

python train_pg.py InvertedPendulum-v2 -n 100 -b 1000 -e 1 -rtg -lr 0.01 --seed 1 --exp_name sb_rtg_na_lr0.01
python train_pg.py InvertedPendulum-v2 -n 100 -b 1000 -e 1 -rtg -lr 0.01 --seed 11 --exp_name sb_rtg_na_lr0.01
python train_pg.py InvertedPendulum-v2 -n 100 -b 1000 -e 1 -rtg -lr 0.01 --seed 21 --exp_name sb_rtg_na_lr0.01
python train_pg.py InvertedPendulum-v2 -n 100 -b 1000 -e 1 -rtg -lr 0.01 --seed 31 --exp_name sb_rtg_na_lr0.01
python train_pg.py InvertedPendulum-v2 -n 100 -b 1000 -e 1 -rtg -lr 0.01 --seed 41 --exp_name sb_rtg_na_lr0.01

python train_pg.py Walker2d-v2 -n 100 -b 1000 -e 1 -rtg -bl -gae --seed 1 --exp_name sb_rtg_na_bl_gae
python train_pg.py Walker2d-v2 -n 100 -b 1000 -e 1 -rtg -bl -gae --seed 11 --exp_name sb_rtg_na_bl_gae
python train_pg.py Walker2d-v2 -n 100 -b 1000 -e 1 -rtg -bl -gae --seed 21 --exp_name sb_rtg_na_bl_gae
python train_pg.py Walker2d-v2 -n 100 -b 1000 -e 1 -rtg -bl --seed 1 --exp_name sb_rtg_na_bl
python train_pg.py Walker2d-v2 -n 100 -b 1000 -e 1 -rtg -bl --seed 11 --exp_name sb_rtg_na_bl
python train_pg.py Walker2d-v2 -n 100 -b 1000 -e 1 -rtg -bl --seed 21 --exp_name sb_rtg_na_bl

python train_pg.py Walker2d-v2 -n 200 -b 1000 -e 1 -rtg -bl -gae --seed 1 --exp_name sb_rtg_na_bl_gae
python train_pg.py Walker2d-v2 -n 200 -b 1000 -e 1 -rtg -bl -gae --seed 11 --exp_name sb_rtg_na_bl_gae
python train_pg.py Walker2d-v2 -n 200 -b 1000 -e 1 -rtg -bl -gae --seed 21 --exp_name sb_rtg_na_bl_gae

python train_pg.py InvertedPendulum-v2 -n 100 -b 1000 -e 1 -rtg -s 64 -lr 0.01 --seed 1 --exp_name sb_rtg_na_lr0.01_s64
python train_pg.py InvertedPendulum-v2 -n 100 -b 1000 -e 1 -rtg -s 64 -lr 0.01 --seed 11 --exp_name sb_rtg_na_lr0.01_s64
python train_pg.py InvertedPendulum-v2 -n 100 -b 1000 -e 1 -rtg -s 64 -lr 0.01 --seed 21 --exp_name sb_rtg_na_lr0.01_s64
python train_pg.py InvertedPendulum-v2 -n 100 -b 1000 -e 1 -rtg -s 64 -lr 0.01 --seed 31 --exp_name sb_rtg_na_lr0.01_s64
python train_pg.py InvertedPendulum-v2 -n 100 -b 1000 -e 1 -rtg -s 64 -lr 0.01 --seed 41 --exp_name sb_rtg_na_lr0.01_s64

python train_pg.py InvertedPendulum-v2 -n 100 -b 5000 -e 1 -rtg -lr 0.01 --seed 1 --exp_name lb_rtg_na_lr0.01
python train_pg.py InvertedPendulum-v2 -n 100 -b 5000 -e 1 -rtg -lr 0.01 --seed 11 --exp_name lb_rtg_na_lr0.01
python train_pg.py InvertedPendulum-v2 -n 100 -b 5000 -e 1 -rtg -lr 0.01 --seed 21 --exp_name lb_rtg_na_lr0.01
python train_pg.py InvertedPendulum-v2 -n 100 -b 5000 -e 1 -rtg -lr 0.01 --seed 31 --exp_name lb_rtg_na_lr0.01
python train_pg.py InvertedPendulum-v2 -n 100 -b 5000 -e 1 -rtg -lr 0.01 --seed 41 --exp_name lb_rtg_na_lr0.01

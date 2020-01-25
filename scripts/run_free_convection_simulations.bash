#!/bin/bash

for dTdz in 0.01 0.05 0.005; do
    for Q in -10 -50 -100; do
        julia --project deepening_mixed_layer.jl -N 256 -V 256 -L 100 -H 100 --dTdz $dTdz -Q $Q --wind-stress 0 --days 8 -d ~/data/
    done
done


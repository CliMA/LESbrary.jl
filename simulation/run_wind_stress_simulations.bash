#!/bin/bash

for tau in 0 0.04 0.1; do
    for dTdz in 0.01 0.001; do
        for Q in 10 0 -75; do
            julia --project deepening_mixed_layer.jl -N 256 -V 256 -L 100 -H 100 -Q $Q --dTdz $dTdz --wind-stress $tau --days 8 -d ~/data/
        done
    done
done


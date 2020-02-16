#!/usr/bin/env bash

inp=$1

grep oracle $inp | \
    awk -F'[\t]' '{print $6}' | \
    awk -F'[ ]' '{print $2}' | \
    awk '{sum += $1} END {print "mean = " sum/NR}'

#!/bin/bash

for data in train dev test
do
    python read_data.py --json="kvret_"$data"_public.json" > $data".txt"
done


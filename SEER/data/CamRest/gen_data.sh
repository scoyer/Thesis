#!/bin/bash

python get_entity_list.py

for data in train dev test
do
    python read_data.py --json="CamRest676_"$data".json" > $data".txt"
done


#!/bin/bash
for data in graph_1.txt graph_2.txt graph_3.txt graph_4.txt graph_5.txt graph_6.txt ibm-5000.txt; do
    python -B hits.py --data $data
    python -B pagerank.py --data $data
    python -B simrank.py --data $data
done
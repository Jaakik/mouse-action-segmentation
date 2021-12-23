#!/bin/bash
n=$1
rm -r data/testFeat*
rm -r data/task_*/*$n/
rm data/task_*/splits/*
rm -r data/task_*/mapping.txt
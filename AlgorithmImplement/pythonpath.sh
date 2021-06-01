#!/bin/bash
export PYTHONPATH=$PWD/AlgorithmImplement
matlab -nodesktop -nosplash -r "cd TestCode;py.sys.setdlopenflags(int32(10));Main;quit"
#!/bin/sh

# Utiliser python
export PYTHON_VERSION="3.9.8"
export PYTHON_MAINVERSION="3.9"
export PYTHON_ROOT=/home/eisti/Irma_Project/python_distrib/Python-${PYTHON_VERSION}
export PYTHON_HOME=${PYTHON_ROOT}/install/Python-${PYTHON_VERSION}
export PATH=${PYTHON_HOME}"/bin:"${PATH}
export LD_LIBRARY_PATH=${PYTHON_HOME}"/lib:"${LD_LIBRARY_PATH}
export PYTHONPATH=${PYTHON_HOME}"/lib/python${PYTHON_MAINVERSION}/site-packages/"

echo -e "\033[31mPython3: $(which python3) \033[0m"
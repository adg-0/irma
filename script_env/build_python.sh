#!/bin/sh

mkdir -p ${PYTHON_ROOT}/build/
cd ${PYTHON_ROOT}/build/

\rm -rf "Python-${PYTHON_VERSION}"
tar -xzvf /home/eisti/Irma_Project/python_distrib/download/Python-${PYTHON_VERSION}.tgz
cd "Python-${PYTHON_VERSION}"

INSTALL_DIR=${PYTHON_ROOT}/install/Python-${PYTHON_VERSION}/
\rm -rf ${INSTALL_DIR}
mkdir -p ${INSTALL_DIR}

./configure --enable-shared --with-ensurepip --prefix=${INSTALL_DIR} --disable-ipv6 --enable-unicode=ucs4
make
make install
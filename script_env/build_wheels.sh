#!/bin/sh

######
# if it doesnt work, check if
#   import setuptools
#   from distutils.core import setup
# are in the setup.py
######

for filename in ./*.tar.gz; do
	#Name of dir after extraction, sufixe and prefixe are deleted
	dirname=${filename%.tar.gz}
	dirname=${dirname##*/}
	echo $dirname
	#Extraction and building wheel
	tar xvfz $filename
	cd $dirname
	python setup.py bdist_wheel
	mv dist/* ../
	cd ..
	rm -r $dirname
done

#todo refactor

for filename in ./*.zip; do
	#Name of dir after extraction, sufixe and prefixe are deleted
	dirname=${filename%.zip}
	dirname=${dirname##*/}
	echo $dirname
	#Extraction and building wheel
	unzip $filename
	cd $dirname
	python setup.py bdist_wheel
	mv dist/* ../
	cd ..
	rm -r $dirname
done

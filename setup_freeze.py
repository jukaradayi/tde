#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy

from cx_Freeze import setup, Executable
from distutils.extension import Extension

readme = open('README.rst').read()
history = open('HISTORY.rst').read().replace('.. :changelog:', '')

requirements = [
    # TODO: put package requirements here
]

test_requirements = [
    # TODO: put package test requirements here
]

np_inc = numpy.get_include()

extensions = [Extension('tde.substrings.ccss', sources=['tde/substrings/ccss.pyx'],
                        extra_compile_args=['-shared', '-pthread', '-fPIC',
                                            '-fwrapv', '-O3', '-Wall',
                                            '-fno-strict-aliasing'],
                        include_dirs=['/usr/include/python2.7', np_inc]),
              Extension('tde.substrings.levenshtein', sources=['tde/substrings/levenshtein'],
                        extra_compile_args=['-shared', '-pthread', '-fPIC',
                                            '-fwrapv', '-O3', '-Wall',
                                            '-fno-strict-aliasing'],
                        include_dirs=['/usr/include/python2.7', np_inc])]


build_exe_options = dict(
    init_script='Console',
    includes=['tde.substrings.ccss', 'tde.substrings.levenshtein']
)

setup(
    name='tde',
    version='0.1.2',
    description='DESCRIPTION',
    long_description=readme + '\n\n' + history,
    author='Maarten Versteegh',
    author_email='maartenversteegh@gmail.com',
    url='https://github.com/mwv/tde',
    packages=[
        'tde',
    ],
    package_dir={'tde':
                 'tde'},
    license="GPLv3",
    keywords='tde',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GPLv3 License',
        'Natural Language :: English',
        "Programming Language :: Python :: 2",
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
    ],
    ext_modules=extensions,
    include_dirs=[numpy.get_include(),
                  os.path.join(numpy.get_include(), 'numpy')],
    executables = [Executable('bin/sample_eval2.py'),
                   Executable('bin/english_eval2.py'),
                   Executable('bin/xitsonga_eval2.py')],
    options={'build_exe': build_exe_options},
)

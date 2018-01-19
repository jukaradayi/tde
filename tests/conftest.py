# -*- coding: utf-8 -*-
"""Defin input data and fixtures common to all tests"""

import os
import pytest
import tde
print(tde.__file__)
import tde.util
from tde.util.reader import tokenlists_to_corpus, read_annotation, annotate_classes, read_classfile, read_split_single
from tde.eval_track2 import load_disc ### TODO MOVE LOAD DISC TO UTIL
_word = '''s0101a 32.217 32.554 okay
s0101a 50.898 51.178 raised
s2002b 226.749 227.126 really
s2002b 334.599 334.760 it
s2002b 334.760 334.969 i
s2002b 334.969 335.279 know
s2002b 335.279 335.409 it
s2002b 335.440 335.649 but
s2002b 335.690 336.029 yknow
s2002b 336.029 336.287 there
s2002b 336.955 337.276 um
s2301b 126.590 126.680 and
s2301b 126.680 126.749 to
s2301b 126.749 127.126 really
'''

_phone = '''s0101a 32.217 32.255 ow
s0101a 32.255 32.395 k
s0101a 32.395 32.554 ey
s0101a 50.898 50.928 r
s0101a 50.928 51.057 ey
s0101a 51.057 51.138 z
s0101a 51.138 51.178 d
s2002b 226.749 226.850 r
s2002b 226.850 226.900 iy
s2002b 226.900 226.999 l
s2002b 226.999 227.126 iy
s2002b 334.599 334.670 ih
s2002b 334.670 334.760 t
s2002b 334.760 334.969 ay
s2002b 334.969 335.070 n
s2002b 335.070 335.279 ow
s2002b 335.279 335.320 ih
s2002b 335.320 335.409 t
s2002b 335.440 335.490 b
s2002b 335.490 335.589 ah
s2002b 335.589 335.649 t
s2002b 335.690 335.760 y
s2002b 335.760 335.820 uw
s2002b 335.820 335.920 n
s2002b 335.920 336.029 ow
s2002b 336.029 336.120 dh
s2002b 336.120 336.170 eh
s2002b 336.170 336.287 r
s2002b 336.955 337.175 ah
s2002b 337.175 337.276 m
s2301b 126.590 126.620 ae
s2301b 126.620 126.650 n
s2301b 126.650 126.680 d
s2301b 126.680 126.710 t
s2301b 126.710 126.749 uw
s2301b 126.749 126.850 r
s2301b 126.850 126.900 iy
s2301b 126.900 126.999 l
s2301b 126.999 127.126 iy
'''

_vad = '''s0101a 32.217 32.554
s0101a 50.898 51.178
s2002b 226.749 227.126
s2002b 334.599 335.409
s2002b 335.440 335.649
s2002b 335.690 336.287
s2002b 336.955 337.276
s2301b 126.590 127.126
'''

_classes = '''Class 0
s0101a 50.867 51.178
s2002b 336.955 337.276
s2301b 126.680 127.126

Class 1
s2002b 334.762 335.277
s2002b 336.960 337.10
s2301b 126.600 126.700

Class 2
s2301b 126.749 127.126
s2002b 226.749 227.126

'''

@pytest.yield_fixture(scope='session')
def vad():
    ### TODO CHANGE UTIL READER TO SIMPLIFY ??
    #return tokenlists_to_corpus(read_annotation(_vad))
    return [read_split_single(_vad)]

@pytest.yield_fixture(scope='session')
def word():
    return tokenlists_to_corpus(read_annotation(_word))

@pytest.yield_fixture(scope='session')
def phone():
    return tokenlists_to_corpus(read_annotation(_phone))

@pytest.yield_fixture(scope='session')
def disc_cls():
    phone_ = phone()
    return annotate_classes(read_classfile(_classes), phone_, split=None) 



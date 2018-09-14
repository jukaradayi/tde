from tde.util.reader import load_split, load_corpus_txt
from tde.eval_track2 import _group_sub, load_disc
from tde.measures.group import *
from conftest import vad, word, disc_cls
import numpy
import sys

reload(sys)
sys.setdefaultencoding('utf8')


def test_group():
    ### Test grouping measure
    ### Based on the (supposed) fact that the measure was originally correct
    ### in TDE.
    __vad = vad()
    __disc_cls = disc_cls()[0]
    p_all, r_all = _group_sub(__disc_cls, __vad, 'all', False, 1)

    # use numpy.isclose, to account approximation error
    assert (numpy.isclose(p_all[0], 0.28571429)), "ERROR: grouping precision not correct"
    assert (numpy.isclose(r_all[0], 1.0)), "ERROR: recall precision not correct"
    return

def test_typeset():
    __disc_cls = disc_cls()[0]

    pclus = make_pclus(__disc_cls, True, False)
    pgoldclus = make_pgoldclus(__disc_cls, True, False)

    ts_disc = make_typeset(pclus, True, False)
    ts_gold = make_typeset(pgoldclus, True, False)

    assert (ts_disc == [('r', 'iy', 'l', 'iy'),
                        ('r', 'ey', 'z', 'd'),
                        ('ah', 'm'),
                        ('t', 'uw', 'r', 'iy', 'l', 'iy'),
                        ('ay', 'n', 'ow'),
                        ('ah',),
                        ('ae', 'n', 'd', 't')])," ERROR, ts_disc not coherent with found classes"
    assert ts_gold == [('r', 'iy', 'l', 'iy')], "ERROR, pgoldclus not coherent with found classes"
                                                 
    return
 
def test_nmatch():
    __disc_cls = disc_cls()[0]

    pclus = make_pclus(__disc_cls, True, False)
    pgoldclus = make_pgoldclus(__disc_cls, True, False)

    pclus_pgoldclus_nmatch = make_pclus_pgoldclus_nmatch(pclus,
                                                         pgoldclus,
                                                         True, False)
    pclus_nmatch = make_pclus_nmatch(pclus, True, False)
    pgoldclus_nmatch = make_pgoldclus_nmatch(pgoldclus,
                                             True, False)
    assert pclus_pgoldclus_nmatch.keys() == [('r', 'iy', 'l', 'iy')], "ERROR, nmatch has wrong keys, {}".format(pclus_pgoldclus_nmatch.keys())
    assert pclus_pgoldclus_nmatch[('r', 'iy', 'l', 'iy')] == 1, "ERROR, nmatch has wrong value"
    assert pclus_nmatch.keys() == [('t', 'uw', 'r', 'iy', 'l', 'iy'), ('ae', 'n', 'd', 't'), ('ah',), ('ah', 'm'), ('r', 'iy', 'l', 'iy')]
    assert pclus_nmatch[('t', 'uw', 'r', 'iy', 'l', 'iy')] == 2
    assert pclus_nmatch[('ae', 'n', 'd', 't')] == 2
    assert pclus_nmatch[('ah',)] == 1
    assert pclus_nmatch[('ah', 'm')] == 1
    assert pclus_nmatch[('r', 'iy', 'l', 'iy')] == 1
    assert pgoldclus_nmatch.keys() == [('r', 'iy', 'l', 'iy')]
    assert pgoldclus_nmatch[('r', 'iy', 'l', 'iy')] == 1
    return

def test_weights():
    __disc_cls = disc_cls()[0]

    pclus = make_pclus(__disc_cls, True, False)
    pgoldclus = make_pgoldclus(__disc_cls, True, False)

    ws_disc = make_weights(pclus, True, False)
    ws_gold = make_weights(pgoldclus, True, False)
    assert ws_disc == {('ae', 'n', 'd', 't'): 0.125,
                       ('ah',): 0.125, ('t', 'uw', 'r', 'iy', 'l', 'iy'): 0.125,
                       ('r', 'ey', 'z', 'd'): 0.125, ('ay', 'n', 'ow'): 0.125,
                       ('ah', 'm'): 0.125, ('r', 'iy', 'l', 'iy'): 0.25}, "ERROR in weights"
    assert ws_gold == {('r', 'iy', 'l', 'iy'): 1.0}, "ERROR in gold weights"
    return

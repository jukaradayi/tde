from contextlib import contextmanager
import time
import sys

def banner(s):
    l = len(s)
    return '-'*l+'\n'+s+'\n'+'-'*l


def pretty_pairs(pclus_set):
    strings = [('<{0} [{1:.3f},{2:.3f}] ({3})>'.format(f1.name, f1.interval.start,
                                                       f1.interval.end, ' '.join(f1.mark)),
                '<{0} [{1:.3f},{2:.3f}] ({3})>'.format(f2.name, f2.interval.start,
                                                       f2.interval.end, ' '.join(f2.mark)))
               for f1, f2 in pclus_set]
    if len(strings) == 0:
        return ''
    longest = max(len(x[0]) for x in strings)
    return '\n'.join('{1:{0}s} - {2:{0}s}'.format(longest, s1, s2)
                     for s1, s2 in strings)


def pretty_score_f(ps, rs, fs, label, nfolds, nsamples):
    r = '{sep}\n'.format(sep=37*'-')
    r += '{label}\n#folds:    {nfolds}\n#samples:  {nsamples}\n'.format(
        label=label, nfolds=nfolds, nsamples=nsamples)
    r += '{sep}\n'.format(sep=37*'-')
    r += '{score:9s}  {mean:5s}  {std:5s}  {min:5s}  {max:5s}\n'.format(
        score="measure", mean="mean", std="std",
        min="min", max="max")
    r += '---------  -----  -----  -----  -----\n'
    r += '{score:9s}  {mean:.3f}  {std:.3f}  {min:.3f}  {max:.3f}\n'.format(
        score="precision", mean=ps.mean(), std=ps.std(),
        min=ps.min(), max=ps.max())
    r += '{score:9s}  {mean:.3f}  {std:.3f}  {min:.3f}  {max:.3f}\n'.format(
        score="recall", mean=rs.mean(), std=rs.std(),
        min=rs.min(), max=rs.max())
    r += '{score:9s}  {mean:.3f}  {std:.3f}  {min:.3f}  {max:.3f}\n'.format(
        score="fscore", mean=fs.mean(), std=fs.std(),
        min=fs.min(), max=fs.max())
    r += '{sep}\n'.format(sep=37*'-')
    return r

def pretty_score(ps, rs, fs, label, nsamples):
    r = '{sep}\n'.format(sep=37*'-')
    r += '{label}\n#samples:  {nsamples}\n'.format(
        label=label, nsamples=nsamples)
    r += '{sep}\n'.format(sep=37*'-')
    r += '{score:9s}  {mean:5s} \n'.format(
        score="measure", mean="mean")
    r += '---------  -----  -----  -----  -----\n'
    r += '{score:9s}  {mean:.3f} \n'.format(
        score="precision", mean=ps.mean())
    r += '{score:9s}  {mean:.3f} \n'.format(
        score="recall", mean=rs.mean())
    r += '{score:9s}  {mean:.3f}\n'.format(
        score="fscore", mean=fs.mean())
    r += '{sep}\n'.format(sep=37*'-')
    return r

def pretty_score_nlp(ned_score, coverage_score, label, nfolds, nsamples):
    r = '{sep}\n'.format(sep=37*'-')
    r += '{label}\n#folds:    {nfolds}\n#samples:  {nsamples}\n'.format(
        label=label, nfolds=nfolds, nsamples=nsamples)
    r += '{sep}\n'.format(sep=37*'-')
    r += '{score:9s}  {mean:5s}  {std:5s}  {min:5s}  {max:5s}\n'.format(
        score="measure", mean="mean", std="std", min="min", max="max")
    r += '---------  -----  -----  -----  -----\n'
    r += '{score:9s}  {mean:.3f}  {std:.3f}  {min:.3f}  {max:.3f}\n'.format(
        score="NED", mean=ned_score.mean(), std=ned_score.std(),
        min=ned_score.min(), max=ned_score.max())
    r += '{score:9s}  {mean:.3f}  {std:.3f}  {min:.3f}  {max:.3f}\n'.format(
        score="coverage", mean=coverage_score.mean(), std=coverage_score.std(),
        min=coverage_score.min(), max=coverage_score.max())
    r += '{sep}\n'.format(sep=37*'-')
    return r


def pretty_score_cov(score, nsamples):
    r = '{sep}\n'.format(sep=37*'-')
    r += '{label}\n#samples:  {nsamples}\n'.format(
        label="COV", nsamples=nsamples)
    r += '{sep}\n'.format(sep=37*'-')
    r += '{score}  {mean}\n'.format(
        score="measure", mean="value")
    r += '---------  -----  -----  -----  -----\n'
    r += '{score}  {mean:.3f}\n'.format(
        score="COVERAGE", mean=ned_score)
    r += '{sep}\n'.format(sep=37*'-')
    return r

def pretty_score_ned(ned_all, ned_w, ned_a,  nsamples):
    r = '{sep}\n'.format(sep=37*'-')
    r += '{label}\n#samples:  {nsamples}\n'.format(
        label="NED", nsamples=nsamples)
    r += '{sep}\n'.format(sep=37*'-')
    r += '{score}  {mean}\n'.format(
        score="measure", mean="value")
    r += '---------  -----  -----  -----  -----\n'
    r += '{score}  {mean:.3f}\n'.format(
        score="NED ALL", mean=ned_all)
    r += '{score}  {mean:.3f}\n'.format(
        score="NED WITHIN", mean=ned_w)
    r += '{score}  {mean:.3f}\n'.format(
        score="NED ACROSS", mean=ned_a)
    r += '{sep}\n'.format(sep=37*'-')
    return r

@contextmanager
def verb_print(label, verbose=False, when_done=False,
               timeit=False, with_dots=False):
    if timeit:
        t0 = time.time()
    if verbose:
        msg = label + ('...' if with_dots else '') + ('' if when_done else '\n')
        print msg,
        sys.stdout.flush()
    try:
        yield
    finally:
        if verbose and when_done:
            if timeit:
                print 'done. Time: {0:.3f}s'.format(time.time() - t0)
            else:
                print 'done.'
            sys.stdout.flush()
        elif verbose and timeit:
            print '{1}: time: {0:.3f}s'.format(time.time() - t0, label)
            sys.stdout.flush()

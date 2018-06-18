"""Evaluate Spoken Term Discovery"""

from __future__ import division

import os
import os.path as path
import sys
from itertools import izip

import numpy as np
from joblib import Parallel, delayed
import ipdb

VERSION = "0.2.1"

from tde.util.reader import load_classes_txt, load_corpus_txt, load_split
from tde.util.printing import verb_print, banner, pretty_score_f, \
    pretty_score_nlp, pretty_score_cov, pretty_score_ned, pretty_score
from tde.util.splits import truncate_intervals, check_intervals
from tde.util.functions import fscore

from tde.nlp.check_consistency import create_silence_tree, parse_class_file
from tde.nlp.compute_ned import ned_from_class
from tde.nlp.compute_cov import cov_from_class
from tde.measures.group import evaluate_group
from tde.measures.boundaries import Boundaries, eval_from_bounds
#from tde.measures.match import eval_from_psets, make_pdisc, make_pgold, \
#    make_psubs
from tde.measures.token_type import evaluate_token_type


#def _load_classes(fname, corpus, split_mapping=None):
#    return load_classes_txt(fname, corpus, split=split_mapping)

def load_names(fname): 
    """ read vad and create list of all the file names in the corpus"""
    all_names = set()
    with open(fname, 'r') as fin:
        vad = fin.readlines()
        for line in vad:
            spkr, _, _ = line.strip('\n').split(' ')
            all_names.add(spkr)
    return list(all_names)

def load_disc(fname, corpus, split_file, truncate, verbose):
    with verb_print('  loading discovered classes',
                             verbose, True, True, True):
        split_mapping = load_split(split_file)
        #disc, errors = load_classes_txt(fname, corpus, split=split_mapping)
        disc, errors = load_classes_txt(fname, corpus, split=None)
        if not truncate:
            errors_found = len(errors) > 0
            if len(errors) > 100:
                print 'There were more than 100 interval errors found.'
                print 'Printing only the first 100.'
                print
                errors = errors[:100]
            for fragment in sorted(errors, key=lambda x: (x.name, x.interval.start)):
                print '  error: {0} [{1:.3f}, {2:.3f}]'.format(
                    fragment.name, fragment.interval.start, fragment.interval.end)
            if not truncate and errors_found:
                print 'There were errors in {0}. Use option -f to'\
                    ' automatically skip invalid intervals.'.format(fname)
                sys.exit()

    if truncate:
        with verb_print('  checking discovered classes and truncating'):
            disc, filename_errors, interval_errors = \
                truncate_intervals(disc, corpus,
                                   split_mapping)
    else:
        with verb_print('  checking discovered classes', verbose, True,
                                 True, True):
            filename_errors, interval_errors = \
                check_intervals(disc, split_mapping)
    if not truncate:
        filename_errors = sorted(filename_errors,
                                 key=lambda x: (x.name, x.interval.start))
        interval_errors = sorted(interval_errors,
                                 key=lambda x: (x.name, x.interval.start))
        interval_error = len(interval_errors) > 0
        filename_error = len(filename_errors) > 0
        errors_found = filename_error or interval_error
        if interval_error:
            print banner('intervals found in {0} outside of valid'
                                      ' splits'.format(fname))
            if len(interval_errors) > 100:
                print 'There were more than 100 interval errors found.'
                print 'Printing only the first 100.'
                print
                interval_errors = interval_errors[:100]
            for fragment in sorted(interval_errors,
                                   key=lambda x: (x.name, x.interval.start)):
                print '  error: {0} [{1:.3f}, {2:.3f}]'.format(
                    fragment.name,
                    fragment.interval.start, fragment.interval.end)
        if filename_error:
            print banner('unknown filenames found in {0}'
                                      .format(fname))
            if len(filename_errors) > 100:
                print 'There were more than 100 filename errors found.'
                print 'Printing only the first 100.'
                print
                filename_errors = filename_errors[:100]
            for fragment in sorted(filename_errors,
                                   key=lambda x: (x.name, x.interval.start)):
                print '  error: {0}'.format(fragment.name)
        if not truncate and errors_found:
            print 'There were errors in {0}. Use option -f to automatically skip invalid intervals.'.format(fname)
            sys.exit()
    return disc


def _group_sub(disc_clsdict, names, label, verbose, n_jobs):
    eg = evaluate_group
    #if verbose:
    #    print '  group ({2}): subsampled {0} files in {1} sets'\
    #        .format(sum(map(len, names)), len(names), label)
    with verb_print('  group ({0}): calculating scores'.format(label),
                             verbose, False, True, False):
        p, r = izip(*(Parallel(n_jobs=n_jobs,
                              verbose=5 if verbose else 0,
                              pre_dispatch='n_jobs')
                     (delayed(eg)(disc_clsdict.restrict(ns, True))
                      for ns in names)))
    p, r = np.fromiter(p, dtype=np.double), np.fromiter(r, dtype=np.double)
    p, r = praggregate(p, r)
    return p, r


def group(disc_clsdict, fragments_all, dest, verbose, n_jobs):
    if verbose:
        print banner('GROUP')
    #TODO CHECK SCORE ACROSS/WITHIN!
    pc, rc = _group_sub(disc_clsdict, fragments_all, 'all', verbose, n_jobs)
    fc = np.fromiter((fscore(pc[i], rc[i]) for i in xrange(pc.shape[0])), dtype=np.double)

    #pw, rw = _group_sub(disc_clsdict, fragments_within, 'within', verbose, n_jobs)
    #fw = np.fromiter((fscore(pw[i], rw[i]) for i in xrange(pw.shape[0])), dtype=np.double)
    with open(path.join(dest, 'group'), 'w') as fid:
        fid.write(pretty_score(pc, rc, fc, 'group total',
                                 sum(map(len, fragments_all))))


def _token_type_sub(clsdict, wrd_corpus, names, label, verbose, n_jobs):
    et = evaluate_token_type
    if verbose:
        print '  token/type ({2}): subsampled {0} files in {1} sets'\
            .format(sum(map(len, names)), len(names), label)
    with verb_print('  token/type ({0}): calculating scores'
                             .format(label), verbose, False, True, False):
        #print wrd_corpus
        #ipdb.set_trace()
        pto, rto, pty, rty = izip(*(et(clsdict.restrict(ns, False),
                                       wrd_corpus.restrict(ns))
                                    for ns in names))
    pto, rto, pty, rty = np.array(pto), np.array(rto), np.array(pty), np.array(rty)
    pto, rto = praggregate(pto, rto)
    pty, rty = praggregate(pty, rty)

    return pto, rto, pty, rty


def token_type(disc_clsdict, wrd_corpus, fragments_cross,
               dest, verbose, n_jobs):
    if verbose:
        print banner('TOKEN/TYPE')
    ptoc, rtoc, ptyc, rtyc = _token_type_sub(disc_clsdict, wrd_corpus,
                                             fragments_cross, 'cross',
                                             verbose, n_jobs)
    ftoc = np.fromiter((fscore(ptoc[i], rtoc[i]) for i in xrange(ptoc.shape[0])),
                       dtype=np.double)
    ftyc = np.fromiter((fscore(ptyc[i], rtyc[i]) for i in xrange(ptyc.shape[0])),
                       dtype=np.double)

    with open(path.join(dest, 'token_type'), 'w') as fid:
        fid.write(pretty_score(ptoc, rtoc, ftoc, 'token total',
                                 sum(map(len, fragments_cross))))
        fid.write('\n')
        fid.write(pretty_score(ptyc, rtyc, ftyc, 'type total',
                                 sum(map(len, fragments_cross))))

def _boundary_sub(disc_clsdict, corpus, names, label, verbose, n_jobs):
    eb = eval_from_bounds
    if verbose:
        print '  boundary ({2}): subsampled {0} files in {1} sets'\
            .format(sum(map(len, names)), len(names), label)
    with verb_print('  boundary ({0}): calculating scores'
                             .format(label), verbose, True, True, True):
        disc_bounds = [Boundaries(disc_clsdict.restrict(ns))
                       for ns in names]
        gold_bounds = [Boundaries(corpus.restrict(ns))
                       for ns in names]
    with verb_print('  boundary ({0}): calculating scores'
                             .format(label), verbose, False, True, False):
        p, r = izip(*Parallel(n_jobs=n_jobs, verbose=5 if verbose else 0,
                              pre_dispatch='2*n_jobs') \
                    (delayed(eb)(disc, gold)
                     for disc, gold in zip(disc_bounds, gold_bounds)))
    p, r = np.fromiter(p, dtype=np.double), np.fromiter(r, dtype=np.double)
    p, r = praggregate(p, r)
    return p, r


def boundary(disc_clsdict, corpus, fragments_cross,
               dest, verbose, n_jobs):
    if verbose:
        print banner('BOUNDARY')
    pc, rc = _boundary_sub(disc_clsdict, corpus, fragments_cross,
                           'cross', verbose, n_jobs)
    fc = np.fromiter((fscore(pc[i], rc[i]) for i in xrange(pc.shape[0])), dtype=np.double)
    with open(path.join(dest, 'boundary'), 'w') as fid:
        fid.write(pretty_score(pc, rc, fc, 'boundary total',
                                 sum(map(len, fragments_cross))))


def ned_cov(disc_clsfile, transcription, phn_corpus_file, dest, verbose):
    ''' wrapper around the nlp.compute_ned function, computes ned'''
    ''' and write output using pretty_score function '''
    ned_all, ned_w, ned_a, trs = ned_from_class(disc_clsfile,
                                        transcription,
                                        phn_corpus_file,
                                        False)
    cov, n_overall = cov_from_class(disc_clsfile,
                                    phn_corpus_file,
                                    True)
                        

    with open(path.join(dest, 'cov'), 'w') as fid:
        fid.write(pretty_score_cov(cov,
                                    n_overall))
    with open(path.join(dest, 'ned'), 'w') as fid:
        fid.write(pretty_score_ned(ned_all, ned_w,
                                   ned_a, 
                                    n_overall))
    with open(path.join(dest, 'transcription.txt'), 'w') as fout:
        for line in trs:
            fout.write(u'{}\n'.format(line))

def aggregate(array, default_score=0.):
    array = np.array(array)
    array = array[np.logical_not(np.isnan(array))]
    if array.shape[0] == 0:
        array = np.array([default_score])
    return array

def praggregate(p_array, r_array, default_score=0.):
    p_array, r_array = np.array(p_array), np.array(r_array)
    p_index = np.logical_not(np.isnan(p_array))
    r_index = np.logical_not(np.isnan(r_array))
    index = np.logical_and(p_index, r_index)
    p_array, r_array = p_array[index], r_array[index]
    if not np.any(index):
        p_array, r_array = np.array([default_score]), np.array([default_score])
    return p_array, r_array

if __name__ == '__main__':
    import argparse
    def parse_args():
        parser = argparse.ArgumentParser(
            prog='english_eval2',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description='Evaluate spoken term discovery on the english dataset',
            epilog="""Example usage:

\t$ ./english_eval2 my_sample.classes /home/user/resource_dir corpus_1 resultsdir/

where /home/user/resource_dir contains corpus_1.phn (phone transcription), 
corpus_1.wrd (word transcription), and corpus_1.vad (vad).

This command evaluates STD output `my_sample.classes` and stores the
output in `resultsdir/`.

Classfiles must be formatted like this:

Class 1 (optional_name)
fileID starttime endtime
fileID starttime endtime
...

Class 2 (optional_name)
fileID starttime endtime

Transcriptions must be formatted like this:
fileID1 onset offset word/phone
fileID1 onset offset word/phone
fileID2 onset offset word/phone
...

and VAD : 
fileID1 onset offset
fileID1 onset offset
fileID2 onset offset
...
...
""")
        #parser.add_argument(
        #        'corpus', metavar='CORPUS',
        #        choices=['english', 'mandarin', 'french',
        #                 'buckeye', 'xitsonga', 'other'],
        #        help='name of the corpus you are evaluating. Choices are '
        #             '"english", "mandarin", "french", "buckeye", '
        #             '"xitsonga" and "other" if you want to use '
        #             ' your own corpus')
        parser.add_argument('disc_clsfile', metavar='DISCCLSFILE',
                            nargs=1,
                            help='discovered classes')
        parser.add_argument('trs',
                            metavar='TRANSCRIPTION',
                            default='None',
                            nargs=2,
                            help='Folder where the phone and word'
                            'transcriptions and vad are, and the name of'
                            ' the files. Example of use is:\n'
                            '\t../bin/resources mandarin\n'
                            'where, in ../bin/resources, there is a'
                            ' mandarin.phn, mandarin.wrd and mandarin.vad')

        parser.add_argument('outdir', metavar='DESTINATION',
                            nargs=1,
                            help='location for the evaluation results')
        parser.add_argument('-f', '--force-truncate',
                            action='store_true',
                            dest='truncate',
                            default=True,
                            help='force truncation of discovered fragments '
                            'outside of splits')
        parser.add_argument('-m', '--measures',
                            action='store',
                            nargs='*',
                            dest='measures',
                            default=[],
                            choices=['boundary', 'group', 
                                     'token/type', 'nlp'],
                            help='select individual measures to perform')
        parser.add_argument('-v', '--verbose',
                            action='store_true',
                            dest='verbose',
                            default=False,
                            help='display progress')
        parser.add_argument('-j', '--n-jobs',
                            action='store',
                            type=int,
                            dest='n_jobs',
                            default=1,
                            help='number of cores to use')
        parser.add_argument('-V', '--version', action='version',
                            version="%(prog)s version {version}".format(version=VERSION))
        return vars(parser.parse_args())
    args = parse_args()

    verbose = args['verbose']
    n_jobs = args['n_jobs']

    disc_clsfile = args['disc_clsfile'][0]

    dest = args['outdir'][0]
    
    resource_dir = args['trs'][0]

    corpus = args['trs'][1]
    
    # if corpus is "other", change resource_dir to get the transcriptions/vad

    phn_corpus_file   = path.join(resource_dir, '{}.phn'.format(corpus))
    wrd_corpus_file   = path.join(resource_dir, '{}.wrd'.format(corpus))
    vad_file          = path.join(resource_dir, '{}.vad'.format(corpus))
    print vad_file

    if verbose:
        print banner('LOADING FILES')

    # load gold phones and gold words
    with verb_print('  loading word corpus file',
                             verbose, True, True, True):
        wrd_corpus = load_corpus_txt(wrd_corpus_file)

    with verb_print('  loading phone corpus file',
                             verbose, True, True, True):
        phn_corpus = load_corpus_txt(phn_corpus_file)
    
    # load across and withing folds
    with verb_print('  loading folds cross',
                             verbose, True, True, True):
        #fragments_cross = load_split(folds_cross_file,
        #                             multiple=False)
        intervals_vad = [load_split(vad_file,
                                     multiple=False)]
    # get list of file names from vad: 
    #    names = load_names(vad_file)
    try:
        os.makedirs(dest)
    except OSError:
        pass

    # Before loading the class file, check its consistency.
    # If intervals that overlaps a little with silence are found,
    # they are trimmed. If intervals that contain silences are found,
    # an error is thrown and the evaluation breaks.
    sil_tree = create_silence_tree(vad_file)
    parse_class_file(disc_clsfile, sil_tree, dest, verbose)

    # load discovered intervals and gold intervals
    truncate = args['truncate']
    #truncate = False
    truncate = True
    disc_clsdict = load_disc(disc_clsfile, phn_corpus, vad_file,
                             truncate, verbose)


    with open(path.join(dest, 'VERSION_{0}'.format(VERSION)), 'w') as fid:
        fid.write('')

    measures = set(args['measures'])
    do_all = len(measures) == 0
    if do_all or 'group' in measures:
        group(disc_clsdict, intervals_vad, dest, verbose,
              n_jobs)
    if do_all or 'token/type' in measures:
        token_type(disc_clsdict, wrd_corpus, intervals_vad,
                   dest, verbose, n_jobs)
    if do_all or 'boundary' in measures:
        boundary(disc_clsdict, wrd_corpus, intervals_vad,
                 dest, verbose, n_jobs)
    if do_all or 'nlp' in measures:
        ned_cov(disc_clsfile, True, phn_corpus_file,
                dest, verbose)
    if verbose:
        print 'All done. Results stored in {0}'.format(dest)

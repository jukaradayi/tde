#!/usr/bin/env python
#
# author : The CoML Team
# 
"""
script to check the coherence of the word and phone transcriptions to avoid
gaps between words or phonemes, check that the timestamps are the same for 
each word in the word transcription and the phone transcription, and also
check that the VAD is coherent with the phone transcription (and extract the
VAD if it doesn't exist).
"""
import ipdb
import sys
import os
import numpy as np
from collections import defaultdict
import argparse

def read_transcription(filename):
    """ read the transcription and output a dictionary 
        {fname: [(on, off, trs) ]} where fname are the names of the speakers
        in the corpus, on, off are the timestamps and trs the transcription.
    """
    output_transcript = defaultdict(list)
    with open(filename, 'r') as fin:
        transcript = fin.readlines()
        for line in transcript:
            # VAD has 3 columns, phone and word transcriptions have 4 
            try :
                fname, on, off, trs = line.strip('\n').split(' ')
                output_transcript[fname].append((float(on), float(off), trs))
            except:
                fname, on, off = line.strip('\n').split(' ')
                output_transcript[fname].append((float(on), float(off)))

    # sort the lists so that the transcriptions are in order
    for fname in output_transcript:
        output_transcript[fname].sort()

    return output_transcript

def check_contiguous(trs):
    """ check that the transcriptions are contiguous, and if not,
    insert a 'SIL'"""
    not_contiguous_flag = False
    output_trs = defaultdict(list)
    for fname in trs:
        i = 0
        for on, off, sym in trs[fname]:
            if i == 0:
                i += 1
                prev_off = off
                output_trs[fname].append((on, off, sym))
                continue
            if on == prev_off:
                output_trs[fname].append((on, off, sym))
                prev_off = off
                continue
            else:
                output_trs[fname].append((prev_off, on, "SIL"))
                prev_off = off
                not_contiguous_flag = True
    return output_trs, not_contiguous_flag

def check_phone_and_word(phn, wrd):
    """ Check that the phone and word transcriptions are aligned. 
        We loop over all the words in the transcriptions, and for 
        each word, we look in the phone transcription to check 
        that they are written there too. 
    """
    _phn_keys = phn.keys()
    # check that word and phone transcriptions have the same filenames
    for fname in wrd:
        assert fname in _phn_keys, """ERROR: word transcriptions has a """\
                                   """filename that doesn't occur in """\
                                   """phone transcription."""

    # first loop over word's filenames
    for fname in wrd:
        phon_starts = np.array([on for on, off, ph in phn[fname]])
        phon_ends = np.array([off for on, off, ph in phn[fname]])
        #ipdb.set_trace()
        for won, woff, wwo in wrd[fname]:
            if wwo == "SIL":
                # skip Silences from search
                continue
            
            phon = np.where(phon_starts == won)
            phoff = np.where(phon_ends == woff)
            if (len(phon[0]) == 0) or (len(phoff[0]) == 0):
                print("ERROR: word transcription has word "
                      "{} {} {} {}, but these timestamps don't "
                      "have an exact match in the phone "
                      "transcription !".format(fname, won,
                                               woff, wwo))
                #raise KeyError


    return

def check_phone_and_vad(phn, vad):
    """ Check that the vad has the same timestamps as the
        phone transcriptions and contains no silences.
    """
    for fname in vad:
        for von, voff in vad[fname]:
            phon = np.search(phn[fname] == von)
            phoff = np.search(phn[fname] == voff)
            
            # First check that the timestamps occur in the 
            # transcription
            if (len(phon[0]) == 0) or (len(phoff[0]) == 0):
                print("ERROR: vad timestamps "
                      "{} {} {} do not occur in the phone "
                      "transcription !".format(fname, von,
                                               voff))
                raise KeyError
            
            # then check that this transcription doesn't contain
            # a silence longer than 300 ms
            for phon_, phoff_, phpho in phn[fname][phon[0][0]:phoff[0][0]]:
                if phpho == 'SIL'and (phoff_ - phon_ >= 0.3):
                    print('ERROR: vad contains silence: '
                          '{} {} {} in vad, {} {} {} {} '
                          ' in phone transcription.'.format(fname, von, voff,
                                                            fname, phon_,
                                                            phoff_, phpho))


    
    return

def create_vad(phn):
    """ if vad doesn't exist, you can create it using
        the phone transcription"""
    return

def write_txt(out, name):
    with open(name, 'w') as fout:
        for fname in out:
            for on, off, sym in out[fname]:
                fout.write(u'{} {} {} {}\n'.format(fname, on, off, sym))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phn',
            help='''phone transcription''')
    parser.add_argument('wrd',
            help='''word transcription''')
    parser.add_argument('--vad', default = None,
            help='''Voice Activity Detection. If not given,'''
            '''this script will create it using the phone transcription''')
    args = parser.parse_args()
    phn = read_transcription(args.phn)
    wrd = read_transcription(args.wrd)

    # Check that the phone transcription is indeed contiguous
    phn, ph_not_contig = check_contiguous(phn)
    #if ph_not_contig:
    #    write_txt(phn,'outputput')
    #
    wrd, wr_not_contig = check_contiguous(wrd)
    #if wr_not_contig:
    #    write_txt(wrd, 'output')

    if args.vad is not None:
        vad = read_transcription(args.vad)

    check_phone_and_word(phn, wrd)

    # if vad is given, compare it to phone transcription, 
    # else create the vad
    if args.vad is not None:
        check_phone_and_vad(phn, vad)
    else:
        vad = create_vad(phn)
        print vad
        #write_txt(vad, 'vadvad')


if __name__ == "__main__":
    main()

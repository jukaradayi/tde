#!/usr/bin/env python
#
# author = julien karadayi
#
# Check the consistency of the class file you're about to evaluate.
# First, check if the format is good (in particular, check if the line
# of the file is a blank line).
# Then, check that none of the intervals found in the class file contains
# silence. To do that, we build an interval tree with the intervals of
# silences, using the vad, and check for each interval found by the
# algorithm that none overlap with the silence. Overlap of a very small
# number of frames (2 ?) is permissible.
# Some solutions are suggested to treat the non conform pairs.
#
# Input
# - Class File : file containing the found pairs, in the ZR challenge format:
#
#       Class1
#       file1 on1 off1
#       file2 on2 off2
#
#       Class2
#       [...]
#
# - VAD : VAD corresponding to the corpus on which you
#
# Output
# - List of all the intervals that overlap with Silence (outputs nothing if
#   all the intervals are okay)

import os
import sys
import intervaltree
import argparse
import ipdb
from collections import defaultdict


def create_silence_tree(vad_file):
    """Read the vad, and create an interval tree
    with the silence intervals.
    We are interested in the silence intervals, so for each
    file in the vad, sort all the intervals, get the min and max
    of the timestamps, and take the complementary of the set 
    of intervals delimitated by the vad.

    Input
        - vad_file: path to the vad file indicating segments of speech
    Output
        - silence_tree: dict {fname: IntervalTree} : for each file in the vad
          keep an interval tree of silences
    """

    assert os.path.isfile(vad_file), "ERROR: VAD file doesn't exist, check path"
    
    vad = defaultdict(list)
    # Read vad and create dict {file: [intervals]}
    with open(vad_file, 'r') as fin:
        speech = fin.readlines()

        for line in speech:
            fname, on, off = line.strip('\n').split()
            vad[fname].append((float(on), float(off)))

    # Sort the intervals by their onset for each file,
    # and create a "Silence activity detection" by 
    # taking the complementary of each set of intervals
    silence_tree = dict()
    for fname in vad:

        silences = []

        # sort intervals by their onset - there is no overlap
        # between intervals.
        vad[fname].sort()
        on_min, on_max = vad[fname][0][0], vad[fname][-1][1]
        if on_min > 0:
            silences.append((0, on_min))
        for it, times in enumerate(vad[fname]):
            on, off = times
            if it == 0:
                # skip first interval
                prev_off = off
                continue

            # add the gap between previous interval and current
            # interval as "silence"
            if prev_off > on:
                silences.append((prev_off, on))
            
            # keep previous offset
            prev_off = off

        # Create interval tree for each file in the vad.
        silence_tree[fname] = intervaltree.IntervalTree.from_tuples(silences)

    return silence_tree

def parse_class_file(class_file, silence_tree, output, verbose=False):
    ''' Read All Intervals in Class File, if on interval overlaps
        with a Silence in the VAD, raise an error and exit.
    '''

    assert os.path.isfile(class_file), "ERROR: class file not found"
    
    n_trimmed = 0 # count the number of intervals trimmed

    intervals_list = defaultdict(list)
    # Read Class file. Keep lines if they're not blank and 
    # don't start by "Class", because those are the ones 
    # that contain the found intervals.
    with open(class_file, 'r') as fin:
        classes = fin.readlines()
        output_class = []
        for line in classes:
            # skip blanks & "Class [...]"
            if line.startswith('Class') or len(line.strip('\n')) == 0:
                output_class.append(line)
                continue
            
            # retrieve file name and interval
            # ipdb.set_trace()
            fname, on, off = line.strip('\n').split()
            #intervals_list[fname].append((float(on), float(off)))

            # Check if interval overlaps with silence
            ov = silence_tree[fname].search(float(on), float(off))

            # Check in which category the overlap is:
            if len(ov) > 0 :
                raise ValueError("ERROR, a Silence overlaps with "
                        "interval {}. This means the Spoken Term Discovery"
                        "system didn't use the VAD.\n"
                        " You can use the script tde/nlp/check_consistency.py"
                        " to remove the part of the intervals that overlaps with"
                        " silence.\nThe script will still fail if one interval"
                        " straddle with silence.".format(
                            line))



    print("Check successful, {} intervals were trimmed".format(n_trimmed))
    
    ## write class file without Silences
    #output_name = os.path.join(output,
    #                           os.path.basename(class_file) + "_no_SIL")
    #                            
    #write_new_class_file(output_class, output_name)
    #return output_name
    return

def correct_class_file(class_file, silence_tree, output, verbose=False):
    ''' Read All Intervals in Class File, if on interval overlaps
        with a Silence in the VAD, check if the interval overlaps 
        only with a silence at one or both of its ends (1), or it 
        completely contains a silence (2).
        In case (2)
    '''

    assert os.path.isfile(class_file), "ERROR: class file not found"
    
    n_trimmed = 0 # count the number of intervals trimmed

    intervals_list = defaultdict(list)
    # Read Class file. Keep lines if they're not blank and 
    # don't start by "Class", because those are the ones 
    # that contain the found intervals.
    with open(class_file, 'r') as fin:
        classes = fin.readlines()
        output_class = []
        for line in classes:
            # skip blanks & "Class [...]"
            if line.startswith('Class') or len(line.strip('\n')) == 0:
                output_class.append(line)
                continue
            
            # retrieve file name and interval
            # ipdb.set_trace()
            fname, on, off = line.strip('\n').split()
            #intervals_list[fname].append((float(on), float(off)))

            # Check if interval overlaps with silence
            ov = silence_tree[fname].search(float(on), float(off))

            # Check in which category the overlap is:
            if len(ov) > 0 :
                new_on, new_off = check_intervals_found(ov,
                                                        float(on),
                                                        float(off))
                if (new_on, new_off) != (float(on), float(off)):
                    n_trimmed += 1
                    if verbose:
                        print("interval({}, {}, {}) has been trimmed to"\
                          " ({}, {}, {}) to remove"\
                          " overlap with silence".format(fname, on, off,
                                                         fname, new_on,
                                                         new_off))
                on, off = new_on, new_off
                
            if on == -1 and off == -1:
                raise ValueError("ERROR, a Silence is in the middle of "
                        "interval {}. This means the system didn't use the "
                        " VAD. Please treat interval accordingly, so that "
                        "no more intervals in your class file contain"
                        " silences and check again.".format(
                            line))
            output_class.append(" ".join([fname, str(on), str(off)]) + '\n')

    print("Check successful, {} intervals were trimmed".format(n_trimmed))
    
    # write class file without Silences
    output_name = os.path.join(output,
                               os.path.basename(class_file) + "_no_SIL")
                                
    write_new_class_file(output_class, output_name)
    return output_name

def write_new_class_file(output_class, output_name):
    """ Write Class File in which Silences were removed """

    with open(output_name, "w") as fout:
        for line in output_class:
            fout.write(u'{}'.format(line))

def check_intervals_found(ov, on, off):
    """Check that None of the intervals found overlap
    with Silence.
    If an interval overlaps with silence, check three cases :
        1 - The silence is completely inside the interval
            found
        2 - The silence overlaps with one border of the
            found interval
        3 - Any combination (with repetition) of these two cases

    Suggested treatements for such pairs:
        - For case 1: cut interval in half (decide how to reconstruct
            the found pairs), remove pair.
        - For case 2: trim the interval by removing the part that
            overlaps with silence
        - For case 3: if only a combination of case 2, trim pair,
            if case 1 is also occuring, cut interval, rinse and repeat.

    Input: 
        - intervals_list: dictionnary {fname: [list of intervals]},
                          that returns a list of all found intervals
                          for a filename
        - silence_tree: dictionnary that returns an interval tree 
                        built with the intervals of silence for each
                        filename.
    Output:
        - bad_pairs: dict {fname: [(on_pair, off_pair, on_SIL, offSIL)]}
                     that returns for each filename a list of tuples 
                     that indicate the intervals that overlap with silence 
                     (and the silences they overlap with)
    """
    for SIL in ov:
        SIL_on, SIL_off = SIL[0], SIL[1]

        if (on <= SIL_on) and (off >= SIL_off):
            # means the intervals found completely contains a silence,
            # which means the system didn't take into account the VAD.
            # Leave the decision to the user about this interval.
            return -1, -1
        elif (on <= SIL_on) and (off < SIL_off):
            return on, SIL_on
        elif (on > SIL_on) and (off >= SIL_off):
            return SIL_off, off


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('vad_file', metavar='<VAD>',
                        help="""text file containing all the intervals of"""
                             """speech""")
    parser.add_argument('class_file', metavar='<CLASS>',
                        help="""Class file, in the ZR Challenge format:"""
                             """       """
                             """       Class1"""
                             """       file1 on1 off1"""
                             """       file2 on2 off2"""
                             """       """
                             """       Class2"""
                             """       [...]"""
                             """       """)
    parser.add_argument('output', metavar='<OUTPUT>',
                        help="""Path to the cleaned output""")
    parser.add_argument('-v', '--verbose', metavar='<VERBOSE>', action="store_true"
                        help="""enable for more verbose""")

    args = parser.parse_args()
    sil_tree = create_silence_tree(args.vad_file)
    disc_int = correct_class_file(args.class_file, sil_tree, args.output, args.verbose)
    #bad_pairs = check_intervals_foud(disc_int, sil_tree)
    #print bad_pairs

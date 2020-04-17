import argparse
from subprocess import call

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--force_redo', action="store_true",)
    parser.add_argument('-f', '--features_onwards', action="store_true", )
    parser.add_argument('-o', '--offset', action="store", type=int, default=0)

    args = parser.parse_args()

    scripts_offset = args.offset

    if args.features_onwards:
        scripts_offset = 5

    scripts = [
        "signal_extractor.py",  # video to signal
        "signal_preprocessor.py",  # preprocess signal
        "signal_beat_separation.py",  # separate beats
        "signal_fiducial_points_detection.py",  # find fiducial points
        "fta_average_beat.py",  # find average beat (needed for failure to acquire)
        "signal_beat_fta.py",  # filter bad beats
        "feature_extractor.py",  # extract features
        "feature_selection1.py",  # feature selection one
        "feature_selection2.py",  # feature selection two
        "classify.py"  # run experiments
    ][scripts_offset:]

    for s in scripts:
        cmd = "python3 {} {}".format(
            s,
            "-r" if args.force_redo else ""
        )
        cmd = cmd.strip()
        print("###### %s ######" % cmd)
        cmd = cmd.split(" ")
        call(cmd)



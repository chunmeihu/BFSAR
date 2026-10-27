#!/usr/bin/env python
"""CIRR BFSAR Stage-1 Banzhaf estimator with OpenCLIP ViT-L-14 (laion2b).

Same recipe as ``train_bm_cirr_base.py`` (12-crop, max_words=32, MC=50);
only the CLIP backbone + image normalization change.
Writes under ``output/cirr_estimator_openclip`` — does not overwrite OpenAI CIRR runs.
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import sys

from bfsar.cirr.openclip_backbone import assert_isolated_output_dir, install_openclip_cirr_patches


def _output_dir_from_argv(argv):
    if "--output_dir" in argv:
        i = argv.index("--output_dir")
        if i + 1 < len(argv):
            return argv[i + 1]
    return None


def main():
    out = _output_dir_from_argv(sys.argv)
    if out:
        assert_isolated_output_dir(out)
    install_openclip_cirr_patches()
    from train_bm_cirr_base import main as estimator_main

    estimator_main()


if __name__ == "__main__":
    main()

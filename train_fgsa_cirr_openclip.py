#!/usr/bin/env python
"""CIRR BFSAR Stage-2 retrieval with OpenCLIP ViT-L-14 (laion2b).

Same recipe as ``train_fgsa_cirr.py``; only the CLIP backbone +
image normalization change. Load Stage-1 teacher from
``output/cirr_estimator_openclip/``.
Does not overwrite ``output/cirr_retrieval`` (OpenAI CIRR SOTA).
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
    from train_fgsa_cirr import main as retrieval_main

    retrieval_main()


if __name__ == "__main__":
    main()

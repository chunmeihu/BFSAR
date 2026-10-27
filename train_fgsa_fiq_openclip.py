#!/usr/bin/env python
"""Fashion-IQ BFSAR Stage-2 retrieval with OpenCLIP ViT-L-14 (laion2b).

Same retriever recipe as train_fgsa_fiq.py; only the CLIP backbone changes.
Does not overwrite output/fiq_retrieval or output/fiq_retrieval_crop6.
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import sys

from bfsar.fiq.openclip_backbone import assert_isolated_output_dir, install_openclip_patches


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
    install_openclip_patches()
    from train_fgsa_fiq import main as retrieval_main

    retrieval_main()


if __name__ == "__main__":
    main()

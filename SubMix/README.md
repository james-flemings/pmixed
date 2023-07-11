# SubMix

This repo implements the SubMix private prediction protocol for generating text from large-scale transformers.


You can download and preprocess the wikitext-103 dataset using ``bash prepare-wikitext-103.sh`` and ``python preprocess-wikitext-103.py``, respectively. Similarly, you can use ``python preprocess-bigpatent.py`` to both download and preprocess the big patent dataset. Refer to ``example.ipynb`` for more details on how to use SubMix as a programmatic python library with PyTorch.

#### Code Acknowledgements

The majority of SubMix is licensed under CC-BY-NC, however portions of the project are available under separate license terms: https://github.com/affjljoo3581/GPT2, https://github.com/pytorch/opacus, and https://huggingface.co/docs/transformers/index are licensed under the Apache 2.0 license, and https://github.com/pytorch/fairseq is licensed under the MIT license.

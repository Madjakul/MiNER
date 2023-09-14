.. MiNER documentation master file, created by
   sphinx-quickstart on Mon Mar 27 16:08:27 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

#################################
Welcome to MiNER's documentation!
#################################

Named entity recognition using weakly labeled data.
Wrote using Python 3.8.


Quickstart
==========

.. toctree::
   :maxdepth: 2

   installation.md
   quickstart.md
   overview.md


Code
====

.. toctree::
   :maxdepth: 1

   miner/modules/transformer.rst
   miner/modules/base_crf.rst
   miner/modules/partial_crf.rst
   miner/modules/partial_ner.rst
   miner/modules/smooth_ner.rst
   miner/trainers/transformer_trainer.rst
   miner/trainers/partial_ner_trainer.rst
   miner/trainers/smooth_ner_trainer.rst
   miner/utils/data/preprocessing.rst
   miner/utils/data/transformer_dataset.rst
   miner/utils/data/partial_ner_dataset.rst
   miner/utils/data/smooth_ner_dataset.rst
   miner/utils/crf_utils.rst
   miner/utils/ner_utils.rst


About
=====

.. toctree::
   :maxdepth: 1

   license.md


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`

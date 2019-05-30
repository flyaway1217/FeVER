FeVER
=====

This repository contains the python implementation of the paper **Beyond Context: A New Perspective for Word Embeddings**.

Before Training
---------------

Overview
~~~~~~~~

The input of this implementation is not the pure text.
Instead, a multi-label data format is used as the input of our system.
Because of this, we need to preprocess the text into a suitable format, which is described in `Dataset`_.

Installation
~~~~~~~~~~~~

1. Go to the directory of the project.
2. Install the code in development mode.

.. code:: console

    python setup.py develop


Tracking Experiments
~~~~~~~~~~~~~~~~~~~~

This python implementation used ExAssist_ to track each experiments.
Every time you run an experiment, all the output files (include experiment settings and details) will be saved in ``Experiments`` directory. If ``Experiments`` directory does not exist, a new one will be created.

Running Example
--------------

In this subsection, a small example is used to show how to use this repository.
The behavior of our code is controlled by a config file.
After installation, you can directly run our code like:

.. code:: console

    python FeVER/main.py example/config.ini

``config.ini`` file contains all the configuation for running.

Dataset
-------

A toy dataset is stored in the ``example/toy``.
Different files in this directory has different usage.



.. _ExAssist: https://exassist.readthedocs.io/en/latest/

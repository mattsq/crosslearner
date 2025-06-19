Quickstart
==========

Follow these steps to install ``crosslearner`` and train your first model.

Prerequisites
-------------

``crosslearner`` requires **Python 3.10** or later.  We recommend working in a
virtual environment::

   python3 -m venv .venv
   source .venv/bin/activate

Install the dependencies:

.. code-block:: bash

   pip install -r requirements.txt
   pip install torch --extra-index-url https://download.pytorch.org/whl/cpu

For development the `black` formatter and `ruff` linter are useful::

   pip install black ruff

Installation
------------

Once the environment is set up you can install ``crosslearner`` from the
repository root:

.. code-block:: bash

   pip install .

Minimal example
---------------

Run the command line entry point to train on a toy dataset and
report :math:`\sqrt{\mathrm{PEHE}}`:

.. code-block:: bash

   crosslearner-train

This launches a short training loop and prints the final metric.
Loss histories are logged to TensorBoard when available.

Testing
-------

To verify the installation run the test suite::

   pytest --cov=crosslearner --cov-report=xml -q

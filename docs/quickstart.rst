Quickstart
==========

Follow these steps to install ``crosslearner`` and train your first model.

Installation
------------

Install the package from source using ``pip``:

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

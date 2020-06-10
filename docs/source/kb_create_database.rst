Analysing multiple pulses
=========================

We provide a command line tool to collate results from multiple single-pulse
analyse. For help running this tool, see

.. code-block:: console

   $ kb_create_database --help

This will generate a database file, by default kb_database.h5. This database
contains a pandas dataframe and can be read in using

.. code-block:: python

   >>> import pandas as pd
   >>> df = pd.read_hdf("kb_database.h5")

The data base contains the median and standard deviation of all parameters from
all runs, along with summary statistics.

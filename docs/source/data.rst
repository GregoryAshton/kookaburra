Data handling
-------------

Data I/O in kookaburra is handled by the :code:`kookaburra.data` module. In
particular, we provide a class :code:`TimeDomainData` with methods for data
reading and internal modification. See the doctstrings for the :code:`from_*`
methods below for available data formats.

.. autoclass:: kookaburra.data.TimeDomainData
   :members: from_file, from_csv, from_h5

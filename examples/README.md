In this directory, we store a python file to create fake data. This can be run with

  $ python make_fake_data.py

We can then analyse the output a single shapelet and order-1 polynomial:

  $ kb_single_pulse fake_data.txt -p 0 -s 5 -b 2 --plot-fit --nlive 1000

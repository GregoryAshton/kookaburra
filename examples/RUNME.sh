# In this directory, we store a python file to create fake data. This can be run with

python make_fake_data.py

# We can then analyse the output a single shapelet and order-1 polynomial:

kb_single_pulse fake_data.txt -p 0 -s 1 -b 1 --plot-fit --nlive 500

# The fit from this run will be poor, we can improve by adding more components

# kb_single_pulse fake_data.txt -p 0 -s 1 -b 1 --plot-fit --nlive 500 # COMMENT OUT THIS LINE TO RUN


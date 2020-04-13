
# Experiment 1:
echo 'Experiment 1'
python basic_rl.py -ep=0.1
python basic_rl.py -ep=0.15
python basic_rl.py -ep=0.95
python basic_rl.py -ep=0.99

# Experiment 2:
echo 'Experiment 2'
python basic_rl.py -ed=0.9
python basic_rl.py -ed=0.95
python basic_rl.py -ed=0.99

# Experiment 3:
echo 'Experiment 3'
python basic_rl.py -ga=0.95
python basic_rl.py -ga=0.99
python basic_rl.py -ga=0.995

# Experiment 4
echo 'Experiment 4'
python basic_rl.py -al=0.1
python basic_rl.py -al=0.9

# Experiment 5
echo 'Experiment 5'
python basic_rl.py -al=0.9 -ga=0.99 -ep=0.1 -ed=0.99
python basic_rl.py -al=0.9 -ga=0.99 -ep=0.1 -ed=0.999
python basic_rl.py -al=0.9 -ga=0.99 -ep=0.99 -ed=0.99
python basic_rl.py -al=0.9 -ga=0.99 -ep=0.99 -ed=0.999
python basic_rl.py -al=0.1 -ga=0.99 -ep=0.1 -ed=0.99
python basic_rl.py -al=0.1 -ga=0.99 -ep=0.1 -ed=0.999
python basic_rl.py -al=0.1 -ga=0.99 -ep=0.99 -ed=0.99
python basic_rl.py -al=0.1 -ga=0.99 -ep=0.99 -ed=0.999
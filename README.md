# DDPG with Expert Policy and Curriculum Learning

This repo runs the DDPG algorithm for training on the gym-truck-backerupper custom OpenAI gym environment. This is NOT a vanilla DDPG algorithm as certain hyperparameters are tuned and some complex machine teaching strategies were employed. The Ornstein-Uhlenbeck noise is used for exploration, but was found to be insufficient for a really simply straight line path for the tractor-trailer. Since DDPG is an off-policy algorithm, an expert policy is used `50%` of the time and is decayed once convergence starts. The rest of the time, the DDPG algorithm predicts and some exploration noise is added.

## Dependencies

```
tensorflow==1.12
gym-truck-backerupper
ast
collections
```

## What to expect

The code is designed so that one can run the code through a list of initial seeds to combat the stochasticity with training. It is recommended to at least run 3 before making any decision, but more is better. The code will run until the convergence criteria is met, which basically consists of a percent error change of less than `5%` and the normalized reward is larger than a continuously checked value. 

The convergence criteria is NOT a part of the vanilla DDPG algorithm, however, early stopping was found to be necessary and allowed the user to sleep at night without worrying about stopping training at the correct moment.

For each initial seed, the weights and biases will be saved as a checkpoint file. Once it is saved, a new tensorflow session is created, loaded up, and evaluated. This is continued until all the seeds in the list is changed

```
SEEDS = [0, 1, 12]
```

## Usage

Train with random tracks, completely.
```
>>> python3 DDPG.py
```

Train with paths specified in `.txt` file, where the scripts will randomly select from only paths from the file.

```
>>> python3 DDPG.py lesson_plan.txt
```

Resume training using previous weights and biases

```
>>> python3 DDPG.py models/<modelname>
```

NOTE: You can run both as arguments at the same time

## Rendering

Rendering does not occur by default because it speeds up training time. Simply type into the CLI the following and press enter. It does not matter that things print out, just type it fast enough and press enter.

```
>>> render
```

If you no longer want to look at rendering, simply type the following again

```
>>> hide
```

## Hyperparameters
```
GAMMA = 0.99
ALPHA_C = .001
ALPHA_A = .0001
EPISODES = 10000
MAX_BUFFER = 1e6
BATCH_SIZE = 64
COPY_STEPS = 1
TRAIN_STEPS = 1
WARM_UP = 10000
N_NEURONS1 = 400
N_NEURONS2 = 300
TAU = .001
BN = False
L2 = False
```

## Settle

If training is not converging and you want to save the model where it is currently at, simply type the following into the CLI even though things are printing out. Press enter.

```
>>> settle
```


## Decay
This will trigger an early convergence where the probability of using the expert policy will decay. It may not save a model if the convergence is not met and will thus reset the expert policy. Type the following into the CLI even though things are printing out. Press enter.

```
>>> decay
```



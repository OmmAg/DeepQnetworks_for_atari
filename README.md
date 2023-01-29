# DeepQnetworks_for_atari

# Steps to run the implementation:
1. Download the folder
2. Create a virtual enviroment and download the required library versions using requirements.txt(ignore any warnings)

# Running the cartpole game:
Simply run the dqn.py python script to train the model to play the classic cartpole game. After it reaches a average reward of 195(which can be changed), the observation automatically starts and you can watch the model play the game

# Running the breakout game:
Run the newdqn.py python script to train the model to play the breakout game. 
The model should be trained for atleast 3 hours
An automatic file will be generated with the name ./atari_model.pack with the saved model parameters
Then run the observe.py file to see the model play the game

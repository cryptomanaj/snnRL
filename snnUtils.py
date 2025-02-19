from collections import deque, namedtuple
import os

from tqdm import tqdm
import pandas as pd
import random, imageio, time, copy
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

import torch.nn as nn
import torch
import snntorch as snn
from snntorch import spikeplot as splt
from snntorch import spikegen


class ReplayMemory(object):
    """
    Implement's the replay memory algorithm.
    """
    def __init__(self, size, dtype, device) -> None:
        """
        Initialize the class with a double ended queue which will contain named tuples.

        Args:
            size (int): The maximum size of the memory buffer.
            dtype (torch.dtype): The data type of the elements in the memory buffer.
            device (torch.device): The device to store the data (CPU or GPU)
        """
        self.exp = deque([], maxlen=size)
        self.size = size # SIze of the memory
        self.len = len(self.exp)
        self.dtype = dtype
        self.device = device

    def exportExpereince(self):
        """
        Exports the experiences to a dictionary of lists.

        Returns:
            Returns a dictionary containing keys "state, action, reward, nextState, done"
        """
        __state = [e.state for e in self.exp]
        __action = [e.action for e in self.exp]
        __reward = [e.reward for e in self.exp]
        __nextState = [e.nextState for e in self.exp]
        __done = [e.done for e in self.exp]

        return {
            "state": __state,
            "action": __action,
            "reward": __reward,
            "nextState": __nextState,
            "done": __done
        }
    
    def loadExperiences(self, state, action, reward, nextState, done):
        """
        Loads previous experiences into the ReplayMemory object.

        Args:   
            Lists of experiences, with the same length
            
        Returns:
            None
        """
        try:
            __experiences = zip(state, action, reward, nextState, done)
            __tempTuple = namedtuple("exp", ["state", "action", "reward", "nextState", "done"])
            __tempDeque = deque([], maxlen = self.size)

            for __state, __action, __reward, __nextState, __done in __experiences:
                __tempDeque.appendleft(
                    __tempTuple(
                        __state, # Current state
                        __action,
                        __reward, # Current state's reward
                        __nextState, # Next state
                        __done
                    )
                )
            
            self.exp = __tempDeque
            self.len = len(__tempDeque)
        except:
            print("Could not load the data to ReplayMemory object")

    def addNew(self, exp:namedtuple) -> None:
        """
        Adding a new iteration to the memory. Note that the most recent values of
        the training will be located at the position 0 and if the list reaches maxlen
        the oldest data will be dropped.

        Args:
            exp: namedtuple: The experience should be a named tuple with keys named
                like this: ["state", "action", "reward", "nextState", "done"]
        """
        self.exp.appendleft(exp)
        self.len = len(self.exp)

    def sample(self, miniBatchSize:int, framework = "pytorch") -> tuple:
        """
        Get a random number of experiences from the entire experience memory.
        The memory buffer is a double ended queue (AKA deque) of named tuples. To make
        this list usable for tensor flow neural networks, this each named tuple inside
        the deque has to be unpacked. we use a iterative method to unpack. It may be
        inefficient and maybe using pandas can improve this process. one caveat of using
        pandas tables instead of deque is expensiveness of appending/deleting rows
        (experiences) from the table.

        Args:
            miniBatchSize: int: The size of returned the sample

        Returns:
            A tuple containing state, action, reward, nextState and done
        """
        if framework == "pytorch":
            miniBatch = random.sample(self.exp, miniBatchSize)
            state = torch.from_numpy(np.array([e.state for e in miniBatch if e != None])).to(self.device, dtype = self.dtype)
            action = torch.from_numpy(np.array([e.action for e in miniBatch if e != None])).to(self.device, dtype = torch.int)
            reward = torch.from_numpy(np.array([e.reward for e in miniBatch if e != None])).to(self.device, dtype = self.dtype)
            nextState = torch.from_numpy(np.array([e.nextState for e in miniBatch if e != None])).to(self.device, dtype = self.dtype)
            done = torch.from_numpy(np.array([e.done for e in miniBatch if e != None]).astype(np.uint8)).to(self.device, dtype = torch.int)
        elif framework == "tensorflow":
            miniBatch = random.sample(self.exp, miniBatchSize)
            state = tf.convert_to_tensor(np.array([e.state for e in miniBatch if e != None]), dtype=tf.float32)
            action = tf.convert_to_tensor(np.array([e.action for e in miniBatch if e != None]), dtype=tf.float32)
            reward = tf.convert_to_tensor(np.array([e.reward for e in miniBatch if e != None]), dtype=tf.float32)
            nextState = tf.convert_to_tensor(np.array([e.nextState for e in miniBatch if e != None]), dtype=tf.float32)
            done = tf.convert_to_tensor(np.array([e.done for e in miniBatch if e != None]).astype(np.uint8), dtype=tf.float32)
        return tuple((state, action, reward, nextState, done))

def decayEbsilon(currE: float, rate:float, minE:float) -> float:
    """
    Decreases ebsilon each time called. It multiplies current ebsilon to decrease rate.
    The decreasing is continued until reaching minE.
    """
    return(max(currE*rate, minE))

def computeLoss(experiences:tuple, gamma:float, qNetwork, target_qNetwork):
    """
    Computes the loss between y targets and Q values. For target network, the Q values are
    calculated using Bellman equation. If the reward of current step is R_i, then y = R_i
    if the episode is terminated, if not, y = R_i + gamma * Q_hat(i+1) where gamma is the
    discount factor and Q_hat is the predicted return of the step i+1 with the
    target_qNetwork.

    For the primary Q network, Q values are acquired from the step taken in the episode
    experiences (Not necessarily MAX(Q value)).

    Args:
        experiences (Tuple): A tuple containing experiences as pytorch tensors.
        gamma (float): The discount factor.
        qNetwork (pytorch NN): The neural network for predicting the Q.
        target_qNetwork (pytorch NN): The neural network for predicting the target-Q.

    Returns:
        loss: float: The Mean squared errors (AKA. MSE) of the Qs.
    """
    # Unpack the experience mini-batch
    state, action, reward, nextState, done = experiences

    # with torch.no_grad():
    target_qNetwork.eval()
    qNetwork.eval()

    # To implement the calculation scheme explained in comments, we multiply Qhat by (1-done).
    # If the episode has terminated done == True so (1-done) = 0.
    Qhat = torch.amax(target_qNetwork(nextState), dim = 1)
    yTarget = reward + gamma *  Qhat * ((1 - done)) # Using the bellman equation

    # IMPORTANT: When getting qValues, we have to account for the ebsilon-greedy algorithm as well.
    # This is why we dont use max(qValues in each state) but instead we use the qValues of the taken
    # action in that step.
    qValues = qNetwork(state)

    qValues = qValues[torch.arange(state.shape[0], dtype = torch.long), action]

    # Calculate the loss
    loss = nn.functional.mse_loss(qValues, yTarget)

    return loss

def fitQNetworks(experience, gamma, qNetwork, target_qNetwork):
    """
    Updates the weights of the neural networks with a custom training loop. The target network is
    updated by a soft update mechanism.

    Args:
        experience (tuple): The data for training networks. This data has to be passed with
            replayMemory.sample() function which returns a tuple of tensorflow tensors in
            the following order: state, action, reward, nextState, done)
        gamma (float): The learning rate.
        qNetwork, target_qNetwork (list): A list of pytorch model and its respective
            optimizer. The first member should be the model, second one its optimizer

    Returns:
        None
    """
    __qNetworkModel = qNetwork[0]
    __qNetworkOptim = qNetwork[1]
    __targetQNetworkModel = target_qNetwork[0]

    # Update the Q network's weights
    loss = computeLoss(experience, gamma, __qNetworkModel, __targetQNetworkModel)

    __qNetworkModel.train()
    __targetQNetworkModel.train()

    __qNetworkOptim.zero_grad()
    loss.backward()
    __qNetworkOptim.step()

    # Update the target Q network's weights using soft updating method
    for targetParams, primaryParams in zip(__targetQNetworkModel.parameters(), __qNetworkModel.parameters()):
        targetParams.data.copy_(targetParams.data * (1 - .001) + primaryParams.data * .001)

def getAction(qVal: list, e:float) -> int:
    """
    Gets the action via an epsilon-greedy algorithm. This entire action state depends on the env.
    With a probability of epsilon, a random choice will be picked, else the action with
    the greatest Q value will be picked.

    Args:
        qVal: list: The q value of actions
        e: float: The epsilon which represents the probability of a random action

    Returns:
        action_: int: 0 for doing nothing, and 1 for left thruster, 2 form main thruster
            and 3 for right thruster.
    """
    rnd = random.random()

    # The actions possible for LunarLander i.e. [DoNothing, leftThruster, MainThruster, RightThruster]
    actions = [0, 1, 2, 3]

    if rnd < e:
        # Take a random step
        action_ = random.randint(0,3)
    else:
        action_ = actions[torch.argmax(qVal)]

    return action_

def updateNetworks(timeStep: int, replayMem: ReplayMemory, miniBatchSize: int, C: int) -> bool:
    """
    Determines if the neural network (qNetwork and target_qNetwork) weights are to be updated.
    The update happens C time steps apart. for performance reasons.

    Args:
        timeStep: int: The time step of the current episode
        replayMem: deque: A double edged queue containing the experiences as named tuples.
            the named tuples should be as follows: ["state", "action", "reward", "nextState", "done"]

    Returns:
        A boolean, True for update and False to not update.
    """

    return True if ((timeStep+1) % C == 0 and miniBatchSize < replayMem.len) else False

def getEbsilon(e:float, eDecay:float, minE: float) -> float:
    """
    Decay epsilon for epsilon-Greedy algorithm. epsilon starts with 1 at the beginning of the
    learning process which indicates that the agent completely acts on a random basis (AKA
    Exploration) but as the learning is continued, the rate at which agent acts randomly decreased
    via multiplying the epsilon by a decay rate which ensures agent acting based on it's learnings
    (AKA Exploitation).

    Args:
        e: float: The current rate of epsilon
        eDecay: float: The decay rate of epsilon
        minE: float: the minimum amount of epsilon. To ensure the exploration possibility of the
            agent, epsilon should't be less than a certain amount.

    Returns: epsilon's value
    """

    return max(minE, eDecay * e)

def renderEpisode(initialState: int, actions:str, envName:str, delay:float = .02) -> None:
    """
    Renders the previously done episode so the user can see what happened. We use Gym to
    render the environment. All the render is done in the "human" mode.

    Args:
        initialState: int: The initial seed that determine's the initial state of the episode
            (The state before we took teh first action)
        actions: string: A string of actions delimited by comma (i.e. 1,2,3,1,3, etc.)
        env: string: The name of the environment to render the actions, It has to be a gymnasium
            compatible environment.
        delay: int: The delay (In seconds) to put between showing each step to make it more
            comprehensive.

    Returns: None
    """
    tempEnv = gym.make(envName, render_mode = "human") # Use render_mode = "human" to render each episode
    state, info = tempEnv.reset(seed=initialState) # Get a sample state of the environment

    # Process the string of actions taken
    actions = actions.split(",") # Split the data
    actions = actions[:-1] # Remove the lat Null member of the list
    actions = list(map(int, actions)) # Convert the strings to ints

    # Take steps
    for action in actions:
        _, _, terminated, truncated, _ = tempEnv.step(action)

        # Exit loop if the simulation has ended
        if terminated or truncated:
            _, _ = tempEnv.reset()
            break

        # Delay showing the next step
        time.sleep(delay)

    tempEnv.close()

def analyzeLearning(episodePointHistory:list, episodeTimeHistory:list) -> None:
    """
    Plots the learning performance of the agent

    Args:
        episodePointHistory: list: The commulative rewards of each episode in consrcutive time steps.
        episodeTimeHistory: list: The time it took to run the episode
    """
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(15, 5))
    ax1.plot(episodePointHistory)
    ax1.set_title("Episode points")

    ax2.plot(episodeTimeHistory)
    ax2.set_title("Episode elapsed time")

def testAgent(envName:str, network, __device, __dtype, saveVideoName:str = "", ) -> int:
    """
    Runs an agent through a predefined gymnasium environment. The actions of the agent are chosen via
    a greedy policy by a trained neural network. To see the agent in action, the environment's render
    mode has to be "human" or  "rgb-array"

    Args:
        envName: string: The name of the environment.
        network: pytorch NN: The trained neural network that accepts state as an input and outputs
            the desired action.
        environment: gymnasium env: The environment for testing.
        saveVideoName:string: The name of the file to be saved. If equals "", No video file will be
            saved; Also remember that the file name should include the file extension.
    """

    def interactionLoop(env_, seed_, V_):
        """
        The loop that lets agent interact with the environment.
        if V_ == True, save the video (requires render_mode == rgb_array)
        """
        state, _ = env_.reset(seed = seed_)
        points = 0
        if V_:
            videoWriter = imageio.get_writer(saveVideoName)

        maxStepN = 1000
        for t in range(maxStepN):
            # Take greedy steps
            # action = np.argmax(network(np.expand_dims(state, axis = 0)))

            action = torch.argmax(network(torch.tensor(state, device = __device, dtype = __dtype)))

            state, reward, terminated, truncated, _ = env_.step(action.item())

            if V_:
                videoWriter.append_data(env_.render())

            points += reward

            # Exit loop if the simulation has ended
            if terminated or truncated:
                _, _ = env_.reset()

                if V_:
                    videoWriter.close()

                return points

    # Get the random seed to get the initial state of the agent.
    seed = random.randint(0, 1_000_000_000)

    # Because gymnasium doesn't let the environment to have two render modes,
    # we run the simulation twice, The first renders the environment with "human"
    # mode and the second run, runs the environment with "egb_array" mode that
    # lets us save the interaction process to a video file. Both loops are run
    # with the same seeds and neural networks so they should have identical outputs.
    environment = gym.make(envName, render_mode = "human")
    point = interactionLoop(environment, seed, False)

    environment = gym.make(envName, render_mode = "rgb_array")
    point = interactionLoop(environment, seed, True if saveVideoName != "" else False)

    return point
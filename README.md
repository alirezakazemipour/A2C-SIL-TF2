[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com)  

# A2C-SIL-TF2

This repository is a TensorFlow2 implementation of **Self-Imitation Learning (SIL)** with **Synchronous Advantage Actor-Critic (A2C)** as the backbone of action-selection policy.

In short, the SIL builds on the intuition that good past experiences should be exploited more and consequently driving to deep exploration. Thus, the SIL is responsible to improve exploitation of good past decisions and is an auxiliary module that can be added to any action-selection policy.

The paper "Self-Imitation Learning" by Oh et al, combines the SIL with the synchronous version of Advantage Actor-Critic method and showed that the SIL is applicable to any other Actor-Critic methods like Proximal Policy Optimization (PPO).

This repository follows the same procedure of the paper and produced results are obtained by combining the SIL with the A2C.

## Demo

<p align="center">
  <img src="Results/Gifs/Freeway/Freeway.gif">
</p>  

## Results

<p align="center">
  <img src="Results/Plots/Freeway/Running_last_10_Reward.svg">
</p>  

> X-axis corresponds episode numbers
> Blue curve has 0.1 for  value of Bias Correction for Prioritized Experience Replay

Rest of the training plots are at [the end](#results-contd) of the current Readme file.

## Important Notes About This Code!

While the current code tries to put the ideas of the SIL into practice, there are some differences between it and the [original implementation](https://github.com/junhyukoh/self-imitation-learning):
-  The SIL paper claims that it stores past experiences in a Replay Buffer and uses only useful and good memories without any constraints or heuristics about domain-specific configurations.  
However, in the original implementation, transitions of an episode are added to the replay buffer **if their corresponding episode contains at least a single transition with positive reward**. To be specific, [this part](https://github.com/junhyukoh/self-imitation-learning/blob/13eb8a79e9585f92761e0a4670bd76c2e0a7bf05/baselines/common/self_imitation.py#L266) of the original code:
```python
def update_buffer(self, trajectory):
    positive_reward = False
    for (ob, a, r) in trajectory:
        if r > 0:
            positive_reward = True
            break
    if positive_reward:
        self.add_episode(trajectory)
```
This heuristic is only valid for Atari domain with its current DeepMind proposed preprocessing and decreases the generality of the SIL.  

A simple example is enough to underline that the above code is **only an aid** for faster training of Atari games with a specific preprocessing on rewards and decreases the generality:
Suppose we hand engineer the rewards of an Atari game like Pong so that we substarct a -2 from every timestep rewards. This intervention moves rewards' range from [-1, 1] to [-3, -1] without losing the expressiveness of expected optimal behavior that should be encoded within the rewards. But, the above hueristic falis on this new environment and **should be tuned again** (= loss of generality)!

**As a result, the current code avoids appling such domain knowledge about positivity of rewards**. 

- To speedup the training, the currect code benefits from using and LSTMCell and the lighter network architecture introduced in A3C paper instead of the larger DQN network.
- The current code uses Adam as its optimizer and no major violation was seen during training time, thus the clippings of the Advantage function, Log Probabilities and Critic Loss of the original code (shown below) that uses RMSprop, were not required:
``` python
clipped_nlogp = tf.stop_gradient(tf.minimum(nlogp, self.max_nlogp) - nlogp) + nlogp
            
adv = tf.stop_gradient(tf.clip_by_value(self.R - tf.squeeze(model_vf), 0.0, clip))

delta = tf.clip_by_value(v_estimate - v_target, -self.clip, 0) * mask
```
>Not required.

- The Learning Rate is different from the paper.
- **One caveat of the current code is that it can not be executed on [Colab](https://colab.research.google.com/)** since there is an unknown issue that the total amount of avaialable RAM is consumed **despite lowering the replay buffer size** as much as possible, thus the whole training was done on [paperspace.com](https://www.paperspace.com/) **without any problem about the RAM or etc**.

## Table of Hyperparameters
>Hyperparameters used for Pong nad Freeway environments.

Parameter| Value
:-----------------------:|:-----------------------:|
lr			     | 2.5e-4
alpha (Exponent for PER)| 0.6
Beta (Bias Correction for PER)| 0.4
entropy coefficient for A2C | 0.01
critic coefficient for A2C | 0.5
max grad norm	          | 0.5
num of sil updates| 4
sil batch size | 512
crtitc coefficient for SIL | 0.01
entropy coefficient for SIL | 0
num parallel workers| 8
k (rollout length) | 80 // num parallel workers
memory size| 1e+5

## Dependencies

- comet_ml == 3.15.3
- gym == 0.17.3
- numpy == 1.19.2
- opencv_contrib_python == 4.4.0.44
- psutil == 5.5.1
- tensorflow == 2.6.0
- tensorflow_probability == 0.13.0
- tqdm == 4.50.0









## Results (Cont'd)

<p align="center">
  <img src="Results/Plots/Freeway/Max_Episode_Reward.svg">
</p>  

> X-axis corresponds episode numbers
> Blue curve has 0.1 for  value of Bias Correction for Prioritized Experience Replay

<p align="center">
  <img src="Results/Plots/Freeway/Running_Total_Episode_Reward.svg">
</p>  

> X-axis corresponds episode numbers
> Blue curve has 0.1 for  value of Bias Correction for Prioritized Experience Replay

<p align="center">
  <img src="Results/Plots/Freeway/Running_Explained_Variance.svg">
</p>  

> X-axis corresponds iteration numbers
> Blue curve has 0.1 for  value of Bias Correction for Prioritized Experience Replay

<p align="center">
  <img src="Results/Plots/Freeway/Running_Grad_Norm.svg">
</p>  

> X-axis corresponds iteration numbers
> Blue curve has 0.1 for  value of Bias Correction for Prioritized Experience Replay

<p align="center">
  <img src="Results/Plots/Freeway/Running_Entropy.svg">
</p>  

> X-axis corresponds iteration numbers
> Blue curve has 0.1 for  value of Bias Correction for Prioritized Experience Replay

<p align="center">
  <img src="Results/Plots/Freeway/Running_PG_Loss.svg">
</p>  

> X-axis corresponds iteration numbers
> Blue curve has 0.1 for  value of Bias Correction for Prioritized Experience Replay

<p align="center">
  <img src="Results/Plots/Freeway/Running_Value_Loss.svg">
</p>  

> X-axis corresponds iteration numbers
> Blue curve has 0.1 for  value of Bias Correction for Prioritized Experience Replay

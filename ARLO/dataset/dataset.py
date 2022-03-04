"""
This module contains the implementation of the Class BaseDataSet and of the Class TabularDataSet. 

The Class BaseDataSet inherits from the Class AbstractUnit and from ABC, while the Class TabularDataSet inherits from the 
Class BaseDataSet.

The Class BaseDataSet is an abstract Class used as base class for all types of data one can have: tabular data, image data, 
text data.

The Class TabularDataSet contains tabular data in its dataset member.
"""

from abc import ABC

from mushroom_rl.core.environment import MDPInfo
from mushroom_rl.utils.dataset import parse_dataset, arrays_as_dataset

from ARLO.abstract_unit.abstract_unit import AbstractUnit


class BaseDataSet(AbstractUnit, ABC):
    """
    This is the base abstract Class for all the datasets. The idea is that the data can be any: text, image, tabular. 
    This Class inherits from the Class AbstractUnit.
    """
    
    def __init__(self, observation_space, action_space, discrete_actions, discrete_observations, gamma, horizon, obj_name, 
                 seeder=2, log_mode='console', checkpoint_log_path=None, verbosity=3, n_jobs=1, job_type='process'):        
        """
        Parameters
        ----------
        observation_space: This must be a space from MushroomRL like Box, Discrete.
            
        action_space: This must be a space from MushroomRL like Box, Discrete.
       
        discrete_actions: This is either True or False. It is True if the action space is discrete.
        
        discrete_observations: This is either True or False. It is True if the observation space is discrete.
        
        gamma: This is the value of the gamma of the MDP, that is in this class.
        
        horizon: This is the horizon of the MDP, that is in this class.
        
        The other parameters and non-parameters members are described in the Class AbstractUnit.
        """
        
        super().__init__(obj_name=obj_name, seeder=seeder, log_mode=log_mode, checkpoint_log_path=checkpoint_log_path, 
                         verbosity=verbosity, n_jobs=n_jobs, job_type=job_type)
                
        self.observation_space = observation_space
        self.action_space = action_space
        self.discrete_actions = discrete_actions
        self.discrete_observations = discrete_observations
        self.gamma = gamma
        self.horizon = horizon
    
    def __repr__(self):
         return 'BaseDataSet('+'observation_space='+str(self.observation_space)+', action_space='+str(self.action_space)\
                 +', discrete_actions='+str(self.discrete_actions)+', discrete_observations='+str(self.discrete_observations)\
                 +', gamma='+str(self.gamma)+', horizon='+str(self.horizon)+', obj_name='+str(self.obj_name)\
                 +', seeder='+str(self.seeder)+', local_prng='+ str(self.local_prng)+', log_mode='+str(self.log_mode)\
                 +', checkpoint_log_path='+str(self.checkpoint_log_path)+', verbosity='+str(self.verbosity)\
                 +', n_jobs='+str(self.n_jobs)+', job_type='+str(self.job_type)+', logger='+str(self.logger)+')'
                 
    @property
    def info(self):
        """
        This is a property method and it returns an object of Class mushroom_rl.environment.MDPInfo containing the action and
        observation spaces, and the gamma and the horizon of the environment.
        """
        
        #each block must modify the observation_space and action_space according to the transformation they did
        return MDPInfo(self.observation_space, self.action_space, self.gamma, self.horizon)
        
    
class TabularDataSet(BaseDataSet):
    """
    This Class is the generic base Class for tabular data. This Class inherits from the Class BaseDataSet.
    """
    
    def __init__(self, dataset, observation_space, action_space, discrete_actions, discrete_observations, gamma, horizon, 
                 obj_name, seeder=2, log_mode='console', checkpoint_log_path=None, verbosity=3, n_jobs=1, job_type='process'):      
        """
        Parameters
        ----------
        dataset: This must be a list where each entry of the list is: current state, drawn action, reward, next state, 
                 absorbing state flag, episode terminal flag.
                                       
        The other parameters and non-parameters members are described in the Class BaseDataSet.
        """
        
        super().__init__(observation_space=observation_space, action_space=action_space, discrete_actions=discrete_actions,
                         discrete_observations=discrete_observations, gamma=gamma, horizon=horizon, obj_name=obj_name, 
                         seeder=seeder, log_mode=log_mode, checkpoint_log_path=checkpoint_log_path, verbosity=verbosity,
                         n_jobs=n_jobs, job_type=job_type)
        
        self.dataset = dataset
                
        #transform tuple into list. This is needed since tuples are immutable:
        if(isinstance(self.dataset, list)):
            self.tuples_to_lists()
        else:
            wrn_msg = 'You created an object of Class \'TabularDataSet\' but the member \'dataset\' is not a \'list\'!'\
                      +' This means that the tuples in the dataset will not be transformed into lists! Transform it to \'list\''\
                      +' to call the method \'tuples_to_lists\'!'
            self.logger.warning(msg=wrn_msg)

    def __repr__(self):
         #no dataset in this return since it is too long
         return 'TabularDataSet('+'observation_space='+str(self.observation_space)+', action_space='+str(self.action_space)\
                 +', discrete_actions='+str(self.discrete_actions)+', discrete_observations='+str(self.discrete_observations)\
                 +', gamma='+str(self.gamma)+', horizon='+str(self.horizon)+', obj_name='+str(self.obj_name)\
                 +', seeder='+str(self.seeder)+', local_prng='+ str(self.local_prng)+', log_mode='+str(self.log_mode)\
                 +', checkpoint_log_path='+str(self.checkpoint_log_path)+', verbosity='+str(self.verbosity)\
                 +', n_jobs='+str(self.n_jobs)+', job_type='+str(self.job_type)+', logger='+str(self.logger)+')'
    
    def tuples_to_lists(self):
        """
        This method transforms a list of tuples into a list of lists. This is needed since tuples are immutable. This method is
        called in the __init__ only when the dataset parameter is not None. 
        """
        
        if(isinstance(self.dataset, list)):
            for i in range(len(self.dataset)):
                if(isinstance(self.dataset[i], tuple)):
                    self.dataset[i] = list(self.dataset[i])
        else:
            exc_msg = '\'tuples_to_lists \' can be called only if \'dataset\' is a \'list\'!'
            self.logger.exception(msg=exc_msg)
            raise TypeError(exc_msg)
    
    @staticmethod
    def arrays_as_data(states, actions, rewards, next_states, absorbings, lasts):
        """
        Parameters
        ----------
        states: This must be an array containing the states. Each state refers to a single time step.
        
        actions: This must be an array containing the actions. Each action refers to a single time step.
            
        rewards: This must be an array containing the rewards. Each reward refers to a single time step.
            
        next_states: This must be an array containing the next state the agent reaches by taking the sampled action in the 
                     current state.
            
        absorbing: This must be an array containing the flags indicating absorbing states.
            
        lasts: This must be an array containing the flags indicating end of episodes states.

        Returns
        -------
        created_dataset: A list where each element of a list is: current state, current action, current reward, next state, 
        absorbing state flag, episode terminal flag.
        """
        
        created_dataset = arrays_as_dataset(states=states, actions=actions, rewards=rewards, next_states=next_states, 
                                            absorbings=absorbings, lasts=lasts)
        
        return created_dataset
      
    def parse_data(self):
        """
        Returns
        -------
        The six arrays that are obtained by parsing the dataset: states, actions, rewards, next_states, absorbing state flags, 
        episode terminals flags.
        """
        
        #from the dataset we build the arrays of attributes using the mushroom_rl.utils.dataset parse_dataset function it
        #returns: states,actions,rewards,next_states,absorbing,episode_terminals
        return parse_dataset(dataset=self.dataset)
 
    def get_states(self):
        """
        Returns
        -------
        states: the current states array.
        """
        
        states = self.parse_data()[0]
        return states
    
    def get_actions(self):
        """
        Returns
        -------
        actions: the current actions array.
        """
        
        actions = self.parse_data()[1]
        return actions

    def get_rewards(self):
        """
        Returns
        -------
        rewards: the current rewards array.
        """
        
        rewards = self.parse_data()[2]
        return rewards
    
    def get_next_states(self):
        """
        Returns
        -------
        next_states: the next states array.
        """
        
        
        next_states = self.parse_data()[3]
        return next_states
    
    def get_absorbing(self):
        """
        Returns
        -------
        absorbing: the absorbing state flags array.
        """
        
        absorbing = self.parse_data()[4]
        return absorbing
 
    def get_episode_terminals(self):
        """
        Returns
        -------
        episode_terminals: the episode terminals flags array.
        """
        
        episode_terminals = self.parse_data()[5]
        return episode_terminals
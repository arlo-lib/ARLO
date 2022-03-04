"""
This module contains the implementation of the Classes: InputLoader, LoadSameEnv, LoadSameTrainData, 
LoadUniformSubSampleWithReplacement, LoadUniformSubSampleWithReplacementAndEnv, LoadDifferentSizeForEachBlock and 
LoadDifferentSizeForEachBlockAndEnv.

The Class InputLoader inherits from the Class AbstractUnit and from ABC.

The Class InputLoader is an abstract Class used as base class for all types of input loaders. An input loader selects and returns
the some specific input for some specific block for some specific tuner.
"""

from abc import ABC, abstractmethod
import copy

from ARLO.dataset.dataset import TabularDataSet
from ARLO.environment.environment import BaseEnvironment
from ARLO.abstract_unit.abstract_unit import AbstractUnit


class InputLoader(AbstractUnit, ABC):
    """
    This is an abstract Class. It is used as generic base class for all input loaders. These are used to generate the right input 
    to the different blocks that are trained by the Tuner.
        
    These Classes are needed because the Tuner are block agnostic and so we need a way to generate the right input for whichever 
    automatic block might have called the Tuner. 

    This Class inherits from the Class AbstractUnit and from ABC.
    """
    
    def __init__(self, obj_name, seeder=2, log_mode='console', checkpoint_log_path=None, verbosity=3, n_jobs=1, 
                 job_type='process'):
        """
        Parameters
        ----------        
        The other parameters and non-parameters members are described in the Class AbstractUnit.
        """
        
        super().__init__(obj_name=obj_name, seeder=seeder, log_mode=log_mode, checkpoint_log_path=checkpoint_log_path, 
                         verbosity=verbosity, n_jobs=n_jobs, job_type=job_type)
        
        #these two are needed for checking the consistency of the metric with an input_loader. In this abstract class i set both
        #to None, but in the specific Classes I will need to set these either to True or False.
        self.returns_dataset = None
        self.returns_env = None
    
    def __repr__(self):
        return 'InputLoader('+'obj_name='+str(self.obj_name)+', seeder='+str(self.seeder)+', local_prng='+str(self.local_prng)\
               +', log_mode='+str(self.log_mode)+', checkpoint_log_path='+str(self.checkpoint_log_path)\
               +', verbosity='+str(self.verbosity)+', n_jobs='+str(self.n_jobs)+', job_type='+str(self.job_type)\
               +', returns_dataset='+str(self.returns_dataset)+', returns_env='+str(self.returns_env)\
               +', logger='+str(self.logger)+')'
                
    @abstractmethod
    def get_input(self, blocks, n_inputs_to_load, train_data=None, env=None):
        raise NotImplementedError
                
        
class LoadSameEnv(InputLoader):
    """    
    This particular Class simply returns n_inputs_to_load times the copy of the input environment. Indeed for online model 
    generation we just have an environment so we can keep using it, anyway the trajectory followed by the agents depends by the
    particular action taken by each agent.
    """
    
    def __init__(self, obj_name, seeder=2, log_mode='console', checkpoint_log_path=None, verbosity=3, n_jobs=1, 
                 job_type='process'):
        """
        The other parameters and non-parameters members are described in the Class InputLoader.
        """
        
        super().__init__(obj_name=obj_name, seeder=seeder, log_mode=log_mode, checkpoint_log_path=checkpoint_log_path, 
                         verbosity=verbosity, n_jobs=n_jobs, job_type=job_type)
        
        #these two are needed for checking the consistency of the metric with an input_loader:
        self.returns_dataset = False
        self.returns_env = True       
    
    def __repr__(self):
         return 'LoadSameEnv('+'obj_name='+str(self.obj_name)+', seeder='+str(self.seeder)+', local_prng='+str(self.local_prng)\
                +', log_mode='+str(self.log_mode)+', checkpoint_log_path='+str(self.checkpoint_log_path)\
                +', verbosity='+str(self.verbosity)+', n_jobs='+str(self.n_jobs)+', job_type='+str(self.job_type)\
                +', returns_dataset='+str(self.returns_dataset)+', returns_env='+str(self.returns_env)\
                +', logger='+str(self.logger)+')'
    
    def get_input(self, blocks, n_inputs_to_load, train_data=None, env=None):
        """
        Parameters
        ----------        
        blocks: This is a list containing the blocks for which we need to load the input. This is used only in some InputLoaders.
                In this InputLoader it is not used.

        n_inputs_to_load: This is the number of environments that will be deep copied.
                
        train_data: This must be an object of a Class inheriting from the Class BaseDataSet.
            
        env: This must be an object of a Class inheriting from the Class BaseEnvironment.
                
        Returns
        -------
        copied_envs: This is just a list with deep copies of the original environment        
        """
        
        if((env is None) or (not isinstance(env, BaseEnvironment))):
            exc_msg = '\'env\' is \'None\' or is an object of a Class not inheriting from the Class \'BaseEnvironment\'!'    
            self.logger.exception(msg=exc_msg)
            raise TypeError(exc_msg) 
        
        copied_envs = []        
        for n in range(n_inputs_to_load):
            #A simple assignment would only yield a shallow copy so i need to deepcopy the env. Indeed in the line afterwards i 
            #change the member obj_name:
            new_tmp_env = copy.deepcopy(env)
            new_tmp_env.obj_name = str(self.obj_name)+'_'+str(new_tmp_env.obj_name)+'_split_'+str(n)
            copied_envs.append(new_tmp_env)
            
        return None, copied_envs        


class LoadSameTrainData(InputLoader):
    """    
    This particular Class simply returns n_inputs_to_load times the copy of the input train_data. 
    """
    
    def __init__(self, obj_name, seeder=2, log_mode='console', checkpoint_log_path=None, verbosity=3, n_jobs=1, 
                 job_type='process'):
        """
        The other parameters and non-parameters members are described in the Class InputLoader.
        """
        
        super().__init__(obj_name=obj_name, seeder=seeder, log_mode=log_mode, checkpoint_log_path=checkpoint_log_path, 
                         verbosity=verbosity, n_jobs=n_jobs, job_type=job_type)
        
        #these two are needed for checking the consistency of the metric with an input_loader:
        self.returns_dataset = True
        self.returns_env = False               
    
    def __repr__(self):
         return 'LoadSameTrainData('+'obj_name='+str(self.obj_name)+', seeder='+str(self.seeder)\
                +', local_prng='+str(self.local_prng)+', log_mode='+str(self.log_mode)\
                +', checkpoint_log_path='+str(self.checkpoint_log_path)+', verbosity='+str(self.verbosity)\
                +', n_jobs='+str(self.n_jobs)+', job_type='+str(self.job_type)+', returns_dataset='+str(self.returns_dataset)\
                +', returns_env='+str(self.returns_env)+', logger='+str(self.logger)+')'
                
    def get_input(self, blocks, n_inputs_to_load, train_data=None, env=None):
        """   
        Parameters
        ----------               
        blocks: This is a list containing the blocks for which we need to load the input. This is used only in some InputLoaders.
                In this InputLoader it is not used.
            
        n_inputs_to_load: This is the number of datasets that will be sub-sampled.

        train_data: This must be an object of a Class inheriting from the Class TabularDataSet.
            
        env: This must be an object of a Class inheriting from the Class BaseEnvironment.
            
        Returns
        -------
        copied_train_data: This method simply returns copies of the original train_data: A list of objects all equal to 
                           train_data is returned.
        """
    
        copied_train_data = []        
        for n in range(n_inputs_to_load):
            new_train_data = copy.deepcopy(train_data)
            copied_train_data.append(new_train_data)
                    
        return copied_train_data, None


class LoadUniformSubSampleWithReplacement(InputLoader):
    """    
    This particular Class sub-samples with replacement the given dataset by using a uniform distribution. In the end a list of 
    objects of Class TabularDataSet is created where each object has in its member dataset the new sub-sampled dataset.
    """
    
    def __init__(self, obj_name, single_split_length, seeder=2, log_mode='console', checkpoint_log_path=None, verbosity=3, 
                 n_jobs=1, job_type='process'):
        """
        Parameters
        ----------    
        single_split_length: This is the length of each new sub-sampled dataset.
                
        The other parameters and non-parameters members are described in the Class InputLoader.
        """
        
        super().__init__(obj_name=obj_name, seeder=seeder, log_mode=log_mode, checkpoint_log_path=checkpoint_log_path, 
                         verbosity=verbosity, n_jobs=n_jobs, job_type=job_type)
        self.single_split_length = single_split_length
        
        #these two are needed for checking the consistency of the metric with an input_loader:
        self.returns_dataset = True
        self.returns_env = False               
    
    def __repr__(self):
         return 'LoadUniformSubSampleWithReplacement('+'obj_name='+str(self.obj_name)+', seeder='+str(self.seeder)\
                +', local_prng='+str(self.local_prng)+', log_mode='+str(self.log_mode)\
                +', single_split_length='+str(self.single_split_length)+', checkpoint_log_path='+str(self.checkpoint_log_path)\
                +', verbosity='+str(self.verbosity)+', n_jobs='+str(self.n_jobs)+', job_type='+str(self.job_type)\
                +', returns_dataset='+str(self.returns_dataset)+', returns_env='+str(self.returns_env)\
                +', logger='+str(self.logger)+')'
                
    def get_input(self, blocks, n_inputs_to_load, train_data=None, env=None):
        """   
        Parameters
        ----------               
        blocks: This is a list containing the blocks for which we need to load the input. This is used only in some InputLoaders.
                In this InputLoader it is not used.
            
        n_inputs_to_load: This is the number of datasets that will be sub-sampled.

        train_data: This must be an object of a Class inheriting from the Class TabularDataSet.
            
        env: This must be an object of a Class inheriting from the Class BaseEnvironment.
            
        Returns
        -------
        splitted_datasets: This method subsamples with replacement the given dataset by using a uniform distribution. In the end
                           a list of objects of Class TabularDataSet is created where each object has in its member dataset the 
                           new sub-sampled dataset.
        """
        
        if((train_data is None) or (not isinstance(train_data, TabularDataSet))):
            exc_msg = '\'train_data\' is \'None\' or is an object of a Class not inheriting from the Class \'TabularDataSet\'!'    
            self.logger.exception(msg=exc_msg)
            raise TypeError(exc_msg)      
                        
        original_data = train_data.dataset
        
        splitted_datasets = []        
        for n in range(n_inputs_to_load):
            tmp_rows = self.local_prng.choice(range(len(original_data)), size=self.single_split_length, replace=True)

            tmp_split = [original_data[x] for x in tmp_rows]
            new_tmp_dataset = TabularDataSet(dataset=tmp_split, observation_space=train_data.observation_space, 
                                             action_space=train_data.action_space, discrete_actions=train_data.discrete_actions, 
                                             discrete_observations=train_data.discrete_observations, gamma=train_data.gamma, 
                                             horizon=train_data.horizon, 
                                             obj_name=str(self.obj_name)+'_'+str(train_data.obj_name)+'_split_'+str(n),
                                             seeder=train_data.seeder, log_mode=train_data.log_mode, 
                                             checkpoint_log_path=train_data.checkpoint_log_path, verbosity=train_data.verbosity)

            splitted_datasets.append(new_tmp_dataset)
                    
        return splitted_datasets, None
    
    
class LoadUniformSubSampleWithReplacementAndEnv(InputLoader):
    """    
    This particular Class sub-samples with replacement the given dataset by using a uniform distribution. In the end a list of 
    objects of Class TabularDataSet is created where each object has in its member dataset the new sub-sampled dataset.
    Moreover also a list of environments is returned so that this input loader can be used with metrics that use the environment,
    such as the DiscountedReward.
    """
    
    def __init__(self, obj_name, single_split_length, seeder=2, log_mode='console', checkpoint_log_path=None, verbosity=3,
                 n_jobs=1, job_type='process'):
        """
        Parameters
        ----------    
        single_split_length: This is the length of each new sub-sampled dataset.
                
        The other parameters and non-parameters members are described in the Class InputLoader.
        """
        
        super().__init__(obj_name=obj_name, seeder=seeder, log_mode=log_mode, checkpoint_log_path=checkpoint_log_path, 
                         verbosity=verbosity, n_jobs=n_jobs, job_type=job_type)
        self.single_split_length = single_split_length
        
        #these two are needed for checking the consistency of the metric with an input_loader:
        self.returns_dataset = True
        self.returns_env = True               
    
    def __repr__(self):
         return 'LoadUniformSubSampleWithReplacementAndEnv('+'obj_name='+str(self.obj_name)+', seeder='+str(self.seeder)\
                +', local_prng='+str(self.local_prng)+', log_mode='+str(self.log_mode)\
                +', single_split_length='+str(self.single_split_length)+', checkpoint_log_path='+str(self.checkpoint_log_path)\
                +', verbosity='+str(self.verbosity)+', n_jobs='+str(self.n_jobs)+', job_type='+str(self.job_type)\
                +', returns_dataset='+str(self.returns_dataset)+', returns_env='+str(self.returns_env)\
                +', logger='+str(self.logger)+')'
                
    def get_input(self, blocks, n_inputs_to_load, train_data=None, env=None):
        """   
        Parameters
        ----------       
        blocks: This is a list containing the blocks for which we need to load the input. This is used only in some InputLoaders.
                In this InputLoader it is not used.
                
        n_inputs_to_load: This is the number of datasets that will be sub-sampled, which also equals the number of deep copied 
                          environments.

        train_data: This must be an object of a Class inheriting from the Class TabularDataSet.
            
        env: This must be an object of a Class inheriting from the Class BaseEnvironment.
            
        Returns
        -------
        splitted_datasets: This method subsamples with replacement the given dataset by using a uniform distribution. In the end
                           a list of objects of Class TabularDataSet is created where each object has in its member dataset the 
                           new sub-sampled dataset.
                           
        copied_envs: This is just a list with deep copies of the original environment        
        """
        
        data_loader_params = dict(obj_name=str(self.obj_name)+'_data_loader', single_split_length=self.single_split_length,
                                  seeder=self.seeder, log_mode=self.log_mode, checkpoint_log_path=self.checkpoint_log_path,
                                  verbosity=self.verbosity)
        data_loader = LoadUniformSubSampleWithReplacement(**data_loader_params)
        
        env_loader_params = dict(obj_name=str(self.obj_name)+'_env_loader', seeder=self.seeder, log_mode=self.log_mode, 
                                 checkpoint_log_path=self.checkpoint_log_path, verbosity=self.verbosity)
        env_loader = LoadSameEnv(**env_loader_params)
        
        splitted_datasets = data_loader.get_input(blocks=blocks, n_inputs_to_load=n_inputs_to_load, train_data=train_data)[0]
        copied_envs = env_loader.get_input(blocks=blocks, n_inputs_to_load=n_inputs_to_load, env=env)[1]
        
        return splitted_datasets, copied_envs
    

class LoadDifferentSizeForEachBlock(InputLoader):
    """    
    This particular Class sub-samples with replacement the given dataset by using a uniform distribution, but it extracts for
    each block a different number of samples. In the end a list of objects of Class TabularDataSet is created where each object 
    has in its member dataset the new sub-sampled dataset.
    """
    
    def __init__(self, obj_name, seeder=2, log_mode='console', checkpoint_log_path=None, verbosity=3, n_jobs=1, 
                 job_type='process'):
        """      
        The other parameters and non-parameters members are described in the Class InputLoader.
        """
        
        super().__init__(obj_name=obj_name, seeder=seeder, log_mode=log_mode, checkpoint_log_path=checkpoint_log_path, 
                         verbosity=verbosity, n_jobs=n_jobs, job_type=job_type)
        
        #these two are needed for checking the consistency of the metric with an input_loader:
        self.returns_dataset = True
        self.returns_env = False               
    
    def __repr__(self):
         return 'LoadDifferentSizeForEachBlock('+'obj_name='+str(self.obj_name)+', seeder='+str(self.seeder)\
                +', local_prng='+str(self.local_prng)+', log_mode='+str(self.log_mode)\
                +', checkpoint_log_path='+str(self.checkpoint_log_path)+', verbosity='+str(self.verbosity)\
                +', n_jobs='+str(self.n_jobs)+', job_type='+str(self.job_type)+', returns_dataset='+str(self.returns_dataset)\
                +', returns_env='+str(self.returns_env)+', logger='+str(self.logger)+')'
                
    def get_input(self, blocks, n_inputs_to_load, train_data=None, env=None):
        """   
        Parameters
        ----------               
        blocks: This is a list containing the blocks for which we need to load the input. This is used only in some InputLoaders.
                In this InputLoader it is used.
                
        n_inputs_to_load: This is the number of datasets that will be sub-sampled.

        train_data: This must be an object of a Class inheriting from the Class TabularDataSet.
            
        env: This must be an object of a Class inheriting from the Class BaseEnvironment.
            
        Returns
        -------
        splitted_datasets: This method subsamples with replacement the given dataset by using a uniform distribution, but it 
                           extracts for each block a different number of samples. In the end a list of objects of Class 
                           TabularDataSet is created where each object has in its member dataset the new sub-sampled dataset.
        """
        
        if((train_data is None) or (not isinstance(train_data, TabularDataSet))):
            exc_msg = '\'train_data\' is \'None\' or is an object of a Class not inheriting from the Class \'TabularDataSet\'!'    
            self.logger.exception(msg=exc_msg)
            raise TypeError(exc_msg)      
            
        block_sizes = []
        for n in range(n_inputs_to_load):
            n_th_block_params = blocks[n].get_params()
            block_sizes.append(n_th_block_params['n_train_samples'].current_actual_value)
                        
        original_data = train_data.dataset
        
        splitted_datasets = []        
        for n in range(n_inputs_to_load):
            tmp_rows = self.local_prng.choice(range(len(original_data)), size=block_sizes[n], replace=True)

            tmp_split = [original_data[x] for x in tmp_rows]
            new_tmp_dataset = TabularDataSet(dataset=tmp_split, observation_space=train_data.observation_space, 
                                             action_space=train_data.action_space, discrete_actions=train_data.discrete_actions, 
                                             discrete_observations=train_data.discrete_observations, gamma=train_data.gamma, 
                                             horizon=train_data.horizon, 
                                             obj_name=str(self.obj_name)+'_'+str(train_data.obj_name)+'_split_'+str(n),
                                             seeder=train_data.seeder, log_mode=train_data.log_mode, 
                                             checkpoint_log_path=train_data.checkpoint_log_path, verbosity=train_data.verbosity)
            splitted_datasets.append(new_tmp_dataset)
                    
        return splitted_datasets, None  
    

class LoadDifferentSizeForEachBlockAndEnv(InputLoader):
    """    
    This particular Class sub-samples with replacement the given dataset by using a uniform distribution, but it extracts for 
    each block a different nmber of samples. In the end a list of objects of Class TabularDataSet is created where each object 
    has in its member dataset the new sub-sampled dataset. Moreover also a list of environments is returned so that this input 
    loader can be used with metrics that use the environment, such as the DiscountedReward.
    """
    
    def __init__(self, obj_name, seeder=2, log_mode='console', checkpoint_log_path=None, verbosity=3,
                 n_jobs=1, job_type='process'):
        """         
        The other parameters and non-parameters members are described in the Class InputLoader.
        """
        
        super().__init__(obj_name=obj_name, seeder=seeder, log_mode=log_mode, checkpoint_log_path=checkpoint_log_path, 
                         verbosity=verbosity, n_jobs=n_jobs, job_type=job_type)
        
        #these two are needed for checking the consistency of the metric with an input_loader:
        self.returns_dataset = True
        self.returns_env = True               
    
    def __repr__(self):
         return 'LoadDifferentSizeForEachBlockAndEnv('+'obj_name='+str(self.obj_name)+', seeder='+str(self.seeder)\
                +', local_prng='+str(self.local_prng)+', log_mode='+str(self.log_mode)\
                +', checkpoint_log_path='+str(self.checkpoint_log_path)+', verbosity='+str(self.verbosity)\
                +', n_jobs='+str(self.n_jobs)+', job_type='+str(self.job_type)+', returns_dataset='+str(self.returns_dataset)\
                +', returns_env='+str(self.returns_env)+', logger='+str(self.logger)+')'
                
    def get_input(self, blocks, n_inputs_to_load, train_data=None, env=None):
        """   
        Parameters
        ----------        
        blocks: This is a list containing the blocks for which we need to load the input. This is used only in some InputLoaders.
                In this InputLoader it is used.
        
        n_inputs_to_load: This is the number of datasets that will be sub-sampled, which also equals the number of deep copied 
                          environments.

        train_data: This must be an object of a Class inheriting from the Class TabularDataSet.
            
        env: This must be an object of a Class inheriting from the Class BaseEnvironment.
            
        Returns
        -------
        splitted_datasets: This method subsamples with replacement the given dataset by using a uniform distribution. In the end
                           a list of objects of Class TabularDataSet is created where each object has in its member dataset the 
                           new sub-sampled dataset.
                           
        copied_envs: This is just a list with deep copies of the original environment        
        """
        
        data_loader_params = dict(obj_name=str(self.obj_name)+'_data_loader', seeder=self.seeder, log_mode=self.log_mode, 
                                  checkpoint_log_path=self.checkpoint_log_path, verbosity=self.verbosity)
        data_loader = LoadDifferentSizeForEachBlock(**data_loader_params)
        
        env_loader_params = dict(obj_name=str(self.obj_name)+'_env_loader', seeder=self.seeder, log_mode=self.log_mode, 
                                 checkpoint_log_path=self.checkpoint_log_path, verbosity=self.verbosity)
        env_loader = LoadSameEnv(**env_loader_params)
        
        splitted_datasets = data_loader.get_input(blocks=blocks, n_inputs_to_load=n_inputs_to_load, train_data=train_data)[0]
        copied_envs = env_loader.get_input(blocks=blocks, n_inputs_to_load=n_inputs_to_load, env=env)[1]
        
        return splitted_datasets, copied_envs    
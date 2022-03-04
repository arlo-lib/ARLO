"""
This module contains the implementation of the Class BlockOutput. This is used to represent the output of a generic block of a 
Class inheriting from the Class Block.

The Class BlockOutput inherits from the Class AbstractUnit.
"""

from mushroom_rl.policy import DeterministicPolicy

from ARLO.abstract_unit.abstract_unit import AbstractUnit
from ARLO.dataset.dataset import BaseDataSet
from ARLO.environment.environment import BaseEnvironment
from ARLO.policy.policy import BasePolicy


class BlockOutput(AbstractUnit):
    """
    All blocks that can make up the pipeline, and the pipeline itself, return an object of Class BlockOutput. This Class is 
    needed to generalise the handling of the output of a generic block.
    
    This Class inherits from the Class AbstractUnit.
    """
    
    def __init__(self, obj_name, seeder=2, log_mode='console', checkpoint_log_path=None, verbosity=3, n_jobs=1, 
                 job_type='process', train_data=None, env=None, policy=None, policy_eval=None):
        """      
        Parameters
        ----------                                            
        train_data: If specified this must be an object of a Class inheriting from the Class BaseDataSet.
                    
                    The default is None.
            
        env: If specified this must be an object of a Class inheriting from the Class BaseEnvironment.
             
             The default is None.
            
        policy: If specified this must be an object of a Class inheriting from the Class BasePolicy. 
                
                The default is None.
            
        policy_eval: If specified this must represent the evaluation of the policy.
                     
                     The default is None.      
                           
        Non-Parameters Members
        ----------------------
        n_outputs: This is an integer greater than, or equal to, zero and it represents the number of actual outputs that a
                   block wants to save in an object of this Class. 
                   
        The other parameters and non-parameters members are described in the Class AbstractUnit.
        """
        
        super().__init__(obj_name=obj_name, seeder=seeder, log_mode=log_mode, checkpoint_log_path=checkpoint_log_path, 
                         verbosity=verbosity, n_jobs=n_jobs, job_type=job_type)
        
        self.train_data = train_data
        self.env = env
        self.policy = policy
        self.policy_eval = policy_eval
        self.n_outputs = 0
        
        if(self.train_data is not None):
            self.n_outputs += 1
            if(not isinstance(self.train_data, BaseDataSet)):
                exc_msg = '\'train_data\' must be an object of a Class inheriting from Class \'BaseDataSet\'!'
                self.logger.exception(msg=exc_msg)
                raise TypeError(exc_msg)
        if(self.env is not None):
            self.n_outputs += 1
            if(not isinstance(self.env, BaseEnvironment)):
                exc_msg = '\'env\' must be an object of a Class inheriting from Class \'BaseEnvironment\'!'
                self.logger.exception(msg=exc_msg)
                raise TypeError(exc_msg)
        if(self.policy is not None):
            self.n_outputs += 1
            if(not isinstance(self.policy, BasePolicy)):
                exc_msg = '\'policy\' must be an object of a Class inheriting from Class \'BasePolicy\'!'
                self.logger.exception(msg=exc_msg)
                raise TypeError(exc_msg)
        if(self.policy_eval is not None):
            self.n_outputs += 1
            
    def __repr__(self):
        return 'BlockOutput('+'obj_name='+str(self.obj_name)+', seeder='+ str(self.seeder)+', local_prng='+ str(self.local_prng)\
               +', log_mode='+str(self.log_mode)+', checkpoint_log_path='+str(self.checkpoint_log_path)\
               +', verbosity='+str(self.verbosity)+', n_jobs='+str(self.n_jobs)  +', job_type='+str(self.job_type)\
               +', train_data='+str(self.train_data)+', env='+str(self.env)+', policy='+str(self.policy)\
               +', policy_eval='+str(self.policy_eval)+', n_outputs='+str(self.n_outputs)+', logger='+str(self.logger)+')'
               
    def make_policy_deterministic(self):
        """
        This method turns the policy contained in an object of this Class into a deterministic policy.
        """
        
        dict_of_attributes_for_different_policies = {'EpsGreedy': '_approximator',
                                                    'Boltzmann': '_approximator', 
                                                    'Mellowmax': '_outer._approximator', 
                                                    'GaussianPolicy': '_approximator', 
                                                    'DiagonalGaussianPolicy': '_approximator', 
                                                    'StateStdGaussianPolicy': '_mu_approximator', 
                                                    'StateLogStdGaussianPolicy': '_mu_approximator', 
                                                    'OrnsteinUhlenbeckPolicy': '_approximator', 
                                                    'ClippedGaussianPolicy': '_approximator',
                                                    'GaussianTorchPolicy': '_mu',
                                                    'BoltzmannTorchPolicy': '_logits',
                                                    'SACPolicy': '_mu_approximator'
                                                    }
        
        policy_name = str(self.policy.policy.__class__.__name__)
        
        if(policy_name in list(dict_of_attributes_for_different_policies.keys())):
            if(hasattr(self.policy.policy, dict_of_attributes_for_different_policies[policy_name])):
                extracted_approximator = getattr(self.policy.policy, dict_of_attributes_for_different_policies[policy_name])
                
                self.policy.policy = DeterministicPolicy(mu=extracted_approximator)
                self.policy.approximator = extracted_approximator 
            else:
                err_msg = 'The \'policy\' is not recognised: the attribute to extract contained in'\
                          +' \'dict_of_attributes_for_different_policies\' is not in the policy object,'\
                          +' and so it cannot be made deterministic!'
                self.logger.error(msg=err_msg)
        else:
            self.logger.error(msg='The \'policy\' is not recognised and so it cannot be made deterministic!')
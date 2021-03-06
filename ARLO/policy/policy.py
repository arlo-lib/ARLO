"""
This module contains the implementation of the Class: BasePolicy.
"""

from ARLO.abstract_unit.abstract_unit import AbstractUnit 


class BasePolicy(AbstractUnit):  
    """
    This is a generic Class used as container for the output of any ModelGeneration block.
    
    This Class inherits from the Class AbstractUnit. 
    
    If needed this can be sub-classed.    
    """
    
    def __init__(self, policy, regressor_type, obj_name, approximator=None, seeder=2, log_mode='console', 
                 checkpoint_log_path=None, verbosity=3, n_jobs=1, job_type='process'):
        """
        Parameters
        ----------
        policy: This is the policy object that is the output from a specific model generation algorithm.
        
                Generally this should be an object of a Class inheriting from the Class mushroom_rl.policy.Policy. 
        
                Alternatively it can also be an object of any Class that exposes the method draw_action() taking as parameter 
                just a single state.
                     
        regressor_type: This is a string and it can either be: 'action_regressor', 'q_regressor' or 'generic_regressor'. This is
                        used to specify which one of the 3 possible kind of regressor, made available by MushroomRL, has been
                        picked.
                        
                        This is needed in the Metric Classes: depending on the regressor_type we handle the evaluation phase
                        differently. To see an example of how regressor_type is used see the implementation of the Class 
                        DiscountedReward.
                               
        approximator: This is the approximator used in the policy: this may be used when doing batch_evaluation in the 
                      DiscountedReward Class for example. 
                       
                      Generally this should be an object of a Class inheriting from the Class 
                      mushroom_rl.approximators.regressor.Regressor.
                      
                      Alternatively it can also be an object of any Class that exposes the method predict() taking as parameter
                      either a single sample, or multiple samples.
            
                      The default is None.
   
        The other parameters and non-parameters members are described in the Class AbstractUnit.
        """
        
        super().__init__(obj_name=obj_name, seeder=seeder, log_mode=log_mode, checkpoint_log_path=checkpoint_log_path, 
                         verbosity=verbosity, n_jobs=n_jobs, job_type=job_type)
        
        self.policy = policy
        self.regressor_type = regressor_type
        self.approximator = approximator
        
    def __repr__(self):
         return str(self.__class__.__name__)+'('+'policy='+str(self.policy)+', regressor_type='+str(self.regressor_type)\
                +', obj_name='+str(self.obj_name)+', approximator='+str(self.approximator)+', seeder='+str(self.seeder)\
                +', local_prng='+str(self.local_prng)+', log_mode='+str(self.log_mode)\
                +', checkpoint_log_path='+str(self.checkpoint_log_path)+', verbosity='+str(self.verbosity)\
                +', n_jobs='+str(self.n_jobs)+', job_type='+str(self.job_type)+', logger='+str(self.logger)+')'

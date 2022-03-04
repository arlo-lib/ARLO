"""
This module contains the implementation of the Classes: HyperParameter, Numerical, Real, Integer and Categorical.

The Class HyperParameter inherits from the Class AbstractUnit and from ABC.

The Class HyperParameter is an abstract Class used as base class for all types of hyperparameters.

Any block that has to be tuned needs to have as parameters objects of a Class inheriting from the Class HyperParameter.
"""

from abc import ABC, abstractmethod

from ARLO.abstract_unit.abstract_unit import AbstractUnit


class HyperParameter(AbstractUnit, ABC):
    """
    This is an abstract Class. It is used as generic base Class for all hyperparameters. These are used to ease the 
    hyperparameters tuning process. 
    
    The Class HyperParameter inherits from the Class AbstractUnit and from ABC.
    """
    
    def __init__(self, hp_name, current_actual_value, obj_name, seeder=2, log_mode='console', checkpoint_log_path=None, 
                 verbosity=3, n_jobs=1, job_type='process', to_mutate=False):
        """
        Parameters
        ----------
        hp_name: This is a string with the exact name of the hyperparameter described by this class.
        
        current_actual_value: This is the current actual value of the parameter.
                    
        to_mutate: This is either True or False. It is True if we want the Tuner to tune this hyperparameter. It is False 
                   otherwise.
                
        Non-Parameters Members
        ----------------------
        block_owner_flag: This is an integer and it is used as flag to mark that a certain hyperparameter belongs to some 
                          specific block. This is needed for assigning the correct hyperparameters in the set method of the 
                          Class RLPipeline.
                    
                          What happens is that in the Tuner Class I extract the parameters of a block with the get_params method 
                          of that block, I mutate the hyperparameters, and then I call the set_params method of that block. 
                          Now if that block is a pipeline it will be composed of multiple blocks. The member block_owner_flag 
                          allows the set_params method of a pipeline block to set the right hyperparameters in the right blocks.
                    
                          The default is None.

        The other parameters and non-parameters members are described in the Class AbstractUnit.
        """
        
        super().__init__(obj_name=obj_name, seeder=seeder, log_mode=log_mode, checkpoint_log_path=checkpoint_log_path, 
                         verbosity=verbosity, n_jobs=n_jobs, job_type=job_type)

        self.hp_name = hp_name
        self.current_actual_value = current_actual_value                
        self.to_mutate = to_mutate
        
        self.block_owner_flag = None
    
    def __repr__(self):
         return str(self.__class__.__name__)+'('+'hp_name='+str(self.hp_name)\
                +', current_actual_value='+str(self.current_actual_value)+', obj_name='+str(self.obj_name)\
                +', seeder='+str(self.seeder)+', local_prng='+str(self.local_prng)+', log_mode='+str(self.log_mode)\
                +', checkpoint_log_path='+str(self.checkpoint_log_path)+', verbosity='+str(self.verbosity)\
                +', n_jobs='+str(self.n_jobs)+', job_type='+str(self.job_type)+', to_mutate='+str(self.to_mutate)\
                +', logger='+str(self.logger)+', block_owner_flag='+str(self.block_owner_flag)+')'
                
    @abstractmethod
    def mutate(self, first_mutation):
        raise NotImplementedError
            

class Numerical(HyperParameter):
    """
    This is a generic abstract Class for all Numerical hyperparameters. 
    """
    
    def __init__(self, hp_name, current_actual_value, obj_name, seeder=2, log_mode='console', checkpoint_log_path=None, 
                 verbosity=3, n_jobs=1, job_type='process', range_of_values=None, to_mutate=False, 
                 type_of_mutation='perturbation'):
        """      
        Parameters
        ----------
        range_of_values: This is a list with two numbers: the bounds for the sensible values of the hyperparameter we are 
                         considering. 
                              
        type_of_mutation: This is a string and it is either 'perturbation' or 'mutation': if 'perturbation' then the next 
                          possible values, occurring as a result of a mutation, can only be in the range: 
                          [0.8*current_actual_value to 1.2*current_actual_value]
                          
                          If 'mutation' then the next possible values, occurring as a result of a mutation, can be in the entire
                          range of the possible values of the hyperparameter.
                          
                          Selecting 'perturbation' we have less dramatic changes in the hyperparameters from one step to the 
                          next.
                                                    
                          The default is 'perturbation'.
                          
        The other parameters and non-parameters members are described in the Class HyperParameter.
        """
        
        super().__init__(hp_name=hp_name, current_actual_value=current_actual_value, obj_name=obj_name, seeder=seeder,
                         log_mode=log_mode, checkpoint_log_path=checkpoint_log_path, verbosity=verbosity, n_jobs=n_jobs,
                         job_type=job_type, to_mutate=to_mutate)
        
        self.range_of_values = range_of_values
        #if you do not specify the range_of_values then you cannot mutate the hyperparameter
        if(self.range_of_values is None):
            self.to_mutate = False
            
        self.type_of_mutation = type_of_mutation
        if(self.type_of_mutation not in ['perturbation', 'mutation']):
            exc_msg = 'The member \'type_of_mutation\' can either be: \'perturbation\' or \'mutation\'!'
            self.logger.exception(msg=exc_msg)
            raise ValueError(exc_msg)
    
    def __repr__(self):
         return str(self.__class__.__name__)+'('+'hp_name='+str(self.hp_name)\
                +', current_actual_value='+str(self.current_actual_value)+', obj_name='+str(self.obj_name)\
                +', seeder='+str(self.seeder)+', local_prng='+str(self.local_prng)+', log_mode='+str(self.log_mode)\
                +', checkpoint_log_path='+str(self.checkpoint_log_path)+', verbosity='+str(self.verbosity)\
                +', n_jobs='+str(self.n_jobs)+', job_type='+str(self.job_type)+', range_of_values='+str(self.range_of_values)\
                +', to_mutate='+str(self.to_mutate)+', type_of_mutation='+str(self.type_of_mutation)\
                +', logger='+str(self.logger)+', block_owner_flag='+str(self.block_owner_flag)+')'
                
    @abstractmethod
    def mutate(self, first_mutation):
        raise NotImplementedError  


class Real(Numerical):
    """
    This is the Class representing all Numerical and Real (i.e: Continuous) hyperparameters.
    """
                
    def mutate(self, first_mutation):
        """
        Parameters
        ----------
        first_mutation: This is True if this is the first time mutating the hyper-parameter, else it is False.
        
        This method updates the value of the member current_actual_value by sampling from a continuous uniform distribution.
        """
        
        #I need to change the value of the hyperparameter only if to_mutate = True
        if(self.to_mutate):      
            if(self.type_of_mutation == 'perturbation'):
                #selects uniformly over the range of values if this is the first mutation, else i just mutate in the
                #range [0.8, 1.2] of the current actual value. This is to avoid too big jumps: it is just a perturbation of the 
                #current value
                if(first_mutation):
                    low = self.range_of_values[0]
                    high = self.range_of_values[1]                
                else:
                    #low must be in the range_of_values:
                    low = max(self.current_actual_value*0.8, self.range_of_values[0])
                    
                    #high must be in the range_of_values:
                    high = min(self.current_actual_value*1.2, self.range_of_values[1])
            elif(self.type_of_mutation == 'mutation'):
                 low = self.range_of_values[0]
                 high = self.range_of_values[1]     
            
            #cast to float: I want to use int e float not numpy ints or numpy floats. This is purely for consistency:
            self.current_actual_value = float(self.local_prng.uniform(low=low, high=high))
            
        
class Integer(Numerical):
    """
    This is the Class representing all Numerical and Integer (i.e: Discrete) hyperparameters.
    """
                
    def mutate(self, first_mutation):
        """
        Parameters
        ----------
        first_mutation: This is True if this is the first time mutating the hyper-parameter, else it is False.
        
        This method updates the value of the member current_actual_value by sampling from a discrete uniform distribution.
        """
        
        #I need to change the value of the hyperparameter only if to_mutate = True
        if(self.to_mutate):  
            if(self.type_of_mutation == 'perturbation'):                
                #selects uniformly over the range of values if this is the first mutation, else i just mutate in the
                #range [0.8, 1.2] of the current actual value. This is to avoid too big jumps: it is just a perturbation of the
                #current value          
                if(first_mutation):
                    low = self.range_of_values[0]
                    high = self.range_of_values[1]
                else:
                    #low must be in the range_of_values:
                    low = max(int(self.current_actual_value*0.8), self.range_of_values[0])
                            
                    #high must be in the range_of_values:
                    high = min(int(self.current_actual_value*1.2), self.range_of_values[1])
            elif(self.type_of_mutation == 'mutation'):
                low = self.range_of_values[0]
                high = self.range_of_values[1]     
            
            #cast to int: I want to use int e float not numpy ints or numpy floats. This is purely for consistency:
            self.current_actual_value = int(self.local_prng.integers(low=low, high=high, endpoint=True))
    
        
class Categorical(HyperParameter):
    """
    This is the Class representing all Categorical hyperparameters.
    """
    
    def __init__(self, hp_name, current_actual_value, obj_name, seeder=2, log_mode='console', checkpoint_log_path=None, 
                 verbosity=3, n_jobs=1, job_type='process', possible_values=None, to_mutate=False):                                  
        """          
        Parameters
        ----------
        possible_values: This is an exhaustive list of all possible values.
        
        The other parameters and non-parameters members are described in the Class HyperParameter.
        """
        
        super().__init__(hp_name=hp_name, current_actual_value=current_actual_value, obj_name=obj_name, seeder=seeder, 
                         log_mode=log_mode, checkpoint_log_path=checkpoint_log_path, verbosity=verbosity, n_jobs=n_jobs,
                         job_type=job_type, to_mutate=to_mutate)
        
        self.possible_values = possible_values
        #you can avoid to specify the possible_values but then you cannot mutate the hyperparameter
        if(self.possible_values is None):
            self.to_mutate = False

    def __repr__(self):
        return 'Categorical('+'hp_name='+str(self.hp_name)+', current_actual_value='+str(self.current_actual_value)\
               +', obj_name='+str(self.obj_name)+', seeder='+str(self.seeder)+', local_prng='+str(self.local_prng)\
               +', log_mode='+str(self.log_mode)+', checkpoint_log_path='+str(self.checkpoint_log_path)\
               +', verbosity='+str(self.verbosity)+', n_jobs='+str(self.n_jobs)+', job_type='+str(self.job_type)\
               +', possible_values='+str(self.possible_values)+', to_mutate='+str(self.to_mutate)+', logger='+str(self.logger)\
               +', block_owner_flag='+str(self.block_owner_flag)+')'
                
    def mutate(self, first_mutation):
        """
        Parameters
        ----------
        first_mutation: This is True if this is the first time mutating the hyper-parameter, else it is False.
        
        This method updates the value of the member current_actual_value by sampling from the possible_values.
        """
        
        #I need to change the value of the hyperparameter only if to_mutate = True
        if(self.to_mutate):
            #mutation for now is: pick uniformly one of the possible values
            new_val = self.local_prng.choice(self.possible_values)
            
            #If this categorical hyperparameter is numerical I cast to int or float: I want to use int e float not numpy ints or
            #numpy floats. This is purely for consistency:
            if(isinstance(new_val, int)):
                new_val = int(new_val)
            elif(isinstance(new_val, float)):
                new_val = float(new_val)
                
            if(hasattr(new_val, 'dtype')):
                if('int' in new_val.dtype.name):
                    new_val = int(new_val)
                elif('float' in new_val.dtype.name):
                    new_val = float(new_val)
                    
            self.current_actual_value = new_val
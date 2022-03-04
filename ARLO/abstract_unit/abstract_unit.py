"""
This module contains the implementation of the AbstractUnit Class. Every Class in this library, except for the Class Logger, 
inherits from this Class.
"""

import os
import numpy as np
import datetime

import cloudpickle

from ARLO.logger.logger import Logger


def load(pickled_file_path):
    """
    Parameters
    ---------- 
    pickled_file_path: This must be an absolute path to the pickled file from which you want to load.
    
    This function loads a saved Class from a pickle file in binary form. 
    """
    
    with open(pickled_file_path, 'rb') as pickled_file:
        loaded_class_obj = cloudpickle.load(pickled_file)
        
    return loaded_class_obj
        

class AbstractUnit:
    """
    This Class used as base Class for every Class of this library, except for the Class Logger.
    """
    
    def __init__(self, obj_name, seeder=2, log_mode='console', checkpoint_log_path=None, verbosity=3, n_jobs=1, 
                 job_type='process'):
        """      
        Parameters
        ----------             
        obj_name: This is a string representing the name of a specific object. This is used as name of the pickle file where the
                  object will be saved to.
        
        seeder: This must be a non-negative integer for seeding: this will be used for setting the state of the local pseudo 
                random number generator (PRNG). Without this numpy.random is not safe to use on concurrent threads or processes. 
                cf. https://numpy.org/doc/stable/reference/random/parallel.html
                cf. https://albertcthomas.github.io/good-practices-random-number-generators/
              
                The default is 2.
                
        log_mode: This is a string and can be: 'console', 'file' or 'both'. 
                  -If 'console' then everything is printed to the console
                  -If 'file' then a log file is created.
                  -If 'both' then everything is printed to the console but also saved in the log file.
              
                  The default is 'console'.
              
        checkpoint_log_path: This is the path where the file containing all logging is saved to. If it is not specified no log 
                             file will be saved.
                  
                             This is also the path where the object will be saved to. If the path is not specified the object 
                             will not be saved. Note that the object will be saved to a pickle file in binary form.
                  
                             The default is None.
               
        verbosity: This is an integer which can be: 0, 1, 2, 3, 4. The higher its value the more informations are logged. 
                   For more information see the Class Logger.  
                   
                   The default is 3.      
                   
        n_jobs: This is the number of jobs to be used to run some parts of the code of the various Classes in parallel.
                The jobs can be processes or threads: to pick between these there is the parameter job_type.
                
                It must be a number greater than or equal to 1, since it may be used for deciding how to parallelise the code. 
                Hence the value -1 is not supported.
            
                The default is 1.
                
        job_type: This is a string and it is either 'process' or 'thread': if 'process' then multiple processes will be created,
                  else if 'thread' then multiple threads will be created. 
                  
                  Threads can be used to parallelise some parts of the code that release the GIL (running non Python code). If
                  however the GIL is not released then it is best to use processes.
                  
                  cf. https://stackoverflow.com/questions/1912557/a-question-on-python-gil
            
                  The default is 'process'.
                   
        Non-Parameters Members
        ----------------------
        local_prng: This is the local pseudo random number generator (PRNG). Without this numpy.random is not safe to use on 
                    concurrent threads or processes. 
                    cf. https://numpy.org/doc/stable/reference/random/parallel.html
                    cf. https://albertcthomas.github.io/good-practices-random-number-generators/

        logger: This is an object of Class Logger that will do the logging.         
        
        backend: This is a string and it is used in Joblib to parallelise the code. It can be: 'multiprocessing', 'loky' or 
                 'threading'. This is assigned based on the value of the parameter job_type.
                 
        prefer: This is a string and it is used in Joblib to parallelise the code. It can be: 'processes' or 'threads'. This is 
                assigned based on the value of the parameter job_type.
        """
        
        self.obj_name = obj_name
        self.log_mode = log_mode
        self.checkpoint_log_path = checkpoint_log_path
        self.verbosity = verbosity
        
        self.logger = Logger(name_obj_logging=self.obj_name, verbosity=self.verbosity, mode=self.log_mode, 
                             log_path=self.checkpoint_log_path)
        
        self.n_jobs = n_jobs
        if(self.n_jobs < 1):
            exc_msg = '\'n_jobs\' must be greater than, or equal to, 1!'
            self.logger.exception(msg=exc_msg)
            raise ValueError(exc_msg)
            
        self.job_type = job_type
        if((self.job_type != 'process') and (self.job_type != 'thread')):
            exc_msg = '\'job_type\' can either be \'process\' or \'thread\'!'
            self.logger.exception(msg=exc_msg)
            raise ValueError(exc_msg)
            
        if(self.job_type == 'process'):
            self.backend = 'loky' #either 'loky' or 'multiprocessing'. 
            #Note that 'multiprocessing' may cause issues in python >= 3.8.0
            self.prefer = 'processes'
        else:
            self.backend = 'threading'
            self.prefer = 'threads'
        
        self.seeder = seeder
        self.local_prng = np.random.default_rng(self.seeder)
        
    def __repr__(self):
        return 'AbstractUnit('+'obj_name='+str(self.obj_name)+', log_mode='+str(self.log_mode)\
               +', checkpoint_log_path='+str(self.checkpoint_log_path)+', verbosity='+str(self.verbosity)\
               +', seeder='+str(self.seeder)+', local_prng='+str(self.local_prng)+', n_jobs='+str(self.n_jobs)\
               +', job_type='+str(self.job_type)+', logger='+str(self.logger)+')'
    
    def set_local_prng(self, new_seeder):
        """
        Parameters
        ---------- 
        new_seeder: This can either be a non-negative integer for seeding or it can be an object of Class SeedSequence. In any 
                    case this will be used for setting the state of the local pseudo random number generator (PRNG). Without this 
                    numpy.random is not safe to use on concurrent threads or processes. 
                    cf. https://numpy.org/doc/stable/reference/random/parallel.html
                    cf. https://albertcthomas.github.io/good-practices-random-number-generators/
        
        After setting the local_prng this method also updates the seeder with the new_seeder.
        """
        
        self.local_prng = np.random.default_rng(new_seeder)
        self.seeder = new_seeder
            
    def save(self):
        """
        Saves the Class in a pickle file in binary form. The name of the file is equal to the name given to the object that is
        trying to be saved, plus the current time and date.
        
        Note that if checkpoint_log_path is not specified then no file will be saved.
        """
        
        if(self.checkpoint_log_path is not None):              
            name_file_obj_to_save = str(self.obj_name)+datetime.datetime.now().strftime('_%H_%M_%S__%d_%m_%Y')+'.pkl'
            with open(os.path.join(self.checkpoint_log_path, name_file_obj_to_save), 'wb') as pickle_file:
                #protocol 4 is to ensure backward compatibility between python3.8 and python3.7
                cloudpickle.dump(self, pickle_file, protocol=4)
                
                #i want to write to disk as soon as i call the save() method. both of the following lines are needed.
                #cf.https://docs.python.org/2/library/stdtypes.html#file.flush
                pickle_file.flush()
                os.fsync(pickle_file)
        else:
            self.logger.warning(msg='You cannot save the object since \'checkpoint_log_path\' is not specified!')
            
    def update_verbosity(self, new_verbosity):
        """
        Parameters
        ----------
        new_verbosity: This is an integer and it represents the new verbosity level.
        
        This method sets the verbosity of the block and of the logger inside the block.
        """
        
        self.verbosity = new_verbosity
        self.logger.verbosity = new_verbosity
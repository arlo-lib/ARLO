"""
This module contains the implementation of the Class Logger. This is used in all the library to perform logging.

The Class Logger is the only one in the library that does not inherit from the Class AbstractUnit.
"""

import os
import datetime


class Logger:
    """
    This class is used for logging in the entire library. There are 5 different levels of verbosity: 0, 1, 2, 3, 4. Where:
    -If verbosity == 0 then nothing is logged.
    -If verbosity > 0 then exceptions are logged.
    -If verbosity == 1 then also informations are logged.
    -If verbosity == 2 then also warnings are logged.
    -If verbosity == 3 then also errors are logged.
    -If verbosity == 4 then also debug comments are logged.
    """
    
    def __init__(self, name_obj_logging, verbosity=3, mode='console', log_path=None):
        """
        Parameters
        ----------
        name_obj_logging: This is a string and it must be specifying in which class is this logger being used.
        
        verbosity: This is the level of verbosity. The higher its value the more informations are displayed. 
                   
                   The default value is 3.
                   
        mode: This is a string and can be: 'console', 'file' or 'both'. 
              -If 'console' then everything is printed to the console
              -If 'file' then the logging is saved to a log file.
              -If 'both' then everything is printed to the console but also saved in a log file.
              
              The default is 'console'.
        
        log_path: This is a string and it must be the path where to save the file 'log.txt'. If not specified the file will not
                  be saved.
                  
                  The default value is None.
        """
        
        self.name_obj_logging = name_obj_logging
        self.verbosity = verbosity
        self.mode = mode
       
        self.log_path = log_path
        if(self.log_path is not None):
            self.file_name = 'ARLO'+datetime.datetime.now().strftime('_%H_%M_%S__%d_%m_%Y')
            self.log_path = os.path.join(self.log_path, str(self.file_name)+'.log')
            
        if((mode != 'console') and (mode != 'file') and (mode != 'both')):
            raise ValueError('In the \'Logger\' Class \'mode\' can only be: \'console\', \'file\' or \'both\'!')
            
        if((mode == 'file' or mode == 'both') and (log_path is None)):
            raise ValueError('You specified \'mode\' equal to \'file\' but you did not provide \'log_path\'!')
    
    def __repr__(self):
        return 'Logger('+'name_obj_logging='+str(self.name_obj_logging)+', verbosity='+str(self.verbosity)\
               +', mode='+str(self.mode)+', log_path='+str(self.log_path)+')'
         
    def _to_console(self, msg, log_level):
        """
        Parameters
        ----------
        msg: This is a string and it is the message that needs to be logged to the console.
        
        log_level: This is a string representing the log level which can be: 'INFO', 'WARNING', 'ERROR', 'DEBUG' or 'EXCEPTION'.
        
        Prints to console the log with the logging level and with the name of the object that has called the logging. 
        """
        
        print(datetime.datetime.now().strftime('[%d-%m-%Y, %H:%M:%S]')+'['+str(log_level)+', '+str(self.name_obj_logging)
              +']: '+str(msg)+'\n')
         
    def _to_file(self, msg, log_level): 
        """
        Parameters
        ----------
        msg: This is a string and it is the message that needs to be logged to file.
        
        log_level: This is a string representing the log level which can be: 'INFO', 'WARNING', 'ERROR', 'DEBUG' or 'EXCEPTION'.
        
        Writes to file the log with the logging level and with the name of the object that has called the logging. If the file
        exists it is opened and the logging is appended to the end of the file. Else if the file does not exist it is created.
        """
        
        if(self.log_path is not None):             
            with open(self.log_path, 'a') as file:
                str_to_write = datetime.datetime.now().strftime('[%d-%m-%Y, %H:%M:%S]')+'['+str(log_level)+', '\
                               +str(self.name_obj_logging)+']: '+str(msg)+'\n'
                file.write(str_to_write)
                
                #i want to write to disk as soon as i call the _to_file() method. both of the following lines are needed.
                #cf.https://docs.python.org/2/library/stdtypes.html#file.flush
                file.flush()
                os.fsync(file)
        else:
            print('You cannot write the log to file since \'log_path\' is not specified!\n')
    
    def _log(self, msg, log_level):
        """
        Parameters
        ----------
        msg: This is a string and it is the message that needs to be logged (either to the console, to file or to both).
        
        log_level: This is a string representing the log level which can be: 'INFO', 'WARNING', 'ERROR', 'DEBUG' or 'EXCEPTION'.
        
        Based on the mode it either calls the method _to_console or _to_file or both.
        """
        
        if((self.mode == 'console') or (self.mode == 'both')):
            self._to_console(msg=msg, log_level=log_level)   
        if((self.mode == 'file') or (self.mode == 'both')):
            self._to_file(msg=msg, log_level=log_level)   

    def info(self, msg):
        """
        Parameters
        ----------
        msg: This is a string and it is the message that needs to be logged. It must be a message of information about something
             that has occurred.
        
        Logs the message with logging level 'INFO'.        
        """        
        
        if(self.verbosity >= 1):
            self._log(msg=msg, log_level='INFO')
            
    def warning(self, msg):
        """
        Parameters
        ----------
        msg: This is a string and it is the message that needs to be logged. It must be a message about some warning.
        
        Logs the message with logging level 'WARNING'.        
        """        
        
        if(self.verbosity >= 2):
            self._log(msg=msg, log_level='WARNING')
    
    def error(self, msg):
        """
        Parameters
        ----------
        msg: This is a string and it is the message that needs to be logged. It must be a message of error about something
             that has occurred.
             
        Logs the message with logging level 'ERROR'.      
        """ 
        
        if(self.verbosity >= 3):
            self._log(msg=msg, log_level='ERROR')
            
    def debug(self, msg):
        """
        Parameters
        ----------
        msg: This is a string and it is the message that needs to be logged. It must be a message used for debug. 
             
        Logs the message with logging level 'DEBUG'.        
        """ 
        
        if(self.verbosity >= 4):
            self._log(msg=msg, log_level='DEBUG')
            
    def exception(self, msg):
        """
        Parameters
        ----------
        msg: This is a string and it is the message that needs to be logged. It must be a message of exception about something
             exception that will be thrown just after the message is logged.
             
        Logs the message with logging level 'EXCEPTION'.        
        """ 
        
        if(self.verbosity >= 0):
            self._log(msg=msg, log_level='EXCEPTION')
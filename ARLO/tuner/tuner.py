"""
This module contains the implementation of the Class: Tuner.

The Class Tuner inherits from the Class AbstractUnit and from ABC.

The Class Tuner is an abstract Class used as base class for all types of tuners.

Any block that wants to be tuned needs to use a tuner.
"""

from abc import ABC, abstractmethod
import os
import datetime
import itertools
import copy
import numpy as np

import plotly.graph_objects as go
from catboost import CatBoostRegressor

from ARLO.abstract_unit.abstract_unit import AbstractUnit, load
from ARLO.block.block import Block
from ARLO.metric.metric import Metric
from ARLO.input_loader.input_loader import InputLoader
from ARLO.hyperparameter.hyperparameter import Real, Categorical


class Tuner(AbstractUnit, ABC):
    """
    This is an abstract Class used as base for all block tuning. The specific tuning algorithms inherit from this Class.
     
    It requires as input: the block to optimise, the evaluation metric and the input loader. The input loader is an object that 
    produces the right input for the agents of a specific block.
    
    This Class inherits from the Class AbstractUnit and from ABC.
    """
    
    def __init__(self, block_to_opt, eval_metric, input_loader, obj_name, create_explanatory_heatmap=False,
                 seeder=2, log_mode='console', checkpoint_log_path=None, verbosity=3, n_jobs=1, job_type='process', 
                 output_save_periodicity=25):
        """
        Parameters
        ----------
        block_to_opt: This must be an object belonging to some Class inherting from the Block base Class.
                      
        eval_metric: This is the evaluation metric that will be used by the tuner. This is used to evaluate the various agents 
                     that are created in the tuning process.
        
        input_loader: This must be an object inheriting from the Class InputLoader. This is used for telling the tuner how to 
                      properly split the input of the automatic block among the various agents present in the Tuner.
 
        output_save_periodicity: This is an integer greater than or equal to 1 and it is the frequency with which to save the 
                                 current block as the tuning procedure takes place. Note that if the member checkpoint_log_path
                                 is not set then nothing will be saved.
                                 
                                 The default is 25.
                                 
        	create_explanatory_heatmap: This is a boolean and if True then at the end of the call of the method tune() the method 
                                    create_explanatory_heatmap_hyperparameters is going to be called: this creates an 
                                    explanatory heatmap of the hyper-parameters.
                                    
                                    The default is False.
    
        Non-Parameters Members
        ----------------------                               
        is_tune_successful: This is used to know whether or not the tuner finished with no errors. 
 
        The other parameters and non-parameters members are described in the Class AbstractUnit.
        """
        
        super().__init__(obj_name=obj_name, seeder=seeder, log_mode=log_mode, checkpoint_log_path=checkpoint_log_path, 
                         verbosity=verbosity, n_jobs=n_jobs, job_type=job_type)

        self.block_to_opt = block_to_opt
        if(not isinstance(self.block_to_opt, Block)):
            raise TypeError('The \'block_to_opt\' must be an object of a Class inheriting from the Class \'Block\'')
        
        self.eval_metric = eval_metric
        if(not isinstance(self.eval_metric, Metric)):
            raise TypeError('The \'eval_metric\' must be an object of a Class inheriting from the Class \'Metric\'')
        
        self.input_loader = input_loader
        if(not isinstance(self.input_loader, InputLoader)):
            raise TypeError('The \'input_loader\' must be an object of a Class inheriting from the Class \'InputLoader\'')
            
        self.output_save_periodicity = output_save_periodicity
        
        self.create_explanatory_heatmap = create_explanatory_heatmap
        
        self.is_tune_successful = False
    
    def __repr__(self):
        return 'Tuner('+'block_to_opt='+str(self.block_to_opt)+', eval_metric='+str(self.eval_metric)\
               +', input_loader='+str(self.input_loader)+', obj_name='+str(self.obj_name)\
               +', create_explanatory_heatmap='+str(self.create_explanatory_heatmap)+', seeder='+str(self.seeder)\
               +', local_prng='+str(self.local_prng)+', log_mode='+str(self.log_mode)\
               +', checkpoint_log_path='+str(self.checkpoint_log_path)+', verbosity='+str(self.verbosity)\
               +', n_jobs='+str(self.n_jobs)+', job_type='+str(self.job_type)\
               +', output_save_periodicity='+str(self.output_save_periodicity)+', logger='+str(self.logger)+')'
                
    def is_metric_consistent_with_input_loader(self):
        """
        This method returns True if the metric in the class Tuner is consistent with the input loader in the class Tuner. A 
        metric and an input_loader are consistent when a metric requires in input something that the input_loader will return.
        
        Otherwise this method returns False.
        """
        
        if((self.eval_metric.requires_dataset and (not self.input_loader.returns_dataset)) or 
           (self.eval_metric.requires_env and (not self.input_loader.returns_env))):
            self.logger.error(msg='The \'metric\' and the \'input_loader\' are not consistent!')
            return False
        
        return True
    
    def tune(self, train_data=None, env=None):
        """
        Parameters
        ----------        
        train_data: This is the dataset that can be used by the Tuner. 
                   
                    It must be an object of a Class inheriting from BaseDataSet.
        
        env: This is the environment that can be used by the Tuner.
           
             It must be an object of a Class inheriting from BaseEnvironment.

        Returns
        -------
        best_agent: This is the tuned agent: the original agent but with the tuned hyper-parameters.
        
        best_agent_eval: This is the evaluation of the tuned agent.
        """
           
        #each time i re-tune the block i need to set is_tune_successful to False
        self.is_tune_successful = False
        
        #check that the metric and the input_loader are consistent:
        if(not self.is_metric_consistent_with_input_loader()):
            self.is_tune_successful = False
            self.logger.error(msg='The \'metric\' and the \'input_loader\' are not consistent!')
            return None, None
         
        if(not self.block_to_opt.is_parametrised):
            self.logger.info(msg='No tuning needed since \'self.block_to_opt.is_parametrised\' is \'False\'!')
            
            best_agent = self.block_to_opt
            best_agent_res = self.block_to_opt.learn(train_data=train_data, env=env)
            
            if(not best_agent.is_learn_successful):
                self.is_tune_successful = False
                self.logger.error(msg='There was an error in the \'learn\' method of an agent!')
                return None, None
            else:
                best_agent_eval = self.eval_metric.evaluate(block_res=best_agent_res, train_data=train_data, env=env)

                self.is_tune_successful = True
                self.logger.info(msg='Tuning complete!')
                return best_agent, best_agent_eval
        
        #if i reach this point then it means this method will be successfully called by a sub-class of Tuner:
        name_new_folder = str(self.obj_name)+datetime.datetime.now().strftime('_%H_%M_%S__%d_%m_%Y')

        tuner_results_path = os.path.join(self.checkpoint_log_path, 'tuner_'+str(name_new_folder))     

        if not os.path.isdir(tuner_results_path):
            os.makedirs(tuner_results_path)
            self.checkpoint_log_path = tuner_results_path
            self.block_to_opt.checkpoint_log_path = self.checkpoint_log_path
        else:
            exc_msg = 'Cannot create a new directory named: \''+str(tuner_results_path)+'\' as this directory already'\
                      +' exists! Therefore the tuning procedure will stop here.'
            self.logger.exception(msg=exc_msg)
            raise RuntimeError(exc_msg)
            
    @abstractmethod
    def _evaluate_a_generation(self, gen):
        raise NotImplementedError
        
    def update_verbosity(self, new_verbosity):
        """
        Parameters
        ----------
        new_verbosity: This is an integer and it represents the new verbosity level.
        
        This method calls the method update_verbosity() implemented in the Class AbstractUnit and then it calls such method on 
        the object block_to_opt and input_loader.
        """
         
        super().update_verbosity(new_verbosity=new_verbosity)
        
        self.block_to_opt.update_verbosity(new_verbosity=new_verbosity)        
        self.input_loader.update_verbosity(new_verbosity=new_verbosity)
        
    def create_explanatory_heatmap_hyperparameters(self):
        """
        This method is only called if create_explanatory_heatmap is set to True.
        
        This method loads the agents that were constructed throughout the tuning procedure, it creates from these a dataset
        where the features are the hyper-parameters values and the target is the obtained block_eval, it fits a CatBoostRegressor
        on such dataset and then via Plotly it create and save an html file containing a heatmap showcasing the effect of 
        changing an hyper-parameter value on the block_eval.
        """
        
        self.logger.info(msg='Now creating the explanatory heatmap of the hyper-parameters...')
        
        #read filenames in folder results:
        filenames = next(os.walk(self.checkpoint_log_path), (None, None, []))[2]  # [] if no file
        
        #create a new folder to contain the heatmap only:
        heatmap_folder_destination = os.path.join(self.checkpoint_log_path, 'heatmap')
        
        os.makedirs(heatmap_folder_destination)
            
        #population of tuned agents:
        agents_pop = []
        for tmp_file in filenames:
            if((self.block_to_opt.obj_name in tmp_file) and ('.pkl' in tmp_file)):
                if(('_result_' not in tmp_file) and ('_new_best_' not in tmp_file)):
                    new_agent = load(os.path.join(self.checkpoint_log_path, str(tmp_file))) 
                    agents_pop.append(new_agent)
        
        best_agent = None
        for tmp_file in filenames:
            if('best_agent_' in tmp_file):
                best_agent = load(os.path.join(self.checkpoint_log_path, tmp_file))
        
        if(best_agent is None):
            exc_msg = 'The \'best_agent\' failed to be loaded!'
            self.logger.exception(msg = exc_msg)
            raise RuntimeError(exc_msg)
        
        tmp_params = best_agent.get_params()
        
        params = {}
        #remove categorical hyperparameters and only keep mutated hyperparameters:
        for tmp_key in list(tmp_params.keys()):
            if((not isinstance(tmp_params[tmp_key], Categorical)) and (tmp_params[tmp_key].to_mutate)):
                params.update({tmp_key: tmp_params[tmp_key]})
                
            if(isinstance(tmp_params[tmp_key], Categorical) and (tmp_params[tmp_key].to_mutate)):
                if(isinstance(tmp_params[tmp_key].current_actual_value, int) or 
                   isinstance(tmp_params[tmp_key].current_actual_value, float)):
                    new_parameter = Real(hp_name=tmp_key, current_actual_value=tmp_params[tmp_key].current_actual_value, 
                                         obj_name=tmp_key+'_artificial_param', 
                                         range_of_values=[tmp_params[tmp_key].possible_values[0], 
                                                          tmp_params[tmp_key].possible_values[-1]], 
                                         to_mutate=True)
                    params.update({tmp_key: new_parameter})
        
        #create dataset containing the values of the hyperparameters of the tuned agents:
        agents_dataset_params = []
        for agent in agents_pop:
            tmp_params_agent = agent.get_params()
            tmp_params_values_agent = {}
            
            tmp_effective_params = []
        
            for tmp_key in list(tmp_params_agent.keys()):
                if((not isinstance(tmp_params_agent[tmp_key], Categorical)) and (tmp_params_agent[tmp_key].to_mutate)):
                    tmp_params_values_agent.update({tmp_key: tmp_params_agent[tmp_key].current_actual_value})
                    tmp_effective_params.append(tmp_params_agent[tmp_key].current_actual_value)
                if(isinstance(tmp_params[tmp_key], Categorical) and (tmp_params[tmp_key].to_mutate)):
                    if(isinstance(tmp_params[tmp_key].current_actual_value,int) or 
                       isinstance(tmp_params[tmp_key].current_actual_value,float)):
                        tmp_params_values_agent.update({tmp_key: tmp_params_agent[tmp_key].current_actual_value})
                        tmp_effective_params.append(tmp_params_agent[tmp_key].current_actual_value)
                    
            agents_dataset_params.append(tmp_effective_params)    
        
        agents_eval = []
        for agent in agents_pop:
            agents_eval.append(agent.block_eval)
            
        #catboost regressor: fit on the hyperparameter dataset where the target is the obtained performance
        cat_reg = CatBoostRegressor(iterations=1500, learning_rate=0.004, loss_function='MAPE', thread_count=-1, silent=True, 
                                    eval_metric='MAPE', subsample=0.8, max_depth=10, colsample_bylevel=0.8,
                                    random_state=3, reg_lambda=0.2, objective='MAPE')
        
        cat_reg.fit(agents_dataset_params, agents_eval)
    
        class cat_reg_estimator():
            def predict(self, samples_to_predict, x_range, y_range, x_index, y_index):
                cat_reg_prediction = cat_reg.predict(samples_to_predict)
                                        
                final_prediction = []
                tmp_prediction = []
                for elem in cat_reg_prediction:
                    tmp_prediction.append(elem)
                    
                    if(len(tmp_prediction) == 100):
                        final_prediction.append(tmp_prediction)
                        tmp_prediction = []
                        
                return final_prediction
            
        estimator = cat_reg_estimator()
                
        def my_approximator(hp_x_name, hp_x_range, hp_y_name, hp_y_range):
            all_hp = copy.deepcopy(params)
            
            generic_sample_to_predict = []
            #a generic sample to predict is a sample where the X and Y hyperparameters are missing, while the others 
            #hyperparameters have the value of the optimal agent:
            for tmp_key in list(all_hp.keys()):
                if(tmp_key not in [hp_x_name, hp_y_name]):    
                    generic_sample_to_predict.append(all_hp[tmp_key].current_actual_value)
                elif(tmp_key == hp_x_name):
                    generic_sample_to_predict.append(np.inf)
                else:
                    generic_sample_to_predict.append(-np.inf)

            samples_to_predict = []
            for i in range(len(hp_y_range)):
                new_sample = copy.deepcopy(generic_sample_to_predict)
                        
                idx_y = np.argwhere(np.array(generic_sample_to_predict) == -np.inf).ravel()[0]
                new_sample[idx_y] = hp_y_range[i]
             
                for k in range(len(hp_x_range)):   
                    new_sample_x = copy.deepcopy(new_sample)
                   
                    idx_x = np.argwhere(np.array(generic_sample_to_predict) == np.inf).ravel()[0]
                    new_sample_x[idx_x] = hp_x_range[k]
                    
                    if(len(samples_to_predict) == 0):
                        samples_to_predict = new_sample_x
                    else:
                        samples_to_predict = np.vstack((samples_to_predict, new_sample_x))
           
            hp_x_idx = list(all_hp.keys()).index(hp_x_name)
            hp_y_idx = list(all_hp.keys()).index(hp_y_name)
            
            eval_prediction = estimator.predict(samples_to_predict=samples_to_predict, x_range=hp_x_range, y_range=hp_y_range, 
                                                x_index=hp_x_idx, y_index=hp_y_idx)
        
            return eval_prediction
        
        default_hp_x = list(params.keys())[0]
        default_hp_y = list(params.keys())[1]
        
        dtype_x = float
        if(isinstance(params[default_hp_x].current_actual_value, int)):             
            dtype_x = int
                        
        dtype_y = float
        if(isinstance(params[default_hp_y].current_actual_value, int)):             
            dtype_y = int
            
        #create heatmap grid:
        hpx = np.linspace(params[default_hp_x].range_of_values[0], params[default_hp_x].range_of_values[1], 100, dtype=dtype_x)
        hpy = np.linspace(params[default_hp_y].range_of_values[0], params[default_hp_y].range_of_values[1], 100, dtype=dtype_y)
        
        data = list(my_approximator(hp_x_name=default_hp_x, hp_x_range=hpx, hp_y_name=default_hp_y, hp_y_range=hpy))   
                
        params_keys = list(params.keys())
        combinations_hp = [] 
        #computes all combinations of 2 hyperparmeters:
        for subset in itertools.combinations(params_keys, 2):
            combinations_hp.append(list(subset))
           
        #updates data for all buttons in the figure:
        def compute_new_data(menu, fig):
            buttons = []
            
            for tmp_combination_hp in combinations_hp:    
                hp_x = tmp_combination_hp[0]
                hp_y = tmp_combination_hp[1]
                
                base_dict = dict(args=None, label=hp_x+'-'+hp_y, method='update')
                
                dtype_x = float
                
                if(isinstance(params[hp_x].current_actual_value, int)):             
                    dtype_x = int           
                       
                dtype_y = float
                if(isinstance(params[hp_y].current_actual_value, int)):             
                    dtype_y = int
            
                #new heatmap grid:
                x = np.linspace(params[hp_x].range_of_values[0], params[hp_x].range_of_values[1], 100,  dtype=dtype_x)
                y = np.linspace(params[hp_y].range_of_values[0], params[hp_y].range_of_values[1], 100,  dtype=dtype_y)
                        
                new_data = []
                
                new_data = list(my_approximator(hp_x_name=hp_x, hp_x_range=x, hp_y_name=hp_y, hp_y_range=y))
                    
                base_dict['args'] = [{"z": [new_data], 'x':[x], 'y':[y]}]
                
                buttons.append(base_dict)
                
            return buttons
        
        #create plotly figure. The following is re-adaptation of the code taken from here:
        #cf. https://plotly.com/python/custom-buttons/
        fig = go.Figure()
        
        fig.add_trace(go.Heatmap(z=data, x=hpx, y=hpy, colorscale='Viridis', visible=True))
        
        fig.update_layout(width=1500, height=800, autosize=False, margin=dict(t=120, b=0, l=500, r=300))
        
        fig.update_scenes(aspectratio=dict(x=1, y=1, z=0.7), aspectmode='manual')
        
        button_layer_1_height = 1.08
        
        fig.update_layout(updatemenus=[#color button
                                       dict(buttons=list([dict(args=[{'colorscale': 'Viridis'}], label='Viridis', 
                                                               method='update'),
                                                          dict(args=[{'colorscale': 'Aggrnyl'}], label='Aggrnyl', 
                                                               method='update'),
                                                          dict(args=[{'colorscale': 'Blues'}], label='Blues', method='update'),
                                                          dict(args=[{'colorscale': 'YlOrRd'}], label='YlOrRd', method='update')
                                                          ]),
                                            direction='down', pad={'r': 0, 't': 10}, showactive=True, x=0.06, xanchor='center', 
                                            y=button_layer_1_height, yanchor='top'
                                            ),
                                       #reverse color scale button
                                       dict(buttons=list([dict(args=[{'reversescale': False}], label='False', method='update'),
                                                          dict(args=[{'reversescale': True}], label='True', method='update')
                                                          ]),
                                            direction='down', pad={'r': 0, 't': 10}, showactive=True, x=0.22, xanchor='center',
                                            y=button_layer_1_height, yanchor='top'
                                            ),
                                       #show-hide contour lines button
                                       dict(buttons=list([dict(args=[{'contours.showlines': False, 'type': 'contour'}], 
                                                               label='Hide lines', method='update'),
                                                          dict(args=[{'contours.showlines': True, 'type': 'contour'}],
                                                               label='Show lines', method='update')
                                                          ]),
                                            direction='down', pad={'r': 0, 't': 10}, showactive=True, x=0.4, xanchor='center', 
                                            y=button_layer_1_height, yanchor='top'
                                            ),
                                       #heatmap or 3d surface button
                                       dict(buttons=list([dict(args=[{'type': 'heatmap'}], label='Heatmap', method='update'),
                                                          dict(args=[{'type': 'surface'}], label='3D Surface', method='update')
                     
                                                        ]),
                                            direction='down', pad={'r': 0, 't': 10}, showactive=True, x=0.6, xanchor='center', 
                                            y=button_layer_1_height, yanchor='top'
                                            ),
                                       #x-hyperparameter button
                                       dict(buttons=compute_new_data(menu=fig.layout.updatemenus, fig=fig),
                                            direction='down', pad={'r': 0, 't': 10}, showactive=True, x=0.925, xanchor='center', 
                                            y=button_layer_1_height, yanchor='top', active=0
                                            )
                                       ]
                          )
        
        fig.update_layout(annotations=[dict(text='Colorscale', x=0.01, xref='paper', y=1.1, yref='paper', align='center', 
                                            showarrow=False),
                                       dict(text='Reverse Colorscale', x=0.14, xref='paper', y=1.1, yref='paper', align='center', 
                                            showarrow=False),
                                       dict(text='Lines', x=0.4, xref='paper', y=1.1, yref='paper', align='center', 
                                            showarrow=False),
                                       dict(text='Trace', x=0.6, xref='paper', y=1.1, yref='paper', align='center', 
                                            showarrow=False),      
                                       dict(text='Hyperparameters', x=1, xref='paper', y=1.1, yref='paper', align='center', 
                                            showarrow=False)
                                       ])
        
        config = {'displaylogo': False}
        
        heatmap_file_name = str(self.obj_name)+'_'+str(self.block_to_opt.obj_name)+'_hyperparameters_heatmap.html'
        fig.write_html(os.path.join(heatmap_folder_destination, heatmap_file_name), config=config)
        
        self.logger.info(msg='Heatmap saved in the file \''+heatmap_file_name+'\'')
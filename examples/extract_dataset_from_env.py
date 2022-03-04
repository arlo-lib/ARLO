from ARLO.block import DataGenerationRandomUniformPolicy
from ARLO.rl_pipeline import OfflineRLPipeline
from ARLO.metric import SomeSpecificMetric
from ARLO.environment import BaseHalfCheetah

if __name__ == '__main__':
    my_data_gen = DataGenerationRandomUniformPolicy(eval_metric=SomeSpecificMetric(obj_name='some_specific_metric'), 
                                                    obj_name='my_data_gen')

    my_pipeline = OfflineRLPipeline(list_of_block_objects=[my_data_gen],
                                    eval_metric=SomeSpecificMetric(obj_name='some_specific_metric_pipeline'), 
                                    obj_name='OfflinePipeline') 

    my_env = BaseHalfCheetah(obj_name='my_cheetah', gamma=1, horizon=1000)
    
    out = my_pipeline.learn(env=my_env)

    #extracted dataset:
    my_data = out.train_data.dataset
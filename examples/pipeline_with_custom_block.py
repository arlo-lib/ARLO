from ARLO.block import ModelGenerationMushroomOnline
from ARLO.rl_pipeline import OnlineRLPipeline
from ARLO.metric import DiscountedReward, SomeSpecificMetric
from ARLO.environment import BaseHalfCheetah

class MyCustomModelGenBlock(ModelGenerationMushroomOnline):
    
    #implement here your custom block

    pass

if __name__ == '__main__':
    my_custom_block = MyCustomModelGenBlock(eval_metric=DiscountedReward(obj_name='my_custom_block_metric', n_episodes=10, 
                                                                         batch=False), 
                                            obj_name='my_custom_block')

    my_pipeline = OnlineRLPipeline(list_of_block_objects=[my_custom_block],
                                   eval_metric=SomeSpecificMetric(obj_name='some_specific_metric_pipeline'), 
                                   obj_name='OnlinePipeline') 

    my_env = BaseHalfCheetah(obj_name='my_cheetah', gamma=1, horizon=1000)
    out = my_pipeline.learn(env=my_env)
    
    #learnt policy:
    my_policy = out.policy.policy
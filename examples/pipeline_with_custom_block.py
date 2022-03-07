from ARLO.block import ModelGenerationMushroomOnline
from ARLO.rl_pipeline import OnlineRLPipeline
from ARLO.metric import DiscountedReward
from ARLO.environment import BaseHalfCheetah

if __name__ == '__main__':
    class MyCustomModelGenBlock(ModelGenerationMushroomOnline):
        
        #implement here your custom block

        pass

    my_custom_block = MyCustomModelGenBlock(eval_metric=DiscountedReward(obj_name='my_custom_block_metric', n_episodes=10), 
                                            obj_name='my_custom_block')

    my_pipeline = OnlineRLPipeline(list_of_block_objects=[my_custom_block],
                                   eval_metric=DiscountedReward(obj_name='my_custom_block_metric', n_episodes=10), 
                                   obj_name='OnlinePipeline') 

    my_env = BaseHalfCheetah(obj_name='my_cheetah', gamma=1, horizon=1000)
    out = my_pipeline.learn(env=my_env)
    
    #learnt policy:
    my_policy = out.policy.policy
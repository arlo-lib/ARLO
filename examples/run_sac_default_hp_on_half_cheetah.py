from ARLO.block import ModelGenerationMushroomOnlineSAC
from ARLO.rl_pipeline import OnlineRLPipeline
from ARLO.metric import DiscountedReward
from ARLO.environment import BaseHalfCheetah

if __name__ == '__main__':
    #uses default hyper-paramters of SAC
    my_sac = ModelGenerationMushroomOnlineSAC(eval_metric=DiscountedReward(obj_name='ddpg_metric', n_episodes=10, batch=True), 
                                              obj_name='my_ddpg', deterministic_output_policy=True)

    my_pipeline = OnlineRLPipeline(list_of_block_objects=[my_sac],
                                   eval_metric=DiscountedReward(obj_name='pipeline_metric', n_episodes=10, batch=True), 
                                   obj_name='OnlinePipeline') 

    my_env = BaseHalfCheetah(obj_name='my_cheetah', gamma=1, horizon=1000)
    my_env.seed(2)
    
    out = my_pipeline.learn(env=my_env)
    
    #learnt policy:
    my_policy = out.policy.policy
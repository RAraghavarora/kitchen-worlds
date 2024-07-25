import config

from cogarch_tools.cogarch_run import run_agent
from cogarch_tools.processes.pddlstream_agent import PDDLStreamAgent
import tracit
import sys
sys.setprofile(tracit.tracefunc)

if __name__ == '__main__':
    seed = 378277
    goal_variations = [1]
    run_agent(
        agent_class=PDDLStreamAgent, config='config_pigi.yaml',
        goal_variations=goal_variations, seed=seed, exp_subdir='piginet_data', problem='test_full_kitchen'
    )
    #TODO: Why is problem and exp_subdir not being loaded by themselves? (Raghav)

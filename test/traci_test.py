import os,sys
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
    print('success')
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")
import traci

sumo_cmd = [os.environ["SUMO_HOME"] + '/bin/sumo-gui', '-c', '../map/ramp3/mapDenseSlow.sumo.cfg', '--start', str(True)]
sumo_cmd_reload = ['-c', '../map/ramp3/mapDenseSlow.sumo.cfg', '--start', str(True)]

traci.start(sumo_cmd)

for j in range(3):

    for i in range(5):
        traci.simulationStep()
    traci.load(sumo_cmd_reload)

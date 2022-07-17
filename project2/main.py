import pybullet
import pybullet_data
import time

# Set up pybullet instance
physicsClient = pybullet.connect(pybullet.GUI)
pybullet.setAdditionalSearchPath(pybullet_data.getDataPath())
pybullet.setPhysicsEngineParameter(enableFileCaching=0)
pybullet.setGravity(0, 0, -9.8)
pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_GUI, False)
pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_SHADOWS, True)
pybullet.resetDebugVisualizerCamera(cameraDistance=1.400, cameraYaw=58.000, cameraPitch=-42.200,
                                    cameraTargetPosition=(0.0, 0.0, 0.0))

kuka = pybullet.loadURDF("assets/kuka/kuka.urdf", basePosition=[0, 0, 0.02], useFixedBase=True)
plane = pybullet.loadURDF("plane.urdf")
box1 = pybullet.loadURDF("assets/box/box.urdf", basePosition=[1, 1, 0.5], useFixedBase=True)
box2 = pybullet.loadURDF("assets/box/box.urdf", basePosition=[2, 0, 1.0], useFixedBase=True)

start_time = 0.0
end_time = 10.0
time_step = 0.01
number_of_simulation_steps = int((end_time - start_time) / time_step)

for t in range(number_of_simulation_steps):
    # Can either step the simulation (nots sure we need to do this, as only using kinematics)
    pybullet.stepSimulation()
    time.sleep(time_step)
    # Or can just sleep
    time.sleep(5)

# End
pybullet.disconnect()


#!/bin/bash

# Calibrate the follower arm
echo "---------- Start calibrating the follower arm ----------"
lerobot-calibrate \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM1 \
    --robot.id=my_awesome_follower_arm


# Calibrate the leader arm
echo "---------- Start calibrating the leader arm ----------"
lerobot-calibrate \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM0 \
    --teleop.id=my_awesome_leader_arm
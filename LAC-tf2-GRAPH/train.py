"""Start the LAC agent training."""

import tensorflow as tf

from variant import (
    LOG_PATH,
    TRAIN_PARAMS,
)
from lac import train

if __name__ == "__main__":

    # Train several agents in the environment and save the results
    for i in range(
        TRAIN_PARAMS["start_of_trial"],
        TRAIN_PARAMS["start_of_trial"] + TRAIN_PARAMS["num_of_trials"],
    ):
        roll_out_log_path = LOG_PATH + "/" + str(i)
        print("logging to " + roll_out_log_path)
        train(roll_out_log_path)
        tf.compat.v1.reset_default_graph()

import argparse


def get_params():
    parser = argparse.ArgumentParser(
        description="Variable parameters based on the configuration of the machine or user's choice")

    parser.add_argument("--env_name", default="PongNoFrameskip-v4", type=str, help="Name of the environment.")

    parser.add_argument("--total_iterations", default=8000, type=int, help="The total number of iterations.")  # Fixed
    parser.add_argument("--interval", default=30, type=int,
                        help="The interval specifies how often different parameters should be saved and printed,"
                             " counted by iterations.")
    parser.add_argument("--do_test", action="store_true",
                        help="The flag determines whether to train the agent or play with it.")
    parser.add_argument("--render", action="store_true",
                        help="The flag determines whether to render each agent or not.")
    parser.add_argument("--train_from_scratch", action="store_true",
                        help="The flag determines whether to train from scratch or continue previous tries.")

    parser_params = parser.parse_args()

    # region default parameters
    default_params = {"state_shape": (84, 84, 4),
                      "lr": 7e-4,  # Fixed
                      "adam_eps": 1e-5,
                      "gamma": 0.99,  # Fixed
                      "ent_coeff": 0.01,
                      "critic_coeff": 0.5,  # Fixed
                      "max_grad_norm": 0.5,  # Fixed
                      "random_seed": 123
                      }

    # endregion
    total_params = {**vars(parser_params), **default_params}
    print("params:", total_params)
    return total_params
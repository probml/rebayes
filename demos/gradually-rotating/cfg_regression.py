import ml_collections

def get_config():
    cfg = ml_collections.ConfigDict()

    cfg.memory = 10
    cfg.forecast = 10

    # LoFi parameters
    cfg.lofi = ml_collections.ConfigDict()
    cfg.lofi.dynamics_weight = 1.0
    cfg.lofi.dynamics_covariance = 0.0
    cfg.lofi.initial_covariance = 0.1

    # Replay-buffer SGD parameters
    cfg.rsgd = ml_collections.ConfigDict()
    cfg.rsgd.learning_rate = 5e-4
    cfg.rsgd.n_inner = 1

    return cfg

if __name__ == "__main__":
    cfg = get_config()
    print(cfg)

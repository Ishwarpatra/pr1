class PipelineConfig:
    def __init__(self, env='local'):
        self.env = env
        if env == 'local':
            self.MAX_ROWS = 2000
            self.SR_ITERATIONS = 10
            self.SR_OPERATORS = ["+", "-", "*", "/"]
            self.OPTUNA_TRIALS = 10
        else:
            self.MAX_ROWS = 100000
            self.SR_ITERATIONS = 40
            self.SR_OPERATORS = ["+", "*", "-", "/", "exp", "sqrt", "log"]
            self.OPTUNA_TRIALS = 100

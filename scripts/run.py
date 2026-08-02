from src import Baseline, train_baseline

if __name__ == "__main__":
    model = Baseline()
    train_baseline(model, 10, 32, 0.0001)
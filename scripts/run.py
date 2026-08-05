from src import Baseline, PoseEstimator, train_baseline, train_model

if __name__ == "__main__":
    model = PoseEstimator()
    train_model(model, 10, 128, 0.0002)
from kerastuner.tuners import RandomSearch
from neuralnets.training_utils.OneCycleScheduler import OneCycleScheduler
from neuralnets.models.onedcnn_functional import OneDCNNModel


class ModelTuner:
    def __init__(self, load_level: int = 0, electrodes: int = 0):
        self.load_level = load_level
        self.electrodes = electrodes

    def tune(self, X_train, y_train, X_val, y_val):
        """
        A tuner method that uses the keras tuner to search for the best hyperparameters for the model.
        Example usage:
        nn.NeuralNets.tuner(
            x,
            y,
            x_val,
            y_val,
            load_level=load_level,
            electrodes=len(config.ch_level[load_level]),
        )
        """
        epochs = 25
        batch_size = 32
        scheduler = OneCycleScheduler(
            max_lr=0.0001,
            steps_per_epoch=len(X_train) // batch_size,
            epochs=epochs,
            verbose=1,
        )
        tuner = RandomSearch(
            lambda hp: OneDCNNModel.create_and_compile_sequential_tune(
                hp, self.load_level, self.electrodes
            ),
            objective="val_accuracy",
            max_trials=100,
            directory="my_dir",
            project_name="OneDCNN_F",
        )

        tuner.search_space_summary()

        tuner.search(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=[scheduler],
        )

        best_model = tuner.get_best_models(num_models=1)[0]

        best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]

        print(best_hyperparameters.values)
        return best_model, best_hyperparameters

import importlib

class ModelManager:
    """
    Handles the instantiation of models. Designed to keep the main code agnostic 
    to the specific model being used.
    """


    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.model_class = self._get_model_class()
        self.kwargs = kwargs

    def _get_model_class(self):
        '''
        Import the model that will be used.
        '''
        
        module = importlib.import_module(f"models.{self.model_name}")
        return getattr(module, self.model_name)

    def infer_input_size(self, first_batch):
        """
        Compute the input size dynamically based on the model type and the first batch of data.
        """

        x = first_batch[0]
        if self.model_name.lower() == "fcn" or self.model_name.lower() == "fcnregressor":
            # FCN expects C channels (assuming shape [N, C, L])
            input_size = x.shape[1]
        elif self.model_name.lower() == "mlp":
            # MLP expects flattened input
            if x.ndim > 2:
                input_size = x.shape[1] * x.shape[2]  # C * L
            else:
                input_size = x.shape[1]  # L
        else:
            # fallback
            input_size = x.shape[1]

        return input_size

    def get_model(self, first_batch=None, enc_in=None, seq_len=None, **override_kwargs):
        """
        Instantiate the model, automatically computing input_size if first_batch is provided.
        """

        kwargs = self.kwargs.copy()
        kwargs.update(override_kwargs)

        # Convtran init needs a config file.
        if self.model_name == 'ConvTran':
            x_batch, y_batch = first_batch
            config = {
                'data_path': 'Dataset/UEA/',
                'output_dir': 'Results/2025-09-18_17-00-00',  # se Setup criar timestamp
                'Norm': False,
                'val_ratio': 0.2,
                'print_interval': 10,
                'Net_Type': ['C-T'],
                'emb_size': 16,
                'dim_ff': 256,
                'num_heads': 8,
                'Fix_pos_encode': 'tAPE',
                'Rel_pos_encode': 'eRPE',
                'epochs': 100,
                'batch_size': 16,
                'lr': 1e-3,
                'dropout': 0.01,
                'val_interval': 2,
                'key_metric': 'accuracy',
                'gpu': 0,
                'console': False,
                'seed': 1234,
                'device': 'cuda:0',  # geralmente Setup adiciona isso
                'Data_shape': x_batch.shape
            }

            return self.model_class(config, num_classes=1)
        
        elif self.model_name == 'TSERMamba':

            # Default parameters from parser (matching TSERMamba signature)
            config = dict(
                enc_in=enc_in,
                seq_len=seq_len,
                projected_space=64,
                rescale_size=64,
                patch_size=8,
                dropout=0.1,
                initial_focus=1.0,
                d_state=256,
                dconv=2,
                e_fact=1,
                num_mambas=1,
                flip_dir=2,
                output_dim=1
            )

            # Allow overrides from constructor kwargs or get_model()
            config.update(kwargs)

            return self.model_class(**config)
        else:
            if first_batch is not None:
                kwargs["input_size"] = self.infer_input_size(first_batch)

            return self.model_class(**kwargs)
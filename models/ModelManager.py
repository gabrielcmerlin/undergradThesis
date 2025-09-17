class ModelManager:

    def __init__(self, model_name: str, **kwargs):

        self.model_name = model_name
        self.model_class = self._get_model_class()
        self.kwargs = kwargs

    def _get_model_class(self):
        
        import importlib
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

    def get_model(self, first_batch=None, **override_kwargs):
        """
        Instantiate the model, automatically computing input_size if first_batch is provided.
        """

        kwargs = self.kwargs.copy()
        kwargs.update(override_kwargs)

        if first_batch is not None:
            kwargs["input_size"] = self.infer_input_size(first_batch)

        return self.model_class(**kwargs)
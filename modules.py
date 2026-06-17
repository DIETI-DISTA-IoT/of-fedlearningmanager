import torch.nn as nn
import torch.nn.init as init


class WeightInitMixin:
    """Shared weight-initialisation logic for all global-model architectures.

    Only ``nn.Linear`` layers are explicitly initialised; other layers
    (e.g. ``nn.Conv1d`` in the CNN) keep their PyTorch defaults.  The global
    model's weights are overwritten by aggregation after the first round, so
    this only seeds the model used before the first successful aggregation.
    """

    def initialize_weights(self, strategy='xavier'):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if strategy == 'xavier':
                    # Xavier Initialization
                    init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        init.zeros_(m.bias)
                elif strategy == 'he':
                    # He Initialization
                    init.kaiming_normal_(m.weight)
                    if m.bias is not None:
                        init.zeros_(m.bias)
                elif strategy == 'normal':
                    # Normal Initialization
                    init.normal_(m.weight, 0, 0.01)
                    if m.bias is not None:
                        init.zeros_(m.bias)
                else:
                    raise ValueError(f"Unknown local initialization strategy: {strategy}")


class MLP(WeightInitMixin, nn.Module):
    def __init__(self, **kwargs):
        super(MLP, self).__init__()
        input_dim = kwargs.get('input_dim', 40)
        h_dim = kwargs.get('h_dim', 128)
        dropout = kwargs.get('dropout', 0.1)
        num_layers = kwargs.get('num_layers', 1)
        layer_norm = kwargs.get('layer_norm', False)
        output_dim = kwargs.get('output_dim', 1)
        self.mode = kwargs.get('mode')

        main_stream = []

        if layer_norm:
            main_stream.append(nn.LayerNorm(input_dim))

        curr_output_dim = h_dim
        curr_main_input_dim = input_dim

        for _ in range(num_layers):
            main_stream.append(nn.Linear(curr_main_input_dim, curr_output_dim))
            main_stream.append(nn.ReLU())
            main_stream.append(nn.Dropout(dropout))
            curr_main_input_dim = curr_output_dim
            curr_output_dim = curr_output_dim // 2

        # manifold layer:
        curr_output_dim = 2
        self.manifold_layer = nn.Linear(curr_main_input_dim, curr_output_dim)
        curr_main_input_dim = curr_output_dim

        # By avoiding non linear activations after the manifold layer, we induce a linear geometry on manifold frontiers!
        self.output_module = nn.Sequential(
            nn.Linear(curr_main_input_dim, output_dim))

        self.main_stream = nn.Sequential(*main_stream)

    def forward(self, x):

        main = self.main_stream(x)
        manifold = self.manifold_layer(main)
        return self.output_module(manifold), manifold.detach()


class CNN1D(WeightInitMixin, nn.Module):
    """1D-CNN classifier sharing the same 2D manifold bottleneck as the MLP.

    Kept parameter-compatible (module names / shapes) with the consumer's
    ``CNN1D`` so the FL global model's ``state_dict`` keys match the weights
    published by CNN participants.
    """

    def __init__(self, **kwargs):
        super(CNN1D, self).__init__()
        input_dim = kwargs.get('input_dim', 40)
        h_dim = kwargs.get('h_dim', 128)
        dropout = kwargs.get('dropout', 0.1)
        num_layers = kwargs.get('num_layers', 1)
        layer_norm = kwargs.get('layer_norm', False)
        output_dim = kwargs.get('output_dim', 1)
        self.mode = kwargs.get('mode')

        self.layer_norm = nn.LayerNorm(input_dim) if layer_norm else None

        conv_stream = []
        in_channels = 1
        out_channels = max(h_dim // (2 ** (num_layers - 1)), 8) if num_layers > 0 else h_dim
        curr_length = input_dim

        for _ in range(num_layers):
            conv_stream.append(nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1))
            conv_stream.append(nn.ReLU())
            conv_stream.append(nn.Dropout(dropout))
            conv_stream.append(nn.MaxPool1d(kernel_size=2))
            curr_length = max(curr_length // 2, 1)
            in_channels = out_channels
            out_channels = out_channels * 2

        self.conv_stream = nn.Sequential(*conv_stream)
        flat_dim = in_channels * curr_length

        # manifold layer:
        self.manifold_layer = nn.Linear(flat_dim, 2)

        # By avoiding non linear activations after the manifold layer, we induce a linear geometry on manifold frontiers!
        self.output_module = nn.Sequential(
            nn.Linear(2, output_dim))

    def forward(self, x):

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # add channel dimension: (batch, 1, input_dim)
        main = self.conv_stream(x.unsqueeze(1))
        main = main.flatten(1)
        manifold = self.manifold_layer(main)
        return self.output_module(manifold), manifold.detach()


class ResNetBlock(nn.Module):
    """Pre-activation residual block, as per Gorishniy et al. (2021)."""

    def __init__(self, dim, hidden_dim, dropout, layer_norm):
        super(ResNetBlock, self).__init__()
        self.normalization = nn.LayerNorm(dim) if layer_norm else nn.Identity()
        self.linear_first = nn.Linear(dim, hidden_dim)
        self.linear_second = nn.Linear(hidden_dim, dim)
        self.dropout_first = nn.Dropout(dropout)
        self.dropout_second = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, x):
        z = self.normalization(x)
        z = self.linear_first(z)
        z = self.activation(z)
        z = self.dropout_first(z)
        z = self.linear_second(z)
        z = self.dropout_second(z)
        return x + z


class TabResNet(WeightInitMixin, nn.Module):
    """Tabular ResNet baseline (Gorishniy et al., 2021, "Revisiting Deep
    Learning Models for Tabular Data"): a stack of pre-activation residual
    MLP blocks operating on a fixed-width main representation.

    Kept parameter-compatible (module names / shapes) with the consumer's
    ``TabResNet`` so the FL global model's ``state_dict`` keys match the
    weights published by ResNet participants."""

    def __init__(self, **kwargs):
        super(TabResNet, self).__init__()
        input_dim = kwargs.get('input_dim', 40)
        h_dim = kwargs.get('h_dim', 128)
        dropout = kwargs.get('dropout', 0.1)
        num_layers = kwargs.get('num_layers', 1)
        layer_norm = kwargs.get('layer_norm', False)
        output_dim = kwargs.get('output_dim', 1)
        self.mode = kwargs.get('mode')

        self.input_layer = nn.Linear(input_dim, h_dim)

        self.blocks = nn.ModuleList([
            ResNetBlock(dim=h_dim, hidden_dim=h_dim * 2, dropout=dropout, layer_norm=layer_norm)
            for _ in range(num_layers)
        ])

        self.final_normalization = nn.LayerNorm(h_dim) if layer_norm else nn.Identity()
        self.final_activation = nn.ReLU()

        # manifold layer:
        self.manifold_layer = nn.Linear(h_dim, 2)

        # By avoiding non linear activations after the manifold layer, we induce a linear geometry on manifold frontiers!
        self.output_module = nn.Sequential(
            nn.Linear(2, output_dim))

    def forward(self, x):

        main = self.input_layer(x)
        for block in self.blocks:
            main = block(main)
        main = self.final_activation(self.final_normalization(main))
        manifold = self.manifold_layer(main)
        return self.output_module(manifold), manifold.detach()


# Dispatch table mirroring the consumer's brain.py model selection.  The FL
# global model MUST be the same architecture as the participants, otherwise
# the aggregated ``state_dict`` keys/shapes will not match the weights the
# consumers publish.
_MODEL_TYPES = {
    'mlp': MLP,
    'cnn': CNN1D,
    'resnet': TabResNet,
}


def build_model(**kwargs):
    """Construct the global model for the architecture selected via config.

    Dispatches on ``model_type`` (``mlp`` / ``cnn`` / ``resnet``) exactly as
    the consumer's ``brain.py`` does, so the global model's parameters line up
    with the per-vehicle weights published for FL aggregation.  Unknown or
    missing values fall back to ``mlp``.
    """
    model_type = str(kwargs.get('model_type', 'mlp')).lower()
    model_cls = _MODEL_TYPES.get(model_type, MLP)
    return model_cls(**kwargs)

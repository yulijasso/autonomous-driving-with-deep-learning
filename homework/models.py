from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

HOMEWORK_DIR = Path(__file__).resolve().parent
INPUT_MEAN = [0.2788, 0.2657, 0.2629]
INPUT_STD = [0.2064, 0.1944, 0.2252]


class MLPPlanner(nn.Module):
    def __init__(
        self,
        n_track: int = 10,
        n_waypoints: int = 3,
    ):
        """
        Args:
            n_track (int): number of points in each side of the track
            n_waypoints (int): number of waypoints to predict
        """
        super().__init__()

        self.n_track = n_track
        self.n_waypoints = n_waypoints

        # Enhanced MLP with additional layers and dropout for regularization
        self.mlp = nn.Sequential(
            nn.Linear(2 * n_track * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_waypoints * 2)
        )

    def forward(
        self,
        track_left: torch.Tensor,
        track_right: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Predicts waypoints from the left and right boundaries of the track.

        Args:
            track_left (torch.Tensor): shape (b, n_track, 2)
            track_right (torch.Tensor): shape (b, n_track, 2)

        Returns:
            torch.Tensor: future waypoints with shape (b, n_waypoints, 2)
        """
        # Concatenate left and right track boundaries along the last dimension
        x = torch.cat((track_left, track_right), dim=-1)  # shape: (b, n_track, 4)
        
        # Flatten the last two dimensions
        x = x.view(x.size(0), -1)  # shape: (b, 2 * n_track * 2)

        # Pass through the MLP
        x = self.mlp(x)

        # Reshape the output to (b, n_waypoints, 2)
        waypoints_pred = x.view(-1, self.n_waypoints, 2)

        return waypoints_pred



class TransformerPlanner(nn.Module):
    def __init__(
        self,
        n_track: int = 10,
        n_waypoints: int = 3,
        d_model: int = 64,
        n_heads: int = 4,
        num_layers: int = 3,
        dropout: float = 0.1,
        use_curvature: bool = False  # New argument for curvature-based features
    ):
        super().__init__()

        self.n_track = n_track
        self.n_waypoints = n_waypoints
        self.d_model = d_model
        self.use_curvature = use_curvature

        # Query embeddings for waypoints (learnable embeddings)
        self.query_embed = nn.Embedding(n_waypoints, d_model)

        # Additional track feature encoding (optional)
        if self.use_curvature:
            self.curvature_encoder = nn.Linear(3, d_model)  # Assuming curvature is a feature

        # Linear layer to encode track points (left/right boundary)
        self.track_encoder = nn.Linear(2, d_model)

        # Transformer Encoder Layer (Multi-head self-attention)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dropout=dropout)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Transformer Decoder Layer (Cross-attention)
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=n_heads, dropout=dropout)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Final linear projection to output 2D waypoints (x, y)
        self.fc_out = nn.Linear(d_model, 2)

    def forward(self, track_left: torch.Tensor, track_right: torch.Tensor, curvature: torch.Tensor = None) -> torch.Tensor:
        """
        Predicts waypoints from the left and right track boundaries.

        Args:
            track_left (torch.Tensor): shape (batch_size, n_track, 2)
            track_right (torch.Tensor): shape (batch_size, n_track, 2)
            curvature (torch.Tensor, optional): curvature features if `use_curvature=True`, shape (batch_size, n_track, 1)

        Returns:
            torch.Tensor: future waypoints (batch_size, n_waypoints, 2)
        """

        batch_size = track_left.shape[0]

        # Combine track boundaries (concat left & right): shape (batch_size, 2*n_track, 2)
        track_boundaries = torch.cat([track_left, track_right], dim=1)

        # Optional curvature-based feature encoding
        if self.use_curvature and curvature is not None:
            curvature_embedded = self.curvature_encoder(curvature)  # shape (batch_size, n_track, d_model)
            track_boundaries = torch.cat([track_boundaries, curvature_embedded], dim=-1)

        # Encode track boundaries into d_model features: shape (batch_size, 2*n_track, d_model)
        track_embedded = self.track_encoder(track_boundaries)

        # Transformer encoder: shape (batch_size, 2*n_track, d_model)
        encoded_track = self.encoder(track_embedded.permute(1, 0, 2))  # (seq_len, batch, d_model)

        # Query embeddings for waypoints (learnable queries)
        queries = self.query_embed.weight.unsqueeze(1).repeat(1, batch_size, 1)  # (n_waypoints, batch_size, d_model)

        # Transformer decoder: attends queries over encoded track features
        decoded_output = self.decoder(queries, encoded_track)  # (n_waypoints, batch_size, d_model)

        # Output layer: transform d_model into (x, y) waypoints
        waypoints = self.fc_out(decoded_output.permute(1, 0, 2))  # (batch_size, n_waypoints, 2)

        return waypoints


class CNNPlanner(nn.Module):
    def __init__(self, n_waypoints: int = 3):
        super().__init__()

        self.n_waypoints = n_waypoints

        # Register input normalization buffers
        self.register_buffer("input_mean", torch.tensor(INPUT_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("input_std", torch.tensor(INPUT_STD).view(1, 3, 1, 1), persistent=False)

        # Convolutional feature extractor
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),  # (B, 16, 48, 64)
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # (B, 32, 24, 32)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # (B, 64, 12, 16)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),# (B, 128, 6, 8)
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),# (B, 256, 3, 4)
            nn.ReLU(),
        )

        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(256 * 3 * 4, 512),  # Flatten CNN output to 512 features
            nn.ReLU(),
            nn.Linear(512, 256), 
            nn.ReLU(),
            nn.Linear(256, n_waypoints * 2)  # Output (B, n_waypoints * 2)
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image (torch.FloatTensor): shape (B, 3, 96, 128), values in [0, 1]

        Returns:
            torch.FloatTensor: future waypoints with shape (B, n_waypoints, 2)
        """
        # Normalize input image
        x = (image - self.input_mean) / self.input_std  

        # Feature extraction
        x = self.conv_layers(x)

        # Flatten and pass through fully connected layers
        x = x.view(x.size(0), -1)  # (B, 256 * 3 * 4)
        x = self.fc_layers(x)

        # Reshape to (B, n_waypoints, 2)
        return x.view(-1, self.n_waypoints, 2)
        
MODEL_FACTORY = {
    "mlp_planner": MLPPlanner,
    "transformer_planner": TransformerPlanner,
    "cnn_planner": CNNPlanner,
}


def load_model(
    model_name: str,
    with_weights: bool = False,
    **model_kwargs,
) -> torch.nn.Module:
    """
    Called by the grader to load a pre-trained model by name
    """
    m = MODEL_FACTORY[model_name](**model_kwargs)

    if with_weights:
        model_path = HOMEWORK_DIR / f"{model_name}.th"
        assert model_path.exists(), f"{model_path.name} not found"

        try:
            m.load_state_dict(torch.load(model_path, map_location="cpu"))
        except RuntimeError as e:
            raise AssertionError(
                f"Failed to load {model_path.name}, make sure the default model arguments are set correctly"
            ) from e

    # limit model sizes since they will be zipped and submitted
    model_size_mb = calculate_model_size_mb(m)

    if model_size_mb > 20:
        raise AssertionError(f"{model_name} is too large: {model_size_mb:.2f} MB")

    return m


def save_model(model: torch.nn.Module) -> str:
    """
    Use this function to save your model in train.py
    """
    model_name = None

    for n, m in MODEL_FACTORY.items():
        if type(model) is m:
            model_name = n

    if model_name is None:
        raise ValueError(f"Model type '{str(type(model))}' not supported")

    output_path = HOMEWORK_DIR / f"{model_name}.th"
    torch.save(model.state_dict(), output_path)

    return output_path


def calculate_model_size_mb(model: torch.nn.Module) -> float:
    """
    Naive way to estimate model size
    """
    return sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024

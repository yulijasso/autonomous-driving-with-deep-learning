import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from datetime import datetime
from pathlib import Path
import torch.utils.tensorboard as tb
from torch.optim import Adam
from models import MLPPlanner, TransformerPlanner, CNNPlanner, load_model, save_model
from homework.datasets.road_dataset import load_data
from metrics import PlannerMetric

def weighted_l1_loss(pred, target, weight_longitudinal, weight_lateral):
    """
    Custom L1 loss function that weights longitudinal and lateral errors differently.

    Args:
        pred (torch.Tensor): Predicted waypoints (batch_size, n_waypoints, 2).
        target (torch.Tensor): Ground truth waypoints (batch_size, n_waypoints, 2).
        weight_longitudinal (float): Weight for longitudinal (x-axis) error.
        weight_lateral (float): Weight for lateral (y-axis) error.
    
    Returns:
        torch.Tensor: Weighted L1 loss.
    """
    diff = pred - target
    loss = (weight_longitudinal * torch.abs(diff[..., 0]) + weight_lateral * torch.abs(diff[..., 1])).mean()
    return loss

def train(
    exp_dir: str = "logs",
    model_name: str = "transformer_planner",
    num_epoch: int = 50,
    lr: float = 1e-3,
    batch_size: int = 128,
    seed: int = 2024,
    transform_pipeline=None, 
    num_workers=2,
    weight_longitudinal=5.0,  # Emphasize longitudinal (x) error
    weight_lateral=1.0,  # Emphasize lateral (y) error
    **kwargs
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)

    log_dir = Path(exp_dir) / f"{model_name}_{datetime.now().strftime('%m%d_%H%M%S')}"
    logger = tb.SummaryWriter(log_dir)

    # Select model based on user choice
    model_class = {
        "transformer_planner": TransformerPlanner,
        "mlp_planner": MLPPlanner,
        "cnn_planner": CNNPlanner,
    }.get(model_name.lower(), TransformerPlanner)  # Default to TransformerPlanner

    model = model_class(n_waypoints=3, **kwargs).to(device)
    model.train()

    try:
        train_loader = load_data("drive_data/train", shuffle=True, batch_size=batch_size, num_workers=num_workers)
        val_loader = load_data("drive_data/val", shuffle=False, batch_size=batch_size, num_workers=num_workers)
    except Exception as e:
        print(f"Data loading error: {e}")
        return

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    global_step = 0

    for epoch in range(num_epoch):
        model.train()
        train_losses = []
        train_metrics = PlannerMetric()

        for batch in train_loader:
            try:
                waypoints = batch.get('waypoints', torch.empty(0)).to(device)
                
                if waypoints.numel() == 0:
                    continue  # Skip if waypoints are empty
                
                optimizer.zero_grad()

                if isinstance(model, CNNPlanner):
                    input_data = batch['image'].to(device)  # Assuming CNNPlanner uses image input
                    predicted_waypoints = model(input_data)
                else:
                    track_left = batch.get('track_left', torch.empty(0)).to(device)
                    track_right = batch.get('track_right', torch.empty(0)).to(device)
                    
                    if track_left.numel() == 0 or track_right.numel() == 0:
                        continue  # Skip if track_left or track_right is empty
                    
                    predicted_waypoints = model(track_left=track_left, track_right=track_right)

                loss = weighted_l1_loss(predicted_waypoints, waypoints, weight_longitudinal, weight_lateral)
                loss.backward()
                optimizer.step()

                train_losses.append(loss.item())
                logger.add_scalar("train_loss", loss.item(), global_step)
                global_step += 1

                labels_mask = torch.ones_like(waypoints[..., 0], dtype=torch.bool)
                train_metrics.add(predicted_waypoints, waypoints, labels_mask)

            except KeyError as e:
                print(f"KeyError during training: {e}")
            except RuntimeError as e:
                print(f"RuntimeError during training: {e}")
            except Exception as e:
                print(f"Unexpected error during training: {e}")

        train_metric_vals = train_metrics.compute()
        logger.add_scalars("Train Metrics", train_metric_vals, epoch)

        model.eval()
        val_losses = []
        val_metrics = PlannerMetric()

        with torch.no_grad():
            for batch in val_loader:
                try:
                    waypoints = batch.get('waypoints', torch.empty(0)).to(device)

                    if waypoints.numel() == 0:
                        continue

                    if isinstance(model, CNNPlanner):
                        input_data = batch['image'].to(device)  # Assuming CNNPlanner uses image input
                        predicted_waypoints = model(input_data)
                    else:
                        track_left = batch.get('track_left', torch.empty(0)).to(device)
                        track_right = batch.get('track_right', torch.empty(0)).to(device)
                        
                        if track_left.numel() == 0 or track_right.numel() == 0:
                            continue  # Skip if track_left or track_right is empty
                        
                        predicted_waypoints = model(track_left=track_left, track_right=track_right)

                    loss = weighted_l1_loss(predicted_waypoints, waypoints, weight_longitudinal, weight_lateral)
                    val_losses.append(loss.item())

                    labels_mask = torch.ones_like(waypoints[..., 0], dtype=torch.bool)
                    val_metrics.add(predicted_waypoints, waypoints, labels_mask)

                except KeyError as e:
                    print(f"KeyError during validation: {e}")
                except RuntimeError as e:
                    print(f"RuntimeError during validation: {e}")
                except Exception as e:
                    print(f"Unexpected error during validation: {e}")

        val_metric_vals = val_metrics.compute()
        logger.add_scalars("Validation Metrics", val_metric_vals, epoch)

        avg_train_loss = np.mean(train_losses) if train_losses else float('inf')
        avg_val_loss = np.mean(val_losses) if val_losses else float('inf')
        logger.add_scalar("average_train_loss", avg_train_loss, epoch)
        logger.add_scalar("average_val_loss", avg_val_loss, epoch)

        print(f"Epoch {epoch + 1}/{num_epoch} | train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f}, "
              f"train_metrics={train_metric_vals}, val_metrics={val_metric_vals}")

    save_model(model)
    torch.save(model.state_dict(), log_dir / f"{model_name}.th")
    print(f"Model saved to {log_dir / f'{model_name}.th'}")


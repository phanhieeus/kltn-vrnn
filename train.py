import os
import torch
from tqdm import tqdm
import wandb

def kl_annealing(epoch, warmup=20):
    return min(1.0, epoch / warmup)

def train(model, dataloader, optimizer, epochs=100, device='cpu'):
    model.train()
    os.makedirs("checkpoints", exist_ok=True)
    
    pbar = tqdm(range(epochs), desc="Training")
    for epoch in pbar:
        epoch_total_loss = 0.0
        epoch_recon_loss = 0.0
        epoch_kld_loss = 0.0
        beta = kl_annealing(epoch)
        
        for x in dataloader:
            # x shape from dataloader: (B, T, x_dim)
            x = x.permute(1, 0, 2) # (T, B, x_dim)
            x = x.to(device)
            
            optimizer.zero_grad()
            loss, recon_loss, kld_loss = model(x, beta)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            
            epoch_total_loss += loss.item()
            epoch_recon_loss += recon_loss.item()
            epoch_kld_loss += kld_loss.item()
        
        avg_loss = epoch_total_loss / len(dataloader)
        avg_recon = epoch_recon_loss / len(dataloader)
        avg_kld = epoch_kld_loss / len(dataloader)

        wandb.log({
            "total_loss": avg_loss,
            "recon_loss": avg_recon,
            "kld_loss": avg_kld,
            "beta": beta
        })
        
        pbar.set_postfix({
            "Loss": f"{avg_loss:.4f}",
            "Recon": f"{avg_recon:.4f}",
            "KLD": f"{avg_kld:.4f}",
            "Beta": f"{beta:.4f}"
        })
        
        # Save checkpoint every epochs / 4
        if (epoch + 1) % (epochs // 4) == 0:
            checkpoint_path = f"checkpoints/vrnn_epoch_{epoch+1}.pth"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            
            # Log as Artifact
            checkpoint_artifact = wandb.Artifact(
                name=f"model-checkpoint-epoch-{epoch+1}", 
                type="model",
                description=f"Model checkpoint at epoch {epoch+1}"
            )
            checkpoint_artifact.add_file(checkpoint_path)
            wandb.log_artifact(checkpoint_artifact)

    
    # Save final model
    final_model_path = "checkpoints/vrnn_final.pth"
    torch.save(model.state_dict(), final_model_path)
    
    final_artifact = wandb.Artifact(
        name="vrnn-final-model", 
        type="model",
        description="Final trained VRNN model"
    )
    final_artifact.add_file(final_model_path)
    wandb.log_artifact(final_artifact)

    model.eval()
    return model

if __name__ == "__main__":
    # This block can be used for local testing if needed
    pass


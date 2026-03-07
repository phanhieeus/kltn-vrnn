def kl_annealing(epoch, warmup_epochs=20):
    return min(1.0, epoch / warmup_epochs)

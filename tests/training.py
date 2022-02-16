import torch
import logging

def train_dynamics(dataset, args, dyn_model, get_state_residual, optimizer):
    """
    Train the dynamics model and validate it self-generated ground truth
    """
    train_size = int(len(dataset) * 0.8)
    val_size = len(dataset) - train_size
    train_subset, val_subset = torch.utils.data.random_split(
        dataset, [train_size, val_size])

    train_dataloader = torch.utils.data.DataLoader(train_subset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_subset, batch_size=args.batch_size, shuffle=True)

    # thaw dyn_model
    for param in dyn_model.parameters():
        param.requires_grad = True

    for epoch in range(args.TRAIN_EPOCH):
        for i, (obs_t, action, reward, obs_tp1, done) in enumerate(train_dataloader):
            obs_tp1_cpy = obs_tp1.detach().clone()
            obs_t_cpy = obs_t.detach().clone()

            Yhat = get_state_residual(obs_t.cuda(), action.cuda(), dyn_model)
            Y = (obs_tp1_cpy - obs_t_cpy).cuda()

            optimizer.zero_grad()
            # MSE loss
            loss = (Y - Yhat).norm(2, dim=1) ** 2
            loss.mean().backward()
            optimizer.step()
            logging.info("iter size %d epoch %d iter %d loss %f", i, epoch, i, loss.mean().item())

    # freeze dyn_model
    for param in dyn_model.parameters():
        param.requires_grad = False

    validate_dynamics(val_dataloader, args, dyn_model, get_state_residual)

def validate_dynamics(val_dataloader, args, dyn_model, get_state_residual):
    for epoch in range(args.TRAIN_EPOCH):
        for i, (obs_t, action, reward, obs_tp1, done) in enumerate(val_dataloader):
            obs_tp1_cpy = obs_tp1.detach().clone()
            obs_t_cpy = obs_t.detach().clone()

            Yhat = get_state_residual(obs_t.cuda(), action.cuda(), dyn_model)
            Y = (obs_tp1_cpy - obs_t_cpy).cuda()

            # MSE loss
            loss = (Y - Yhat).norm(2, dim=1) ** 2
            logging.info("[validation] iter size %d epoch %d iter %d loss %f", i, epoch, i, loss.mean().item())


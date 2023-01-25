import torch
import torch.nn.functional as F

class SCELoss(torch.nn.Module):
    def __init__(self, alpha=0.5, beta=0.5):
        super(SCELoss, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.alpha = alpha
        self.beta = beta
        self.binary_cross_entropy=torch.nn.BCELoss(reduction='none')   

    def forward(self, predictions,ground_truths):
        # CCE
        ce = torch.sum(self.binary_cross_entropy(predictions, ground_truths))

        # RCE
        ground_truths_clamped=torch.where(ground_truths==0,torch.Tensor([1e-4]).cuda(),ground_truths) 
        rce = -1*torch.sum(predictions * torch.log(ground_truths_clamped))

        # Loss
        loss = self.alpha * ce + self.beta * rce

        return loss



import torch
import torch.nn




def coefficient_loss(anchor, anchor_rgb, anchor_bev, pos, pos_rgb, 
                      pos_bev, neg, neg_rgb, neg_bev, margin, device, a = 0.3, b=0.05):
    
    loss = torch.nn.functional.triplet_margin_loss(anchor.to(device), 
                                                           pos.to(device), 
                                                           neg.to(device), 
                                                           margin = margin,
                                                           p=1)   
    loss_rgb = torch.nn.functional.triplet_margin_loss(anchor_rgb.to(device), 
                                                           pos_rgb.to(device), 
                                                           neg_rgb.to(device), 
                                                           margin = margin,
                                                           p=1)   
    loss_bev = torch.nn.functional.triplet_margin_loss(anchor_bev.to(device), 
                                                           pos_bev.to(device), 
                                                           neg_bev.to(device), 
                                                           margin = margin,
                                                           p = 1)
    loss_all = (1-a-b)*loss+a*loss_rgb+b*loss_bev
    return loss_all, loss_rgb, loss_bev, loss


def quadruplet_loss(anchor, anchor_rgb, anchor_bev, 
                    pos, pos_rgb, pos_bev, 
                    neg, neg_rgb, neg_bev, neg_nearst,
                    neg_nearst_rgb, neg_nearst_bev, 
                    margin1, margin2, device):
    
    triplet_loss, loss_rgb, loss_bev, loss = coefficient_loss(anchor, anchor_rgb, anchor_bev, pos, pos_rgb, pos_bev,
                                                              neg, neg_rgb, neg_bev, margin1, device, a = 0.35, b=0.0)
    
    # triplet_loss2,_,_,_ = coefficient_loss(anchor, anchor_rgb, anchor_bev, pos, pos_rgb, pos_bev, 
    #                                  neg_nearst, neg_nearst_rgb, neg_nearst_bev, margin2, device, a = 0.4, b=0.0)
    
    triplet_loss2,_,_,_ = coefficient_loss( pos, pos_rgb, pos_bev, anchor, anchor_rgb, anchor_bev,
                                     neg_nearst, neg_nearst_rgb, neg_nearst_bev, margin2, device, a = 0.35, b=0.0)
    
    
    loss_all = triplet_loss + 2*triplet_loss2
    
    return loss_all, loss_rgb, loss_bev, loss, triplet_loss2
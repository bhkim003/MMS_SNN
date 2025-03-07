
import torch

import modules
    
class LIF_layer(torch.nn.Module):
    def __init__ (self, v_decay, v_threshold, v_reset_mode, sg_width, surrogate):
        super(LIF_layer, self).__init__()
        self.v_decay = v_decay
        self.v_threshold = v_threshold
        self.v_reset_mode = v_reset_mode
        self.sg_width = sg_width
        self.surrogate = surrogate
    def forward(self, input_current):
        Time = input_current.shape[0]
        v = torch.full_like(input_current[0], fill_value = 0.0, dtype = torch.float, requires_grad=False)
        post_spike = torch.full_like(input_current, fill_value = 0.0, device=input_current.device, dtype = torch.float, requires_grad=False) 
        
        for t in range(Time):
            v = v * self.v_decay + input_current[t]
            post_spike[t] = FIRE.apply(v - self.v_threshold, self.surrogate, self.sg_width) 
            if (self.v_reset_mode == 'soft_reset'):
                v = v - post_spike[t].detach() * self.v_threshold
            elif (self.v_reset_mode == 'hard_reset'):
                v = v*(1-post_spike[t].detach())
            else:
                assert False, f'{self.v_reset_mode} doesn\'t exist'

        return post_spike
    

class FIRE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v_minus_threshold, surrogate, sg_width):
        if surrogate == 'sigmoid':
            surrogate = 1
        elif surrogate == 'rectangle':
            surrogate = 2
        elif surrogate == 'rough_rectangle':
            surrogate = 3
        elif surrogate == 'hard_sigmoid':
            surrogate = 4
        else:
            assert False, 'surrogate doesn\'t exist'
        ctx.save_for_backward(v_minus_threshold,
                            torch.tensor([surrogate], requires_grad=False),
                            torch.tensor([sg_width], requires_grad=False)) # save before reset
        return (v_minus_threshold >= 0.0).float()

    @staticmethod
    def backward(ctx, grad_output):
        v_minus_threshold, surrogate, sg_width = ctx.saved_tensors
        surrogate=surrogate.item()
        sg_width=sg_width.item()

        if (surrogate == 1):
            #===========surrogate gradient function (sigmoid)
            alpha = sg_width 
            sig = torch.sigmoid(alpha*v_minus_threshold)
            grad_input = alpha*sig*(1-sig)*grad_output
        elif (surrogate == 2):
            # ===========surrogate gradient function (rectangle)
            grad_input = grad_output * (v_minus_threshold.abs() < sg_width/2).float() / sg_width
        elif (surrogate == 3):
            #===========surrogate gradient function (rough rectangle)
            grad_input[v_minus_threshold.abs() > sg_width/2] = 0
            grad_input = grad_output / sg_width
        elif (surrogate == 4):
            #===========surrogate gradient function (hard sigmoid)
            alpha = sg_width 
            sig = torch.clamp(alpha*v_minus_threshold * 0.2 + 0.5, min=0, max=1)
            grad_input = alpha*sig*(1-sig)*grad_output
        return grad_input, None, None
    

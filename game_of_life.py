# %%
import plotly.graph_objects as go
import torch
import numpy as np

HEIGHT, WIDTH = 16,16
num_frames = 100  
generations = torch.zeros((num_frames,HEIGHT, WIDTH), dtype=torch.int) 
#generations[0] = torch.randint(4, (HEIGHT, WIDTH), dtype=torch.int) < 1 # Fill the tensor with 25% ones and 75% zeros

generations[0][:3, :3] = torch.tensor([[0, 1, 0], [0, 0, 1], [1, 1, 1]], dtype=torch.int) #glider

def next_generation(current_gen, wrap=True):
    """
    Calculates the next generation of the Game of Life based on the current generation.

    Args:
        current_gen (torch.Tensor): The current generation of the Game of Life represented as a 2D tensor.
        wrap (bool, optional): Determines whether the grid wraps around at the edges. Defaults to True.

    Returns:
        torch.Tensor: The next generation of the Game of Life represented as a 2D tensor.
    """
    base_height, base_width = current_gen.shape
    pad_height, pad_width = base_height + 2, base_width + 2
    if wrap:
        current_gen = current_gen.unsqueeze(0).unsqueeze(0)
        padded_gen = torch.nn.functional.pad(current_gen, (1, 1, 1, 1), mode = 'circular')
        padded_gen = padded_gen.squeeze()
    else:
        padded_gen = torch.nn.functional.pad(current_gen, (1, 1, 1, 1), value=0)
        
    # could also just directly convolve with kernel [[1,1,1],[1,0,1],[1,1,1]
    # but for the sake of the exercise, only use torch.as_strided and torch.sum
    neighbours = torch.as_strided(padded_gen, 
                                  size=(base_height, base_width, 3, 3), 
                                  stride=(pad_width, 1, pad_width, 1))
    alive = neighbours.sum(dim=(-1,-2)) - current_gen 
    #need to minus current_gen to not accidentally count the current cell in alive count
    
    alive_two, alive_three = alive==2 , alive==3
    alive_stay_alive = current_gen & (alive_two | alive_three)
    dead_become_alive = (1-current_gen) & alive_three
    return alive_stay_alive | dead_become_alive

for i in range(1, num_frames):
    generations[i] = next_generation(generations[i-1])

# %%


# Convert the tensor to numpy for Plotly
data_np = generations.numpy()

# Create the frames for each time step
import plotly.graph_objects as go

frames = [go.Frame(data=go.Heatmap(z=data_np[i]),
                   name=str(i)) for i in range(num_frames)]


# Create the initial figure with looping enabled
fig = go.Figure(
    data=[go.Heatmap(z=data_np[0])],
    layout=go.Layout(
        updatemenus=[dict(type="buttons",
        buttons=[dict(label="Play",
                    method="animate",
                    args=[None, {"frame": {"duration": 100, "redraw": True}, "fromcurrent": True, "repeat": True}]),
                dict(label="Pause",
                    method="animate",
                    args=[[None], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}])])],
        sliders=[dict(steps=[dict(method='animate',
                            args=[[str(i)], 
                                {"frame": {"duration": 10, "redraw": True},
                                    "mode": "immediate"}],
                            label=str(i)) for i in range(num_frames)],
                transition={"duration": 10},
                currentvalue={"prefix": "Frame: "},
                len=1)]
    ),
    frames=frames
)

fig.show()

# %%

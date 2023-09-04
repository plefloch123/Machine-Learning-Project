import torch

# ============================================================== #
#                         Tensor indexing                        #
# ============================================================== #

batch_size = 10
features = 25
x = torch.rand((batch_size, features))

print(x[0].shape)  # x[0,:]

print(x[:, 0].shape)

print(x[2, 0:10])  # look at 3rd batch and take the first 10 features

x[0, 0] = 100  # Set tensor value

# Fancy indexing
x = torch.arange(10)
indices = [2, 5, 8]
print(x[indices])

x = torch.rand((3, 5))
rows = torch.tensor([1, 0])  # Second row, first row
cols = torch.tensor([4, 0])  # Fith column, first column
print(x[rows, cols].shape)

# More advanced indexing
x = torch.arange(10)
print(x[(x < 2) | (x > 8)])
print(x[(x < 2) & (x > 8)])
print(x[x.remainder(2) == 0])

# Useful operations
print(torch.where(x > 5, x, x*2))
print(torch.tensor([0, 0, 1, 2, 2, 4, 4]).unique())
print(x.ndimension())  # 5x5x5 (ndimension = 3)
print(x.numel())

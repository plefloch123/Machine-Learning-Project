import torch

# ============================================================== #
#               Tensor Math & Comparison Operations              #
# ============================================================== #

x = torch.tensor([1, 2, 3])
y = torch.tensor([9, 8, 7])

# Addition
z1 = torch.empty(3)
torch.add(x, y, out=z1)
print(z1)
z2 = torch.add(x, y)
print(z2)
z = x + y
print(z)

# Subtraction
z = x - y

# Division
z = torch.true_divide(x, y)

# Inplace operations
t = torch.zeros(3)
t.add_(x)
t += x  # t = t + x (not inplace operation, will create copy)

# Exponentiation
z = x.pow(2)
z = x ** 2

# Simple comparison
z = x > 0
z = x < 0
print(z)

# Matrix Multiplication
x1 = torch.rand((2, 5))
x2 = torch.rand((5, 3))
x3 = torch.mm(x1, x2) # 2*3 matrix
x3 = x1.mm(x2)
print(x3)

# Matrix exponentiation
matrix_exp = torch.rand(5, 5)
print(matrix_exp.matrix_power(3))

# Element wise mult
z = x * y
print(z)

# dot product
z = torch.dot(x, y)
print(z)

# Batch Matrix Multiplication
batch = 32
n = 10
m = 20
p = 30

tensor1 = torch.rand((batch, n, m))
tensor2 = torch.rand((batch, m, p))
out_bmm = torch.bmm(tensor1, tensor2)  # (batch, n, p)

# Example of Broadcasting
x1 = torch.rand((5, 5))
x2 = torch.rand((1, 5))

z = x1 - x2  # Does that make sense ? A matrix - vector ? Will actually expand x2 to 5 rows to make the calculus possible

# Other useful tensor operations
sum_x = torch.sum(x, dim=0)
values, indices = torch.max(x, dim=0)
print(values)
print(indices)
values, indices = torch.min(x, dim=0)
print(values)
print(indices)
abs_x = torch.abs(x)  # absolute value
z = torch.argmax(x, dim=0)
z = torch.argmin(x, dim=0)
mean_x = torch.mean(x.float(), dim=0)
z = torch.eq(x, y)  # Return True or False if equal at same index position (True,False,True)
sorted_y, indices = torch.sort(y, dim=0, descending=False)  # Ascending order
print(sorted_y)
print(indices)

z = torch.clamp(x, min=0, max=10)

x = torch.tensor([1, 0, 1, 1, 1], dtype=torch.bool)
z = torch.any(x)  # If at least one is greater then 0
print(z)
z = torch.all(x)  # If all greater then 0
print(z)




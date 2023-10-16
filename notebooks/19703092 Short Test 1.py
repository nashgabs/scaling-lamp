# %%
import numpy as np
import csv
import matplotlib.pyplot as plt
np.set_printoptions(precision=2)

# %%
def gauss_elim(N,augmat): #where N gives height
    matdim = (N, N+1)
    prow = 0 #pivot row
    x = np.zeros(N)
    
    for prow in range(0,N): #for each pivot row
        for row in range(prow+1, N): #for each row underneath the pivot row
            ratio = augmat[row, prow]/augmat[prow, prow] #find the ratio needed to gain 0 in the first position
            for col in range(prow, N+1): #for each element in current row
                augmat[row, col] -= ratio*augmat[prow,col] #M'_ij = M_ij-(R*M_kj) where N > j > k and N is the width of M
                
    for i in reversed(range(0,N)): #for each element in x-vector, starting with the last
        sum=0
        for j in reversed(range(0,N)): #for each factor in the row
            if(i!=j): #exclude factor of itself
                sum += x[j]*augmat[i,j] #sum of other x-vector elements * factors
        x[i] = (augmat[i,N]-sum)/augmat[i,i]
    return x

# %%
#open csv data
with open('data.csv', 'r') as f:
    r = csv.reader(f)
    data = list(r)
#create array containing values
val = np.array(data,dtype=float)

# %%
""" Question Two """
augmat = np.zeros([3,4])

#for the A matrix
def sum_x(x, n): #takes an x-value array and an index e.g. sum_x(val[:,0], diag)
    sum = 0
    for i in range(0,x.shape[0]):
        sum += x[i]**n
    return sum

def sum_xy(x, y, n): #takes an x-value array, y-value array and an index e.g. sum_y(val[:,0], val[:,1], diag)
    sum = 0
    for i in range(0, x.shape[0]):
        sum += y[i]*(x[i]**n)
    return sum

for diag in range(0,augmat.shape[0]+2): #for each +ve diagonal set (0 to 4)
    for htl in range(0, 3): #for the horizontal (0 to 2)
        for vtl in range(0, 3): #for the vertical (0 to 2)
            if htl+vtl == diag: #if (htl,vtl) lies on the +ve diagonal
                augmat[vtl,htl] = sum_x(val[:,0], diag) #set the element to sum

for i in range(0, 3): #for the b matrix
    augmat[i, 3] = sum_xy(val[:,0], val[:,1], i)

augmat

# %%
a = gauss_elim(augmat.shape[0], augmat)

X1, X2 = val[:,0], np.linspace(-10,10,23)
Y1, Y2 = val[:,1], (a[0] + a[1]*X2 + a[2]*X2**2)
fig, ax = plt.subplots()
ax.plot(X2, Y2, color="c", alpha=0.5, linestyle='--', label='Fitted line')
ax.plot(X1, Y1, color="C1", marker='x', linestyle='none', label='Data')
ax.set_title('Fitting a quadratic function')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.legend(loc="upper left")
print('A0:', a[0], '\nA1:', a[1],
      '\nA2', a[2])



import numpy as np
import matplotlib.pyplot as plt
Y, X = np.ogrid[:100, :100]
x = 50
y = 60
z = np.sqrt((X - x)**2 + (Y-y)**2)
z_min, z_max = 0, np.abs(z).max()
c = plt.imshow(z, vmin = z_min, vmax = z_max,
    extent =[X.min(), X.max(), Y.min(), Y.max()],
    interpolation ='nearest', origin ='lower')
plt.colorbar(c)
  
plt.title('matplotlib.pyplot.imshow() function Example', 
                                     fontweight ="bold")
plt.show()


# coding: utf-8

# # Intro

# In this post, I will talk about one of unsupervised learning techniques called K-Means Clustering and this is quite easy to understand and to implement from scratch.

# Given some data points (2D in this example), we are passing an integer value for $k$ and divide the whole points into $k$ partitions that data points in each partition are closest to that k points. And once the model is made and done training, we can determine which group or partition (or class) a testing input data belongs to.

# ## Load libraries 

# In[723]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# <code>np.random.normal</code> function creates data points of size = size and mean of sigma, normally distributed at the center of first parameter.<br>In the following cell, x is centered at 0 with mean of 2 and size of 200 points. y is centered at -0.5 with mean of 3 and size of 200.<br>
# x and y should have the same size since we are to scatter plot of them.

# In[724]:


sigma, size = 1.2, 200
x = np.random.normal(0, 2, size)
y = np.random.normal(-0.5, 3, size)


# If we only plot our created data, it looks like this. (Note that the graph may differ on your end since <code>np.random.normal</code> creates points randomly.

# In[725]:


plt.scatter(x, y)
plt.xlim(-7, 7)
plt.ylim(-7, 7);


# # Function Definitions

# In[708]:


# compute the Euclidean distance between two points
def get_distance(x, y):
    return np.sqrt((y[0]-x[0])**2 + (y[1]-x[1])**2)


# return k randomly generated points
def k_points(n):
    return [{'prev':None, 'curr':(np.random.randint(-4, 4, 1)[0],np.random.randint(-4, 4, 1)[0]), 'points':[]} for _ in range(n)]


def update_position():
    """
    Compute the mean of data points in each k points.
    After reset points in each point since previous and current points don't depend on each other.
    """
    for point in p:
        point['prev'] = point['curr']
        point['curr'] = np.mean(point['points'], axis=0) if len(point['points']) > 0 else (0, 0)
        point['points'] = []


def done_moving():
    """
    Check if it needs another iteration.
    If previous coordinate and current is the same, more iterations are needed.
    It will return false if even one of k points have different prev and curr.
    """
    for point in p:
        if np.any(point['curr'] != point['prev']):
            return False
    return True


def assign_points(x, y):
    """
    Compute each distance between k points and given x1, y1 coordinate and assign it to the closest point.
    """
    for x1, y1 in zip(x, y):
        
        dist = [get_distance(point['curr'], [x1,y1]) for point in p]
        
        # Get the closest kth point from current x,y
        closest_point = dist.index(min(dist))
        p[closest_point]['points'].append([x1,y1])
    


# As mentioned, <code>get_distance</code> function assumes that the given point is in two dimension. If you want to work with larger dimension, you can just modify the function and <code>assign_points</code> only and everything else remains the same.

# # Application with 5 points

# Now let's create three points and plot with earlier x and y values.<br>
# Also these $k$ points are created randomly so it will differ.

# In[740]:


p = k_points(5)


# In[741]:


plt.xlim(-7, 7)
plt.ylim(-7, 7)
plt.scatter(x, y, c='y')

for i in range(len(p)):
    plt.scatter(p[i]['curr'][0], p[i]['curr'][1], s=80)
plt.show();


# should mention that it can have different result based on where the initial $k$ points are generated

# Next, let us loop 1000 times and see different plot images of $k$ points updating and moving to different positions. We are looping as well as checking for <code>done_moving</code> because though most times, the algorithm is guaranteed to converge, sometimes it doesn't and we will see an easy example after.

# In[742]:


plt.xlim(-7, 7)
plt.ylim(-7, 7)
i = 0
while not done_moving():
        
    assign_points(x, y)
    update_position()
    
    plt.scatter(x, y, c='y')
    
    for j in range(len(p)):
        plt.scatter(p[j]['curr'][0], p[j]['curr'][1], s=80)
    
    plt.pause(0.2)
    i += 1

print("Done at {}th iteration".format(i))
plt.show();


# Now that we've made a trained model p, let's input five randomly generated data points and predict each of its class (0, 1, 2).

# In[743]:


test_x = np.random.normal(0, 2, 5)
test_y = np.random.normal(-0.5, 3, 5)


# In[744]:


def predict_class(x, y):
    assign_points(test_x, test_y)
    for i, point in enumerate(p):
        for dat in point['points']:
            print('Data point at {} belongs in class {}'.format(dat,i))
        point['points'] = []
    


# In[746]:


predict_class(test_x, test_y)


# It works! You should note that depending on where these $k$ points are generated, the final graph can be different from each other. Let's look at one simple example next.

# In[747]:


x = np.array([[-1],[1]])
y = np.array([[0], [0]])
kx1, kx2 = np.array([0]), np.array([0])
ky1, ky2 = np.array([0]), np.array([1])


# In[748]:


plt.xlim(-2, 2)
plt.scatter(x, y)
plt.scatter(kx1, ky1, s=70)
plt.scatter(kx2, ky2, s=70);


# As you can see above, the yellow point is already in the center of two given data points and these two are closer to it than the green one that no update is needed. Let's look at same data points with different coordinates of two points.

# In[749]:


kx1, kx2 = np.array([0.5]), np.array([-0.5])
ky1, ky2 = np.array([0]), np.array([0])

plt.xlim(-2, 2)
plt.scatter(x, y)
plt.scatter(kx1, ky1, s=70)
plt.scatter(kx2, ky2, s=70);


# Left data point is closer to green while the right is closer to orange. Just like this example, the initial locations of $k$ points affect the final graph of the K-Means Clustering.

# # End Note

# The codes above can be much efficiently implemented but since this post is about giving intuition and showing how the algorithm works, I'm just going to leave it like as is.

# Again, thank you for reading the post and please let me know if you find any error.

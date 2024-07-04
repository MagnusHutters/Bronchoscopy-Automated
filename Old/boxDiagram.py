


import matplotlib.pyplot as plt
import numpy as np

# Generating random data for four different categories
np.random.seed(10)
#category1 = [320, 140, 125, 165, 240, 155, 280, 420]
#category2 = [340, 280, 210, 320, 260, 190, 250, 290]
#category3 = [450, 380, 410, 220, 580, 390, 360, 430]
#category4 = [290, 510, 460, 430, 520, 620, 520, 250]


#category1 = [280, 620, 420, 430, 360, 240, 260, 680]
#category2 = [390, 340, 510, 370, 340, 610, 535, 330]
#category3 = [430, 620, 460, 540, 680, 690, 320, 640]
#category4 = [430, 680, 690, 440, 640, 620, 810, 440]

#category1 = [500, 460, 240, 330, 450, 600, 350, 360]
#category2 = [400, 430, 330, 600, 550, 290, 520, 380]
#category3 = [490, 330, 300, 550, 520, 240, 370, 380]
#category4 = [130, 120, 140, 120, 130, 280, 110, 120]

#category1 = [640, 270, 460, 380, 230, 390, 710, 330]
#category2 = [520, 295, 570, 605, 330, 420, 355, 380]
#category3 = [480, 470, 690, 530, 430, 680, 615, 550]
#category4 = [525, 455, 675, 530, 795, 685, 595, 315]


category1 = [1.8, 1.2, 1.6, 1.5, 2.2, 2.1, 1.9, 2.0, 1.8, 2.6]
category2 = [0.9, 0.3, 0.8, 1.1, 1.3, 0.7, 0.8, 0.8, 0.8, 1.0]
category3 = [1.3, 0.7, 0.7, 0.9, 1.0, 1.5, 2.7, 1.2, 1.3, 1.2]
category4 = [0.9, 0.7, 0.8, 0.7, 1.1, 1.0, 1.3, 1.5, 2.0, 1.9]
category5 = [1.0, 1.1, 0.2, 0.3, 0.2, 0.3, 0.4, 0.0, 0.7, 0.1]
category6 = [0.0, 0.1, 0.0, 0.1, 0.2, 0.0, 0.1, 0.0, 0.0, 0.1]


# Combining the data into a list
data = [category1, category2, category3, category4, category5,category6]
for i in range(4):
    for j in range(8):
        pass
        #data[i][j] = data[i][j] *0.1

# Creating the box plot
plt.figure(figsize=(10, 6))
#plt.boxplot(data, labels=['Side-arm 1', 'Lower arm', 'Side-arm 2', 'Side-arm 3'],whis=[0, 100])
plt.boxplot(data, labels=['Upper arm', 'Side-arm 1','Middle Arm', 'Lower arm', 'Side-arm 2', 'Side-arm 3'], whis=[10, 90])
#plt.boxplot(data, labels=['up-up','up-down' , 'down-up', 'down-down'],whis=[0, 100])

#set the limit for the y-axis to 0-100
plt.ylim(0,3)


# Adding title and labels
plt.title('Distance reached in one second')
plt.xlabel('Location')
plt.ylabel('Distance (cm)')

# Showing the plot
plt.show()
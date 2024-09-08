#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

x0 = np.arange(0, 11)
y0 = x0 ** 3

mean = [69, 0]
cov = [[15, 8], [8, 15]]
np.random.seed(5)
x1, y1 = np.random.multivariate_normal(mean, cov, 2000).T
y1 += 180

x2 = np.arange(0, 28651, 5730)
r2 = np.log(0.5)
t2 = 5730
y2 = np.exp((r2 / t2) * x2)

x3 = np.arange(0, 21000, 1000)
r3 = np.log(0.5)
t31 = 5730
t32 = 1600
y31 = np.exp((r3 / t31) * x3)
y32 = np.exp((r3 / t32) * x3)

np.random.seed(5)
student_grades = np.random.normal(68, 15, 50)

#red line
plt.subplot(321)
plt.axis((0, 10, 0, 1000))
plt.plot(x0, y0, 'r-')

#magenta scatter
plt.subplot(322, title="Men's Height vs Weight")
plt.xlabel('Height (in)', fontsize=6)
plt.ylabel('Weight (lbs)', fontsize=6)
plt.plot(x1, y1, 'm.')

#exponential decay
plt.subplot(323, title='Exponential Decay of C-14')
plt.xlabel('Time (years)', fontsize=6)
plt.ylabel('Fraction Remaining', fontsize=6)
plt.plot(x2, y2,)
plt.xlim((0, 28650))
plt.yscale('log')


#radio active elements
plt.subplot(324, title='Exponential Decay of Radioactive Elements')
plt.plot(x3, y31, 'r--', label='C-14')
plt.plot(x3, y32, 'g', label='Ra-226')
plt.xlim((0, 20000))
plt.ylim((0, 1))
#legend
plt.legend(loc='upper right', frameon=True)
#labels
plt.xlabel('Time (years)', fontsize=6)
plt.ylabel('Fraction Remaining', fontsize=6)

#project A
plt.subplot(313, title='Project A')
plt.hist(student_grades, bins=10, edgecolor='black', range=(0, 100))
plt.xticks(np.arange(0, 101, 10))
plt.xlim((0, 100))
plt.ylabel('Grades', fontsize=6)
plt.xlabel('Number of Students', fontsize=6)


plt.suptitle('All in one')
plt.tight_layout()
plt.show()
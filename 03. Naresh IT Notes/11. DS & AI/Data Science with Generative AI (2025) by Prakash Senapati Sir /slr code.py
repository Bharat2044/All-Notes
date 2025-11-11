import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv(r"C:\Users\A3MAX SOFTWARE TECH\Desktop\WORK\1. KODI WORK\1. NARESH\1. MORNING BATCH\N_Batch -- 8.30AM _ AUG25\3. Apr' 25\21st- SLR\SIMPLE LINEAR REGRESSION\Salary_Data.csv")

x = dataset.iloc[:, :-1]  
y = dataset.iloc[:, -1] 

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

y_pred = regressor.predict(x_test)

plt.scatter(x_test, y_test, color = 'red')  # Real salary data (testing)
plt.plot(x_train, regressor.predict(x_train), color = 'blue')  # Regression line from training set
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# i want to predict the future 

m_slope = regressor.coef_
print(m_slope)

c_intercept = regressor.intercept_
print(c_intercept)

pred_12yr_emp_exp = m_slope * 12 + c_intercept
print(pred_12yr_emp_exp)

pred_20yr_emp_exp = m_slope * 20 + c_intercept
print(pred_20yr_emp_exp)







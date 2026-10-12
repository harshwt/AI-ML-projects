import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso #to remove overfitting 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import seaborn as sns


from google.colab import files
uploaded = files.upload()

df = pd.read_csv("data.csv")
df.dropna(inplace=True)
df.dropna(axis=1, inplace=True)

X = df.iloc[:, :-1]
y = df.iloc[: ,-1]

print(df)


xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.3, random_state=2)
reg = LinearRegression()
reg.fit(xtrain, ytrain)
lasso_reg = Lasso()
parameter = {'alpha': [10, 20, 30, 40, 60, 80, 100]}
lassocv = GridSearchCV(lasso_reg, parameter, scoring='neg_mean_squared_error', cv=5)
lassocv.fit(xtrain, ytrain)
#print(ridgecv.best_params_)
#print(ridgecv.best_score_)
print("Training score: ", reg.score(xtrain, ytrain))
print("Testing score: ", reg.score(xtest, ytest))

print(lassocv.score(xtrain, ytrain))
print(lassocv.best_score_)


#sns.displot(ridge_pred-ytest, kind='kde')

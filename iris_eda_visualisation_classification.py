
import pandas as pd 

#Load dataset

iris_df = pd.read_csv('Iris.csv') # memuat file csv sebagai data frame
iris_df.head()


# iris_df = iris_df.drop(columns='Id')
iris_df.drop(columns='Id', inplace=True) # menghapus kolom bernama 'Id'

"""#### Identify the shape of the datatset"""

info_baris_kolom =iris_df.shape # bentuk/dimensi dataset (baris,kolom)

"""#### Get the list of columns"""

nama_coloms = iris_df.columns # daftar nama kolom

"""#### Identify data types for each column"""

type_kolom =iris_df.dtypes # tipe data untuk tiap kolom

"""#### Get bassic dataset information"""

iris_df.info() # informasi dataset

"""#### Identify missing values"""


nilai_kosong = iris_df.isna().values.any() # mendeteksi keberadaan nilai kosong

"""#### Identify duplicate entries/rows"""

# iris_df[iris_df.duplicated(keep=False)] # tampilkan seluruh baris dengan duplikasi 
baris_duplicat= iris_df[iris_df.duplicated()] # tampilkan hanya baris duplikasi sekunder

jumlah_duplicat = iris_df.duplicated().value_counts() # hitung jumlah duplikasi data

"""#### Drop duplicate entries/rows"""

iris_df.drop_duplicates(inplace=True) # menghapus duplikasi data

"""#### Describe the dataset"""

iris_df.describe() # deskripsi data

"""#### Correlation Matrix"""

korelasi =iris_df.corr() # korelasi antar kolom

"""## Iris Dataset: Data Visualisation

#### Import Modules
"""

# Commented out IPython magic to ensure Python compatibility.
import matplotlib.pyplot as plt # visualisasi data
import seaborn as sns # visualisasi data

# output dari visualisasi data akan diarahkan ke notebook
# %matplotlib inline

"""#### Heatmap"""

sns.heatmap(data=iris_df.corr())

"""#### Bar Plot"""

iris_df['Species'].value_counts() # menghitung jumlah setiap species

iris_df['Species'].value_counts().plot.bar()
plt.tight_layout()
plt.show()

sns.countplot(data=iris_df, x='Species')
plt.tight_layout()
# sns.countplot?

"""#### Pie Chart"""

iris_df['Species'].value_counts().plot.pie(autopct='%1.1f%%', labels=None, legend=True)
plt.tight_layout()

"""#### Line Plot"""

fig,ax = plt.subplots(nrows=2, ncols=2, figsize=(8,8))

iris_df['SepalLengthCm'].plot.line(ax=ax[0][0])
ax[0][0].set_title('Sepal Length')

iris_df['SepalWidthCm'].plot.line(ax=ax[0][1])
ax[0][1].set_title('Sepal Width')

iris_df.PetalLengthCm.plot.line(ax=ax[1][0])
ax[1][0].set_title('Petal Length')

iris_df.PetalWidthCm.plot.line(ax=ax[1][1])
ax[1][1].set_title('Petal Width')

iris_df.plot()
plt.tight_layout()

"""#### Histogram"""

iris_df.hist(figsize=(6,6), bins=10)
plt.tight_layout()

"""#### Boxplot"""

iris_df.boxplot()
plt.tight_layout()

iris_df.boxplot(by="Species", figsize=(8,8))
plt.tight_layout()

"""#### Scatter Plot"""

sns.scatterplot(x='SepalLengthCm', y='SepalWidthCm', data=iris_df, hue='Species')
plt.tight_layout()

"""#### Pair Plot"""

sns.pairplot(iris_df, hue='Species', markers='+')
plt.tight_layout()

"""#### Violin Plot"""

sns.violinplot(data=iris_df, y='Species', x='SepalLengthCm', inner='quartile')
plt.tight_layout()

"""## Iris Dataset: Classification Models

#### Import Modules
"""

from sklearn.model_selection import train_test_split # pembagi dataset menjadi training dan testing set
from sklearn.metrics import accuracy_score,confusion_matrix, classification_report # evaluasi performa model

"""#### Dataset: Features & Class Label"""

X = iris_df.drop(columns='Species') # menempatkan features ke dalam variable X
X.head() # tampilkan 5 baris pertama

y = iris_df['Species'] # menempatkan class label (target) ke dalam variabel y
y.head() # tampilkan 5 baris pertama

"""#### Split the dataset into a training set and a testing set"""

# membagi dataset ke dalam training dan testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=10)

print('training dataset')
print(X_train.shape)
print(y_train.shape)
print()
print('testing dataset:')
print(X_test.shape)
print(y_test.shape)

"""#### K Nearest Neighbors"""

from sklearn.neighbors import KNeighborsClassifier

k_range = list(range(1,26))
scores = []
for k in k_range:
    model_knn = KNeighborsClassifier(n_neighbors=k) # konfigurasi algoritma
    model_knn.fit(X_train, y_train) # training model/classifier
    y_pred = model_knn.predict(X_test) # melakukan prediksi
    scores.append(accuracy_score(y_test, y_pred)) # evaluasi performa

plt.plot(k_range, scores)
plt.xlabel('Value of k for KNN')
plt.ylabel('Accuracy Score')
plt.title('Accuracy Scores for Values of k of k-Nearest-Neighbors')
plt.tight_layout()
plt.show()

model_knn = KNeighborsClassifier(n_neighbors=3) # konfigurasi algoritma
model_knn.fit(X_train,y_train) # training model/classifier
y_pred = model_knn.predict(X_test) # melakukan prediksi

"""##### Accuracy Score"""

print(accuracy_score(y_test, y_pred)) # evaluasi akurasi

"""##### Confusion Matrix"""

print(confusion_matrix(y_test, y_pred)) # evaluasi confusion matrix

"""##### Classification Report"""

print(classification_report(y_test, y_pred)) # evaluasi klasifikasi

"""#### Logistic Regression"""

from sklearn.linear_model import LogisticRegression

# model_logreg = LogisticRegression()
model_logreg = LogisticRegression(solver='lbfgs', multi_class='auto')
model_logreg.fit(X_train,y_train)
y_pred = model_logreg.predict(X_test)

"""##### Accuracy Score"""

print(accuracy_score(y_test, y_pred))

"""##### Confusion Matrix"""

print(confusion_matrix(y_test, y_pred))

"""##### Classification Report"""

print(classification_report(y_test, y_pred))

"""#### Support Vector Classifier"""

from sklearn.svm import SVC

# model_svc = SVC()
model_svc = SVC(gamma='scale')
model_svc.fit(X_train,y_train)
y_pred = model_svc.predict(X_test)

"""#### Decision Tree Classifier"""

from sklearn.tree import DecisionTreeClassifier

model_dt = DecisionTreeClassifier()
model_dt.fit(X_train,y_train)
y_pred = model_dt.predict(X_test)

"""#### Random Forest Classifier"""

from sklearn.ensemble import RandomForestClassifier

# model_rf = RandomForestClassifier()
model_rf = RandomForestClassifier(n_estimators=100)
model_rf.fit(X_train,y_train)
pred_rf = model_rf.predict(X_test)

"""#### Accuracy comparision for various models."""

models = [model_knn, model_logreg, model_svc, model_dt, model_rf]
accuracy_scores = []
for model in models:
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_scores.append(accuracy)
    
print(accuracy_scores)

plt.bar(['KNN', 'LogReg', 'SVC', 'DT', 'RF'],accuracy_scores)
plt.ylim(0.90,1.01)
plt.title('Accuracy comparision for various models', fontsize=15, color='r')
plt.xlabel('Models', fontsize=18, color='g')
plt.ylabel('Accuracy Score', fontsize=18, color='g')
plt.tight_layout()
plt.show()

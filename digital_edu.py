import pandas as pd

df = pd.read_csv('train.csv')
df.drop(['bdate', 'has_photo', 'has_mobile', 'followers_count', 'graduation', 'relation', 'life_main', 'people_main', 'city', 'last_seen', 'occupation_name', 'career_start', 'career_end'], axis = 1, inplace = True)

def  sex_apply(sex):
    if sex == 2:
        return 0
    return 1

df['sex'] = df['sex'].apply(sex_apply)

df['education_form'].fillna('Full-time', inplace = True)
df[list(pd.get_dummies(df['education_form']).columns)] = pd.get_dummies(df['education_form'])
df.drop(['education_form'], axis = 1, inplace = True)

def status_apply(status):
    if status == 'Undergraduate applicant':
        return 0
    if status == "Alumus (Speciallist)" or "Alumnus (Bachelor's)" or "Alumnus (Master's)":
        return 1
    if status == "Student (Specialist)" or "Student (Bachelor's)" or "student (Master's)":
        return 2
    return 3
df['education_status'] = df['education_status'].apply(status_apply)

print(df['langs'].value_counts())
def langs_apply(langs):
    if langs.find('English') != -1 and langs.find('Русский') != -1:
        return 1
    else:
        return 0
df['langs'] = df['langs'].apply(langs_apply)  

df['occupation_type'].fillna('university', inplace = True)
def occupation_type_apply(occupation_type):
    if occupation_type == 'university':
        return 0
    else:
        return 1
df['occupation_type'] = df['occupation_type'].apply(occupation_type_apply)
df.info()


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

x = df.drop('result', axis = 1)
y = df['result']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25)

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

classifier = KNeighborsClassifier(n_neighbors = 5)
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)
print('Процент правильно предсказанных исходов:', round(accuracy_score(y_test, y_pred), 2) * 100)
print('Confusion matrix:')
print(confusion_matrix(y_test, y_pred))

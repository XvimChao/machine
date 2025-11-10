import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

data = load_breast_cancer()

X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

Cs = np.logspace(-4, 4, 20)

acc_log = []
for c in Cs:
    model = LogisticRegression(penalty='l2', C=c, max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train)
    pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, pred)
    acc_log.append(acc)

best_c_log = Cs[np.argmax(acc_log)]
print(f"Лучший C для LogReg: {best_c_log}\nТочность: {max(acc_log)}")

acc_svm = []
for c in Cs:
    model = LinearSVC(penalty='l2', C=c, max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train)
    pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, pred)
    acc_svm.append(acc)

best_c_svm = Cs[np.argmax(acc_svm)]
print(f"Лучший C для LinearSVC: {best_c_svm}\nТочность: {max(acc_svm)}")


#Графики
plt.figure(figsize=(10, 6))
plt.plot(np.log10(Cs), acc_log, marker='o', label='Logistic Regression')
plt.plot(np.log10(Cs), acc_svm, marker='s', label='LinearSVC')
plt.xlabel('$log_{10}(C)$')
plt.ylabel('$Точность$')
plt.title('$LogReg$ $vs$ $LinearSVC$')
plt.legend()
plt.grid(True)
plt.show()


df_features = list(data.feature_names)
selected_cs_log = [0.001, best_c_log, 1000]
for c in selected_cs_log:
    model = LogisticRegression(C=c, max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train)
    coef = model.coef_[0]

    plt.figure(figsize=(12, 6))
    plt.bar(range(len(coef)), coef)
    plt.xticks(range(len(coef)), df_features, rotation=90)
    plt.xlabel('Признаки')
    plt.ylabel('Значение коэффициента')
    plt.title(f'Коэффициенты $LogReg$ для $C$ = {c}')
    plt.tight_layout()
    plt.show()


selected_cs_svm = [0.001, best_c_svm, 1000]
for c in selected_cs_svm:
    model = LinearSVC(C=c, random_state=42)
    model.fit(X_train_scaled, y_train)
    coef = model.coef_[0]

    plt.figure(figsize=(12, 6))
    plt.bar(range(len(coef)), coef)
    plt.xticks(range(len(coef)), df_features, rotation=90)
    plt.xlabel('Признаки')
    plt.ylabel('Значение коэффициента')
    plt.title(f'Коэффициенты логистической регрессии для $C$ = {c}')
    plt.title(f'Коэффициенты $LinearSVC$ для $C$ = {c}')
    plt.tight_layout()
    plt.show()


model_opt_log = LogisticRegression(C=best_c_log, max_iter=1000, random_state=42)
model_opt_log.fit(X_train_scaled, y_train)
abs_coef = np.abs(model_opt_log.coef_[0])
top2_idx = np.argsort(abs_coef)[-2:]
feat1, feat2 = df_features[top2_idx[0]], df_features[top2_idx[1]]
print(f"Выбранные признаки: {feat1}, {feat2}")

X_train_2d = X_train_scaled[:, top2_idx]
X_test_2d = X_test_scaled[:, top2_idx]



def plot_decision_boundary(ax, model, X, y, title):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.8, cmap='coolwarm')
    ax.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap='coolwarm')
    ax.set_xlabel(feat1)
    ax.set_ylabel(feat2)
    ax.set_title(title)


model_log_2d = LogisticRegression(penalty='l2', C=best_c_log, max_iter=1000, random_state=42)
model_log_2d.fit(X_train_2d, y_train)

model_svm_2d = LinearSVC(penalty='l2', C=best_c_svm, max_iter=1000, random_state=42)
model_svm_2d.fit(X_train_2d, y_train)

fig, axs = plt.subplots(1, 2, figsize=(16, 6))
plot_decision_boundary(axs[0], model_log_2d, X_train_2d, y_train, 'Граница принятия решений $LogReg$')
plot_decision_boundary(axs[1], model_svm_2d, X_train_2d, y_train, 'Граница принятия решений $LinearSVC$')
plt.tight_layout()
plt.show()
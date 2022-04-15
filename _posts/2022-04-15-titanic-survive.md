---
jupyter:
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
  language_info:
    codemirror_mode:
      name: ipython
      version: 3
    file_extension: .py
    mimetype: text/x-python
    name: python
    nbconvert_exporter: python
    pygments_lexer: ipython3
    version: 3.6.4
  nbformat: 4
  nbformat_minor: 4
---

::: {.cell .code _cell_guid="b1076dfc-b9ad-4769-8c92-a6c4dae69d19" _uuid="8f2839f25d086af736a60e9eeb907d3b93b6e0e5" execution="{\"iopub.status.busy\":\"2022-04-11T08:27:54.093282Z\",\"shell.execute_reply.started\":\"2022-04-11T08:27:54.093697Z\",\"iopub.status.idle\":\"2022-04-11T08:27:54.12605Z\",\"iopub.execute_input\":\"2022-04-11T08:27:54.093784Z\",\"shell.execute_reply\":\"2022-04-11T08:27:54.125408Z\"}" trusted="true"}
``` {.python}
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
```
:::

::: {.cell .markdown}
# 타이타닉 생존자 예측기
:::

::: {.cell .markdown}
연습: 타이타닉 데이터셋 다루기.
[https://homl.info/titanic.tgz에서](https://homl.info/titanic.tgz에서){.uri}
데이터를 받아 2챕터에서 한 것처럼 압축해제 하면 된다. \_train.csv와
test.csv 두개의 CSV파일을 받을 수 있다. 목표는 다른 열을 기반으로 생존
열을 예측할 수 있는 분류기를 훈련하는 것이다.
:::

::: {.cell .code execution="{\"iopub.status.busy\":\"2022-04-11T08:27:54.127413Z\",\"shell.execute_reply.started\":\"2022-04-11T08:27:54.127726Z\",\"iopub.status.idle\":\"2022-04-11T08:27:54.136428Z\",\"iopub.execute_input\":\"2022-04-11T08:27:54.127756Z\",\"shell.execute_reply\":\"2022-04-11T08:27:54.135498Z\"}" trusted="true"}
``` {.python}
from pathlib import Path
import pandas as pd
import tarfile
import urllib.request

def load_titanic_data():
    tarball_path = Path("datasets/titanic.tgz")
    if not tarball_path.is_file():
        Path("datasets").mkdir(parents=True, exist_ok=True)
        url = "https://github.com/ageron/data/raw/main/titanic.tgz"
        urllib.request.urlretrieve(url, tarball_path)
        with tarfile.open(tarball_path) as titanic_tarball:
            titanic_tarball.extractall(path="datasets")
    return [pd.read_csv(Path("datasets/titanic") / filename)
            for filename in ("train.csv", "test.csv")]
```
:::

::: {.cell .code execution="{\"iopub.status.busy\":\"2022-04-11T08:27:54.1377Z\",\"shell.execute_reply.started\":\"2022-04-11T08:27:54.137897Z\",\"iopub.status.idle\":\"2022-04-11T08:27:54.861904Z\",\"iopub.execute_input\":\"2022-04-11T08:27:54.137925Z\",\"shell.execute_reply\":\"2022-04-11T08:27:54.861147Z\"}" trusted="true"}
``` {.python}
train_data, test_data = load_titanic_data()
```
:::

::: {.cell .markdown}
데이터는 이미 텍스트셋과 훈련셋으로 나뉘어져 있지만, 테스트 데이터에는
레이블이 포함되어 있지 않다. 훈련 데이터를 이용해 최선의 모델을 만들어
테스트 데이터에 예측하여 케글에 올려 점수를 확인하는 것이다.

훈련셋의 맨 위 몇행을 확인해보자.
:::

::: {.cell .code execution="{\"iopub.status.busy\":\"2022-04-11T08:27:54.864076Z\",\"shell.execute_reply.started\":\"2022-04-11T08:27:54.864399Z\",\"iopub.status.idle\":\"2022-04-11T08:27:54.889265Z\",\"iopub.execute_input\":\"2022-04-11T08:27:54.864442Z\",\"shell.execute_reply\":\"2022-04-11T08:27:54.888379Z\"}" trusted="true"}
``` {.python}
train_data.head()
```
:::

::: {.cell .markdown}
각 속성의 의미는 다음과 같다.

-   **PassengerId**: 각 승객의 고유번호
-   \***Survived**: 타깃: 0는 사망, 1은 생존을 의미
-   **Pclass**: 승객의 클래스
-   **Name, Sex, Age**: 개인정보
-   **SibSp**: 승객의 형제, 배우자 수
-   **Parch**: 승객의 자식 및 부모 수
-   **Ticket**: 티켓 ID
-   **Fare**: 비용(파운드)
-   **Cabin**: 승객의 객실 번호
-   **Embarked**: 승객의 승선 장소목표는 이러한 속성들을 이용해 승객이
    살아남았는지 아닌지를 예측하는 것이다.

PassengerId 열을 인덱스 열로 명시적으로 설정해 보자.
:::

::: {.cell .markdown}
목표는 이러한 속성들을 이용해 승객이 살아남았는지 아닌지를 예측하는
것이다.
:::

::: {.cell .code execution="{\"iopub.status.busy\":\"2022-04-11T08:27:54.890593Z\",\"shell.execute_reply.started\":\"2022-04-11T08:27:54.892394Z\",\"iopub.status.idle\":\"2022-04-11T08:27:54.918888Z\",\"iopub.execute_input\":\"2022-04-11T08:27:54.892433Z\",\"shell.execute_reply\":\"2022-04-11T08:27:54.918047Z\"}" trusted="true"}
``` {.python}
# test_data의 PassengerID속성을 사용하기 위해 미리 복사해두기.
test_data_=test_data.copy()
# Survived속성이 없는 것이 확인 가능하다.
test_data_.info()
```
:::

::: {.cell .markdown}
PassengerId 열을 인덱스 열로 명시적으로 설정해 보자.
:::

::: {.cell .code execution="{\"iopub.status.busy\":\"2022-04-11T08:27:54.920575Z\",\"shell.execute_reply.started\":\"2022-04-11T08:27:54.92085Z\",\"iopub.status.idle\":\"2022-04-11T08:27:54.928249Z\",\"iopub.execute_input\":\"2022-04-11T08:27:54.920893Z\",\"shell.execute_reply\":\"2022-04-11T08:27:54.927217Z\"}" trusted="true"}
``` {.python}
train_data = train_data.set_index("PassengerId")
test_data = test_data.set_index("PassengerId")
```
:::

::: {.cell .code execution="{\"iopub.status.busy\":\"2022-04-11T08:27:54.930331Z\",\"shell.execute_reply.started\":\"2022-04-11T08:27:54.930834Z\",\"iopub.status.idle\":\"2022-04-11T08:27:54.948772Z\",\"iopub.execute_input\":\"2022-04-11T08:27:54.930882Z\",\"shell.execute_reply\":\"2022-04-11T08:27:54.947742Z\"}" trusted="true"}
``` {.python}
# 누락데이터 확인
train_data.info()
```
:::

::: {.cell .code execution="{\"iopub.status.busy\":\"2022-04-11T08:27:54.950293Z\",\"shell.execute_reply.started\":\"2022-04-11T08:27:54.950667Z\",\"iopub.status.idle\":\"2022-04-11T08:27:54.959299Z\",\"iopub.execute_input\":\"2022-04-11T08:27:54.950711Z\",\"shell.execute_reply\":\"2022-04-11T08:27:54.958589Z\"}" trusted="true"}
``` {.python}
train_data[train_data["Sex"]=="female"]["Age"].median()
```
:::

::: {.cell .markdown}
Age, Cabin, Embarked가 누락이 있다(특히, Cabin은 77%가 null이다).
그러므로 Cabin은 무시하고 나머지에 집중하자. Age속성은 19%의 null값을
가지므로 이 null을 어떻게 처리할지 정해야 한다.

null값을 중앙값으로 바꾸는것이 합리적으로 보인다. 다른 속성을 기반으로
나이를 예측하면 좀 더 똑똑해 질 것이다.(예를 들어, 연령의 중앙값은
1급에서 37세, 2급에서 29세, 3급에서 24세이다).

이름 및 티켓 속성은 가치가 있을 수 있지만 모델이 사용할 수 있는 유용한
숫자로 변환하기가 약간 까다로울 것 같으므로, 앞으로 무시한다.하지만
우리는 단순하게 하기 위해 전체 연령의 중앙값을 사용할 것이다.
:::

::: {.cell .markdown}
이름 및 티켓 속성은 가치가 있을 수 있지만 모델이 사용할 수 있는 유용한
숫자로 변환하기가 약간 까다로울 것 같으므로, 앞으로 무시한다.
:::

::: {.cell .code execution="{\"iopub.status.busy\":\"2022-04-11T08:27:54.960717Z\",\"shell.execute_reply.started\":\"2022-04-11T08:27:54.961077Z\",\"iopub.status.idle\":\"2022-04-11T08:27:54.997873Z\",\"iopub.execute_input\":\"2022-04-11T08:27:54.961131Z\",\"shell.execute_reply\":\"2022-04-11T08:27:54.997281Z\"}" trusted="true"}
``` {.python}
train_data.describe()
```
:::

::: {.cell .markdown}
생존률이 40%에 가깝기 때문에 정확도는 모델 평가의 합리적인 척도가 될
것이다.
:::

::: {.cell .code execution="{\"iopub.status.busy\":\"2022-04-11T08:27:55.000255Z\",\"shell.execute_reply.started\":\"2022-04-11T08:27:55.000584Z\",\"iopub.status.idle\":\"2022-04-11T08:27:55.007339Z\",\"iopub.execute_input\":\"2022-04-11T08:27:55.000629Z\",\"shell.execute_reply\":\"2022-04-11T08:27:55.006506Z\"}" trusted="true"}
``` {.python}
# 훈련셋의 타겟 정보 확인
train_data["Survived"].value_counts()
```
:::

::: {.cell .markdown}
범주형 속성을 간단히 살펴보자.
:::

::: {.cell .code execution="{\"iopub.status.busy\":\"2022-04-11T08:27:55.008494Z\",\"shell.execute_reply.started\":\"2022-04-11T08:27:55.008823Z\",\"iopub.status.idle\":\"2022-04-11T08:27:55.028318Z\",\"iopub.execute_input\":\"2022-04-11T08:27:55.008864Z\",\"shell.execute_reply\":\"2022-04-11T08:27:55.027479Z\"}" trusted="true"}
``` {.python}
train_data["Pclass"].value_counts()
```
:::

::: {.cell .code execution="{\"iopub.status.busy\":\"2022-04-11T08:27:55.029544Z\",\"shell.execute_reply.started\":\"2022-04-11T08:27:55.029751Z\",\"iopub.status.idle\":\"2022-04-11T08:27:55.03966Z\",\"iopub.execute_input\":\"2022-04-11T08:27:55.029779Z\",\"shell.execute_reply\":\"2022-04-11T08:27:55.039044Z\"}" trusted="true"}
``` {.python}
train_data["Sex"].value_counts()
```
:::

::: {.cell .code execution="{\"iopub.status.busy\":\"2022-04-11T08:27:55.040693Z\",\"shell.execute_reply.started\":\"2022-04-11T08:27:55.041148Z\",\"iopub.status.idle\":\"2022-04-11T08:27:55.056994Z\",\"iopub.execute_input\":\"2022-04-11T08:27:55.041189Z\",\"shell.execute_reply\":\"2022-04-11T08:27:55.056064Z\"}" trusted="true"}
``` {.python}
# C=Cherbourg, Q=Queenstown, S=Southampton.
train_data["Embarked"].value_counts()
```
:::

::: {.cell .code execution="{\"iopub.status.busy\":\"2022-04-11T08:27:55.058401Z\",\"shell.execute_reply.started\":\"2022-04-11T08:27:55.058687Z\",\"iopub.status.idle\":\"2022-04-11T08:27:56.293459Z\",\"iopub.execute_input\":\"2022-04-11T08:27:55.058721Z\",\"shell.execute_reply\":\"2022-04-11T08:27:56.292386Z\"}" trusted="true"}
``` {.python}
# 숫자형 속성을 처리하는 전처리 파이프라인
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
```
:::

::: {.cell .code execution="{\"iopub.status.busy\":\"2022-04-11T08:27:56.295264Z\",\"shell.execute_reply.started\":\"2022-04-11T08:27:56.295539Z\",\"iopub.status.idle\":\"2022-04-11T08:27:56.302717Z\",\"iopub.execute_input\":\"2022-04-11T08:27:56.295587Z\",\"shell.execute_reply\":\"2022-04-11T08:27:56.301447Z\"}" trusted="true"}
``` {.python}
# 범주형 속성을 처리하는 전처리 파이프라인
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder

cat_pipeline = Pipeline([
        ("ordinal_encoder", OrdinalEncoder()),    
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("cat_encoder", OneHotEncoder(sparse=False)),
    ])
```
:::

::: {.cell .code execution="{\"iopub.status.busy\":\"2022-04-11T08:27:56.30432Z\",\"shell.execute_reply.started\":\"2022-04-11T08:27:56.304942Z\",\"iopub.status.idle\":\"2022-04-11T08:27:56.321615Z\",\"iopub.execute_input\":\"2022-04-11T08:27:56.304989Z\",\"shell.execute_reply\":\"2022-04-11T08:27:56.320527Z\"}" trusted="true"}
``` {.python}
# 두 속성 파이프라인 합치기
from sklearn.compose import ColumnTransformer

num_attribs = ["Age", "SibSp", "Parch", "Fare"]
cat_attribs = ["Pclass", "Sex", "Embarked"]

preprocess_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", cat_pipeline, cat_attribs),
    ])
```
:::

::: {.cell .markdown}
이제 원시 데이터를 가져와 원하는 기계 학습 모델에 제공할 수 있는 수치
입력 기능을 출력하는 전처리 파이프라인이 생성되었다.
:::

::: {.cell .code execution="{\"iopub.status.busy\":\"2022-04-11T08:27:56.323269Z\",\"shell.execute_reply.started\":\"2022-04-11T08:27:56.323492Z\",\"iopub.status.idle\":\"2022-04-11T08:27:56.347336Z\",\"iopub.execute_input\":\"2022-04-11T08:27:56.323523Z\",\"shell.execute_reply\":\"2022-04-11T08:27:56.346458Z\"}" trusted="true"}
``` {.python}
X_train = preprocess_pipeline.fit_transform(train_data)
X_train
```
:::

::: {.cell .code execution="{\"iopub.status.busy\":\"2022-04-11T08:27:56.348546Z\",\"shell.execute_reply.started\":\"2022-04-11T08:27:56.348743Z\",\"iopub.status.idle\":\"2022-04-11T08:27:56.353348Z\",\"iopub.execute_input\":\"2022-04-11T08:27:56.348778Z\",\"shell.execute_reply\":\"2022-04-11T08:27:56.352244Z\"}" trusted="true"}
``` {.python}
y_train = train_data["Survived"]
```
:::

::: {.cell .markdown}
이제 분류기를 훈련시킬 준비가 되었다. RandomForestClassifier부터
시작해보자.
:::

::: {.cell .code execution="{\"iopub.status.busy\":\"2022-04-11T08:27:56.354674Z\",\"shell.execute_reply.started\":\"2022-04-11T08:27:56.355548Z\",\"iopub.status.idle\":\"2022-04-11T08:27:56.680017Z\",\"iopub.execute_input\":\"2022-04-11T08:27:56.355594Z\",\"shell.execute_reply\":\"2022-04-11T08:27:56.679188Z\"}" trusted="true"}
``` {.python}
from sklearn.ensemble import RandomForestClassifier


forest_clf = RandomForestClassifier(n_estimators=100, random_state=42)
forest_clf.fit(X_train, y_train)
```
:::

::: {.cell .code execution="{\"iopub.status.busy\":\"2022-04-11T08:27:56.681324Z\",\"shell.execute_reply.started\":\"2022-04-11T08:27:56.681752Z\",\"iopub.status.idle\":\"2022-04-11T08:27:56.714206Z\",\"iopub.execute_input\":\"2022-04-11T08:27:56.681786Z\",\"shell.execute_reply\":\"2022-04-11T08:27:56.713171Z\"}" trusted="true"}
``` {.python}
# 테스트셋 예측에 활용
X_test = preprocess_pipeline.transform(test_data)
y_pred = forest_clf.predict(X_test)
```
:::

::: {.cell .markdown}
이제 CSV파일을 만들어 케글에 업로드해보자(케글에서 형식 존중). 그전에
교차검증을 통해 모델이 얼마나 좋은지 확인해 보자.
:::

::: {.cell .code execution="{\"iopub.status.busy\":\"2022-04-11T08:27:56.715652Z\",\"shell.execute_reply.started\":\"2022-04-11T08:27:56.715863Z\",\"iopub.status.idle\":\"2022-04-11T08:27:58.956232Z\",\"iopub.execute_input\":\"2022-04-11T08:27:56.715891Z\",\"shell.execute_reply\":\"2022-04-11T08:27:58.955178Z\"}" trusted="true"}
``` {.python}
from sklearn.model_selection import cross_val_score

forest_scores = cross_val_score(forest_clf, X_train, y_train, cv=10)
forest_scores.mean()
```
:::

::: {.cell .code execution="{\"iopub.status.busy\":\"2022-04-11T08:27:58.957752Z\",\"shell.execute_reply.started\":\"2022-04-11T08:27:58.958284Z\",\"iopub.status.idle\":\"2022-04-11T08:27:58.969417Z\",\"iopub.execute_input\":\"2022-04-11T08:27:58.958336Z\",\"shell.execute_reply\":\"2022-04-11T08:27:58.968321Z\"}" trusted="true"}
``` {.python}
# 예측한 결과를 바탕으로 CSV파이 생성하기
# test_data의 PassengerID와 예측결과인 y_pred를 케글에서 필요한 형식으로 변환
predict = pd.DataFrame({
    "PassengerID":test_data_["PassengerId"],
    "Survived":y_pred
})

predict.to_csv('submission.csv', index=False)
```
:::

::: {.cell .code execution="{\"iopub.status.busy\":\"2022-04-11T08:27:58.970756Z\",\"shell.execute_reply.started\":\"2022-04-11T08:27:58.971326Z\",\"iopub.status.idle\":\"2022-04-11T08:27:58.98871Z\",\"iopub.execute_input\":\"2022-04-11T08:27:58.971364Z\",\"shell.execute_reply\":\"2022-04-11T08:27:58.987989Z\"}" trusted="true"}
``` {.python}
#케글에 업로드 할 때 418개의 값이 필요함을 확인.
predict.info()
```
:::

::: {.cell .markdown}
**업로드 결과**
:::

::: {.cell .markdown}
![image.png](vertopal_6097a477950343aface07ec2fa1b7457/1328455a-3d4a-4d2b-8d49-71845476c244.png)
:::

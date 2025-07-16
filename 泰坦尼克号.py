import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 创建示例数据
data = pd.DataFrame([
    [1,0,3,"Braund, Mr. Owen Harris","male",22,1,0,"A/5 21171",7.25,None,"S"],
    [2,1,1,"Cumings, Mrs. John Bradley (Florence Briggs Thayer)","female",38,1,0,"PC 17599",71.2833,"C85","C"],
    [3,1,3,"Heikkinen, Miss. Laina","female",26,0,0,"STON/O2. 3101282",7.925,None,"S"],
    [4,1,1,"Futrelle, Mrs. Jacques Heath (Lily May Peel)","female",35,1,0,"113803",53.1,"C123","S"],
    [5,0,3,"Allen, Mr. William Henry","male",35,0,0,"373450",8.05,None,"S"],
    [6,0,3,"Moran, Mr. James","male",None,0,0,"330877",8.4583,None,"Q"],
    [7,0,1,"McCarthy, Mr. Timothy J","male",54,0,0,"17463",51.8625,"E46","S"],
    [8,0,3,"Palsson, Master. Gosta Leonard","male",2,3,1,"349909",21.075,None,"S"],
    [9,1,3,"Johnson, Mrs. Oscar W (Elisabeth Vilhelmina Berg)","female",27,0,2,"347742",11.1333,None,"S"],
    [10,1,2,"Nasser, Mrs. Nicholas (Adele Achem)","female",14,1,0,"237736",30.0708,None,"C"]
], columns=["PassengerId","Survived","Pclass","Name","Sex","Age","SibSp","Parch","Ticket","Fare","Cabin","Embarked"])

# 特征工程函数
def feature_engineering(df):
    # 1. 头衔提取与合并
    df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    title_map = {'Lady':'Rare','Countess':'Rare','Capt':'Rare','Col':'Rare',
                 'Don':'Rare','Dr':'Rare','Major':'Rare','Rev':'Rare',
                 'Sir':'Rare','Jonkheer':'Rare','Dona':'Rare',
                 'Mlle':'Miss','Ms':'Miss','Mme':'Mrs'}
    df['Title'] = df['Title'].replace(title_map)
    
    # 2. 客舱处理
    df['Deck'] = df['Cabin'].str[0].fillna('X')  # 缺失值用'X'表示
    
    # 3. 家庭特征
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    
    # 4. 票价分箱
    df['FareBin'] = pd.qcut(df['Fare'], 4, labels=False, duplicates='drop')
    
    # 5. 年龄分组与填充
    df['Age'] = df['Age'].fillna(df['Age'].median())
    age_bins = [0, 12, 20, 40, 120]
    df['AgeGroup'] = pd.cut(df['Age'], bins=age_bins, labels=['Child', 'Teen', 'Adult', 'Senior'])
    
    # 6. 登船港口填充
    df['Embarked'] = df['Embarked'].fillna('S')  # 最常见值
    
    # 7. 亲子组合特征
    df['ParentChild'] = ((df['Age'] < 12) & (df['Parch'] > 0)).astype(int)
    
    return df

# 应用特征工程
df = feature_engineering(data)

# 定义特征和目标变量
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 
            'Title', 'Deck', 'FamilySize', 'IsAlone', 'FareBin', 'AgeGroup', 'ParentChild']
X = df[features]
y = df['Survived']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# 预处理管道
categorical_features = ['Sex', 'Embarked', 'Title', 'Deck', 'AgeGroup']
numerical_features = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'FamilySize', 'IsAlone', 'FareBin', 'ParentChild']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ])

# 创建完整管道
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(
        n_estimators=300,
        max_depth=5,
        min_samples_split=5,
        max_features='sqrt',
        class_weight='balanced',
        random_state=42
    ))
])

# 训练模型
pipeline.fit(X_train, y_train)

# 预测与评估
y_pred = pipeline.predict(X_test)

print(f"准确率: {accuracy_score(y_test, y_pred):.2f}")
print("\n分类报告:")
print(classification_report(y_test, y_pred))

# 特征重要性分析
# 获取特征名称
cat_encoder = pipeline.named_steps['preprocessor'].named_transformers_['cat']
cat_features = cat_encoder.get_feature_names_out(categorical_features)
all_features = np.concatenate([numerical_features, cat_features])

# 提取特征重要性
importances = pipeline.named_steps['classifier'].feature_importances_
indices = np.argsort(importances)[::-1]

print("\nTop 5特征重要性:")
for i in range(5):
    print(f"{i+1}. {all_features[indices[i]]}: {importances[indices[i]]:.4f}")

# 混淆矩阵可视化
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['未存活', '存活'], 
            yticklabels=['未存活', '存活'])
plt.xlabel('预测值')
plt.ylabel('真实值')
plt.title('混淆矩阵')
plt.show()

# 特征重要性可视化
plt.figure(figsize=(12, 8))
sns.barplot(x=importances[indices][:10], y=all_features[indices][:10])
plt.title('Top 10特征重要性')
plt.xlabel('重要性分数')
plt.tight_layout()
plt.show()

# 输出处理后的数据集
print("\n处理后的数据集:")
print(df[features + ['Survived']].head())